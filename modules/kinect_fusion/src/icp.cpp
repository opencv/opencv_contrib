//TODO: add license

#include "precomp.hpp"
#include "icp.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace std;

ICP::ICP(const Intr _intrinsics, const std::vector<int>& _iterations, float _angleThreshold, float _distanceThreshold) :
    iterations(_iterations), angleThreshold(_angleThreshold), distanceThreshold(_distanceThreshold),
    intrinsics(_intrinsics)
{ }

///////// GPU implementation /////////

class ICPCPU : public ICP
{
public:
    ICPCPU(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const;

    virtual ~ICPCPU() { }

private:
    void getAb(const Points &oldPts, const Normals &oldNrm, const Points &newPts, const Normals &newNrm,
               cv::Affine3f pose, int level, cv::Matx66f& A, cv::Vec6f& b) const;
};


ICPCPU::ICPCPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }

bool ICPCPU::estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> _oldFrame, cv::Ptr<Frame> _newFrame) const
{
    ScopeTime st("icp");

    cv::Ptr<FrameCPU> oldFrame = _oldFrame.dynamicCast<FrameCPU>();
    cv::Ptr<FrameCPU> newFrame = _newFrame.dynamicCast<FrameCPU>();

    const std::vector<Points>& oldPoints   = oldFrame->points;
    const std::vector<Normals>& oldNormals = oldFrame->normals;
    const std::vector<Points>& newPoints  = newFrame->points;
    const std::vector<Normals>& newNormals = newFrame->normals;

    transform = Affine3f::Identity();
    for(int level = iterations.size() - 1; level >= 0; level--)
    {
        Points  oldPts = oldPoints [level], newPts = newPoints [level];
        Normals oldNrm = oldNormals[level], newNrm = newNormals[level];

        for(int iter = 0; iter < iterations[level]; iter++)
        {
            Matx66f A;
            Vec6f b;

            getAb(oldPts, oldNrm, newPts, newNrm, transform, level, A, b);

            double det = cv::determinant(A);

            if (abs (det) < 1e-15 || cvIsNaN(det))
                return false;

            Vec6f x;
            // theoretically, any method of solving is applicable
            // since there are usual least square matrices
            solve(A, b, x, DECOMP_SVD);
            Affine3f tinc(Vec3f(x.val), Vec3f(x.val+3));
            transform = tinc * transform;
        }
    }

    return true;
}

static inline bool fastCheck(const Point3f& p)
{
    // 1 coord to check is enough since we know the generation
    return !cvIsNaN(p.x);
}

static inline Point3f bilinear(const Points& m, cv::Point2f pt)
{
    if(pt.x < 0 || pt.x >= m.cols-1 ||
       pt.y < 0 || pt.y >= m.rows-1)
        return nan3;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

    const Point3f* row0 = m[yi+0];
    const Point3f* row1 = m[yi+1];

    Point3f v00 = row0[xi+0];
    Point3f v01 = row0[xi+1];
    Point3f v10 = row1[xi+0];
    Point3f v11 = row1[xi+1];

    //do not fix missing data
    if(fastCheck(v00) && fastCheck(v01) &&
       fastCheck(v10) && fastCheck(v11))
    {
        float tx  = pt.x - xi, ty = pt.y - yi;

        //TODO: check speed
//        float tx1 = 1.f-tx, ty1 = 1.f-ty;
//        return v00*tx1*ty1 + v01*tx*ty1 + v10*tx1*ty + v11*tx*ty;

        float txty = tx*ty;
        Point3f d001 = v00 - v01;
        return v00 + tx*d001 + ty*(v10-v00) + txty*(d001 - v10 + v11);
    }
    else
    {
        return nan3;
    }
}

//TODO: optimize it to use only 27 elems
typedef Matx<float, 6, 7> ABtype;

struct GetAbInvoker : ParallelLoopBody
{
    GetAbInvoker(ABtype& _globalAb, Mutex& _mtx,
                 const Points& _oldPts, const Normals& _oldNrm, const Points& _newPts, const Normals& _newNrm,
                 Affine3f _pose, Intr::Projector _proj, float _sqDistanceThresh, float _minCos) :
        ParallelLoopBody(),
        globalSumAb(_globalAb), mtx(_mtx),
        oldPts(_oldPts), oldNrm(_oldNrm), newPts(_newPts), newNrm(_newNrm), pose(_pose),
        proj(_proj), sqDistanceThresh(_sqDistanceThresh), minCos(_minCos)
    { }

    virtual void operator ()(const Range& range) const
    {
        ABtype sumAB = ABtype::zeros();

        for(int y = range.start; y < range.end; y++)
        {
            const Point3f* newPtsRow = newPts[y];
            const Point3f* newNrmRow = newNrm[y];

            for(int x = 0; x < newPts.cols; x++)
            {
                Point3f newP = newPtsRow[x];
                Point3f newN = newNrmRow[x];

                Point3f oldP(nan3), oldN(nan3);

                //if(!(isNaN(newP) || isNaN(newN)))
                if(fastCheck(newP) && fastCheck(newN))
                {
                    //transform to old coord system
                    newP = pose * newP;
                    newN = pose.rotation() * newN;

                    //find correspondence
                    Point2f oldCoords = proj(newP);
                    oldP = bilinear(oldPts, oldCoords);
                    oldN = bilinear(oldNrm, oldCoords);
                }
                else
                {
                    continue;
                }

                if(fastCheck(oldP) && fastCheck(oldN))
                {
                    //filter by distance
                    if((newP - oldP).dot(newP - oldP) > sqDistanceThresh)
                    {
                        continue;
                    }

                    //filter by angle
                    if(abs(newN.dot(oldN)) < minCos)
                    {
                        continue;
                    }

                    // build point-wise vector ab = [ A | b ]

                    //try to optimize
                    Point3f VxN = newP.cross(oldN);
                    float ab[7] = {VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, oldN.dot(oldP - newP)};

                    // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                    // which is [A^T*A | A^T*b]
                    //TODO: optimize it to use only 27 elems
                    ABtype aab = ABtype::zeros();
                    for(int i = 0; i < 6; i++)
                    {
                        for(int j = i; j < 7; j++)
                        {
                            aab(i, j) = ab[i]*ab[j];
                        }
                    }
                    //TODO: optimize it to use only 27 elems
                    sumAB += aab;
                }
            }
        }

        AutoLock al(mtx);
        globalSumAb += sumAB;
    }

    ABtype& globalSumAb;
    Mutex& mtx;
    const Points& oldPts;
    const Normals& oldNrm;
    const Points& newPts;
    const Normals& newNrm;
    Affine3f pose;
    const Intr::Projector proj;
    float sqDistanceThresh;
    float minCos;
};


void ICPCPU::getAb(const Points& oldPts, const Normals& oldNrm, const Points& newPts, const Normals& newNrm,
                   Affine3f pose, int level, Matx66f &A, Vec6f &b) const
{
    ScopeTime st("icp: get ab", false);

    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    //TODO: optimize it to use only 27 elems
    ABtype sumAB = ABtype::zeros();

    Mutex mutex;
    GetAbInvoker invoker(sumAB, mutex, oldPts, oldNrm, newPts, newNrm, pose,
                         intrinsics.scale(level).makeProjector(),
                         distanceThreshold*distanceThreshold, cos(angleThreshold));
    Range range(0, newPts.rows);
    parallel_for_(range, invoker);

    // splitting AB matrix to A and b
    for(int i = 0; i < 6; i++)
    {
        // augment lower triangle of A by symmetry
        for(int j = i; j < 6; j++)
        {
            A(i, j) = A(j, i) = sumAB(i, j);
        }

        b(i) = sumAB(i, 6);
    }
}

///////// GPU implementation /////////

class ICPGPU : public ICP
{
public:
    ICPGPU(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const;

    virtual ~ICPGPU() { }
};

ICPGPU::ICPGPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }


bool ICPGPU::estimateTransform(cv::Affine3f& /*transform*/, cv::Ptr<Frame> /*_oldFrame*/, cv::Ptr<Frame> /*newFrame*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<ICP> makeICP(cv::kinfu::KinFu::KinFuParams::PlatformType t,
                     const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold)
{
    switch (t)
    {
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_CPU:
        return cv::makePtr<ICPCPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_GPU:
        return cv::makePtr<ICPGPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    default:
        return cv::Ptr<ICP>();
    }
}
