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

///////// CPU implementation /////////

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

typedef Matx<float, 6, 7> ABtype;

struct GetAbInvoker : ParallelLoopBody
{
    enum
    {
        UTSIZE = 27
    };

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
        float upperTriangle[UTSIZE];
        for(int i = 0; i < UTSIZE; i++)
            upperTriangle[i] = 0;

        for(int y = range.start; y < range.end; y++)
        {
            const Point3f* newPtsRow = newPts[y];
            const Point3f* newNrmRow = newNrm[y];

            for(int x = 0; x < newPts.cols; x++)
            {
                Point3f newP = newPtsRow[x];
                Point3f newN = newNrmRow[x];

                Point3f oldP(nan3), oldN(nan3);

                if(!(fastCheck(newP) && fastCheck(newN)))
                    continue;

                //transform to old coord system
                newP = pose * newP;
                newN = pose.rotation() * newN;

                //find correspondence
                Point2f oldCoords = proj(newP);
                if(!(oldCoords.x >= 0 && oldCoords.x < oldPts.cols - 1 &&
                     oldCoords.y >= 0 && oldCoords.y < oldPts.rows - 1))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                int xi = cvFloor(oldCoords.x), yi = cvFloor(oldCoords.y);
                float tx  = oldCoords.x - xi, ty = oldCoords.y - yi;
                float tx1 = 1.f-tx, ty1 = 1.f-ty;
                float t00 = tx1*ty1, t01 = tx*ty1, t10 = tx1*ty, t11 = tx*ty;

                const Point3f* prow0 = oldPts[yi+0];
                const Point3f* prow1 = oldPts[yi+1];

                Point3f p00 = prow0[xi+0];
                Point3f p01 = prow0[xi+1];
                Point3f p10 = prow1[xi+0];
                Point3f p11 = prow1[xi+1];

                //do not fix missing data
                if(!(fastCheck(p00) && fastCheck(p01) &&
                     fastCheck(p10) && fastCheck(p11)))
                    continue;

                const Point3f* nrow0 = oldNrm[yi+0];
                const Point3f* nrow1 = oldNrm[yi+1];

                Point3f n00 = nrow0[xi+0];
                Point3f n01 = nrow0[xi+1];
                Point3f n10 = nrow1[xi+0];
                Point3f n11 = nrow1[xi+1];

                if(!(fastCheck(n00) && fastCheck(n01) &&
                     fastCheck(n10) && fastCheck(n11)))
                    continue;

                oldP = p00*t00 + p01*t01 + p10*t10 + p11*t11;
                oldN = n00*t00 + n01*t01 + n10*t10 + n11*t11;

                if(!(fastCheck(oldP) && fastCheck(oldN)))
                    continue;

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
                float ut[UTSIZE];
                int pos = 0;
                for(int i = 0; i < 6; i++)
                {
                    for(int j = i; j < 7; j++)
                    {
                        ut[pos++] = ab[i]*ab[j];
                    }
                }
                // gather sum
                for(int i = 0; i < UTSIZE; i++)
                    upperTriangle[i] += ut[i];
            }
        }

        ABtype sumAB = ABtype::zeros();
        int pos = 0;
        for(int i = 0; i < 6; i++)
        {
            for(int j = i; j < 7; j++)
            {
                sumAB(i, j) = upperTriangle[pos++];
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
