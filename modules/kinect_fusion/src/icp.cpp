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
    void getAb(const Points oldPts, const Normals oldNrm, const Points newPts, const Normals newNrm,
               cv::Affine3f pose, int level, cv::Matx66f& A, cv::Vec6f& b) const;
};


ICPCPU::ICPCPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }

bool ICPCPU::estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> _oldFrame, cv::Ptr<Frame> _newFrame) const
{
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


void ICPCPU::getAb(const Points oldPts, const Normals oldNrm, const Points newPts, const Normals newNrm,
                   Affine3f pose, int level, Matx66f &A, Vec6f &b) const
{
    Cv32suf s; s.u = 0x7fc00000;
    const float vnan = s.f;
    const p3type qnan3(vnan, vnan, vnan);

    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    //TODO: optimize it to use only 27 elems
    typedef Matx<float, 6, 7> ABtype;
    ABtype sumAB = ABtype::zeros();

    Intr::Projector proj = intrinsics.scale(level).makeProjector();

    float sqDistanceThresh = distanceThreshold*distanceThreshold;
    float minCos = cos(angleThreshold);

    for(int y = 0; y < newPts.rows; y++)
    {
        const p3type* newPtsRow = newPts[y];
        const p3type* newNrmRow = newNrm[y];

        for(int x = 0; x < newPts.cols; x++)
        {
            p3type newP = newPtsRow[x];
            p3type newN = newNrmRow[x];

            p3type oldP(qnan3), oldN(qnan3);

            if(!(isNaN(newP) || isNaN(newN)))
            {
                //transform to old coord system
                newP = pose * newP;
                newN = pose.rotation() * newN;

                //find correspondence
                Point2f oldCoords = proj(newP);
                oldP = bilinear<p3type, Points >(oldPts, oldCoords);
                oldN = bilinear<p3type, Normals>(oldNrm, oldCoords);
            }
            else
            {
                continue;
            }

            if(!(isNaN(oldP) || isNaN(oldN)))
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
                Point3f VxN = newP.cross(oldN);
                Vec<float, 7> ab(VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, oldN.dot(oldP - newP));

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                //TODO: optimize it to use only 27 elems
                ABtype aab = ABtype::zeros();
                for(int i = 0; i < 6; i++)
                {
                    for(int j = i; j < 7; j++)
                    {
                        aab(i, j) = ab(i)*ab(j);
                    }
                }
                //TODO: optimize it to use only 27 elems
                sumAB += aab;
            }
        }
    }

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
