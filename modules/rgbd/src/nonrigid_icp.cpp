// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <algorithm>
#include "precomp.hpp"
#include "nonrigid_icp.hpp"

#define MAD_SCALE 1.4826
#define TUKEY_B 4.6851

namespace cv
{
namespace dynafu
{
using namespace kinfu;

NonRigidICP::NonRigidICP(const Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations) :
iterations(_iterations), volume(_volume), intrinsics(_intrinsics)
{}

class ICPImpl : public NonRigidICP
{
public:
    ICPImpl(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations);

    virtual bool estimateWarpNodes(WarpField& currentWarp, const Affine3f &pose,
                                   InputArray vertImage, InputArray normImage,
                                   InputArray newPoints) const override;

    virtual ~ICPImpl() {}

private:
    float median(std::vector<float>& v) const;
    float tukeyWeight(float x, float sigma) const;
};

ICPImpl::ICPImpl(const Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations) :
NonRigidICP(_intrinsics, _volume, _iterations)
{}

float ICPImpl::median(std::vector<float>& v) const
{
    size_t n = v.size()/2;
    std::nth_element(v.begin(), v.end()+n, v.end());
    float vn = v[n];

    if(n%2 == 0)
    {
        std::nth_element(v.begin(), v.end()+n-1, v.end());
        return (vn+v[n-1])/2.f;
    }
    else
    {
        return vn;
    }
}

float ICPImpl::tukeyWeight(float r, float sigma) const
{
    float x = r/sigma;
    if(std::abs(x) < TUKEY_B)
    {
        float y = 1 - (x*x)/(TUKEY_B * TUKEY_B);
        return y*y;

    } else return 0;
}

bool ICPImpl::estimateWarpNodes(WarpField& currentWarp, const Affine3f &pose,
                                InputArray _vertImage, InputArray _normImage,
                                InputArray _newPoints) const
{
    CV_Assert(_vertImage.isMat());
    CV_Assert(_normImage.isMat());
    CV_Assert(_newPoints.isMat());

    Mat vertImage = _vertImage.getMat();
    Mat normImage = _normImage.getMat();
    Mat newPoints = _newPoints.getMat();

    CV_Assert(!vertImage.empty() && !normImage.empty());
    CV_Assert(vertImage.size() == normImage.size());

    NodeVectorType warpNodes = currentWarp.getNodes();

    std::vector<Matx66f> A(warpNodes.size(), Matx66f::all(0));
    std::vector<Matx61f> b(warpNodes.size(), Matx61f::all(0));

    std::vector<float> residuals;

    Mat Tu_Vg(vertImage.size(), CV_32FC3);
    Mat Tu_Ng(vertImage.size(), CV_32FC3);

    for (int y = 0; y < vertImage.size().height; y++)
    {
        for (int x = 0; x < vertImage.size().width; x++)
        {
            Vec3f Vg = vertImage.at<Vec3f>(y, x);

            if (Vg == Vec3f::all(0))
                continue;

            Vg[0] *= volume->volSize.x;
            Vg[1] *= volume->volSize.y;
            Vg[2] *= volume->volSize.z;

            Vec3f Ng = normImage.at<Vec3f>(y, x) * 2 - Vec3f(1, 1, 1);

            Vec3f Vc = fromPtype(newPoints.at<ptype>(y, x));

            Point3i p(Vg[0], Vg[1], Vg[2]);
            int n;
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
            Tu_Vg.at<Vec3f>(y, x) = pose * currentWarp.applyWarp(p, neighbours, n);

            Tu_Ng.at<Vec3f>(y, x) = pose.rotation() * currentWarp.applyWarp(Ng, neighbours, n, true);

            Vec3f diff = Tu_Vg.at<Vec3f>(y, x) - Vc;
            float rd = Tu_Vg.at<Vec3f>(y, x).dot(diff);

            residuals.push_back(rd);

        }
    }

    float med = median(residuals);
    std::for_each(residuals.begin(), residuals.end(), [med](float& x){x =  std::abs(x-med);});
    float sigma = MAD_SCALE * median(residuals);

    for(int y = 0; y < vertImage.size().height; y++)
    {
        for(int x = 0; x < vertImage.size().width; x++)
        {
            Vec3f Vg = vertImage.at<Vec3f>(y, x);

            if (Vg == Vec3f::all(0))
                continue;

            Vg[0] *= volume->volSize.x;
            Vg[1] *= volume->volSize.y;
            Vg[2] *= volume->volSize.z;

            Vec3f Ng = normImage.at<Vec3f>(y, x) * 2 - Vec3f(1, 1, 1);
            Vec3f Vc = fromPtype(newPoints.at<ptype>(y, x));

            Point3i p(Vg[0], Vg[1], Vg[2]);
            int n;
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
            Vec3f diff = Tu_Vg.at<Vec3f>(y, x) - Vc;

            float rd = Tu_Vg.at<Vec3f>(y, x).dot(diff);

            float totalNeighbourWeight = 0.f;
            float neighWeights[DYNAFU_MAX_NEIGHBOURS];
            for (const int i : neighbours)
            {
                neighWeights[i] = warpNodes[i]->weight(Vg);
                totalNeighbourWeight += neighWeights[i];

                Vec3f Tj_Vg = pose * (warpNodes[i]->transform * Point3f(Vg));
                Vec3f Tj_Ng = pose.rotation() * (warpNodes[i]->transform.rotation() * Point3f(Ng));

                Vec3f v1 = Tj_Ng.cross(diff) + Tj_Vg.cross(Tu_Ng.at<Vec3f>(y, x));
                Vec3f v2 = diff + Tu_Ng.at<Vec3f>(y, x);

                Matx16f J_data(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                Matx66f H_data = J_data.t() * J_data;

                float w = (neighWeights[i] / totalNeighbourWeight);

                float robustWeight = tukeyWeight(rd, sigma);
                A[i] += robustWeight * w * w * H_data;

                b[i] += -robustWeight * rd * w * J_data.t();
            }

        }
    }

    return true;
}

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations)
{
    return makePtr<ICPImpl>(_intrinsics, _volume, _iterations);
}

} // namespace dynafu
} // namespace cv
