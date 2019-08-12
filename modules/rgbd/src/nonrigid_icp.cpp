// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <algorithm>
#include "precomp.hpp"
#include "nonrigid_icp.hpp"

#define MAD_SCALE 1.4826f
#define TUKEY_B 4.6851f

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

static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}

float ICPImpl::median(std::vector<float>& v) const
{
    size_t n = v.size()/2;
    if(n == 0) return 0;

    std::nth_element(v.begin(), v.begin()+n, v.end());
    float vn = v[n];

    if(n%2 == 0)
    {
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
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
    if(std::abs(x) <= TUKEY_B)
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

    const NodeVectorType& warpNodes = currentWarp.getNodes();

    Affine3f T_lw = pose.inv();
    std::vector<Matx66f> A(warpNodes.size(), Matx66f::all(0));
    std::vector<Matx61f> b(warpNodes.size(), Matx61f::all(0));

    std::vector<float> residuals;

    Mat Tu_Vg(vertImage.size(), CV_32FC3);
    Mat Tu_Ng(vertImage.size(), CV_32FC3);

    Mat Vc(vertImage.size(), CV_32FC3);

    for (int y = 0; y < vertImage.size().height; y++)
    {
        for (int x = 0; x < vertImage.size().width; x++)
        {
            Vec3f Vg = vertImage.at<Vec3f>(y, x);
            Vc.at<Vec3f>(y, x) = nan3;

            if (Vg == Vec3f::all(0) || cvIsNaN(Vg[0]))
                continue;

            Vg[0] *= volume->volResolution.x;
            Vg[1] *= volume->volResolution.y;
            Vg[2] *= volume->volResolution.z;

            Vec3f Ng = normImage.at<Vec3f>(y, x) * 2 - Vec3f(1, 1, 1);

            Point3i p(Vg[0], Vg[1], Vg[2]);
            int n;
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);

            // TODO: Tu_Vg and Tu_Ng don't need to be calculated and can be
            // obtained by calling makeFrameFromDepth on the predicted depth
            Tu_Vg.at<Vec3f>(y, x) = T_lw * volume->pose *
                                    currentWarp.applyWarp(p*volume->voxelSize, neighbours, n);

            Tu_Ng.at<Vec3f>(y, x) = T_lw.rotation() * volume->pose.rotation() *
                                    currentWarp.applyWarp(Ng, neighbours, n, true);

            // Obtain correspondence by projecting Tu_Vg
            cv::kinfu::Intr::Projector proj = intrinsics.makeProjector();

            Point2f newCoords = proj(Tu_Vg.at<Point3f>(y, x));

            if(!(newCoords.x >= 0 && newCoords.x < newPoints.cols - 1 &&
                 newCoords.y >= 0 && newCoords.y < newPoints.rows - 1))
                continue;

            // bilinearly interpolate newPoints under newCoords point
            int xi = cvFloor(newCoords.x), yi = cvFloor(newCoords.y);
            float tx  = newCoords.x - xi, ty = newCoords.y - yi;

            const ptype* prow0 = newPoints.ptr<ptype>(yi+0);
            const ptype* prow1 = newPoints.ptr<ptype>(yi+1);

            Point3f p00 = fromPtype(prow0[xi+0]);
            Point3f p01 = fromPtype(prow0[xi+1]);
            Point3f p10 = fromPtype(prow1[xi+0]);
            Point3f p11 = fromPtype(prow1[xi+1]);

            //do not fix missing data
            if(!(fastCheck(p00) && fastCheck(p01) &&
                fastCheck(p10) && fastCheck(p11)))
                continue;

            Point3f p0 = p00 + tx*(p01 - p00);
            Point3f p1 = p10 + tx*(p11 - p10);
            Point3f newP = (p0 + ty*(p1 - p0));

            Vec3f diff = Tu_Vg.at<Vec3f>(y, x) - Vec3f(newP);

            Vc.at<Point3f>(y, x) = newP;

            float rd = Tu_Vg.at<Vec3f>(y, x).dot(diff);

            residuals.push_back(rd);

        }
    }

    float med = median(residuals);
    std::for_each(residuals.begin(), residuals.end(), [med](float& x){x =  std::abs(x-med);});
    std::cout << "median: " << med << " from " << residuals.size() << " residuals " << std::endl;
    float sigma = MAD_SCALE * median(residuals);

    float total_error = 0;
    int pix_count = 0;

    for(int y = 0; y < vertImage.size().height; y++)
    {
        for(int x = 0; x < vertImage.size().width; x++)
        {
            Vec3f Vg = vertImage.at<Vec3f>(y, x);

            if (Vg == Vec3f::all(0) || cvIsNaN(Vg[0]))
                continue;

            if(!fastCheck(Vc.at<Point3f>(y, x)))
                continue;

            Vg[0] *= volume->volResolution.x;
            Vg[1] *= volume->volResolution.y;
            Vg[2] *= volume->volResolution.z;

            Vec3f Ng = normImage.at<Vec3f>(y, x) * 2 - Vec3f(1, 1, 1);

            Point3i p(Vg[0], Vg[1], Vg[2]);
            int n;
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
            Vec3f diff = Tu_Vg.at<Vec3f>(y, x) - Vc.at<Vec3f>(y, x);

            float rd = Tu_Ng.at<Vec3f>(y, x).dot(diff);
            total_error += tukeyWeight(rd, sigma) * rd * rd;
            pix_count++;

            float totalNeighbourWeight = 0.f;
            float neighWeights[DYNAFU_MAX_NEIGHBOURS];
            for (int i = 0; i < n; i++)
            {
                int neigh = neighbours[i];
                neighWeights[i] = warpNodes[neigh]->weight(Point3f(Vg)*volume->voxelSize);
                if(neighWeights[i] < 0.01) continue; // TODO: remove this line

                totalNeighbourWeight += neighWeights[i];
            }

            if(totalNeighbourWeight == 0) continue;

            for (int i = 0; i < n; i++)
            {
                if(neighWeights[i] < 0.01) continue;
                int neigh = neighbours[i];

                Vec3f Tj_Vg_Vj = volume->pose * (warpNodes[neigh]->transform *
                                             (Point3f(Vg)*volume->voxelSize - warpNodes[neigh]->pos));
                Vec3f Tj_Ng = volume->pose.rotation() *
                              (warpNodes[neigh]->transform.rotation() * Point3f(Ng));

                Vec3f v1 = Tu_Ng.at<Vec3f>(y, x);


                Matx33f Tj_Ng_x(0, -Tj_Ng[2], Tj_Ng[1],
                                Tj_Ng[2], 0, -Tj_Ng[0],
                                -Tj_Ng[1], Tj_Ng[0], 0);

                Matx33f Tj_Vg_Vj_x(0, -Tj_Vg_Vj[2], Tj_Vg_Vj[1],
                                   Tj_Vg_Vj[2], 0, -Tj_Vg_Vj[0],
                                   -Tj_Vg_Vj[1], Tj_Vg_Vj[0], 0);

                Vec3f v2 = Tj_Ng_x * T_lw.rotation().t() * diff + Tj_Vg_Vj_x * T_lw.rotation().t() * v1;

                v1 = T_lw.rotation().t() * v1;

                Matx16f J_data(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                Matx66f H_data = J_data.t() * J_data;

                float w = (neighWeights[i] / totalNeighbourWeight);

                float robustWeight = tukeyWeight(rd, sigma);
                A[neigh] += robustWeight * w * w * H_data;

                b[neigh] += -robustWeight * rd * w * J_data.t();
            }

        }
    }

    // solve Ax = b for each warp node and update its transform
    float total_det = 0;
    int count = 0, nanCount = 0;
    for(size_t i = 0; i < warpNodes.size(); i++)
    {
        double det = cv::determinant(A[i]);

        if (abs (det) < 1e-15 || cvIsNaN(det))
        {
            if(cvIsNaN(det)) nanCount++;
            continue;
        }
        total_det += abs(det);
        count++;

        Vec6f nodeTwist;
        solve(A[i], b[i], nodeTwist, DECOMP_SVD);
        Affine3f tinc(expSE3(nodeTwist));
        warpNodes[i]->transform = tinc * warpNodes[i]->transform;
    }

    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Avg det: " << total_det/count << " from "  << count << std::endl;
    std::cout << "Total energy: " << total_error << " from " << pix_count << " pixels";
    std::cout << "(Average: " << total_error/pix_count << ")" << std::endl;
    std::cout << "Nan count: " << nanCount << "/" << warpNodes.size() << "\n" << std::endl;;

    return true;
}

Affine3f NonRigidICP::expSE3(const Vec6f& v) const
{
    const Vec3f w(v[3], v[4], v[5]);

    Matx33f  wx(
        0, -v[5], v[4],
        v[5], 0, -v[3],
        -v[4], v[3], 0);

    float theta = norm(w);

    float coeff1 = (1-cos(theta))/(theta*theta);
    float coeff2 = (theta-sin(theta))/(theta*theta*theta);
    Matx33f V = Matx33f::eye() + coeff1 * wx + coeff2 * wx * wx;

    const Vec3f t = V*Vec3f(v[0], v[1], v[2]);
    float fac = volume->voxelSize;
    return Affine3f(w, t*fac);
}

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations)
{
    return makePtr<ICPImpl>(_intrinsics, _volume, _iterations);
}

} // namespace dynafu
} // namespace cv
