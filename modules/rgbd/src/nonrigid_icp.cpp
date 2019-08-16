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
                                   InputArray vertImage, InputArray oldPoints,
                                   InputArray oldNormals, InputArray newPoints,
                                   InputArray newNormals) const override;

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
                                InputArray _vertImage, InputArray _oldPoints,
                                InputArray _oldNormals, InputArray _newPoints,
                                InputArray _newNormals) const
{
    CV_Assert(_vertImage.isMat());
    CV_Assert(_oldPoints.isMat());
    CV_Assert(_newPoints.isMat());
    CV_Assert(_newNormals.isMat());

    Mat vertImage = _vertImage.getMat();
    Mat oldPoints = _oldPoints.getMat();
    Mat newPoints = _newPoints.getMat();
    Mat newNormals = _newNormals.getMat();
    Mat oldNormals = _oldNormals.getMat();

    CV_Assert(!vertImage.empty());
    CV_Assert(!oldPoints.empty());
    CV_Assert(!newPoints.empty());
    CV_Assert(!newNormals.empty());

    const NodeVectorType& warpNodes = currentWarp.getNodes();

    Affine3f T_lw = pose.inv() * volume->pose;
    std::vector<Matx66f> A(warpNodes.size(), Matx66f::all(0));
    std::vector<Matx61f> b(warpNodes.size(), Matx61f::all(0));

    std::vector<float> residuals;

    Mat Vg(oldPoints.size(), CV_32FC3, nan3);

    Mat Vc(oldPoints.size(), CV_32FC3, nan3);
    Mat Nc(oldPoints.size(), CV_32FC3, nan3);
    cv::kinfu::Intr::Projector proj = intrinsics.makeProjector();

    for (int y = 0; y < oldPoints.size().height; y++)
    {
        for (int x = 0; x < oldPoints.size().width; x++)
        {
            // Obtain correspondence by projecting Tu_Vg
            Vec3f curV = oldPoints.at<Vec3f>(y, x);
            if (curV == Vec3f::all(0) || cvIsNaN(curV[0]) || cvIsNaN(curV[1]) || cvIsNaN(curV[2]))
                continue;

            Point2f newCoords = proj(oldPoints.at<Point3f>(y, x));
            if(!(newCoords.x >= 0 && newCoords.x < newPoints.cols - 1 &&
                 newCoords.y >= 0 && newCoords.y < newPoints.rows - 1))
                continue;

            // TODO: interpolate Vg instead of simply converting projected coords to int
            Vg.at<Vec3f>(y, x) = vertImage.at<Vec3f>((int)newCoords.y, (int)newCoords.x);

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

            const ptype* nrow0 = newNormals.ptr<ptype>(yi+0);
            const ptype* nrow1 = newNormals.ptr<ptype>(yi+1);

            Point3f n00 = fromPtype(nrow0[xi+0]);
            Point3f n01 = fromPtype(nrow0[xi+1]);
            Point3f n10 = fromPtype(nrow1[xi+0]);
            Point3f n11 = fromPtype(nrow1[xi+1]);

            if(!(fastCheck(n00) && fastCheck(n01) &&
                fastCheck(n10) && fastCheck(n11)))
                continue;

            Point3f n0 = n00 + tx*(n01 - n00);
            Point3f n1 = n10 + tx*(n11 - n10);
            Point3f newN = n0 + ty*(n1 - n0);

            Vc.at<Point3f>(y, x) = newP;
            Nc.at<Point3f>(y, x) = newN;

            Vec3f diff = oldPoints.at<Vec3f>(y, x) - Vec3f(newP);
            if(diff.dot(diff) > 0.0004f) continue;
            if(abs(newN.dot(oldNormals.at<Point3f>(y, x))) < cos((float)CV_PI / 2)) continue;

            float rd = newN.dot(diff);


            residuals.push_back(rd);

        }
    }

    float med = median(residuals);
    std::for_each(residuals.begin(), residuals.end(), [med](float& x){x =  std::abs(x-med);});
    std::cout << "median: " << med << " from " << residuals.size() << " residuals " << std::endl;
    float sigma = MAD_SCALE * median(residuals);

    float total_error = 0;
    int pix_count = 0;

    for(int y = 0; y < oldPoints.size().height; y++)
    {
        for(int x = 0; x < oldPoints.size().width; x++)
        {
            Vec3f curV = oldPoints.at<Vec3f>(y, x);
            if (curV == Vec3f::all(0) || cvIsNaN(curV[0]))
                continue;

            Vec3f V = Vg.at<Vec3f>(y, x);
            if (V == Vec3f::all(0) || cvIsNaN(V[0]))
                continue;

            V[0] *= volume->volResolution.x;
            V[1] *= volume->volResolution.y;
            V[2] *= volume->volResolution.z;

            if(!fastCheck(Vc.at<Point3f>(y, x)))
                continue;

            if(!fastCheck(Nc.at<Point3f>(y, x)))
                continue;

            Point3i p(V[0], V[1], V[2]);
            Vec3f diff = oldPoints.at<Vec3f>(y, x) - Vc.at<Vec3f>(y, x);

            float rd = Nc.at<Vec3f>(y, x).dot(diff);

            total_error += tukeyWeight(rd, sigma) * rd * rd;
            pix_count++;

            int n;
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
            float totalNeighbourWeight = 0.f;
            float neighWeights[DYNAFU_MAX_NEIGHBOURS];
            for (int i = 0; i < n; i++)
            {
                int neigh = neighbours[i];
                neighWeights[i] = warpNodes[neigh]->weight(Point3f(V)*volume->voxelSize);

                totalNeighbourWeight += neighWeights[i];
            }

            if(totalNeighbourWeight < 1e-5) continue;

            for (int i = 0; i < n; i++)
            {
                if(neighWeights[i] < 0.01) continue;
                int neigh = neighbours[i];

                Vec3f Tj_Vg_Vj = (warpNodes[neigh]->transform *
                                 (Point3f(V)*volume->voxelSize - warpNodes[neigh]->pos));

                Matx33f Tj_Vg_Vj_x(0, -Tj_Vg_Vj[2], Tj_Vg_Vj[1],
                                   Tj_Vg_Vj[2], 0, -Tj_Vg_Vj[0],
                                   -Tj_Vg_Vj[1], Tj_Vg_Vj[0], 0);

                Vec3f v1 = (Tj_Vg_Vj_x * T_lw.rotation().t()) * Nc.at<Vec3f>(y, x);
                Vec3f v2 = T_lw.rotation().t() * Nc.at<Vec3f>(y, x);

                Matx61f J_dataT(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                Matx16f J_data = J_dataT.t();
                Matx66f H_data = J_dataT * J_data;

                float w = (neighWeights[i] / totalNeighbourWeight);

                float robustWeight = tukeyWeight(rd, sigma);
                A[neigh] += robustWeight * w * w * H_data;

                b[neigh] += -robustWeight * rd * w * J_dataT;
            }

        }
    }

    // solve Ax = b for each warp node and update its transform
    float total_det = 0;
    int count = 0, nanCount = 0;
    for(size_t i = 0; i < warpNodes.size(); i++)
    {
        double det = cv::determinant(A[i]);

        if (abs (det) < 1e-5 || cvIsNaN(det))
        {
            if(cvIsNaN(det)) nanCount++;
            continue;
        }
        total_det += abs(det);
        count++;

        Vec6f nodeTwist;
        solve(A[i], b[i], nodeTwist, DECOMP_SVD);

        Affine3f tinc(Vec3f(nodeTwist.val), Vec3f(nodeTwist.val+3));
        warpNodes[i]->transform = warpNodes[i]->transform * tinc;
    }

    std::cout << "Sigma: " << sigma << std::endl;
    std::cout << "Avg det: " << total_det/count << " from "  << count << std::endl;
    std::cout << "Total energy: " << total_error << " from " << pix_count << " pixels";
    std::cout << "(Average: " << total_error/pix_count << ")" << std::endl;
    std::cout << "Nan count: " << nanCount << "/" << warpNodes.size() << "\n" << std::endl;;

    return true;
}

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations)
{
    return makePtr<ICPImpl>(_intrinsics, _volume, _iterations);
}

} // namespace dynafu
} // namespace cv
