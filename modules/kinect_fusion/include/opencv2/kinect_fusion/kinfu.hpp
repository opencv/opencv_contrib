//TODO: correct license

//TODO: maybe pragma once?

#ifndef __OPENCV_KINECT_FUSION_HPP__
#define __OPENCV_KINECT_FUSION_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv {
namespace kinfu {
//! @addtogroup kinect_fusion
//! @{

//TODO: document it properly

// Camera intrinsics
struct Intr
{
    struct Reprojector
    {
        Reprojector() {}
        inline Reprojector(Intr intr)
        {
            fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;
            cx = intr.cx, cy = intr.cy;
        }
        template<typename T>
        inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
        {
            T x = p.z * (p.x - cx) * fxinv;
            T y = p.z * (p.y - cy) * fyinv;
            return cv::Point3_<T>(x, y, p.z);
        }

        float fxinv, fyinv, cx, cy;
    };
    struct Projector
    {
        inline Projector(Intr intr) : fx(intr.fx), fy(intr.fy), cx(intr.cx), cy(intr.cy) { }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p) const
        {
            T x = fx*(p.x/p.z) + cx;
            T y = fy*(p.y/p.z) + cy;
            return cv::Point_<T>(x, y);
        }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p, cv::Point3_<T>& pixVec) const
        {
            pixVec = cv::Point3_<T>(p.x/p.z, p.y/p.z, 1);
            T x = fx*pixVec.x + cx;
            T y = fy*pixVec.y + cy;
            return cv::Point_<T>(x, y);
        }
        float fx, fy, cx, cy;
    };
    Intr() : fx(), fy(), cx(), cy() { }
    Intr(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy) { }
    // scale intrinsics to pyramid level
    inline Intr scale(int pyr) const
    {
        float factor = (1.f /(1 << pyr));
        return Intr(fx*factor, fy*factor, cx*factor, cy*factor);
    }
    inline Reprojector makeReprojector() const { return Reprojector(*this); }
    inline Projector   makeProjector()   const { return Projector(*this);   }

    float fx, fy, cx, cy;
};

class CV_EXPORTS KinFu
{
public:
    struct KinFuParams
    {
        static KinFuParams defaultParams();

        // frame size in pixels
        Size frameSize;

        // camera intrinsics
        Intr intr;

        // pre-scale for input values
        // 5000 per 1 meter for the 16-bit PNG files of TUM database
        // 1 per 1 meter for the 32-bit float images in the ROS bag files
        float depthFactor;

        // meters
        float bilateral_sigma_depth;
        // pixels
        float bilateral_sigma_spatial;
        // pixels
        int   bilateral_kernel_size;

        // used for ICP
        int pyramidLevels;

        // number of voxels
        int volumeDims;
        // size of side in meter
        float volumeSize;

        // meters, integrate only if exceedes
        float tsdf_min_camera_movement;

        // initial volume pose in meters
        Affine3f volumePose;

        // distance to truncate in meters
        float tsdf_trunc_dist;

        // max # of frames per voxel
        int tsdf_max_weight;

        // how much voxel sizes we skip
        float raycast_step_factor;

        // gradient delta in voxel sizes
        float gradient_delta_factor;

        // light pose for rendering in meters
        Vec3f lightPose;

        // distance theshold for ICP in meters
        float icpDistThresh;
        // angle threshold for ICP in radians
        float icpAngleThresh;
        // number of ICP iterations for each pyramid level
        std::vector<int> icpIterations;

        // depth truncation is not used by default
        // float icp_truncate_depth_dist; //meters

    };

    KinFu(const KinFuParams& _params);
    virtual ~KinFu();

    const KinFuParams& getParams() const;
    KinFuParams& getParams();

    /** @brief Renders a volume into an image

      Renders a 0-surface of TSDF using Phong shading into a CV_8UC3 Mat.
      Light pose is fixed in KinFu settings.

        @param image resulting image
        @param cameraPose pose of camera to render from. If empty then render from current pose
        which is a last frame camera pose.
    */

    void render(OutputArray image, const Affine3f cameraPose = Affine3f::Identity()) const;

    void fetchCloud(OutputArray points, OutputArray normals) const;
    void fetchPoints(OutputArray points) const;
    void fetchNormals(InputArray points, OutputArray normals) const;

    void reset();

    const Affine3f getPose() const;

    bool operator()(InputArray depth);

private:

    struct KinFuImpl;
    cv::Ptr<KinFuImpl> impl;
};

//! @}
}
}
#endif
