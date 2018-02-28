//TODO: correct license

//TODO: maybe pragma once?

#ifndef __OPENCV_KINECT_FUSION_HPP__
#define __OPENCV_KINECT_FUSION_HPP__

#include "opencv2/core.hpp"
#include "utils.hpp"

namespace cv {
namespace kinfu {
//! @addtogroup kinect_fusion
//! @{

//TODO: document it properly

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

        //TODO: find out what we need of that

        /*
        Affine3f volume_pose; //meters, inital pose

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3


        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

        */
    };

    KinFu(const KinFuParams& _params);
    virtual ~KinFu();

    const KinFuParams& getParams() const;
    KinFuParams& getParams();

    //TODO: implement
    Points fetchCloud() const;

    //TODO: enable this when (if) features are ready

    /*

    const TSDFVolume& tsdf() const;
    TSDFVolume& tsdf();

    void reset();

    void renderImage(cuda::Image& image, int flags = 0);
    void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

    Affine3f getCameraPose (int time = -1) const;
    */

    bool operator()(InputArray depth);

private:

    struct KinFuImpl;
    cv::Ptr<KinFuImpl> impl;
};

//! @}
}
}
#endif
