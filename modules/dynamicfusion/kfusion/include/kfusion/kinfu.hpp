#pragma once

#include <kfusion/types.hpp>
#include <vector>
#include <string>
#include <dual_quaternion.hpp>
#include <quaternion.hpp>
#include <kfusion/cuda/projective_icp.hpp>
#include <kfusion/cuda/tsdf_volume.hpp>
#include <kfusion/warp_field.hpp>

namespace kfusion
{
    namespace cuda
    {
        KF_EXPORTS int getCudaEnabledDeviceCount();
        KF_EXPORTS void setDevice(int device);
        KF_EXPORTS std::string getDeviceName(int device);
        KF_EXPORTS bool checkIfPreFermiGPU(int device);
        KF_EXPORTS void printCudaDeviceInfo(int device);
        KF_EXPORTS void printShortCudaDeviceInfo(int device);
    }

    struct KF_EXPORTS KinFuParams
    {
        static KinFuParams default_params();
        static KinFuParams default_params_dynamicfusion();

        int cols;  //pixels
        int rows;  //pixels

        Intr intr;  //Camera parameters

        Vec3i volume_dims; //number of voxels
        Vec3f volume_size; //meters
        Affine3f volume_pose; //meters, inital pose

        float bilateral_sigma_depth;   //meters
        float bilateral_sigma_spatial;   //pixels
        int   bilateral_kernel_size;   //pixels

        float icp_truncate_depth_dist; //meters
        float icp_dist_thres;          //meters
        float icp_angle_thres;         //radians
        std::vector<int> icp_iter_num; //iterations for level index 0,1,..,3

        float tsdf_min_camera_movement; //meters, integrate only if exceedes
        float tsdf_trunc_dist;             //meters;
        int tsdf_max_weight;               //frames

        float raycast_step_factor;   // in voxel sizes
        float gradient_delta_factor; // in voxel sizes

        Vec3f light_pose; //meters

    };

    class KF_EXPORTS KinFu
    {
    public:        
        typedef cv::Ptr<KinFu> Ptr;

        KinFu(const KinFuParams& params);

        const KinFuParams& params() const;
        KinFuParams& params();

        const cuda::TsdfVolume& tsdf() const;
        cuda::TsdfVolume& tsdf();

        const cuda::ProjectiveICP& icp() const;
        cuda::ProjectiveICP& icp();

        const WarpField& getWarp() const;
        WarpField& getWarp();

        void reset();

        bool operator()(const cuda::Depth& depth, const cuda::Image& image = cuda::Image());

        void renderImage(cuda::Image& image, int flags = 0);
        void dynamicfusion(cuda::Depth& depth, cuda::Cloud current_frame, cuda::Normals current_normals);
        void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);
        void reprojectToDepth();

        Affine3f getCameraPose (int time = -1) const;
    private:
        void allocate_buffers();

        int frame_counter_;
        KinFuParams params_;

        std::vector<Affine3f> poses_;

        cuda::Dists dists_;
        cuda::Frame curr_, prev_, first_;

        cuda::Cloud points_;
        cuda::Normals normals_;
        cuda::Depth depths_;

        cv::Ptr<cuda::TsdfVolume> volume_;
        cv::Ptr<cuda::ProjectiveICP> icp_;
        cv::Ptr<WarpField> warp_;
        std::vector<std::pair<utils::DualQuaternion<float>, utils::DualQuaternion<float>>> edges;
    };
}
