#pragma once

#include <opencv2/dynamicfusion/types.hpp>
#include <opencv2/dynamicfusion/utils/dual_quaternion.hpp>

namespace cv
{
    namespace kfusion
    {
        class WarpField;
        namespace cuda
        {
            class TsdfVolume
            {
                public:
                TsdfVolume(const cv::Vec3i& dims);
                virtual ~TsdfVolume();

                void create(const Vec3i& dims);

                Vec3i getDims() const;
                Vec3f getVoxelSize() const;

                const CudaData data() const;
                CudaData data();

                cv::Mat get_cloud_host() const;
                cv::Mat get_normal_host() const;

                cv::Mat* get_cloud_host_ptr() const;
                cv::Mat* get_normal_host_ptr() const;

                Vec3f getSize() const;
                void setSize(const Vec3f& size);

                float getTruncDist() const;
                void setTruncDist(float distance);

                int getMaxWeight() const;
                void setMaxWeight(int weight);

                Affine3f getPose() const;
                void setPose(const Affine3f& pose);

                float getRaycastStepFactor() const;
                void setRaycastStepFactor(float factor);

                float getGradientDeltaFactor() const;
                void setGradientDeltaFactor(float factor);

                Vec3i getGridOrigin() const;
                void setGridOrigin(const Vec3i& origin);

                std::vector<float> psdf(const std::vector<Vec3f>& warped, Dists& depth_img, const Intr& intr);
//            float psdf(const std::vector<Vec3f>& warped, Dists& dists, const Intr& intr);
                float weighting(const std::vector<float>& dist_sqr, int k) const;
                void surface_fusion(const WarpField& warp_field,
                std::vector<Vec3f> warped,
                std::vector<Vec3f> canonical,
                cuda::Depth &depth,
                const Affine3f& camera_pose,
                const Intr& intr);

                virtual void clear();
                virtual void applyAffine(const Affine3f& affine);
                virtual void integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr);
                virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals);
                virtual void raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals);

                void swap(CudaData& data);
                cv::kfusion::cuda::DeviceArray<cv::kfusion::Point> fetchCloud(cv::kfusion::cuda::DeviceArray<cv::kfusion::Point>& cloud_buffer) const;
                void fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const;
                void compute_points();
                void compute_normals();


                private:
                CudaData data_;
//            need to make this smart pointers
                cuda::DeviceArray<Point> *cloud_buffer;
                cuda::DeviceArray<Point> *cloud;
                cuda::DeviceArray<Normal> *normal_buffer;
                cv::Mat *cloud_host;
                cv::Mat *normal_host;

                float trunc_dist_;
                float max_weight_;
                Vec3i dims_;
                Vec3f size_;
                Affine3f pose_;
                float gradient_delta_factor_;
                float raycast_step_factor_;
                // TODO: remember to add entry when adding a new node
                struct Entry
                {
                    float tsdf_value;
                    float tsdf_weight;
                };

                std::vector<Entry> tsdf_entries;
            };
        }
    }
}
