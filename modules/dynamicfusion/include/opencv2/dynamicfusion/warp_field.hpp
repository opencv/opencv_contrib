#ifndef KFUSION_WARP_FIELD_HPP
#define KFUSION_WARP_FIELD_HPP

/**
 * \brief
 * \details
 */
#include <opencv2/dynamicfusion/utils/dual_quaternion.hpp>
#include <opencv2/dynamicfusion/types.hpp>
#include <nanoflann/nanoflann.hpp>
#include <opencv2/dynamicfusion/utils/knn_point_cloud.hpp>
#include <opencv2/dynamicfusion/cuda/tsdf_volume.hpp>
#define KNN_NEIGHBOURS 8

namespace cv
{
    namespace kfusion
    {
        typedef nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<float, utils::PointCloud>,
                utils::PointCloud,
                3 /* dim */
        > kd_tree_t;


        //    TODO: remember to rewrite this with proper doxygen formatting (e.g <sub></sub> rather than _
        /*!
         * \struct node
         * \brief A node of the warp field
         * \details The state of the warp field Wt at time t is defined by the values of a set of n
         * deformation nodes Nt_warp = {dg_v, dg_w, dg_se3}_t. Here, this is represented as follows
         *
         * \var node::index
         * Index of the node in the canonical frame. Equivalent to dg_v
         *
         * \var node::transform
         * Transform from canonical point to warped point, equivalent to dg_se in the paper.
         *
         * \var node::weight
         * Equivalent to dg_w
         */
        struct deformation_node
        {
            Vec3f vertex;
            kfusion::utils::DualQuaternion<float> transform;
            float weight = 0;
        };
        class WarpField
        {
        public:
            WarpField();
            ~WarpField();

            void init(const cv::Mat& first_frame, const cv::Mat& normals);
            void init(const std::vector<Vec3f>& first_frame, const std::vector<Vec3f>& normals);
            void energy(const cuda::Cloud &frame,
                        const cuda::Normals &normals,
                        const Affine3f &pose,
                        const cuda::TsdfVolume &tsdfVolume,
                        const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                                kfusion::utils::DualQuaternion<float>>> &edges
            );

            float energy_data(const std::vector<Vec3f> &canonical_vertices,
                              const std::vector<Vec3f> &canonical_normals,
                              const std::vector<Vec3f> &live_vertices,
                              const std::vector<Vec3f> &live_normals);
            void energy_reg(const std::vector<std::pair<kfusion::utils::DualQuaternion<float>,
                    kfusion::utils::DualQuaternion<float>>> &edges);

            float tukeyPenalty(float x, float c = 0.01) const;

            float huberPenalty(float a, float delta) const;

            void warp(std::vector<Vec3f>& points, std::vector<Vec3f>& normals) const;
            void warp(cuda::Cloud& points) const;

            utils::DualQuaternion<float> DQB(const Vec3f& vertex) const;
            utils::DualQuaternion<float> DQB(const Vec3f& vertex, double epsilon[KNN_NEIGHBOURS * 6]) const;

            void getWeightsAndUpdateKNN(const Vec3f& vertex, float weights[KNN_NEIGHBOURS]);

            float weighting(float squared_dist, float weight) const;
            void KNN(Vec3f point) const;

            void clear();

            const std::vector<deformation_node>* getNodes() const;
            const cv::Mat getNodesAsMat() const;
            void setWarpToLive(const Affine3f &pose);


            std::vector<float> out_dist_sqr;
            std::vector<size_t> ret_index;

        private:
            std::vector<deformation_node>* nodes;
            kd_tree_t* index;
            nanoflann::KNNResultSet<float> *resultSet;
            Affine3f warp_to_live;
            void buildKDTree();
        };
    }
}
#endif //KFUSION_WARP_FIELD_HPP
