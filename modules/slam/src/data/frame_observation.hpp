#ifndef SLAM_DATA_FRAME_OBSERVATION_H
#define SLAM_DATA_FRAME_OBSERVATION_H

#include "type.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace cv::slam {
namespace data {

struct frame_observation {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    frame_observation() = default;
    frame_observation(const cv::Mat& descriptors,
                      const std::vector<cv::KeyPoint>& undist_keypts, const eigen_alloc_vector<Vec3_t>& bearings,
                      const std::vector<float>& stereo_x_right, const std::vector<float>& depths)
        : descriptors_(descriptors), undist_keypts_(undist_keypts), bearings_(bearings),
          stereo_x_right_(stereo_x_right), depths_(depths) {}

    //! descriptors
    cv::Mat descriptors_;
    //! undistorted keypoints of monocular or stereo left image
    std::vector<cv::KeyPoint> undist_keypts_;
    //! bearing vectors
    eigen_alloc_vector<Vec3_t> bearings_;
    //! disparities
    std::vector<float> stereo_x_right_;
    //! depths
    std::vector<float> depths_;
    //! keypoint indices in each of the cells
    std::vector<std::vector<std::vector<unsigned int>>> keypt_indices_in_cells_;
    //! number of columns of grid to accelerate reprojection matching
    unsigned int num_grid_cols_;
    //! number of rows of grid to accelerate reprojection matching
    unsigned int num_grid_rows_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_FRAME_OBSERVATION_H
