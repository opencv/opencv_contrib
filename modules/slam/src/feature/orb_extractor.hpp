#ifndef SLAM_FEATURE_ORB_EXTRACTOR_H
#define SLAM_FEATURE_ORB_EXTRACTOR_H

#include "feature/orb_params.hpp"
#include "feature/orb_impl.hpp"

#include <stdexcept>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
#include <cuda_efficient_descriptors.h>
#endif

namespace cv::slam {
namespace feature {

enum class descriptor_type {
    ORB,
    HASH_SIFT
};

inline descriptor_type descriptor_type_from_string(const std::string& desc_type_str) {
    if (desc_type_str == "ORB") {
        return descriptor_type::ORB;
    }
    else if (desc_type_str == "HASH_SIFT" || desc_type_str == "HashSIFT") {
        return descriptor_type::HASH_SIFT;
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

inline std::string descriptor_type_to_string(descriptor_type desc_type) {
    if (desc_type == descriptor_type::ORB) {
        return "ORB";
    }
    else if (desc_type == descriptor_type::HASH_SIFT) {
        return "HashSIFT";
    }
    else {
        throw std::runtime_error("Invalid descriptor_type");
    }
}

class orb_extractor {
public:
    orb_extractor() = delete;

    //! Constructor
    orb_extractor(const orb_params* orb_params,
                  const unsigned int min_area,
                  const descriptor_type desc_type = descriptor_type::ORB,
                  const std::vector<std::vector<float>>& mask_rects = {});

    //! Destructor
    virtual ~orb_extractor() = default;

    //! Extract keypoints and each descriptor of them
    void extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                 std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors);

    //! parameters for ORB extraction
    const orb_params* orb_params_;

    //! A vector of keypoint area represents mask area
    //! Each areas are denoted as form of [x_min / cols, x_max / cols, y_min / rows, y_max / rows]
    std::vector<std::vector<float>> mask_rects_;

    //! Image pyramid
    std::vector<cv::Mat> image_pyramid_;

private:
    //! Calculate scale factors and sigmas
    void calc_scale_factors();

    //! Create a mask matrix that constructed by rectangles
    void create_rectangle_mask(const unsigned int cols, const unsigned int rows);

    //! Compute image pyramid
    void compute_image_pyramid(const cv::Mat& image);

    //! Compute fast keypoints for cells in each image pyramid
    void compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const;

    //! Pick computed keypoints on the image uniformly
    std::vector<cv::KeyPoint> distribute_keypoints(const std::vector<cv::KeyPoint>& keypts_to_distribute,
                                                   const int min_x, const int max_x, const int min_y, const int max_y,
                                                   const float scale_factor) const;

    //! Compute orientation for each keypoint
    void compute_orientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypts) const;

    //! Correct keypoint's position to comply with the scale
    void correct_keypoint_scale(std::vector<cv::KeyPoint>& keypts_at_level, const unsigned int level) const;

    //! Compute the gradient direction of pixel intensity in a circle around the point
    float ic_angle(const cv::Mat& image, const cv::Point2f& point) const;

    //! Compute orb descriptor of a keypoint
    void compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const;

    //! Area of node occupied by one feature point
    unsigned int min_area_sqrt_;

    //! size of maximum ORB patch radius
    static constexpr unsigned int orb_patch_radius_ = 19;

    //! rectangle mask has been already initialized or not
    bool mask_is_initialized_ = false;
    cv::Mat rect_mask_;

    descriptor_type desc_type_;

    //! feature descriptor implementations
    orb_impl orb_impl_;
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
    cv::Ptr<cv::cuda::HashSIFT> hash_sift_;
#endif
};

} // namespace feature
} // namespace cv::slam

#endif // SLAM_FEATURE_ORB_EXTRACTOR_H
