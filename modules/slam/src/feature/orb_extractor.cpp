#include "feature/orb_extractor.hpp"
#include "type.hpp"

#include <stdexcept>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <mutex>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace feature {

orb_extractor::orb_extractor(const orb_params* orb_params,
                             const unsigned int min_area,
                             const descriptor_type desc_type,
                             const std::vector<std::vector<float>>& mask_rects)
    : orb_params_(orb_params), mask_rects_(mask_rects), min_area_sqrt_(std::sqrt(min_area)), desc_type_(desc_type) {
    // resize buffers according to the number of levels
    image_pyramid_.resize(orb_params_->num_levels_);
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
    hash_sift_ = cv::cuda::HashSIFT::create(1.0, cv::cuda::HashSIFT::SIZE_256_BITS);
#endif
}

void orb_extractor::extract(const cv::_InputArray& in_image, const cv::_InputArray& in_image_mask,
                            std::vector<cv::KeyPoint>& keypts, const cv::_OutputArray& out_descriptors) {
    if (in_image.empty()) {
        return;
    }

    // get cv::Mat of image
    const auto image = in_image.getMat();
    assert(image.type() == CV_8UC1);

    // build image pyramid
    compute_image_pyramid(image);

    // mask initialization
    if (!mask_is_initialized_ && !mask_rects_.empty()) {
        create_rectangle_mask(image.cols, image.rows);
        mask_is_initialized_ = true;
    }

    std::vector<std::vector<cv::KeyPoint>> all_keypts;

    // select mask to use
    if (!in_image_mask.empty()) {
        // Use image_mask if it is available
        const auto image_mask = in_image_mask.getMat();
        assert(image_mask.type() == CV_8UC1);
        compute_fast_keypoints(all_keypts, image_mask);
    }
    else if (!rect_mask_.empty()) {
        // Use rectangle mask if it is available and image_mask is not used
        assert(rect_mask_.type() == CV_8UC1);
        compute_fast_keypoints(all_keypts, rect_mask_);
    }
    else {
        // Do not use any mask if all masks are unavailable
        compute_fast_keypoints(all_keypts, cv::Mat());
    }

    cv::Mat descriptors;

    unsigned int num_keypts = 0;
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        num_keypts += all_keypts.at(level).size();
    }
    if (num_keypts == 0) {
        out_descriptors.release();
    }
    else {
        out_descriptors.create(num_keypts, 32, CV_8U);
        descriptors = out_descriptors.getMat();
    }

    keypts.clear();
    keypts.reserve(num_keypts);

    unsigned int offset = 0;
    std::vector<unsigned int> offsets;
    offsets.push_back(0);
    for (unsigned int level = 0; level < orb_params_->num_levels_ - 1; ++level) {
        offset += all_keypts.at(level).size();
        offsets.push_back(offset);
    }

    cv::parallel_for_(cv::Range(0, static_cast<int>(orb_params_->num_levels_)), [&](const cv::Range& range) {
        for (int level = range.start; level < range.end; ++level) {
            auto& keypts_at_level = all_keypts.at(level);
            const auto num_keypts_at_level = keypts_at_level.size();

            if (num_keypts_at_level == 0) {
                continue;
            }

            cv::Mat blurred_image;
            cv::GaussianBlur(image_pyramid_.at(level), blurred_image, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

            cv::Mat descriptors_at_level = descriptors.rowRange(offsets[level], offsets[level] + num_keypts_at_level);
            descriptors_at_level = cv::Mat::zeros(num_keypts_at_level, 32, CV_8UC1);

            if (desc_type_ == feature::descriptor_type::ORB) {
                for (unsigned int i = 0; i < keypts_at_level.size(); ++i) {
                    compute_orb_descriptor(keypts_at_level[i], blurred_image, descriptors_at_level.ptr(i));
                }
            }
            else if (desc_type_ == feature::descriptor_type::HASH_SIFT) {
#ifdef USE_CUDA_EFFICIENT_DESCRIPTORS
                hash_sift_->compute(blurred_image, keypts_at_level, descriptors_at_level);
#else
                throw std::runtime_error("cuda_efficient_features is not available");
#endif
            }
            else {
                throw std::runtime_error("Invalid descriptor_type");
            }

            correct_keypoint_scale(keypts_at_level, level);
        }
    });

    // Collect keypoints for every scale
    for (unsigned int level = 0; level < orb_params_->num_levels_; ++level) {
        auto& keypts_at_level = all_keypts.at(level);
        keypts.insert(keypts.end(), keypts_at_level.begin(), keypts_at_level.end());
    }
}

void orb_extractor::create_rectangle_mask(const unsigned int cols, const unsigned int rows) {
    if (rect_mask_.empty()) {
        rect_mask_ = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    }
    // draw masks
    for (const auto& mask_rect : mask_rects_) {
        // draw black rectangle
        const unsigned int x_min = std::round(cols * mask_rect.at(0));
        const unsigned int x_max = std::round(cols * mask_rect.at(1));
        const unsigned int y_min = std::round(rows * mask_rect.at(2));
        const unsigned int y_max = std::round(rows * mask_rect.at(3));
        cv::rectangle(rect_mask_, cv::Point2i(x_min, y_min), cv::Point2i(x_max, y_max), cv::Scalar(0), -1, cv::LINE_AA);
    }
}

void orb_extractor::compute_image_pyramid(const cv::Mat& image) {
    image_pyramid_.at(0) = image;
    for (unsigned int level = 1; level < orb_params_->num_levels_; ++level) {
        // determine the size of an image
        const double scale = orb_params_->scale_factors_.at(level);
        const cv::Size size(std::round(image.cols * 1.0 / scale), std::round(image.rows * 1.0 / scale));
        // resize
        cv::resize(image_pyramid_.at(level - 1), image_pyramid_.at(level), size, 0, 0, cv::INTER_LINEAR);
    }
}

void orb_extractor::compute_fast_keypoints(std::vector<std::vector<cv::KeyPoint>>& all_keypts, const cv::Mat& mask) const {
    all_keypts.resize(orb_params_->num_levels_);

    // An anonymous function which checks mask(image or rectangle)
    auto is_in_mask = [&mask](const unsigned int y, const unsigned int x, const float scale_factor) {
        return mask.at<unsigned char>(y * scale_factor, x * scale_factor) == 0;
    };

    constexpr unsigned int overlap = 6;
    constexpr unsigned int cell_size = 64;

    static std::mutex g_fast_mtx;
    cv::parallel_for_(cv::Range(0, static_cast<int>(orb_params_->num_levels_)), [&](const cv::Range& range) {
        for (int lvl = range.start; lvl < range.end; ++lvl) {
            const auto level = static_cast<int64_t>(lvl);
            const float scale_factor = orb_params_->scale_factors_.at(level);

            constexpr unsigned int min_border_x = orb_patch_radius_;
            constexpr unsigned int min_border_y = orb_patch_radius_;
            const unsigned int max_border_x = image_pyramid_.at(level).cols - orb_patch_radius_;
            const unsigned int max_border_y = image_pyramid_.at(level).rows - orb_patch_radius_;

            const unsigned int width = max_border_x - min_border_x;
            const unsigned int height = max_border_y - min_border_y;

            const unsigned int num_cols = width / cell_size + 1;
            const unsigned int num_rows = height / cell_size + 1;

            std::vector<cv::KeyPoint> keypts_to_distribute;
            keypts_to_distribute.reserve(500);

            for (int64_t i = 0; i < num_rows; ++i) {
                const unsigned int min_y = min_border_y + i * cell_size;
                if (max_border_y - overlap <= min_y) {
                    continue;
                }
                unsigned int max_y = min_y + cell_size + overlap;
                if (max_border_y < max_y) {
                    max_y = max_border_y;
                }

                for (int64_t j = 0; j < num_cols; ++j) {
                    const unsigned int min_x = min_border_x + j * cell_size;
                    if (max_border_x - overlap <= min_x) {
                        continue;
                    }
                    unsigned int max_x = min_x + cell_size + overlap;
                    if (max_border_x < max_x) {
                        max_x = max_border_x;
                    }

                    // Pass FAST computation if one of the corners of a patch is in the mask
                    if (!mask.empty()) {
                        if (is_in_mask(min_y, min_x, scale_factor) || is_in_mask(max_y, min_x, scale_factor)
                            || is_in_mask(min_y, max_x, scale_factor) || is_in_mask(max_y, max_x, scale_factor)) {
                            continue;
                        }
                    }

                    std::vector<cv::KeyPoint> keypts_in_cell;
                    cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                             keypts_in_cell, orb_params_->ini_fast_thr_, true);

                    // Re-compute FAST keypoint with reduced threshold if enough keypoint was not got
                    if (keypts_in_cell.empty()) {
                        cv::FAST(image_pyramid_.at(level).rowRange(min_y, max_y).colRange(min_x, max_x),
                                 keypts_in_cell, orb_params_->min_fast_thr_, true);
                    }

                    if (keypts_in_cell.empty()) {
                        continue;
                    }

                    for (auto& keypt : keypts_in_cell) {
                        keypt.pt.x += j * cell_size;
                        keypt.pt.y += i * cell_size;
                    }

                    if (!mask.empty()) {
                        std::vector<cv::KeyPoint> keypts_in_cell_masked;
                        for (auto&& keypt : keypts_in_cell) {
                            // Check if the keypoint is in the mask
                            if (is_in_mask(min_border_y + keypt.pt.y, min_border_x + keypt.pt.x, scale_factor)) {
                                continue;
                            }
                            keypts_in_cell_masked.push_back(std::move(keypt));
                        }
                        keypts_in_cell = std::move(keypts_in_cell_masked);
                    }

                    {
                        std::lock_guard<std::mutex> lock(g_fast_mtx);
                        keypts_to_distribute.insert(keypts_to_distribute.end(), keypts_in_cell.begin(), keypts_in_cell.end());
                    }
                }
            }

            std::vector<cv::KeyPoint>& keypts_at_level = all_keypts.at(level);

            // Distribute keypoints via tree
            keypts_at_level = distribute_keypoints(keypts_to_distribute, min_border_x, max_border_x, min_border_y, max_border_y, scale_factor);
            CV_LOG_DEBUG(&g_log_tag, "keypts_at_level " << keypts_to_distribute.size());

            // Keypoint size is patch size modified by the scale factor
            const unsigned int scaled_patch_size = orb_impl_.fast_patch_size_ * scale_factor;

            for (auto& keypt : keypts_at_level) {
                // Translation correction (scale will be corrected after ORB description)
                keypt.pt.x += min_border_x;
                keypt.pt.y += min_border_y;
                // Set the other information
                keypt.octave = level;
                keypt.size = scaled_patch_size;
            }

            compute_orientation(image_pyramid_.at(level), all_keypts.at(level));
        }
    });
}

std::vector<cv::KeyPoint> orb_extractor::distribute_keypoints(const std::vector<cv::KeyPoint>& keypts_to_distribute,
                                                              const int min_x, const int max_x, const int min_y, const int max_y,
                                                              const float scale_factor) const {
    double scaled_min_area_sqrt = min_area_sqrt_ / scale_factor;
    unsigned int num_x_grid = std::ceil((max_x - min_x) / scaled_min_area_sqrt);
    unsigned int num_y_grid = std::ceil((max_y - min_y) / scaled_min_area_sqrt);
    double delta_x = static_cast<double>(max_x - min_x) / num_x_grid;
    double delta_y = static_cast<double>(max_y - min_y) / num_y_grid;
    std::vector<cv::KeyPoint> result_keypts;
    result_keypts.reserve(num_x_grid * num_y_grid);
    std::unordered_map<unsigned int, std::pair<cv::KeyPoint, double>> keypt_response_map;
    std::vector<std::vector<cv::KeyPoint>> keypts_on_grid(num_x_grid * num_y_grid);

    for (const auto& keypt : keypts_to_distribute) {
        const unsigned int ix = keypt.pt.x / delta_x;
        const unsigned int iy = keypt.pt.y / delta_y;
        const unsigned int idx = ix + iy * num_x_grid;
        keypts_on_grid[idx].push_back(keypt);
    }

    for (unsigned int i = 0; i < keypts_on_grid.size(); ++i) {
        auto& keypts = keypts_on_grid[i];
        if (keypts.empty()) {
            continue;
        }
        auto& selected_keypt = keypts.at(0);
        double max_response = selected_keypt.response;

        for (unsigned int k = 1; k < keypts.size(); ++k) {
            const auto& keypt = keypts[k];
            if (keypt.response > max_response) {
                selected_keypt = keypt;
                max_response = keypt.response;
            }
        }

        result_keypts.push_back(selected_keypt);
    }

    return result_keypts;
}

void orb_extractor::compute_orientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypts) const {
    for (auto& keypt : keypts) {
        keypt.angle = ic_angle(image, keypt.pt);
    }
}

void orb_extractor::correct_keypoint_scale(std::vector<cv::KeyPoint>& keypts_at_level, const unsigned int level) const {
    if (level == 0) {
        return;
    }
    const float scale_at_level = orb_params_->scale_factors_.at(level);
    for (auto& keypt_at_level : keypts_at_level) {
        keypt_at_level.pt *= scale_at_level;
    }
}

float orb_extractor::ic_angle(const cv::Mat& image, const cv::Point2f& point) const {
    return orb_impl_.ic_angle(image, point);
}

void orb_extractor::compute_orb_descriptor(const cv::KeyPoint& keypt, const cv::Mat& image, uchar* desc) const {
    orb_impl_.compute_orb_descriptor(keypt, image, desc);
}

} // namespace feature
} // namespace cv::slam
