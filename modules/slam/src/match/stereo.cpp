#include "match/stereo.hpp"

#include <opencv2/core.hpp>

namespace cv::slam {
namespace match {

stereo::stereo(const std::vector<cv::Mat>& left_image_pyramid, const std::vector<cv::Mat>& right_image_pyramid,
               const std::vector<cv::KeyPoint>& keypts_left, const std::vector<cv::KeyPoint>& keypts_right,
               const cv::Mat& descs_left, const cv::Mat& descs_right,
               const std::vector<float>& scale_factors, const std::vector<float>& inv_scale_factors,
               const float focal_x_baseline, const float true_baseline)
    : left_image_pyramid_(left_image_pyramid), right_image_pyramid_(right_image_pyramid),
      num_keypts_(keypts_left.size()), keypts_left_(keypts_left), keypts_right_(keypts_right),
      descs_left_(descs_left), descs_right_(descs_right),
      scale_factors_(scale_factors), inv_scale_factors_(inv_scale_factors),
      focal_x_baseline_(focal_x_baseline), true_baseline_(true_baseline),
      min_disp_(0.0f), max_disp_(focal_x_baseline_ / true_baseline_) {}

void stereo::compute(std::vector<float>& stereo_x_right, std::vector<float>& depths) const {
    // Save keypoint indices on the right image in each image row
    const auto indices_right_in_row = get_right_keypoint_indices_in_each_row(2.0);

    // Compute the parallax and depth for each keypoint on the left image in a subpixel precision
    stereo_x_right.resize(num_keypts_, -1.0f);
    depths.resize(num_keypts_, -1.0f);
    std::vector<std::pair<int, int>> correlation_and_idx_left;
    correlation_and_idx_left.reserve(num_keypts_);

#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int64_t idx_left = 0; idx_left < num_keypts_; ++idx_left) {
        const auto& keypt_left = keypts_left_.at(idx_left);
        const auto scale_level_left = keypt_left.octave;
        const float y_left = keypt_left.pt.y;
        const float x_left = keypt_left.pt.x;

        // Acquire the index of the keypoint on the right image which is observed at the same height level of the one on the left image
        // This is candidate matching
        const auto& candidate_indices_right = indices_right_in_row.at(y_left);
        if (candidate_indices_right.empty()) {
            continue;
        }

        // Compute x value range on the right image
        const float min_x_right = x_left - max_disp_;
        const float max_x_right = x_left - min_disp_;
        if (max_x_right < 0) {
            continue;
        }

        // Search the best candidate index on the right image whose feature vector is the closest to that on the left
        unsigned int best_idx_right = 0;
        unsigned int best_hamm_dist = hamm_dist_thr_;
        find_closest_keypoints_in_stereo(idx_left, scale_level_left, candidate_indices_right,
                                         min_x_right, max_x_right, best_idx_right, best_hamm_dist);
        // Discard if the hamming distance threshold isn't satisfied
        if (hamm_dist_thr_ <= best_hamm_dist) {
            continue;
        }
        const auto& keypt_right = keypts_right_.at(best_idx_right);

        // Compute the parallax of the subpixel order by patch correlation
        float best_x_right = -1.0f;
        float best_disp = -1.0f;
        float best_correlation = std::numeric_limits<float>::max();
        const auto is_valid = compute_subpixel_disparity(keypt_left, keypt_right, best_x_right, best_disp, best_correlation);
        // Discard if it's not found
        if (!is_valid) {
            continue;
        }
        // Discard if the parallax lies outside the valid range
        if (best_disp < min_disp_ || max_disp_ <= best_disp) {
            continue;
        }

        // Save the information if the parallax is within the valid range
        if (best_disp <= 0.0f) {
            // Set a low value if the parallax is 0 (zero)
            best_disp = 0.01f;
            best_x_right = x_left - best_disp;
        }

        // Set the results
        depths.at(idx_left) = focal_x_baseline_ / best_disp;
        stereo_x_right.at(idx_left) = best_x_right;
#ifdef USE_OPENMP
#pragma omp critical
#endif
        {
            correlation_and_idx_left.emplace_back(std::make_pair(best_correlation, idx_left));
        }
    }

    // Acquire the median of correlation
    std::sort(correlation_and_idx_left.begin(), correlation_and_idx_left.end());
    const auto median_i = correlation_and_idx_left.size() / 2;
    const float median_correlation = correlation_and_idx_left.empty()
                                         ? 0.0f
                                         : correlation_and_idx_left.at(median_i).first;
    // Discard if the correlation is weaker than the double median value of the correlation
    const float correlation_thr = 2.0 * median_correlation;

    // Start with 'median_i' because the threshold is set to the double median value of the correlation
    for (unsigned int i = median_i; i < correlation_and_idx_left.size(); ++i) {
        const auto correlation = correlation_and_idx_left.at(i).first;
        const auto idx_left = correlation_and_idx_left.at(i).second;
        if (correlation_thr < correlation) {
            stereo_x_right.at(idx_left) = -1;
            depths.at(idx_left) = -1;
        }
    }
}

std::vector<std::vector<unsigned int>> stereo::get_right_keypoint_indices_in_each_row(const float margin) const {
    // Save keypoint indices on the right image in each image row
    const unsigned int num_img_rows = left_image_pyramid_.at(0).rows;

    std::vector<std::vector<unsigned int>> indices_right_in_row(num_img_rows, std::vector<unsigned int>());
    for (unsigned int row = 0; row < num_img_rows; ++row) {
        indices_right_in_row.at(row).reserve(100);
    }

    const unsigned int num_keypts_right = keypts_right_.size();
    for (unsigned int idx_right = 0; idx_right < num_keypts_right; ++idx_right) {
        // Acquire the cordinates y of the keypoint on the right image
        const auto& keypt_right = keypts_right_.at(idx_right);
        const float y_right = keypt_right.pt.y;
        // Compute uncertainty of the cordinates according to scale
        const float r = margin * scale_factors_.at(keypts_right_.at(idx_right).octave);
        // Compute the max and the min values
        const int max_r = cvCeil(y_right + r);
        const int min_r = cvFloor(y_right - r);

        // Save the index of the keypoint for all the row numbers between the max and the min values
        for (int row_right = min_r; row_right <= max_r; ++row_right) {
            indices_right_in_row.at(row_right).push_back(idx_right);
        }
    }

    return indices_right_in_row;
}

void stereo::find_closest_keypoints_in_stereo(const unsigned int idx_left, const int scale_level_left,
                                              const std::vector<unsigned int>& candidate_indices_right,
                                              const float min_x_right, const float max_x_right,
                                              unsigned int& best_idx_right, unsigned int& best_hamm_dist) const {
    best_idx_right = 0;
    best_hamm_dist = hamm_dist_thr_;

    const cv::Mat& desc_left = descs_left_.row(idx_left);

    // Compute each hamming distance between the keypoints on the right and left images
    // For each of the keypoints on the left image, acquire the index of the closest keypoint on the right image
    for (const auto idx_right : candidate_indices_right) {
        const auto& keypt_right = keypts_right_.at(idx_right);
        // Discard if the ORB scale becomes significantly different
        if (keypt_right.octave < scale_level_left - 1 || keypt_right.octave > scale_level_left + 1) {
            continue;
        }

        // Discard if the parallax lies outside the valid range
        const float x_right = keypt_right.pt.x;
        if (x_right < min_x_right || max_x_right < x_right) {
            continue;
        }

        // Compute the hamming distance
        const auto& desc_right = descs_right_.row(idx_right);
        const unsigned int hamm_dist = match::compute_descriptor_distance_32(desc_left, desc_right);

        if (hamm_dist < best_hamm_dist) {
            best_idx_right = idx_right;
            best_hamm_dist = hamm_dist;
        }
    }
}

bool stereo::compute_subpixel_disparity(const cv::KeyPoint& keypt_left, const cv::KeyPoint& keypt_right,
                                        float& best_x_right, float& best_disp, float& best_correlation) const {
    // The keypoint on the right image whose hamming distance is cloest
    const float x_right = keypt_right.pt.x;
    // Convert cordinates to multiple scaling to compute patch correlation on the scaled image
    const float inv_scale_factor = inv_scale_factors_.at(keypt_left.octave);
    const int scaled_x_left = cvRound(keypt_left.pt.x * inv_scale_factor);
    const int scaled_y_left = cvRound(keypt_left.pt.y * inv_scale_factor);
    const int scaled_x_right = cvRound(x_right * inv_scale_factor);

    // Discard if computation of the patch movement is outside of the range
    constexpr int win_size = 5;
    constexpr int slide_width = 5;
    const int ini_x = scaled_x_right - slide_width - win_size;
    const int end_x = scaled_x_right + slide_width + win_size;
    if (ini_x < 0 || right_image_pyramid_.at(keypt_left.octave).cols <= end_x) {
        return false;
    }

    // Compute the pixel correlation surrounding the keypoint, and compute the parallax in subpixel precision by parabolic fitting
    best_correlation = std::numeric_limits<float>::max();
    int best_offset = 0;
    std::vector<float> correlations(2 * slide_width + 1, -1);

    // Extract a patch on the left image
    auto patch_left = left_image_pyramid_.at(keypt_left.octave)
                          .rowRange(scaled_y_left - win_size, scaled_y_left + win_size + 1)
                          .colRange(scaled_x_left - win_size, scaled_x_left + win_size + 1);
    patch_left.convertTo(patch_left, CV_32F);
    patch_left -= patch_left.at<float>(win_size, win_size) * cv::Mat::ones(patch_left.rows, patch_left.cols, CV_32F);

    for (int offset = -slide_width; offset <= +slide_width; ++offset) {
        // Extract a patch on the right image
        auto patch_right = right_image_pyramid_.at(keypt_left.octave)
                               .rowRange(scaled_y_left - win_size, scaled_y_left + win_size + 1)
                               .colRange(scaled_x_right + offset - win_size, scaled_x_right + offset + win_size + 1);
        patch_right.convertTo(patch_right, CV_32F);
        patch_right -= patch_right.at<float>(win_size, win_size) * cv::Mat::ones(patch_right.rows, patch_right.cols, CV_32F);

        // Acquire correlation L1
        const float correlation = cv::norm(patch_left, patch_right, cv::NORM_L1);
        if (correlation < best_correlation) {
            best_correlation = correlation;
            best_offset = offset;
        }

        correlations.at(slide_width + offset) = correlation;
    }

    if (best_offset == -slide_width || best_offset == slide_width) {
        return false;
    }

    // Apply parabolic fitting to the three-point correlation value centering the point with the strongest correlation
    const float correlation_1 = correlations.at(slide_width + best_offset - 1);
    const float correlation_2 = correlations.at(slide_width + best_offset);
    const float correlation_3 = correlations.at(slide_width + best_offset + 1);

    // Compute the best offset so that the correlation becomes minimum
    // Parabolic vertex coordinates passing through the three points: x_delta = (-1, correlation_1), (0, correlation_2), (+1, correlation_1)
    const float x_delta = (correlation_1 - correlation_3) / (2.0 * (correlation_1 + correlation_3) - 4.0 * correlation_2);

    if (x_delta < -1.0 || 1.0 < x_delta) {
        return false;
    }

    // Compute the parallax
    best_x_right = scale_factors_.at(keypt_left.octave) * (scaled_x_right + best_offset + x_delta);
    best_disp = keypt_left.pt.x - best_x_right;

    return true;
}

} // namespace match
} // namespace cv::slam
