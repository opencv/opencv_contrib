#ifndef SLAM_UTIL_IMAGE_CONVERTER_H
#define SLAM_UTIL_IMAGE_CONVERTER_H

#include "camera/base.hpp"

#include <opencv2/core/mat.hpp>

namespace cv::slam {
namespace util {

void convert_to_grayscale(cv::Mat& img, const camera::color_order_t in_color_order);

void convert_to_true_depth(cv::Mat& img, const double depthmap_factor);

void equalize_histogram(cv::Mat& img);

} // namespace util
} // namespace cv::slam

#endif // SLAM_UTIL_IMAGE_CONVERTER_H
