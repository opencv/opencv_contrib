#include "util/image_converter.hpp"

#include <opencv2/imgproc.hpp>

namespace cv::slam {
namespace util {

void convert_to_grayscale(cv::Mat& img, const camera::color_order_t in_color_order) {
    if (img.channels() == 3) {
        switch (in_color_order) {
            case camera::color_order_t::Gray: {
                break;
            }
            case camera::color_order_t::RGB: {
                cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
                break;
            }
            case camera::color_order_t::BGR: {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                break;
            }
        }
    }
    else if (img.channels() == 4) {
        switch (in_color_order) {
            case camera::color_order_t::Gray: {
                break;
            }
            case camera::color_order_t::RGB: {
                cv::cvtColor(img, img, cv::COLOR_RGBA2GRAY);
                break;
            }
            case camera::color_order_t::BGR: {
                cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
                break;
            }
        }
    }
}

void convert_to_true_depth(cv::Mat& img, const double depthmap_factor) {
    img.convertTo(img, CV_32F, 1.0 / depthmap_factor);
}

void equalize_histogram(cv::Mat& img) {
    assert(img.type() == CV_8UC1 || img.type() == CV_16UC1);
    if (img.type() == CV_16UC1) {
        std::vector<unsigned short> vec(img.begin<unsigned short>(), img.end<unsigned short>());
        std::sort(vec.begin(), vec.end());
        const auto l = vec.at(static_cast<unsigned int>(0.05 * vec.size()));
        const auto h = vec.at(static_cast<unsigned int>(0.95 * vec.size()));
        img.convertTo(img, CV_8UC1, 255.0 / (h - l), -255.0 * l / (h - l)); // 255*(img-l)/(h-l)
    }
    else if (img.type() == CV_8UC1) {
        std::vector<unsigned char> vec(img.begin<unsigned char>(), img.end<unsigned char>());
        std::sort(vec.begin(), vec.end());
        const auto l = vec.at(static_cast<unsigned int>(0.05 * vec.size()));
        const auto h = vec.at(static_cast<unsigned int>(0.95 * vec.size()));
        img.convertTo(img, CV_8UC1, 255.0 / (h - l), -255.0 * l / (h - l)); // 255*(img-l)/(h-l)
    }
}

} // namespace util
} // namespace cv::slam
