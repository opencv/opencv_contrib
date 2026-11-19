#ifndef SLAM_MATCH_BASE_H
#define SLAM_MATCH_BASE_H

#include "type.hpp"

#include <array>
#include <algorithm>
#include <numeric>

#include <opencv2/core/mat.hpp>

namespace cv::slam {
namespace match {

static constexpr unsigned int HAMMING_DIST_THR_LOW = 50;
static constexpr unsigned int HAMMING_DIST_THR_HIGH = 100;
static constexpr unsigned int MAX_HAMMING_DIST = 256;


inline unsigned int compute_descriptor_distance_32(const cv::Mat& desc_1, const cv::Mat& desc_2) {
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    constexpr uint32_t mask_1 = 0x55555555U;
    constexpr uint32_t mask_2 = 0x33333333U;
    constexpr uint32_t mask_3 = 0x0F0F0F0FU;
    constexpr uint32_t mask_4 = 0x01010101U;

    const auto* pa = desc_1.ptr<uint32_t>();
    const auto* pb = desc_2.ptr<uint32_t>();

    unsigned int dist = 0;

    for (unsigned int i = 0; i < 8; ++i, ++pa, ++pb) {
        auto v = *pa ^ *pb;
        v -= ((v >> 1) & mask_1);
        v = (v & mask_2) + ((v >> 2) & mask_2);
        dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 24;
    }

    return dist;
}


inline unsigned int compute_descriptor_distance_64(const cv::Mat& desc_1, const cv::Mat& desc_2) {
    // https://stackoverflow.com/questions/21826292/t-sql-hamming-distance-function-capable-of-decimal-string-uint64?lq=1

    constexpr uint64_t mask_1 = 0x5555555555555555UL;
    constexpr uint64_t mask_2 = 0x3333333333333333UL;
    constexpr uint64_t mask_3 = 0x0F0F0F0F0F0F0F0FUL;
    constexpr uint64_t mask_4 = 0x0101010101010101UL;

    const auto* pa = desc_1.ptr<uint64_t>();
    const auto* pb = desc_2.ptr<uint64_t>();

    unsigned int dist = 0;

    for (unsigned int i = 0; i < 4; ++i, ++pa, ++pb) {
        auto v = *pa ^ *pb;
        v -= (v >> 1) & mask_1;
        v = (v & mask_2) + ((v >> 2) & mask_2);
        dist += (((v + (v >> 4)) & mask_3) * mask_4) >> 56;
    }

    return dist;
}

inline bool check_epipolar_constraint(const Vec3_t& bearing_1, const Vec3_t& bearing_2,
                                      const Mat33_t& E_12, float residual_rad_thr,
                                      const float bearing_1_scale_factor) {
    // Normal vector of the epipolar plane on keyframe 1
    const Vec3_t epiplane_in_1 = E_12 * bearing_2;

    // Acquire the angle formed by the normal vector and the bearing
    const auto cos_residual = std::min(1.0, std::max(-1.0, epiplane_in_1.dot(bearing_1) / epiplane_in_1.norm()));
    const auto residual_rad = std::abs(M_PI / 2.0 - std::acos(cos_residual));

    // The larger keypoint scale permits less constraints
    return residual_rad < residual_rad_thr * bearing_1_scale_factor;
}

class base {
public:
    base(const float lowe_ratio, const bool check_orientation)
        : lowe_ratio_(lowe_ratio), check_orientation_(check_orientation) {}

    virtual ~base() = default;

protected:
    const float lowe_ratio_;
    const bool check_orientation_;
};

} // namespace match
} // namespace cv::slam

#endif // SLAM_MATCH_BASE_H
