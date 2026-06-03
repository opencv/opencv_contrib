#ifndef SLAM_UTIL_FANCY_INDEX_H
#define SLAM_UTIL_FANCY_INDEX_H

#include "type.hpp"

#include <vector>
#include <type_traits>

namespace cv::slam {
namespace util {

template<typename T, typename U>
std::vector<T> resample_by_indices(const std::vector<T>& elements, const std::vector<U>& indices) {
    static_assert(std::is_integral<U>(), "the element type of indices must be integer");

    std::vector<T> resampled;
    resampled.reserve(elements.size());
    for (const auto idx : indices) {
        resampled.push_back(elements.at(idx));
    }

    return resampled;
}

template<typename T, typename U>
eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T>& elements, const std::vector<U>& indices) {
    static_assert(std::is_integral<U>(), "the element type of indices must be integer");

    eigen_alloc_vector<T> resampled;
    resampled.reserve(elements.size());
    for (const auto idx : indices) {
        resampled.push_back(elements.at(idx));
    }

    return resampled;
}

template<typename T>
std::vector<T> resample_by_indices(const std::vector<T>& elements, const std::vector<bool>& indices) {
    assert(elements.size() == indices.size());

    std::vector<T> resampled;
    resampled.reserve(elements.size());
    for (unsigned int idx = 0; idx < elements.size(); ++idx) {
        if (indices.at(idx)) {
            resampled.push_back(elements.at(idx));
        }
    }

    return resampled;
}

template<typename T>
eigen_alloc_vector<T> resample_by_indices(const eigen_alloc_vector<T>& elements, const std::vector<bool>& indices) {
    assert(elements.size() == indices.size());

    eigen_alloc_vector<T> resampled;
    resampled.reserve(elements.size());
    for (unsigned int idx = 0; idx < elements.size(); ++idx) {
        if (indices.at(idx)) {
            resampled.push_back(elements.at(idx));
        }
    }

    return resampled;
}

} // namespace util
} // namespace cv::slam

#endif // SLAM_UTIL_FANCY_INDEX_H
