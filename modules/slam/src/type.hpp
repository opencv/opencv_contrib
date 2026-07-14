#ifndef SLAM_TYPE_H
#define SLAM_TYPE_H

#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/types.hpp>

#ifndef M_PI
// M_PI is not part of the C++ standard. Rather it is part of the POSIX standard. As such,
// it is not directly available on Visual C++ (although _USE_MATH_DEFINES does exist).
#define M_PI 3.14159265358979323846
#endif

namespace cv::slam {

// helper function for creating an object in a unique_ptr.

template<typename T, typename... ArgTs>
std::unique_ptr<T> make_unique(ArgTs&&... args) {
    return std::unique_ptr<T>(new T(std::forward<ArgTs>(args)...));
}

// floating point type

typedef float real_t;

// Eigen matrix types

template<size_t R, size_t C>
using MatRC_t = Eigen::Matrix<double, R, C>;

using Mat22_t = Eigen::Matrix2d;

using Mat33_t = Eigen::Matrix3d;

using Mat44_t = Eigen::Matrix4d;

using Mat55_t = MatRC_t<5, 5>;

using Mat66_t = MatRC_t<6, 6>;

using Mat77_t = MatRC_t<7, 7>;

using Mat34_t = MatRC_t<3, 4>;

using MatX_t = Eigen::MatrixXd;

// Eigen vector types

template<size_t R>
using VecR_t = Eigen::Matrix<double, R, 1>;

using Vec2_t = Eigen::Vector2d;

using Vec3_t = Eigen::Vector3d;

using Vec4_t = Eigen::Vector4d;

using Vec5_t = VecR_t<5>;

using Vec6_t = VecR_t<6>;

using Vec7_t = VecR_t<7>;

using Vec9_t = VecR_t<9>;

using VecX_t = Eigen::VectorXd;

// Eigen Quaternion type

using Quat_t = Eigen::Quaterniond;

// STL with Eigen custom allocator

template<typename T>
using eigen_alloc_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template<typename T, typename U>
using eigen_alloc_map = std::map<T, U, std::less<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_set = std::set<T, std::less<T>, Eigen::aligned_allocator<const T>>;

template<typename T, typename U>
using eigen_alloc_unord_map = std::unordered_map<T, U, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<std::pair<const T, U>>>;

template<typename T>
using eigen_alloc_unord_set = std::unordered_set<T, std::hash<T>, std::equal_to<T>, Eigen::aligned_allocator<const T>>;

// vector operators

template<typename T>
inline Vec2_t operator+(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return {v1(0) + v2.x, v1(1) + v2.y};
}

template<typename T>
inline Vec2_t operator+(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v2 + v1;
}

template<typename T>
inline Vec2_t operator-(const Vec2_t& v1, const cv::Point_<T>& v2) {
    return v1 + (-v2);
}

template<typename T>
inline Vec2_t operator-(const cv::Point_<T>& v1, const Vec2_t& v2) {
    return v1 + (-v2);
}

// Comparators to allow ordering of classes with id_ member (keyframes, landmarks) to be ordered by ID.
// Assume that null pointers have an ID of infinity, i.e. nullptr > any valid pointer.
template<class T>
struct id_less;

template<class T>
struct id_less<std::shared_ptr<T>> {
    bool operator()(const std::shared_ptr<T>& a, const std::shared_ptr<T>& b) const {
        return a != nullptr && (b == nullptr || a->id_ < b->id_);
    }
};

template<class T>
struct id_less<std::weak_ptr<T>> {
    bool operator()(const std::weak_ptr<T>& a, const std::weak_ptr<T>& b) const {
        return !a.expired() && (b.expired() || a.lock()->id_ < b.lock()->id_);
    }
};

template<class T, class U>
struct less_number_and_id_object_pairs {
    bool operator()(const std::pair<T, std::shared_ptr<U>>& a, const std::pair<T, std::shared_ptr<U>>& b) {
        return a.first < b.first || (a.first == b.first && a.second != nullptr && (b.second == nullptr || a.second->id_ < b.second->id_));
    }
};

template<class T, class U>
struct greater_number_and_id_object_pairs {
    bool operator()(const std::pair<T, std::shared_ptr<U>>& a, const std::pair<T, std::shared_ptr<U>>& b) {
        return a.first > b.first || (a.first == b.first && a.second != nullptr && (b.second == nullptr || a.second->id_ < b.second->id_));
    }
};

template<class T>
using id_ordered_set = std::set<T, id_less<T>>;

template<class T, class U>
using id_ordered_map = std::map<T, U, id_less<T>>;

namespace nondeterministic {
#ifdef DETERMINISTIC
// For deterministic behavior, use set and map ordered by id.
template<class T>
using unordered_set = std::set<T, id_less<T>>;
template<class T, class U>
using unordered_map = std::map<T, U, id_less<T>>;
#else
template<class T>
using unordered_set = std::unordered_set<T>;
template<class T, class U>
using unordered_map = std::unordered_map<T, U>;
#endif
} // namespace nondeterministic

} // namespace cv::slam

#endif // SLAM_TYPE_H
