#ifndef SLAM_SOLVE_TRIANGULATOR_H
#define SLAM_SOLVE_TRIANGULATOR_H

#include "type.hpp"

#include <Eigen/SVD>
#include <opencv2/core/types.hpp>

namespace cv::slam {
namespace solve {

class triangulator {
public:
    /**
     * Triangulate using two points and two perspective projection matrices
     * @param pt_1
     * @param pt_2
     * @param P_1
     * @param P_2
     * @return triangulated point in the world reference
     */
    static inline Vec3_t triangulate(const cv::Point2d& pt_1, const cv::Point2d& pt_2, const Mat34_t& P_1, const Mat34_t& P_2);

    /**
     * Triangulate using two bearings and relative rotation & translation
     * @param bearing_1
     * @param bearing_2
     * @param rot_21
     * @param trans_21
     * @return triangulated point in the camera 1 coordinates
     */
    static inline Vec3_t triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat33_t& rot_21, const Vec3_t& trans_21);

    /**
     * Triangulate using two bearings and absolute camera poses
     * @param bearing_1
     * @param bearing_2
     * @param cam_pose_1
     * @param cam_pose_2
     * @return
     */
    static inline Vec3_t triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat44_t& cam_pose_1, const Mat44_t& cam_pose_2);
};

Vec3_t triangulator::triangulate(const cv::Point2d& pt_1, const cv::Point2d& pt_2, const Mat34_t& P_1, const Mat34_t& P_2) {
    Mat44_t A;

    A.row(0) = pt_1.x * P_1.row(2) - P_1.row(0);
    A.row(1) = pt_1.y * P_1.row(2) - P_1.row(1);
    A.row(2) = pt_2.x * P_2.row(2) - P_2.row(0);
    A.row(3) = pt_2.y * P_2.row(2) - P_2.row(1);

    const Eigen::JacobiSVD<Mat44_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Vec4_t v = svd.matrixV().col(3);
    return v.block<3, 1>(0, 0) / v(3);
}

Vec3_t triangulator::triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat33_t& rot_21, const Vec3_t& trans_21) {
    const Vec3_t trans_12 = -rot_21.transpose() * trans_21;
    const Vec3_t bearing_2_in_1 = rot_21.transpose() * bearing_2;

    Mat22_t A;
    A(0, 0) = bearing_1.dot(bearing_1);
    A(1, 0) = bearing_1.dot(bearing_2_in_1);
    A(0, 1) = -A(1, 0);
    A(1, 1) = -bearing_2_in_1.dot(bearing_2_in_1);

    const Vec2_t b{bearing_1.dot(trans_12), bearing_2_in_1.dot(trans_12)};

    const Vec2_t lambda = A.inverse() * b;
    const Vec3_t pt_1 = lambda(0) * bearing_1;
    const Vec3_t pt_2 = lambda(1) * bearing_2_in_1 + trans_12;
    return (pt_1 + pt_2) / 2.0;
}

Vec3_t triangulator::triangulate(const Vec3_t& bearing_1, const Vec3_t& bearing_2, const Mat44_t& cam_pose_1, const Mat44_t& cam_pose_2) {
    Mat44_t A;
    A.row(0) = bearing_1(0) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(0);
    A.row(1) = bearing_1(1) * cam_pose_1.row(2) - bearing_1(2) * cam_pose_1.row(1);
    A.row(2) = bearing_2(0) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(0);
    A.row(3) = bearing_2(1) * cam_pose_2.row(2) - bearing_2(2) * cam_pose_2.row(1);


    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    Eigen::JacobiSVD<Mat44_t> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Vec4_t singular_vector = svd.matrixV().block<4, 1>(0, 3);

    return singular_vector.block<3, 1>(0, 0) / singular_vector(3);
}

} // namespace solve
} // namespace cv::slam

#endif // SLAM_SOLVE_TRIANGULATOR_H
