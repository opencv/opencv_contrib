#include "opencv2/slam/optimizer.hpp"
#include <iostream>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <opencv2/calib3d.hpp>

// If g2o is enabled (CMake defines HAVE_G2O), compile and use the g2o-based
// implementation. Otherwise fall back to a simplified OpenCV-based implementation.

#if defined(HAVE_G2O)
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Core>
#include <map>
#include <memory>
#elif defined(HAVE_SFM)
#include "opencv2/sfm.hpp"
#endif

#include <opencv2/core.hpp>

namespace cv {
namespace vo {

Optimizer::Optimizer() {}

#if defined(HAVE_G2O)
void Optimizer::localBundleAdjustmentG2O(
    std::vector<KeyFrame> &keyframes,
    std::vector<MapPoint> &mappoints,
    const std::vector<int> &localKfIndices,
    const std::vector<int> &fixedKfIndices,
    double fx, double fy, double cx, double cy,
    int iterations) {

    if(localKfIndices.empty()) return;

    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
    auto blockSolver = std::make_unique<Block>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // Camera parameter (id = 0)
    g2o::CameraParameters* cam = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    cam->setId(0);
    optimizer.addParameter(cam);

    // Poses
    std::map<int, g2o::VertexSE3Expmap*> poseVertices;
    for (int idx : localKfIndices) {
        if (idx < 0 || idx >= static_cast<int>(keyframes.size())) continue;
        KeyFrame &kf = keyframes[idx];
        Eigen::Matrix3d R;
        cv2eigen(kf.R_w, R);
        Eigen::Vector3d t;
        cv2eigen(kf.t_w, t);
        g2o::SE3Quat pose(R, t);
        auto *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setId(idx);
        vSE3->setEstimate(pose);
        if (std::find(fixedKfIndices.begin(), fixedKfIndices.end(), idx) != fixedKfIndices.end()) vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        poseVertices[idx] = vSE3;
    }

    // Points
    const int POINT_ID_OFFSET = 10000;
    std::vector<int> mpVertexIds(mappoints.size(), -1);
    for (size_t i = 0; i < mappoints.size(); ++i) {
        const MapPoint &mp = mappoints[i];
        auto *vPoint = new g2o::VertexPointXYZ();
        vPoint->setEstimate(Eigen::Vector3d(mp.p.x, mp.p.y, mp.p.z));
        int vid = POINT_ID_OFFSET + static_cast<int>(i);
        vPoint->setId(vid);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);
        mpVertexIds[i] = vid;
    }

    // Observations / edges
    for (size_t i = 0; i < mappoints.size(); ++i) {
        const MapPoint &mp = mappoints[i];
        int pid = mpVertexIds[i];
        if (pid < 0) continue;
        for (const auto &obs : mp.observations) {
            int kfIdx = obs.first;
            int kpIdx = obs.second;
            if (poseVertices.find(kfIdx) == poseVertices.end()) continue;
            if (kfIdx < 0 || kfIdx >= static_cast<int>(keyframes.size())) continue;
            KeyFrame &kf = keyframes[kfIdx];
            if (kpIdx < 0 || kpIdx >= static_cast<int>(kf.kps.size())) continue;

            const KeyPoint &kp = kf.kps[kpIdx];
            Eigen::Vector2d meas(kp.pt.x, kp.pt.y);

            auto *edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, optimizer.vertex(pid));
            edge->setVertex(1, optimizer.vertex(kfIdx));
            edge->setMeasurement(meas);
            edge->information() = Eigen::Matrix2d::Identity();
            edge->setParameterId(0, cam->id());
            auto *rk = new g2o::RobustKernelHuber();
            rk->setDelta(1.0);
            edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(iterations);

    // Write back poses
    for (auto &kv : poseVertices) {
        int idx = kv.first;
        g2o::VertexSE3Expmap* v = kv.second;
        g2o::SE3Quat est = v->estimate();
        Eigen::Matrix3d R = est.rotation().toRotationMatrix();
        Eigen::Vector3d t = est.translation();
        Mat Rcv, tcv;
        eigen2cv(R, Rcv);
        eigen2cv(t, tcv);
        keyframes[idx].R_w = Rcv.clone();
        keyframes[idx].t_w = tcv.clone();
    }

    // Write back points
    for (size_t i = 0; i < mappoints.size(); ++i) {
        int pid = mpVertexIds[i];
        if (pid < 0) continue;
        auto *v = dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(pid));
        if (!v) continue;
        Eigen::Vector3d p = v->estimate();
        mappoints[i].p.x = p[0];
        mappoints[i].p.y = p[1];
        mappoints[i].p.z = p[2];
    }
}
#endif
#if defined(HAVE_SFM)
void Optimizer::localBundleAdjustmentSFM(
    std::vector<KeyFrame> &keyframes,
    std::vector<MapPoint> &mappoints,
    const std::vector<int> &localKfIndices,
    const std::vector<int> &fixedKfIndices,
    double fx, double fy, double cx, double cy,
    int iterations) {
    // Simple SFM-based local bundle adjustment using OpenCV routines.
    // This implementation alternates:
    //  - point-only Gauss-Newton updates (poses kept fixed)
    //  - pose-only optimization using PnP (points kept fixed)
    // It uses available jacobians w.r.t. point coordinates from
    // computeReprojectionError and the existing optimizePose() helper
    // to avoid depending on g2o / ceres. This provides an effective
    // local refinement for SLAM windows when full BA backends are
    // not available.

    if (localKfIndices.empty() || mappoints.empty()) return;

    // Fast lookup for whether a keyframe id is in the local window
    std::unordered_set<int> localSet(localKfIndices.begin(), localKfIndices.end());
    std::unordered_set<int> fixedSet(fixedKfIndices.begin(), fixedKfIndices.end());

    const double HUBER_DELTA = 3.0; // pixels
    const double MAX_POINT_STEP = 1.0; // meters (safeguard per-iter)

    // Outer iterations: alternate point updates and pose updates
    for (int iter = 0; iter < iterations; ++iter) {
        // --- Point-only Gauss-Newton update (poses fixed) ---
        for (size_t pid = 0; pid < mappoints.size(); ++pid) {
            MapPoint &mp = mappoints[pid];
            if (mp.isBad) continue;

            Mat JtJ = Mat::zeros(3, 3, CV_64F);
            Mat Jtr = Mat::zeros(3, 1, CV_64F);
            int obsCount = 0;

            for (const auto &obs : mp.observations) {
                int kfIdx = obs.first;
                int kpIdx = obs.second;
                if (localSet.find(kfIdx) == localSet.end()) continue;
                if (kfIdx < 0 || kfIdx >= static_cast<int>(keyframes.size())) continue;
                KeyFrame &kf = keyframes[kfIdx];
                if (kpIdx < 0 || kpIdx >= static_cast<int>(kf.kps.size())) continue;

                Point2f observed = kf.kps[kpIdx].pt;
                // Project current point into this keyframe
                Point2f proj = project(mp.p, kf.R_w, kf.t_w, fx, fy, cx, cy);
                if (proj.x < 0 || proj.y < 0) continue;
                // Compute jacobians (fills jacobianPoint)
                Mat jacPose, jacPoint;
                computeReprojectionError(mp.p, kf.R_w, kf.t_w, observed, fx, fy, cx, cy, jacPose, jacPoint);
                // jacPoint is 2x3, residual r = [u - u_obs; v - v_obs]
                Mat r = (Mat_<double>(2,1) << proj.x - observed.x, proj.y - observed.y);

                // robust weight (Huber)
                double err = std::sqrt(r.at<double>(0,0)*r.at<double>(0,0) + r.at<double>(1,0)*r.at<double>(1,0));
                double w = 1.0;
                if (err > HUBER_DELTA) w = HUBER_DELTA / err;

                // accumulate JtJ and Jtr
                Mat Jt = jacPoint.t(); // 3x2
                Mat W = Mat::eye(2,2,CV_64F) * w;
                Mat JtW = Jt * W; // 3x2
                JtJ += JtW * jacPoint; // 3x3
                Jtr += JtW * r; // 3x1
                obsCount++;
            }

            if (obsCount < 2) continue; // not enough constraints

            // Solve for delta using SVD for stability
            Mat delta; // 3x1
            bool solved = false;
            if (cv::determinant(JtJ) != 0) {
                solved = cv::solve(JtJ, -Jtr, delta, cv::DECOMP_SVD);
            } else {
                solved = cv::solve(JtJ, -Jtr, delta, cv::DECOMP_SVD);
            }
            if (!solved) continue;

            double step = std::sqrt(delta.at<double>(0,0)*delta.at<double>(0,0) +
                                    delta.at<double>(1,0)*delta.at<double>(1,0) +
                                    delta.at<double>(2,0)*delta.at<double>(2,0));
            if (step > MAX_POINT_STEP) {
                double scale = MAX_POINT_STEP / step;
                delta *= scale;
            }

            mp.p.x += delta.at<double>(0,0);
            mp.p.y += delta.at<double>(1,0);
            mp.p.z += delta.at<double>(2,0);
        }

        // --- Pose-only optimization (points fixed) ---
        // For each local keyframe that is not fixed, call optimizePose
        for (int kfId : localKfIndices) {
            if (fixedSet.find(kfId) != fixedSet.end()) continue;
            if (kfId < 0 || kfId >= static_cast<int>(keyframes.size())) continue;
            KeyFrame &kf = keyframes[kfId];
            // Build matchedMpIndices: for each keypoint in this KF, which mappoint index it corresponds to
            std::vector<int> matchedMpIndices(kf.kps.size(), -1);
            for (size_t mpIdx = 0; mpIdx < mappoints.size(); ++mpIdx) {
                const MapPoint &mp = mappoints[mpIdx];
                if (mp.isBad) continue;
                for (const auto &obs : mp.observations) {
                    if (obs.first == kfId) {
                        int kpIdx = obs.second;
                        if (kpIdx >= 0 && kpIdx < static_cast<int>(matchedMpIndices.size()))
                            matchedMpIndices[kpIdx] = static_cast<int>(mpIdx);
                    }
                }
            }
            std::vector<bool> inliers;
            // Use the same number of iterations as outer loop for RANSAC attempts
            optimizePose(kf, mappoints, matchedMpIndices, fx, fy, cx, cy, inliers, std::max(20, iterations));
        }
    }
}
#endif

bool Optimizer::optimizePose(
    KeyFrame &kf,
    const std::vector<MapPoint> &mappoints,
    const std::vector<int> &matchedMpIndices,
    double fx, double fy, double cx, double cy,
    std::vector<bool> &inliers,
    int iterations) {

    if(matchedMpIndices.empty()) return false;
    inliers.assign(matchedMpIndices.size(), false);

    std::vector<Point3d> objectPoints;
    std::vector<Point2f> imagePoints;
    for(size_t i = 0; i < matchedMpIndices.size(); ++i) {
        int mpIdx = matchedMpIndices[i];
        if(mpIdx < 0 || mpIdx >= static_cast<int>(mappoints.size())) continue;
        const MapPoint &mp = mappoints[mpIdx];
        if(mp.isBad) continue;
        if(i < kf.kps.size()) {
            objectPoints.emplace_back(mp.p.x, mp.p.y, mp.p.z);
            imagePoints.push_back(kf.kps[i].pt);
        }
    }
    if(objectPoints.size() < 4) return false;

    Mat K = (Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    Mat rvec, tvec, inliersMask;
    bool success = solvePnPRansac(objectPoints, imagePoints, K, Mat(), rvec, tvec,
                                      false, iterations, 2.0, 0.99, inliersMask);
    if(!success) return false;
    Mat R;
    Rodrigues(rvec, R);
    kf.R_w = R;
    kf.t_w = tvec;
    for(int i = 0; i < inliersMask.rows && i < static_cast<int>(inliers.size()); ++i)
        inliers[i] = (inliersMask.at<uchar>(i,0) != 0);

    return true;
}

#if defined(HAVE_SFM)
void Optimizer::globalBundleAdjustmentSFM(
    std::vector<KeyFrame> &keyframes,
    std::vector<MapPoint> &mappoints,
    double fx, double fy, double cx, double cy,
    int iterations) {

    std::cout << "Optimizer: Global BA with " << keyframes.size()
              << " KFs and " << mappoints.size() << " map points" << std::endl;
    std::vector<int> localKfIndices;
    for(size_t i = 1; i < keyframes.size(); ++i) localKfIndices.push_back(static_cast<int>(i));
    std::vector<int> fixedKfIndices = {0};
    localBundleAdjustmentSFM(keyframes, mappoints, localKfIndices, fixedKfIndices, fx, fy, cx, cy, iterations);
    std::cout << "Optimizer: Global BA completed" << std::endl;
}
#endif

double Optimizer::computeReprojectionError(
    const Point3d &point3D,
    const Mat &R, const Mat &t,
    const Point2f &observed,
    double fx, double fy, double cx, double cy,
    Mat &jacobianPose,
    Mat &jacobianPoint) {

    Mat Xw = (Mat_<double>(3,1) << point3D.x, point3D.y, point3D.z);
    Mat Xc = R.t() * (Xw - t);
    double x = Xc.at<double>(0,0), y = Xc.at<double>(1,0), z = Xc.at<double>(2,0);
    if(z <= 0) return std::numeric_limits<double>::max();
    double u = fx * (x / z) + cx;
    double v = fy * (y / z) + cy;
    double du = u - observed.x, dv = v - observed.y;
    jacobianPose = Mat::zeros(2,6,CV_64F);
    jacobianPoint = Mat::zeros(2,3,CV_64F);
    double invZ = 1.0 / z; double invZ2 = invZ * invZ;
    jacobianPoint.at<double>(0,0) = fx * invZ;
    jacobianPoint.at<double>(0,1) = 0;
    jacobianPoint.at<double>(0,2) = -fx * x * invZ2;
    jacobianPoint.at<double>(1,0) = 0;
    jacobianPoint.at<double>(1,1) = fy * invZ;
    jacobianPoint.at<double>(1,2) = -fy * y * invZ2;
    return std::sqrt(du*du + dv*dv);
}

Point2f Optimizer::project(
    const Point3d &point3D,
    const Mat &R, const Mat &t,
    double fx, double fy, double cx, double cy) {

    Mat Xw = (Mat_<double>(3,1) << point3D.x, point3D.y, point3D.z);
    Mat Xc = R.t() * (Xw - t);
    double z = Xc.at<double>(2,0);
    if(z <= 0) return Point2f(-1,-1);
    float u = static_cast<float>(fx * (Xc.at<double>(0,0) / z) + cx);
    float v = static_cast<float>(fy * (Xc.at<double>(1,0) / z) + cy);
    return Point2f(u,v);
}

} // namespace vo
} // namespace cv