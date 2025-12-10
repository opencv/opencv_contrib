#include "opencv2/slam/map.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <algorithm>
#include <limits>

namespace cv {
namespace vo {

MapManager::MapManager() {}

void MapManager::addKeyFrame(const KeyFrame &kf){
    keyframes_.push_back(kf);
    id2idx_[kf.id] = static_cast<int>(keyframes_.size()) - 1;
}

void MapManager::addMapPoints(const std::vector<MapPoint> &pts){
    for(auto p: pts) {
        if(p.id <= 0) p.id = next_mappoint_id_++;
        mpid2idx_[p.id] = static_cast<int>(mappoints_.size());
        mappoints_.push_back(p);
    }
}

std::vector<int> MapManager::findVisibleCandidates(const Mat &lastR, const Mat &lastT,
                                                   double fx, double fy, double cx, double cy,
                                                   int imgW, int imgH) const {
    std::vector<int> candidates; candidates.reserve(mappoints_.size());
    for(size_t mi=0; mi<mappoints_.size(); ++mi){
        const auto &mp = mappoints_[mi];
        Mat Xw = (Mat_<double>(3,1) << mp.p.x, mp.p.y, mp.p.z);
        Mat Xc = lastR.t() * (Xw - lastT);
        double z = Xc.at<double>(2,0);
        if(z <= 0) continue;
        double u = fx * (Xc.at<double>(0,0)/z) + cx;
        double v = fy * (Xc.at<double>(1,0)/z) + cy;
        if(u >= 0 && u < imgW && v >= 0 && v < imgH) candidates.push_back((int)mi);
    }
    return candidates;
}

std::vector<MapPoint> MapManager::triangulateBetweenLastTwo(const std::vector<Point2f> &pts1n,
                                                            const std::vector<Point2f> &pts2n,
                                                            const std::vector<int> &pts1_kp_idx,
                                                            const std::vector<int> &pts2_kp_idx,
                                                            const KeyFrame &lastKf, const KeyFrame &curKf,
                                                            double fx, double fy, double cx, double cy){
    std::vector<MapPoint> newPoints;
    if(pts1n.empty() || pts2n.empty()) return newPoints;
    // Relative pose from last -> current (we expect lastKf and curKf poses to be in world)
    // compute relative transformation
    // P1 = [I|0], P2 = [R_rel|t_rel] where R_rel = R_last^{-1} * R_cur, t_rel = R_last^{-1}*(t_cur - t_last)
    Mat R_last = lastKf.R_w, t_last = lastKf.t_w;
    Mat R_cur = curKf.R_w, t_cur = curKf.t_w;
    Mat R_rel = R_last.t() * R_cur;
    Mat t_rel = R_last.t() * (t_cur - t_last);
    Mat P1 = Mat::eye(3,4,CV_64F);
    Mat P2(3,4,CV_64F);
    for(int r=0;r<3;++r){
        for(int c=0;c<3;++c) P2.at<double>(r,c) = R_rel.at<double>(r,c);
        P2.at<double>(r,3) = t_rel.at<double>(r,0);
    }
    Mat points4D;
    try{ triangulatePoints(P1, P2, pts1n, pts2n, points4D); }
    catch(...) { points4D.release(); }
    if(points4D.empty()) return newPoints;
    Mat p4d64;
    if(points4D.type() != CV_64F) points4D.convertTo(p4d64, CV_64F); else p4d64 = points4D;
    for(int c=0;c<p4d64.cols; ++c){
        double w = p4d64.at<double>(3,c);
        if(std::abs(w) < 1e-8) continue;
        double Xx = p4d64.at<double>(0,c)/w;
        double Xy = p4d64.at<double>(1,c)/w;
        double Xz = p4d64.at<double>(2,c)/w;
        if(Xz <= 0) continue;
        Mat Xc = (Mat_<double>(3,1) << Xx, Xy, Xz);
        Mat Xw = lastKf.R_w * Xc + lastKf.t_w;
        MapPoint mp; mp.p = Point3d(Xw.at<double>(0,0), Xw.at<double>(1,0), Xw.at<double>(2,0));
        // compute reprojection error in both views (pixel coords)
        // project into last
        Mat Xc_last = Xc;
        double u1 = fx * (Xc_last.at<double>(0,0)/Xc_last.at<double>(2,0)) + cx;
        double v1 = fy * (Xc_last.at<double>(1,0)/Xc_last.at<double>(2,0)) + cy;
        // project into current: Xc_cur = R_rel * Xc + t_rel (we computed P2 earlier)
        Mat Xc_cur = R_rel * Xc + t_rel;
        double u2 = fx * (Xc_cur.at<double>(0,0)/Xc_cur.at<double>(2,0)) + cx;
        double v2 = fy * (Xc_cur.at<double>(1,0)/Xc_cur.at<double>(2,0)) + cy;
        // obtain observed pixel locations
        double obs_u1 = -1, obs_v1 = -1, obs_u2 = -1, obs_v2 = -1;
        if(c < static_cast<int>(pts1_kp_idx.size())){
            // pts1n/pts2n are normalized coords; but we were given kp indices - use them if valid
            int kp1 = pts1_kp_idx[c];
            if(kp1 >= 0 && kp1 < static_cast<int>(lastKf.kps.size())){
                obs_u1 = lastKf.kps[kp1].pt.x; obs_v1 = lastKf.kps[kp1].pt.y;
            }
        }
        if(c < static_cast<int>(pts2_kp_idx.size())){
            int kp2 = pts2_kp_idx[c];
            if(kp2 >= 0 && kp2 < static_cast<int>(curKf.kps.size())){
                obs_u2 = curKf.kps[kp2].pt.x; obs_v2 = curKf.kps[kp2].pt.y;
            }
        }
        // if we have observed pixel locations, check reprojection error
        bool pass = true;
        const double MAX_REPROJ_PX = 2.0;
        if(obs_u1 >= 0 && obs_v1 >= 0){
            double e1 = std::hypot(u1 - obs_u1, v1 - obs_v1);
            if(e1 > MAX_REPROJ_PX) pass = false;
        }
        if(obs_u2 >= 0 && obs_v2 >= 0){
            double e2 = std::hypot(u2 - obs_u2, v2 - obs_v2);
            if(e2 > MAX_REPROJ_PX) pass = false;
        }
        // parallax check: angle between viewing rays (in last frame)
        Mat ray1 = Xc_last / norm(Xc_last);
        Mat ray2 = Xc_cur / norm(Xc_cur);
        double cos_par = ray1.dot(ray2);
        double parallax = std::acos(std::min(1.0, std::max(-1.0, cos_par)));
        const double MIN_PARALLAX_RAD = 1.0 * CV_PI / 180.0; // 1 degree
        if(parallax < MIN_PARALLAX_RAD) pass = false;
        if(!pass) continue;
        // attach observations using provided keypoint indices when available
        int kp1idx = (c < static_cast<int>(pts1_kp_idx.size())) ? pts1_kp_idx[c] : -1;
        int kp2idx = (c < static_cast<int>(pts2_kp_idx.size())) ? pts2_kp_idx[c] : -1;
        if(kp1idx >= 0) mp.observations.emplace_back(lastKf.id, kp1idx);
        if(kp2idx >= 0) mp.observations.emplace_back(curKf.id, kp2idx);
        // assign id
        mp.id = next_mappoint_id_++;
        mpid2idx_[mp.id] = static_cast<int>(mappoints_.size()) + static_cast<int>(newPoints.size());
        newPoints.push_back(mp);
    }
    // append to internal map
    for(const auto &p: newPoints) mappoints_.push_back(p);
    return newPoints;
}

int MapManager::keyframeIndex(int id) const{
    auto it = id2idx_.find(id);
    if(it == id2idx_.end()) return -1;
    return it->second;
}

int MapManager::mapPointIndex(int id) const{
    auto it = mpid2idx_.find(id);
    if(it == mpid2idx_.end()) return -1;
    return it->second;
}

void MapManager::applyOptimizedKeyframePose(int keyframeId, const Mat &R, const Mat &t){
    int idx = keyframeIndex(keyframeId);
    if(idx < 0 || idx >= static_cast<int>(keyframes_.size())) return;
    keyframes_[idx].R_w = R.clone();
    keyframes_[idx].t_w = t.clone();
}

void MapManager::applyOptimizedMapPoint(int mappointId, const Point3d &p){
    int idx = mapPointIndex(mappointId);
    if(idx < 0 || idx >= static_cast<int>(mappoints_.size())) return;
    mappoints_[idx].p = p;
}

void MapManager::cullBadMapPoints() {
    const double MAX_REPROJ_ERROR = 3.0;  // pixels
    const int MIN_OBSERVATIONS = 2;
    const float MIN_FOUND_RATIO = 0.25f;

    for(auto &mp : mappoints_) {
        if(mp.isBad) continue;

        // 1. Check observation count
        mp.nObs = static_cast<int>(mp.observations.size());
        if(mp.nObs < MIN_OBSERVATIONS) {
            mp.isBad = true;
            continue;
        }

        // 2. Check found ratio (avoid points rarely tracked)
        if(mp.nVisible > 10 && mp.getFoundRatio() < MIN_FOUND_RATIO) {
            mp.isBad = true;
            continue;
        }

        // 3. Check reprojection error across observations
        // Sample a few keyframes to check reprojection
        int errorCount = 0;
        int checkCount = 0;
        for(const auto &obs : mp.observations) {
            int kfId = obs.first;
            int kfIdx = keyframeIndex(kfId);
            if(kfIdx < 0 || kfIdx >= static_cast<int>(keyframes_.size())) continue;

            const KeyFrame &kf = keyframes_[kfIdx];
            // Use default camera params (should be passed in production code)
            double fx = 500.0, fy = 500.0, cx = 320.0, cy = 240.0;
            double error = computeReprojError(mp, kf, fx, fy, cx, cy);

            checkCount++;
            if(error > MAX_REPROJ_ERROR) {
                errorCount++;
            }

            // Sample up to 3 observations for efficiency
            if(checkCount >= 3) break;
        }

        // If majority of samples have high error, mark as bad
        if(checkCount > 0 && errorCount > checkCount / 2) {
            mp.isBad = true;
        }
    }

    // Remove bad points
    size_t before = mappoints_.size();
    mappoints_.erase(
        std::remove_if(mappoints_.begin(), mappoints_.end(),
            [](const MapPoint &p) { return p.isBad; }),
        mappoints_.end()
    );
    size_t after = mappoints_.size();

    if(before - after > 0) {
        std::cout << "MapManager: culled " << (before - after) << " bad map points ("
                  << after << " remain)" << std::endl;
    }
}

double MapManager::computeReprojError(const MapPoint &mp, const KeyFrame &kf,
                                      double fx, double fy, double cx, double cy) const {
    // Transform world point to camera frame
    Mat Xw = (Mat_<double>(3,1) << mp.p.x, mp.p.y, mp.p.z);
    Mat Xc = kf.R_w.t() * (Xw - kf.t_w);

    double z = Xc.at<double>(2, 0);
    if(z <= 0) return std::numeric_limits<double>::max();

    // Project to image
    double u = fx * (Xc.at<double>(0, 0) / z) + cx;
    double v = fy * (Xc.at<double>(1, 0) / z) + cy;

    // Find corresponding observation in this keyframe
    Point2f observed(-1, -1);
    for(const auto &obs : mp.observations) {
        if(obs.first == kf.id) {
            int kpIdx = obs.second;
            if(kpIdx >= 0 && kpIdx < static_cast<int>(kf.kps.size())) {
                observed = kf.kps[kpIdx].pt;
                break;
            }
        }
    }

    if(observed.x < 0) return std::numeric_limits<double>::max();

    // Compute reprojection error
    double dx = u - observed.x;
    double dy = v - observed.y;
    return std::sqrt(dx * dx + dy * dy);
}

void MapManager::updateMapPointDescriptor(MapPoint &mp) {
    if(mp.observations.empty()) return;

    // Collect all descriptors from observations
    std::vector<Mat> descriptors;
    for(const auto &obs : mp.observations) {
        int kfIdx = keyframeIndex(obs.first);
        if(kfIdx < 0) continue;

        const KeyFrame &kf = keyframes_[kfIdx];
        int kpIdx = obs.second;
        if(kpIdx >= 0 && kpIdx < kf.desc.rows) {
            descriptors.push_back(kf.desc.row(kpIdx));
        }
    }

    if(descriptors.empty()) return;

    // Compute median descriptor (for binary descriptors, use majority voting per bit)
    if(descriptors[0].type() == CV_8U) {
        // Binary descriptor (ORB)
        int bytes = descriptors[0].cols;
        Mat median = Mat::zeros(1, bytes, CV_8U);

        for(int b = 0; b < bytes; ++b) {
            int bitCount[8] = {0};
            for(const auto &desc : descriptors) {
                uchar byte = desc.at<uchar>(0, b);
                for(int bit = 0; bit < 8; ++bit) {
                    if(byte & (1 << bit)) bitCount[bit]++;
                }
            }

            uchar medianByte = 0;
            int threshold = descriptors.size() / 2;
            for(int bit = 0; bit < 8; ++bit) {
                if(bitCount[bit] > threshold) {
                    medianByte |= (1 << bit);
                }
            }
            median.at<uchar>(0, b) = medianByte;
        }

        mp.descriptor = median;
    } else {
        // Fallback: use first descriptor
        mp.descriptor = descriptors[0].clone();
    }
}

int MapManager::countGoodMapPoints() const {
    int count = 0;
    for(const auto &mp : mappoints_) {
        if(!mp.isBad) count++;
    }
    return count;
}

} // namespace vo
} // namespace cv