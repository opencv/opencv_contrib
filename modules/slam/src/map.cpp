#include "opencv2/slam/map.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <limits>

namespace cv {
namespace vo {

MapManager::MapManager() {}

void MapManager::setCameraIntrinsics(double fx, double fy, double cx, double cy){
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    hasIntrinsics_ = (fx_ > 1e-9 && fy_ > 1e-9);
}

void MapManager::setMapPointCullingParams(int minObservations, double minFoundRatio, double maxReprojErrorPx, int maxPointsKeep){
    mpMinObservations_ = std::max(2, minObservations);
    mpMinFoundRatio_ = std::max(0.0, std::min(1.0, minFoundRatio));
    mpMaxReprojErrorPx_ = std::max(0.1, maxReprojErrorPx);
    mpMaxPointsKeep_ = std::max(100, maxPointsKeep);
}

void MapManager::setRedundantKeyframeCullingParams(int minKeyframeObs, int minPointObs, double redundantRatio){
    kfRedundantMinObs_ = std::max(10, minKeyframeObs);
    kfRedundantMinPointObs_ = std::max(2, minPointObs);
    kfRedundantRatio_ = std::max(0.0, std::min(1.0, redundantRatio));
}

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

void MapManager::rebuildKeyframeIndex_(){
    id2idx_.clear();
    id2idx_.reserve(keyframes_.size());
    for(size_t i = 0; i < keyframes_.size(); ++i){
        id2idx_[keyframes_[i].id] = static_cast<int>(i);
    }
}

void MapManager::rebuildMapPointIndex_(){
    mpid2idx_.clear();
    mpid2idx_.reserve(mappoints_.size());
    for(size_t i = 0; i < mappoints_.size(); ++i){
        mpid2idx_[mappoints_[i].id] = static_cast<int>(i);
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
    // Relative pose from last camera -> current camera.
    // Pose convention: R_w is camera->world rotation, t_w is camera center in world (C_w).
    // For triangulatePoints we need: X_cur = R * X_last + t, both in camera coordinates.
    // With this convention:
    //   R = R_cur^T * R_last
    //   t = R_cur^T * (C_last - C_cur)
    Mat R_last = lastKf.R_w, C_last = lastKf.t_w;
    Mat R_cur = curKf.R_w, C_cur = curKf.t_w;
    Mat R_rel = R_cur.t() * R_last;
    Mat t_rel = R_cur.t() * (C_last - C_cur);
    Mat P1 = Mat::eye(3,4,CV_64F);
    Mat P2(3,4,CV_64F);
    for(int r=0;r<3;++r){
        for(int c=0;c<3;++c) P2.at<double>(r,c) = R_rel.at<double>(r,c);
        P2.at<double>(r,3) = t_rel.at<double>(r,0);
    }
    Mat points4D;
    try {
        triangulatePoints(P1, P2, pts1n, pts2n, points4D);
    } catch(const cv::Exception &e) {
        CV_LOG_DEBUG(NULL, "triangulatePoints failed");
        points4D.release();
    } catch(const std::exception &e) {
        CV_LOG_DEBUG(NULL, "triangulatePoints failed");
        points4D.release();
    } catch(...) {
        CV_LOG_DEBUG(NULL, "triangulatePoints failed");
        points4D.release();
    }
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
        MapPoint mp; 
        mp.p = Point3d(Xw.at<double>(0,0), Xw.at<double>(1,0), Xw.at<double>(2,0));
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
        // DO NOT assign global id or insert into internal containers here.
        // Return newly triangulated points to caller and let the caller add them
        // into the MapManager via `addMapPoints` so keyframes are present first.
        newPoints.push_back(mp);
    }
    // Note: do NOT append to internal map here. Caller should add returned points
    // via MapManager::addMapPoints after ensuring the corresponding keyframe
    // has been inserted into the map (avoids transient observations to missing KFs).
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
    const int MIN_OBSERVATIONS = mpMinObservations_;
    const float MIN_FOUND_RATIO = static_cast<float>(mpMinFoundRatio_);
    const size_t MAX_POINTS_KEEP = static_cast<size_t>(mpMaxPointsKeep_);
    const double MAX_REPROJ_ERROR = mpMaxReprojErrorPx_;

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

        // 3. Check reprojection error across observations (if intrinsics known)
        if(hasIntrinsics_){
            int errorCount = 0;
            int checkCount = 0;
            for(const auto &obs : mp.observations) {
                int kfId = obs.first;
                int kfIdx = keyframeIndex(kfId);
                if(kfIdx < 0 || kfIdx >= static_cast<int>(keyframes_.size())) continue;
                const KeyFrame &kf = keyframes_[kfIdx];
                double error = computeReprojError(mp, kf, fx_, fy_, cx_, cy_);

                checkCount++;
                if(error > MAX_REPROJ_ERROR) errorCount++;
                if(checkCount >= 3) break;
            }

            if(checkCount > 0 && errorCount > checkCount / 2) mp.isBad = true;
        }
    }

    // Gather good points
    std::vector<size_t> goodIdx; goodIdx.reserve(mappoints_.size());
    for(size_t i = 0; i < mappoints_.size(); ++i){
        if(!mappoints_[i].isBad) goodIdx.push_back(i);
    }

    // Enforce hard cap with score-based retention when too many points survive
    if(goodIdx.size() > MAX_POINTS_KEEP){
        struct ScoredIdx { double score; size_t idx; };
        std::vector<ScoredIdx> scored; scored.reserve(goodIdx.size());
        for(size_t idx : goodIdx){
            const auto &mp = mappoints_[idx];
            double found = static_cast<double>(std::max(0.05f, mp.getFoundRatio()));
            // Prefer points with many observations and high found ratio.
            // If intrinsics are known, also prefer low reprojection error.
            double reproj = 0.0;
            if(hasIntrinsics_){
                std::vector<double> errs;
                errs.reserve(3);
                int checked = 0;
                for(const auto &obs : mp.observations){
                    int kfIdx = keyframeIndex(obs.first);
                    if(kfIdx < 0 || kfIdx >= static_cast<int>(keyframes_.size())) continue;
                    errs.push_back(computeReprojError(mp, keyframes_[kfIdx], fx_, fy_, cx_, cy_));
                    if(++checked >= 3) break;
                }
                if(!errs.empty()){
                    size_t mid = errs.size()/2;
                    std::nth_element(errs.begin(), errs.begin()+mid, errs.end());
                    reproj = errs[mid];
                }
            }
            double score = static_cast<double>(mp.nObs) * found / (1.0 + reproj);
            scored.push_back({score, idx});
        }
        std::nth_element(scored.begin(), scored.begin() + static_cast<long>(MAX_POINTS_KEEP), scored.end(),
                         [](const ScoredIdx &a, const ScoredIdx &b){ return a.score > b.score; });
        for(size_t i = MAX_POINTS_KEEP; i < scored.size(); ++i){
            mappoints_[scored[i].idx].isBad = true;
        }
    }
    size_t before = mappoints_.size();
    mappoints_.erase(
        std::remove_if(mappoints_.begin(), mappoints_.end(),
            [](const MapPoint &p) { return p.isBad; }),
        mappoints_.end()
    );
    size_t after = mappoints_.size();

    if(after != before) rebuildMapPointIndex_();
}

void MapManager::cullRedundantKeyFrames(int maxCullPerCall){
    if(maxCullPerCall <= 0) return;
    if(keyframes_.size() <= 3) return;

    const int protectFirst = 2;
    const int protectLast = 2;
    if(static_cast<int>(keyframes_.size()) <= protectFirst + protectLast) return;

    int removed = 0;
    for(int kfi = protectFirst; kfi < static_cast<int>(keyframes_.size()) - protectLast; ++kfi){
        const int kfId = keyframes_[kfi].id;

        int totalObs = 0;
        int redundantObs = 0;
        for(auto &mp : mappoints_){
            if(mp.isBad) continue;
            bool observedHere = false;
            for(const auto &obs : mp.observations){
                if(obs.first == kfId){ observedHere = true; break; }
            }
            if(!observedHere) continue;
            totalObs++;
            if(static_cast<int>(mp.observations.size()) >= kfRedundantMinPointObs_) redundantObs++;
        }
        if(totalObs < kfRedundantMinObs_) continue;
        const double ratio = static_cast<double>(redundantObs) / static_cast<double>(totalObs);
        if(ratio < kfRedundantRatio_) continue;

        // Remove this keyframe: erase its observations from map points
        for(auto &mp : mappoints_){
            if(mp.isBad) continue;
            auto &obs = mp.observations;
            obs.erase(std::remove_if(obs.begin(), obs.end(), [&](const std::pair<int,int> &o){ return o.first == kfId; }), obs.end());
            mp.nObs = static_cast<int>(obs.size());
            if(mp.nObs < 2) mp.isBad = true;
        }
        keyframes_.erase(keyframes_.begin() + kfi);
        rebuildKeyframeIndex_();
        removed++;
        if(removed >= maxCullPerCall) break;
        kfi--; // re-check current index after erase
    }

    if(removed > 0) cullBadMapPoints();
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
        int numDesc = static_cast<int>(descriptors.size());
        int threshold = numDesc / 2;

        cv::parallel_for_(cv::Range(0, bytes), [&](const cv::Range &r){
            for(int b = r.start; b < r.end; ++b){
                int bitCount[8] = {0};
                for(int i = 0; i < numDesc; ++i){
                    const uchar *data = descriptors[i].ptr<uchar>(0);
                    uchar byte = data[b];
                    for(int bit = 0; bit < 8; ++bit){
                        if(byte & (1 << bit)) bitCount[bit]++;
                    }
                }
                uchar medianByte = 0;
                for(int bit = 0; bit < 8; ++bit){
                    if(bitCount[bit] > threshold) medianByte |= (1 << bit);
                }
                median.at<uchar>(0, b) = medianByte;
            }
        });

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