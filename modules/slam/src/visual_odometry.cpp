// SPDX-License-Identifier: Apache-2.0
#include "opencv2/slam/visual_odometry.hpp"
#include "opencv2/slam/initializer.hpp"
#include "opencv2/slam/localizer.hpp"
#include "opencv2/slam/pose.hpp"
#include "opencv2/slam/map.hpp"
#include "opencv2/slam/keyframe.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <algorithm>
#include <numeric>

namespace cv {
namespace vo {

class VisualOdometry::Impl {
public:
    Impl(Ptr<Feature2D> det, Ptr<DescriptorMatcher> matcher)
        : detector_(std::move(det)), matcher_(std::move(matcher)) {
        if(!detector_) detector_ = ORB::create(2000);
        if(!matcher_) matcher_ = BFMatcher::create(NORM_HAMMING);
    }

    TrackingResult track(InputArray frame, double timestamp, MapManager* map, bool allowMapping);
    void setOptions(const VisualOdometryOptions& opts) { options_ = opts; }
    VisualOdometryOptions getOptions() const { return options_; }
    void setIntrinsics(double fx, double fy, double cx, double cy){
        fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy; hasIntrinsics_ = (fx_ > 1e-9 && fy_ > 1e-9);
    }
    void setScale(double s){ scale_ = std::max(1e-9, s); }
    void reset(){
        prevGray_.release(); prevDesc_.release(); prevKps_.clear(); prevColor_.release();
        frameId_ = 0; state_ = TrackingState::NOT_INITIALIZED;
        R_w_ = Mat::eye(3,3,CV_64F); t_w_ = Mat::zeros(3,1,CV_64F);
    }
    int getFrameId() const { return frameId_; }
    int getState() const { return static_cast<int>(state_); }
    void getPose(OutputArray R_out, OutputArray t_out) const {
        if(R_out.needed()) R_w_.copyTo(R_out);
        if(t_out.needed()) t_w_.copyTo(t_out);
    }

private:
    Ptr<Feature2D> detector_;
    Ptr<DescriptorMatcher> matcher_;
    VisualOdometryOptions options_;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    bool hasIntrinsics_ = false;
    double scale_ = 1.0;

    int frameId_ = 0;
    TrackingState state_ = TrackingState::NOT_INITIALIZED;
    Mat R_w_ = Mat::eye(3,3,CV_64F);
    Mat t_w_ = Mat::zeros(3,1,CV_64F);

    Mat prevGray_;
    Mat prevDesc_;
    std::vector<KeyPoint> prevKps_;
    Mat prevColor_;

    Initializer initializer_;
    PoseEstimator poseEst_;
    Localizer localizer_{0.7f};
};

static std::vector<DMatch> mutualRatioMatches(const Mat& desc1, const Mat& desc2, DescriptorMatcher& matcher, float ratio){
    std::vector<std::vector<DMatch>> knn12, knn21;
    matcher.knnMatch(desc1, desc2, knn12, 2);
    matcher.knnMatch(desc2, desc1, knn21, 2);
    std::vector<DMatch> good;
    good.reserve(knn12.size());
    for(size_t qi = 0; qi < knn12.size(); ++qi){
        if(knn12[qi].empty()) continue;
        const DMatch &best = knn12[qi][0];
        if(knn12[qi].size() >= 2){
            const DMatch &second = knn12[qi][1];
            if(second.distance > 0.0f && best.distance / second.distance > ratio) continue;
        }
        int trainIdx = best.trainIdx;
        if(trainIdx < 0 || trainIdx >= (int)knn21.size() || knn21[trainIdx].empty()) continue;
        const DMatch &rbest = knn21[trainIdx][0];
        if(rbest.trainIdx == (int)qi) good.push_back(best);
    }
    return good;
}

TrackingResult VisualOdometry::Impl::track(InputArray frameIn, double timestamp, MapManager* map, bool allowMapping){
    TrackingResult res; res.frameId = frameId_; res.timestamp = timestamp; res.state = static_cast<int>(state_);

    if(!hasIntrinsics_){
        res.state = static_cast<int>(state_);
        return res;
    }

    Mat frame = frameIn.getMat();
    Mat color = frame.channels() == 1 ? Mat() : frame;
    if(color.empty()){ cvtColor(frame, color, COLOR_GRAY2BGR); }
    Mat gray = frame;
    if(gray.channels() > 1) cvtColor(gray, gray, COLOR_BGR2GRAY);

    std::vector<KeyPoint> kps; Mat desc;
    detector_->detect(gray, kps);
    detector_->compute(gray, kps, desc);

    if(prevGray_.empty() || prevDesc_.empty()){
        prevGray_ = gray.clone(); prevDesc_ = desc.clone(); prevKps_ = kps; prevColor_ = color.clone();
        state_ = TrackingState::INITIALIZING;
        res.state = static_cast<int>(state_);
        frameId_++;
        return res;
    }

    if(desc.empty() || prevDesc_.empty()){
        state_ = TrackingState::LOST;
        res.state = static_cast<int>(state_);
        prevGray_ = gray.clone(); prevDesc_ = desc.clone(); prevKps_ = kps; prevColor_ = color.clone();
        frameId_++;
        return res;
    }

    auto matches = mutualRatioMatches(prevDesc_, desc, *matcher_, 0.75f);
    std::vector<Point2f> pts1, pts2;
    pts1.reserve(matches.size()); pts2.reserve(matches.size());
    for(const auto &m: matches){
        pts1.push_back(prevKps_[m.queryIdx].pt);
        pts2.push_back(kps[m.trainIdx].pt);
    }

    double median_flow = 0.0;
    if(!pts1.empty()){
        std::vector<double> flows; flows.reserve(pts1.size());
        for(size_t i=0;i<pts1.size();++i){
            double dx = pts2[i].x - pts1[i].x;
            double dy = pts2[i].y - pts1[i].y;
            flows.push_back(std::sqrt(dx*dx + dy*dy));
        }
        auto tmp = flows; size_t mid = tmp.size()/2; std::nth_element(tmp.begin(), tmp.begin()+mid, tmp.end()); median_flow = tmp[mid];
    }

    // Localization mode: do not modify the map, use PnP against an existing map with quality checks.
    if(map && !allowMapping){
        Mat R_pnp, t_pnp; int inliers_pnp = 0; bool ok_pnp = false;
        int postMatches = 0; double meanReproj = std::numeric_limits<double>::infinity();
        if(!map->keyframes().empty()){
            ok_pnp = localizer_.tryPnP(*map, desc, kps, fx_, fy_, cx_, cy_, gray.cols, gray.rows,
                                       options_.minInliers, R_pnp, t_pnp, inliers_pnp, frameId_, nullptr, "",
                                       nullptr, &postMatches, &meanReproj);
            if(ok_pnp){
                if(postMatches > 0 && inliers_pnp < static_cast<int>(postMatches * options_.minInlierRatio)) ok_pnp = false;
                if(meanReproj > 5.0) ok_pnp = false; // loose but guards against bad PnP
            }
        }
        if(ok_pnp){
            R_pnp.convertTo(R_w_, CV_64F);
            t_pnp.convertTo(t_w_, CV_64F);
            state_ = TrackingState::TRACKING;
            res.ok = true;
            res.R_w = R_w_.clone();
            res.t_w = t_w_.clone();
            res.numMatches = 0;
            res.numInliers = inliers_pnp;
        } else {
            state_ = TrackingState::LOST;
        }
        res.state = static_cast<int>(state_);
        prevGray_ = gray.clone(); prevDesc_ = desc.clone(); prevKps_ = kps; prevColor_ = color.clone();
        frameId_++;
        return res;
    }

    // try two-view initialization if map is empty and this is second frame
    if(map && allowMapping && map->keyframes().empty() && frameId_ == 1){
        Mat R_init, t_init; std::vector<Point3d> pts3D; std::vector<bool> isTri;
        bool okInit = initializer_.initialize(prevKps_, kps, matches, fx_, fy_, cx_, cy_, R_init, t_init, pts3D, isTri);
        if(okInit){
            Mat prevImg = prevColor_.empty() ? prevGray_ : prevColor_;
            KeyFrame kf0(frameId_ - 1, prevImg, prevKps_, prevDesc_, Mat::eye(3,3,CV_64F), Mat::zeros(3,1,CV_64F));
            Mat Rwc1 = R_init.t();
            Mat Cw1 = (-Rwc1 * t_init) * scale_;
            KeyFrame kf1(frameId_, color, kps, desc, Rwc1, Cw1);

            std::vector<MapPoint> newMps; newMps.reserve(pts3D.size());
            for(size_t i=0;i<pts3D.size();++i){
                if(!isTri[i]) continue;
                MapPoint mp; mp.p = Point3d(pts3D[i].x * scale_, pts3D[i].y * scale_, pts3D[i].z * scale_);
                if(i < matches.size()){
                    const DMatch &m = matches[i];
                    mp.observations.emplace_back(kf0.id, m.queryIdx);
                    mp.observations.emplace_back(kf1.id, m.trainIdx);
                }
                newMps.push_back(mp);
            }
            map->addKeyFrame(kf0);
            map->addKeyFrame(kf1);
            if(!newMps.empty()) map->addMapPoints(newMps);

            R_w_ = kf1.R_w.clone();
            t_w_ = kf1.t_w.clone();
            state_ = TrackingState::TRACKING;
            res.ok = true; res.state = static_cast<int>(state_);
            res.R_w = R_w_.clone(); res.t_w = t_w_.clone();
            res.keyframeInserted = true; res.numMatches = static_cast<int>(matches.size());
            res.numInliers = static_cast<int>(matches.size());
            prevGray_ = gray.clone(); prevDesc_ = desc.clone(); prevKps_ = kps; prevColor_ = color.clone();
            frameId_++;
            return res;
        }
    }

    Mat R_est, t_est, mask_est; int inliers_est = 0; bool ok_est = false;
    if(pts1.size() >= static_cast<size_t>(std::max(8, options_.minMatches))){
        ok_est = poseEst_.estimate(pts1, pts2, fx_, fy_, cx_, cy_, R_est, t_est, mask_est, inliers_est);
    }

    Mat R_pnp, t_pnp; int inliers_pnp = 0; bool ok_pnp = false; int pnpMatches = 0; double pnpMeanReproj = std::numeric_limits<double>::infinity();
    if(map && !map->keyframes().empty()){
        ok_pnp = localizer_.tryPnP(*map, desc, kps, fx_, fy_, cx_, cy_, gray.cols, gray.rows,
                                   options_.minInliers, R_pnp, t_pnp, inliers_pnp, frameId_, nullptr, "",
                                   nullptr, &pnpMatches, &pnpMeanReproj);
        if(ok_pnp){
            if(pnpMatches > 0 && inliers_pnp < static_cast<int>(pnpMatches * options_.minInlierRatio)) ok_pnp = false;
            if(pnpMeanReproj > 5.0) ok_pnp = false;
        }
    }

    Mat R_use, t_use; int inliers_use = 0; int matchCount = static_cast<int>(matches.size());
    if(ok_est){ R_use = R_est; t_use = t_est; inliers_use = inliers_est; }
    else if(ok_pnp){ R_use = R_pnp; t_use = t_pnp; inliers_use = inliers_pnp; }

    bool integrate = ok_est || ok_pnp;
    if(integrate){
        if(inliers_use < options_.minInliers || matchCount < options_.minMatches) integrate = false;
        if(ok_est){
            Mat t_d; t_use.convertTo(t_d, CV_64F);
            Mat R_d; R_use.convertTo(R_d, CV_64F);
            double t_norm = norm(t_d);
            double trace = R_d.at<double>(0,0) + R_d.at<double>(1,1) + R_d.at<double>(2,2);
            double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
            double rot_angle = std::acos(cos_angle);
            if(t_norm < options_.minTranslationNorm && std::abs(rot_angle) < options_.minRotationRad && median_flow < options_.flowZeroThresh){
                integrate = false;
            }
        }
    }

    if(integrate){
        if(ok_est){
            Mat R_d, t_d; R_use.convertTo(R_d, CV_64F); t_use.convertTo(t_d, CV_64F);
            Mat C2_in_c1 = (-R_d.t() * t_d) * scale_;
            t_w_ = t_w_ + R_w_ * C2_in_c1;
            R_w_ = R_w_ * R_d.t();
        } else {
            R_use.convertTo(R_w_, CV_64F);
            t_use.convertTo(t_w_, CV_64F);
        }
        state_ = TrackingState::TRACKING;
        res.ok = true;
        res.R_w = R_w_.clone(); res.t_w = t_w_.clone();
    } else {
        state_ = ok_est || ok_pnp ? TrackingState::INITIALIZING : TrackingState::LOST;
    }

    res.state = static_cast<int>(state_);
    res.numMatches = matchCount; res.numInliers = inliers_use;

    // keyframe decision
    if(map && allowMapping && res.ok){
        bool insertKf = false;
        const auto &kfs = map->keyframes();
        const int minGap = std::max(1, options_.keyframeMinGap);
        const int maxGap = std::max(minGap, options_.keyframeMaxGap);
        const int lastKfId = kfs.empty() ? -100000 : kfs.back().id;
        const int gap = frameId_ - lastKfId;
        if(gap >= maxGap) insertKf = true;

        // simplistic parallax proxy based on median flow
        if(!insertKf && gap >= minGap){
            if(median_flow >= options_.keyframeMinParallaxPx) insertKf = true;
        }

        // require reasonable inlier ratio before inserting a keyframe
        if(matchCount > 0 && inliers_use < static_cast<int>(matchCount * options_.minInlierRatio)) insertKf = false;

        if(insertKf){
            KeyFrame kf(frameId_, color, kps, desc, R_w_, t_w_);
            map->addKeyFrame(kf);
            res.keyframeInserted = true;

            // triangulate with last two keyframes if possible
            if(map->keyframes().size() >= 2){
                const KeyFrame &lastKf = map->keyframes()[map->keyframes().size() - 2];
                const KeyFrame &curKf = map->keyframes()[map->keyframes().size() - 1];

                // Build normalized coordinate matches from last -> cur using descriptors
                std::vector<DMatch> kfMatches = mutualRatioMatches(lastKf.desc, curKf.desc, *matcher_, 0.75f);
                std::vector<Point2f> pts1n, pts2n;
                std::vector<int> idx1, idx2;
                pts1n.reserve(kfMatches.size()); pts2n.reserve(kfMatches.size());
                idx1.reserve(kfMatches.size()); idx2.reserve(kfMatches.size());
                for(const auto &m : kfMatches){
                    Point2f p1 = lastKf.kps[m.queryIdx].pt;
                    Point2f p2 = curKf.kps[m.trainIdx].pt;
                    pts1n.emplace_back((p1.x - cx_) / fx_, (p1.y - cy_) / fy_);
                    pts2n.emplace_back((p2.x - cx_) / fx_, (p2.y - cy_) / fy_);
                    idx1.push_back(m.queryIdx);
                    idx2.push_back(m.trainIdx);
                }
                auto newMps = map->triangulateBetweenLastTwo(pts1n, pts2n, idx1, idx2, lastKf, curKf, fx_, fy_, cx_, cy_);
                if(!newMps.empty()) map->addMapPoints(newMps);
            }
        }
    }

    prevGray_ = gray.clone(); prevDesc_ = desc.clone(); prevKps_ = kps; prevColor_ = color.clone();
    frameId_++;
    return res;
}

VisualOdometry::VisualOdometry(Ptr<Feature2D> detector, Ptr<DescriptorMatcher> matcher)
    : impl_(std::make_unique<Impl>(std::move(detector), std::move(matcher))) {}

VisualOdometry::~VisualOdometry() = default;

void VisualOdometry::setCameraIntrinsics(double fx, double fy, double cx, double cy){ impl_->setIntrinsics(fx, fy, cx, cy); }
void VisualOdometry::setWorldScale(double scale){ impl_->setScale(scale); }
void VisualOdometry::setOptions(const VisualOdometryOptions& options){ impl_->setOptions(options); }
VisualOdometryOptions VisualOdometry::getOptions() const { return impl_->getOptions(); }
void VisualOdometry::reset(){ impl_->reset(); }
TrackingResult VisualOdometry::track(InputArray frame, double timestamp){ return impl_->track(frame, timestamp, nullptr, true); }
TrackingResult VisualOdometry::track(InputArray frame, double timestamp, MapManager* map){ return impl_->track(frame, timestamp, map, true); }
TrackingResult VisualOdometry::track(InputArray frame, double timestamp, MapManager* map, bool allowMapping){ return impl_->track(frame, timestamp, map, allowMapping); }
int VisualOdometry::getState() const { return impl_->getState(); }
int VisualOdometry::getFrameId() const { return impl_->getFrameId(); }
void VisualOdometry::getCurrentPose(OutputArray R_out, OutputArray t_out) const { impl_->getPose(R_out, t_out); }

} // namespace vo
} // namespace cv

