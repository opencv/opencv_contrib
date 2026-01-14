// SPDX-License-Identifier: Apache-2.0
#include "opencv2/slam/slam_system.hpp"
#include "opencv2/slam/optimizer.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <iomanip>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/calib3d.hpp>
#if defined(HAVE_DBOW3)
#include <DBoW3/DBoW3.h>
#endif
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

namespace cv {
namespace vo {

struct TrajEntry { int frameId = -1; double ts = 0.0; Mat R; Mat t; };

static cv::Vec4d rotationMatrixToQuaternion_(const cv::Mat& R_in)
{
    cv::Quatd q = cv::Quatd::createFromRotMat(R_in);
    // OpenCV Quat stores [w, x, y, z]; we export [qx, qy, qz, qw].
    return cv::Vec4d(q.at(1), q.at(2), q.at(3), q.at(0));
}

class SlamSystem::Impl {
public:
    Impl(Ptr<Feature2D> det, Ptr<DescriptorMatcher> matcher)
        : frontend_(std::move(det), std::move(matcher)) {
        loopMatcher_ = BFMatcher::create(NORM_HAMMING);
    }
    ~Impl(){ stopBackend(); }

    void setFrontendOptions(const VisualOdometryOptions& o){ feOpts_ = o; frontend_.setOptions(o); }
    void setSystemOptions(const SlamSystemOptions& o){
        sysOpts_ = o;
        map_.setMapPointCullingParams(sysOpts_.mapMinObservations, sysOpts_.mapMinFoundRatio,
                                      sysOpts_.mapMaxReprojErrorPx, sysOpts_.mapMaxPointsKeep);
        map_.setRedundantKeyframeCullingParams(sysOpts_.redundantKeyframeMinObs,
                                               sysOpts_.redundantKeyframeMinPointObs,
                                               sysOpts_.redundantKeyframeRatio);
    }
    VisualOdometryOptions frontendOptions() const { return feOpts_; }
    SlamSystemOptions systemOptions() const { return sysOpts_; }

    void setIntrinsics(double fx, double fy, double cx, double cy){
        fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
        frontend_.setCameraIntrinsics(fx, fy, cx, cy);
        map_.setCameraIntrinsics(fx, fy, cx, cy);
    }
    void setScale(double s){ scale_ = s; frontend_.setWorldScale(s); }

    void setMode(int mode){
        mode_ = mode;
        if(mode_ == MODE_LOCALIZATION){
            // Localization mode should be lightweight and map-frozen.
            stopBackend();
        }
    }
    int getMode() const { return mode_; }

    TrackingResult track(InputArray frame, double ts){
        const bool isLocalization = (mode_ == MODE_LOCALIZATION);
        if(!isLocalization) ensureBackend();

        TrackingResult res;
        {
            // Protect map_ against concurrent backend snapshots/writeback.
            std::lock_guard<std::mutex> lk(mapMutex_);
            res = frontend_.track(frame, ts, &map_, !isLocalization);
        }

        if(!isLocalization && sysOpts_.enableBackend && res.keyframeInserted){
            kfSinceBackend_++;
            const int interval = std::max(1, sysOpts_.backendTriggerInterval);
            if(kfSinceBackend_ >= interval){
                kfSinceBackend_ = 0;
                backendRequests_.fetch_add(1);
                backendCv_.notify_one();
            }
        }

        if(!isLocalization && sysOpts_.enableMapMaintenance && res.keyframeInserted){
            kfSinceMaintenance_++;
            const int interval = std::max(1, sysOpts_.maintenanceInterval);
            if(kfSinceMaintenance_ >= interval){
                kfSinceMaintenance_ = 0;
                std::lock_guard<std::mutex> lk(mapMutex_);
                map_.cullRedundantKeyFrames(std::max(1, sysOpts_.maxKeyframeCullsPerMaintenance));
                map_.cullBadMapPoints();
            }
        }
#if defined(HAVE_DBOW3)
        if(res.keyframeInserted && bowReady_){
            std::lock_guard<std::mutex> lk(mapMutex_);
            const auto &kfs = map_.keyframes();
            if(!kfs.empty()) addKeyframeToBoW(kfs.back());
        }
#endif
        if(res.ok){
            TrajEntry te; te.frameId = res.frameId; te.ts = ts; te.R = res.R_w; te.t = res.t_w;
            trajectory_.push_back(te);
        }
        return res;
    }

    bool saveTrajectoryTUM(const std::string& path) const {
        std::ofstream out(path, std::ios::out);
        if(!out.is_open()) return false;
        out << "timestamp,tx,ty,tz,qx,qy,qz,qw\n";
        for(const auto &te : trajectory_){
            if(te.R.empty() || te.t.empty()) continue;
            cv::Vec4d q = rotationMatrixToQuaternion_(te.R);
            cv::Mat t; te.t.convertTo(t, CV_64F);
            if(t.rows < 3) continue;
            out << std::fixed << std::setprecision(9)
                << te.ts << "," << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << ","
                << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "\n";
        }
        return true;
    }

    bool saveOptimizedTrajectoryTUM(const std::string& path) const {
        std::ofstream out(path, std::ios::out);
        if(!out.is_open()) return false;
        out << "timestamp,tx,ty,tz,qx,qy,qz,qw\n";
        
        // Build optimized keyframe poses with timestamps (use KeyFrame::timestamp directly)
        struct KFPose { double ts; int id; Mat R; Mat t; };
        std::vector<KFPose> kfPoses;
        const auto& kfs = map_.keyframes();
        for(const auto &kf : kfs){
            if(kf.R_w.empty() || kf.t_w.empty()) continue;
            // Use KeyFrame::timestamp if available, otherwise fallback to trajectory_ lookup
            double ts = kf.timestamp;
            if(ts <= 0.0){
                // Fallback: find timestamp from trajectory_ for backward compatibility
                for(const auto &te : trajectory_){
                    if(te.frameId == kf.id){ ts = te.ts; break; }
                }
            }
            if(ts > 0.0) kfPoses.push_back({ts, kf.id, kf.R_w.clone(), kf.t_w.clone()});
        }
        
        // Sort by timestamp
        std::sort(kfPoses.begin(), kfPoses.end(), [](const KFPose &a, const KFPose &b){ return a.ts < b.ts; });
        
        if(kfPoses.empty()){
            // Fallback: output original trajectory
            for(const auto &te : trajectory_){
                if(te.R.empty() || te.t.empty()) continue;
                cv::Vec4d q = rotationMatrixToQuaternion_(te.R);
                Mat t; te.t.convertTo(t, CV_64F);
                if(t.rows < 3) continue;
                out << std::fixed << std::setprecision(9)
                    << te.ts << "," << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << ","
                    << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "\n";
            }
            return true;
        }
        
        // For each frame in trajectory_, find nearest optimized keyframe pose by timestamp
        for(const auto &te : trajectory_){
            if(te.R.empty() || te.t.empty()) continue;
            
            // Find nearest keyframe by timestamp
            size_t bestIdx = 0;
            double bestDist = std::abs(te.ts - kfPoses[0].ts);
            for(size_t i = 1; i < kfPoses.size(); ++i){
                double dist = std::abs(te.ts - kfPoses[i].ts);
                if(dist < bestDist){ bestDist = dist; bestIdx = i; }
            }
            
            // Use the nearest keyframe's optimized pose
            const auto &bestKF = kfPoses[bestIdx];
            cv::Vec4d q = rotationMatrixToQuaternion_(bestKF.R);
            Mat t; bestKF.t.convertTo(t, CV_64F);
            if(t.rows < 3) continue;
            
            out << std::fixed << std::setprecision(9)
                << te.ts << "," << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << ","
                << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "\n";
        }
        return true;
    }

    bool saveMap(const std::string& path) const { return map_.save(path); }
    bool loadMap(const std::string& path){ bool ok = map_.load(path); return ok; }
    void reset(){ stopBackend(); map_.clear(); frontend_.reset(); trajectory_.clear(); }
    const MapManager& map() const { return map_; }
    MapManager& mapMutable() { return map_; }

#if defined(HAVE_DBOW3)
    bool setLoopVocabulary(const std::string& path){
        try {
            bowVocab_ = std::make_unique<DBoW3::Vocabulary>(path);
            if(!bowVocab_ || bowVocab_->empty()){
                bowVocab_.reset(); bowDb_.reset(); bowEntryToKfId_.clear(); bowReady_ = false;
                return false;
            }
            bowDb_ = std::make_unique<DBoW3::Database>(*bowVocab_, false, 0);
            bowEntryToKfId_.clear();
            bowReady_ = true;

            bowKeyframesAdded_.store(0);
            bowQueries_.store(0);
            bowCandidatesTotal_.store(0);
            loopAttempts_.store(0);
            loopAccepted_.store(0);

            CV_LOG_INFO(NULL, "DBoW3 vocabulary loaded: '" << path << "' (words=" << bowVocab_->size() << ")");
            return true;
        } catch(const std::exception &){
            bowVocab_.reset(); bowDb_.reset(); bowEntryToKfId_.clear(); bowReady_ = false;
            return false;
        }
    }

    void addKeyframeToBoW(const KeyFrame& kf){
        if(!bowReady_ || !bowDb_ || kf.desc.empty()) return;
        int entryId = bowDb_->add(kf.desc);
        if(entryId >= static_cast<int>(bowEntryToKfId_.size())) bowEntryToKfId_.resize(entryId + 1, -1);
        bowEntryToKfId_[entryId] = kf.id;

        const int n = bowKeyframesAdded_.fetch_add(1) + 1;
        if(n == 1 || (n % 50) == 0) {
            CV_LOG_INFO(NULL, "DBoW3 DB entries: keyframes_added=" << n << ", last_kf_id=" << kf.id);
        }
    }

    std::vector<int> queryLoopCandidates(const KeyFrame& kf, int minGap, int topK){
        std::vector<int> ids;
        if(!bowReady_ || !bowDb_ || kf.desc.empty()) return ids;
        bowQueries_.fetch_add(1);
        DBoW3::QueryResults ret;
        // DBoW3 API: query(features, QueryResults&, maxResults, maxId)
        // NOTE: max_id < 0 means "all" in DBoW3. Passing 0 would restrict to entryId <= 0
        // and effectively break loop-candidate retrieval once the DB has more entries.
        bowDb_->query(kf.desc, ret, topK + 10, -1); // extra results for filtering
        for(const auto &r : ret){
            if(static_cast<size_t>(r.Id) < bowEntryToKfId_.size()){
                int candId = bowEntryToKfId_[static_cast<size_t>(r.Id)];
                if(candId < 0) continue;
                if(std::abs(kf.id - candId) < minGap) continue;
                ids.push_back(candId);
                if(static_cast<int>(ids.size()) >= topK) break;
            }
        }
        bowCandidatesTotal_.fetch_add(static_cast<int>(ids.size()));
        return ids;
    }
#endif

#if !defined(HAVE_DBOW3)
    bool setLoopVocabulary(const std::string&){ return false; }
#endif

private:
    // Mutual ratio test with cross-check for loop candidate scoring.
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

    // Try to detect a loop with the latest keyframe and optimize poses if successful.
    bool tryLoopClosureAndOptimize(std::vector<KeyFrame> &kfs, std::vector<MapPoint> &mps){
        const int K = static_cast<int>(kfs.size());
        if(K < 5 || fx_ <= 1e-6 || fy_ <= 1e-6) return false;
        const int lastIdx = K - 1;
        const int lastId = kfs[lastIdx].id;
        const int minGap = 30; // do not loop with very recent keyframes
        if(lastIdx < minGap) return false;

        const int minMatches = 80;
        const int minInliers = 50;
        const double minInlierRatio = 0.30;

        loopAttempts_.fetch_add(1);

        std::unordered_map<int,int> idToIdx; idToIdx.reserve(kfs.size());
        for(int i = 0; i < K; ++i) idToIdx[kfs[i].id] = i;

        std::vector<int> candidates;
#if defined(HAVE_DBOW3)
        if(bowReady_){
            candidates = queryLoopCandidates(kfs[lastIdx], minGap, 5);
        }
#endif

        if(candidates.empty()){
            int bestIdx = -1;
            int bestMatches = 0;
            std::vector<DMatch> bestDMatches;
            for(int i = 0; i <= lastIdx - minGap; ++i){
                if(kfs[i].desc.empty() || kfs[lastIdx].desc.empty()) continue;
                auto dm = mutualRatioMatches(kfs[i].desc, kfs[lastIdx].desc, *loopMatcher_, 0.75f);
                if((int)dm.size() > bestMatches){
                    bestMatches = static_cast<int>(dm.size());
                    bestIdx = i;
                    bestDMatches = std::move(dm);
                }
            }
            if(bestIdx >= 0 && bestMatches >= minMatches) candidates.push_back(kfs[bestIdx].id);
            else return false;
        }

        for(int candId : candidates){
            auto it = idToIdx.find(candId);
            if(it == idToIdx.end()) continue;
            int candIdx = it->second;
            if(kfs[candIdx].desc.empty() || kfs[lastIdx].desc.empty()) continue;

            auto dm = mutualRatioMatches(kfs[candIdx].desc, kfs[lastIdx].desc, *loopMatcher_, 0.75f);
            if((int)dm.size() < minMatches) continue;

            std::vector<Point2f> pts1, pts2;
            pts1.reserve(dm.size()); pts2.reserve(dm.size());
            for(const auto &m : dm){
                pts1.push_back(kfs[candIdx].kps[m.queryIdx].pt);
                pts2.push_back(kfs[lastIdx].kps[m.trainIdx].pt);
            }

            Mat mask;
            Mat E = findEssentialMat(pts1, pts2, fx_, Point2d(cx_, cy_), RANSAC, 0.999, 1.0, mask);
            if(E.empty()) continue;
            int inliers = mask.empty() ? 0 : cv::countNonZero(mask);
            if(inliers < minInliers || inliers < static_cast<int>(pts1.size() * minInlierRatio)) continue;

            Mat Rlc, tlc;
            int validPts = recoverPose(E, pts1, pts2, Rlc, tlc, fx_, Point2d(cx_, cy_), mask);
            if(validPts < minInliers) continue;

            // Compute actual baseline between candidate and last keyframe for consistent scale
            Mat Ri = kfs[candIdx].R_w, Ci = kfs[candIdx].t_w;
            Mat Rj = kfs[lastIdx].R_w, Cj = kfs[lastIdx].t_w;
            double actualBaseline = cv::norm(Cj - Ci);
            
            // Normalize tlc and apply actual baseline for consistent scale
            Mat tlc_norm = tlc;
            if(tlc_norm.type() != CV_64F) tlc_norm.convertTo(tlc_norm, CV_64F);
            double tlcNorm = cv::norm(tlc_norm);
            if(tlcNorm > 1e-9){
                tlc_norm = tlc_norm / tlcNorm * actualBaseline;
            } else {
                tlc_norm = Mat::zeros(3,1,CV_64F);
            }

            std::vector<int> allIdx(K); std::iota(allIdx.begin(), allIdx.end(), 0);
            std::vector<int> fixed{0}; if(K > 1) fixed.push_back(1);
            Optimizer::localBundleAdjustmentSFM(kfs, mps, allIdx, fixed, fx_, fy_, cx_, cy_, sysOpts_.backendIterations * 2);

            // Add pose-graph edge with consistent scale
            // Use actual relative translation (Ri^T * (Cj - Ci)) as the constraint
            // This ensures scale consistency with the current map
            PoseGraphEdge edge;
            edge.i = kfs[candIdx].id; edge.j = kfs[lastIdx].id;
            edge.R_ij = Rlc.clone();
            // Use actual relative translation in frame i (consistent scale)
            edge.t_ij = Ri.t() * (Cj - Ci);
            // Optionally blend with geometric direction from loop estimate (with consistent scale)
            // This provides geometric constraint while maintaining scale consistency
            Mat t_geometric = Ri * tlc_norm;
            // Blend: 70% actual baseline, 30% geometric direction (weighted average)
            edge.t_ij = edge.t_ij * 0.7 + t_geometric * 0.3;
            edge.weight = 1.0;
            poseGraphEdges_.push_back(edge);

            // Pose-graph optimization on all keyframes
            Optimizer::poseGraphOptimize(kfs, poseGraphEdges_, fixed, std::max(5, sysOpts_.backendIterations), 0.5);

#if defined(HAVE_SFM)
            // Refine with global BA if available (uses updated poses as initial guess)
            Optimizer::globalBundleAdjustmentSFM(kfs, mps, fx_, fy_, cx_, cy_, sysOpts_.backendIterations);
#endif
            // Write back optimized poses/points
            {
                std::lock_guard<std::mutex> lk(mapMutex_);
                for(const auto &kf : kfs){ map_.applyOptimizedKeyframePose(kf.id, kf.R_w, kf.t_w); }
                for(const auto &mp : mps){ map_.applyOptimizedMapPoint(mp.id, mp.p); }
            }

            loopAccepted_.fetch_add(1);
            CV_LOG_INFO(NULL, "[LoopClosure] kf" << lastId << "-kf" << kfs[candIdx].id << ": " << inliers << "/" << dm.size() << " inliers");
            return true;
        }
        return false;
    }
    void ensureBackend(){
        if(backendStarted_ || !sysOpts_.enableBackend) return;
        backendStop_.store(false);
        backendThread_ = std::thread([this](){ backendLoop(); });
        backendStarted_ = true;
    }

    void stopBackend(){
        if(!backendStarted_) return;
        backendStop_.store(true);
        backendCv_.notify_one();
        if(backendThread_.joinable()) backendThread_.join();
        backendStarted_ = false;

#if defined(HAVE_DBOW3)
        if(bowReady_){
            CV_LOG_INFO(NULL,
                        "Loop/BoW stats: bow_kf_added=" << bowKeyframesAdded_.load()
                        << ", bow_queries=" << bowQueries_.load()
                        << ", bow_candidates_total=" << bowCandidatesTotal_.load()
                        << ", loop_attempts=" << loopAttempts_.load()
                        << ", loop_accepted=" << loopAccepted_.load());
        }
#endif
    }

    void backendLoop(){
        while(!backendStop_.load()){
            std::unique_lock<std::mutex> lk(backendMutex_);
            backendCv_.wait(lk, [&]{ return backendStop_.load() || backendRequests_.load() > 0; });
            if(backendStop_.load()) break;
            backendRequests_.store(0);
            backendBusy_ = true;

            std::vector<KeyFrame> kfs;
            std::vector<MapPoint> mps;
            {
                std::lock_guard<std::mutex> mapLk(mapMutex_);
                // Copy snapshots under lock to avoid data races.
                kfs = map_.keyframes();
                mps = map_.mappoints();
            }
            // Preserve pre-optimization state for sanity checks before writeback.
            const std::vector<KeyFrame> kfsPrior = kfs;
            const std::vector<MapPoint> mpsPrior = mps;
            lk.unlock();

            int K = static_cast<int>(kfs.size());
            if(K > 0){
                int lastIdx = K - 1; int lastId = kfs[lastIdx].id;
                std::unordered_map<int,int> idToIdx; idToIdx.reserve(kfs.size());
                for(int i=0;i<K;++i) idToIdx[kfs[i].id] = i;
                std::unordered_map<int,int> shared;
                for(const auto &mp : mps){
                    bool hasLast = false;
                    for(const auto &o : mp.observations){ if(o.first == lastId){ hasLast = true; break; } }
                    if(!hasLast) continue;
                    for(const auto &o : mp.observations){ if(o.first == lastId) continue; auto it = idToIdx.find(o.first); if(it != idToIdx.end()) shared[it->second]++; }
                }
                std::vector<int> localIdx; localIdx.reserve(sysOpts_.backendWindow);
                auto pushU = [&](int idx){ if(idx<0||idx>=K) return; if(std::find(localIdx.begin(), localIdx.end(), idx)!=localIdx.end()) return; localIdx.push_back(idx); };
                pushU(lastIdx);
                // Always include the first two keyframes to anchor gauge (if present).
                if(K > 0) pushU(0);
                if(K > 1) pushU(1);
                std::vector<std::pair<int,int>> scored(shared.begin(), shared.end());
                std::sort(scored.begin(), scored.end(), [](auto a, auto b){ return a.second > b.second; });
                for(const auto &p : scored){ if((int)localIdx.size() >= sysOpts_.backendWindow) break; pushU(p.first); }
                // Fill remaining slots with the most recent keyframes to enforce a sliding window.
                const int maxRecent = std::max(2, lastIdx - sysOpts_.backendWindow + 1);
                for(int idx = lastIdx - 1; idx >= maxRecent && (int)localIdx.size() < sysOpts_.backendWindow; --idx){
                    pushU(idx);
                }
                if(localIdx.empty()) localIdx.push_back(lastIdx);
                std::vector<int> fixed{0}; if(K>1) fixed.push_back(1);
                // Keep only map points observed by the local window to reduce BA size and outliers.
                std::unordered_set<int> localSet(localIdx.begin(), localIdx.end());
                std::vector<MapPoint> mpsLocal; mpsLocal.reserve(mps.size());
                for(const auto &mp : mps){
                    MapPoint mpFiltered = mp;
                    mpFiltered.observations.erase(
                        std::remove_if(mpFiltered.observations.begin(), mpFiltered.observations.end(),
                                       [&](const std::pair<int,int> &o){ return localSet.find(idToIdx[o.first]) == localSet.end(); }),
                        mpFiltered.observations.end());
                    if(mpFiltered.observations.size() < 2) continue; // need at least two obs in window
                    mpsLocal.push_back(std::move(mpFiltered));
                }
        #if defined(HAVE_SFM)
                if(localIdx.size() >= 1){
                    // Step 1: First do local BA for local consistency
                    Optimizer::localBundleAdjustmentSFM(kfs, mpsLocal, localIdx, fixed, fx_, fy_, cx_, cy_, sysOpts_.backendIterations);
                    
                    // Step 2: Then check for loop closure (global consistency)
                    bool loopOptimized = false;
                    if(lastLoopCheckKfId_ != kfs[lastIdx].id) {
                        lastLoopCheckKfId_ = kfs[lastIdx].id;
                        loopOptimized = tryLoopClosureAndOptimize(kfs, mps);
                    }
                    
                    // Step 3: If loop closure succeeded, do a light local BA refinement
                    // This helps integrate the global correction into the local window
                    if(loopOptimized){
                        Optimizer::localBundleAdjustmentSFM(kfs, mpsLocal, localIdx, fixed, fx_, fy_, cx_, cy_, std::max(5, sysOpts_.backendIterations / 2));
                    }

                    // Build lookup tables for validation against prior state.
                    struct PosePrior { Mat R; Mat t; };
                    std::unordered_map<int, PosePrior> posePrior; posePrior.reserve(kfsPrior.size());
                    for(const auto &kf : kfsPrior) posePrior.emplace(kf.id, PosePrior{kf.R_w.clone(), kf.t_w.clone()});
                    std::unordered_map<int, Point3d> pointPrior; pointPrior.reserve(mpsPrior.size());
                    for(const auto &mp : mpsPrior) pointPrior.emplace(mp.id, mp.p);

                    auto isFinitePose = [](const Mat &R, const Mat &t){
                        return !R.empty() && !t.empty() && R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1
                               && cv::checkRange(R) && cv::checkRange(t);
                    };
                    auto poseAcceptable = [&](int id, const Mat &R, const Mat &t){
                        if(!isFinitePose(R, t)) return false;
                        auto it = posePrior.find(id);
                        if(it == posePrior.end()) return true;
                        Mat diff = t - it->second.t;
                        double jump = cv::norm(diff);
                        return jump < 1000.0 * scale_;
                    };
                    auto isFinitePoint = [](const Point3d &p){
                        return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
                    };
                    auto pointAcceptable = [&](int id, const Point3d &p){
                        if(!isFinitePoint(p)) return false;
                        double normp = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
                        if(normp > 1e6 * scale_) return false;
                        auto it = pointPrior.find(id);
                        if(it == pointPrior.end()) return true;
                        Point3d dp(p.x - it->second.x, p.y - it->second.y, p.z - it->second.z);
                        double step = std::sqrt(dp.x*dp.x + dp.y*dp.y + dp.z*dp.z);
                        return step < 2000.0 * scale_;
                    };

                    const std::vector<MapPoint>* mpsWrite = loopOptimized ? &mps : &mpsLocal;
                    int poseApplied = 0, pointApplied = 0;
                    {
                        std::lock_guard<std::mutex> mapLk(mapMutex_);
                        for(const auto &kf : kfs){
                            if(!poseAcceptable(kf.id, kf.R_w, kf.t_w)){
                                continue;
                            }
                            map_.applyOptimizedKeyframePose(kf.id, kf.R_w, kf.t_w);
                            poseApplied++;
                        }
                        for(const auto &mp : *mpsWrite){
                            if(mp.id <= 0) continue;
                            if(!pointAcceptable(mp.id, mp.p)){
                                continue;
                            }
                            map_.applyOptimizedMapPoint(mp.id, mp.p);
                            pointApplied++;
                        }
                    }
                    if(loopOptimized) {
                        CV_LOG_INFO(NULL, "[PoseGraph] Poses=" << poseApplied << " Points=" << pointApplied);
                    }
                }
        #endif
            }
            lk.lock();
            backendBusy_ = false;
            backendCv_.notify_all();
        }
    }

    VisualOdometry frontend_;
    MapManager map_;
    VisualOdometryOptions feOpts_;
    SlamSystemOptions sysOpts_;
    int mode_ = MODE_SLAM;
    int kfSinceBackend_ = 0;
    int kfSinceMaintenance_ = 0;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    double scale_ = 1.0;

    Ptr<DescriptorMatcher> loopMatcher_;
    int lastLoopCheckKfId_ = -1;

    std::vector<PoseGraphEdge> poseGraphEdges_;

#if defined(HAVE_DBOW3)
    std::unique_ptr<DBoW3::Vocabulary> bowVocab_;
    std::unique_ptr<DBoW3::Database> bowDb_;
    std::vector<int> bowEntryToKfId_;
    bool bowReady_ = false;

    std::atomic<int> bowKeyframesAdded_{0};
    std::atomic<int> bowQueries_{0};
    std::atomic<int> bowCandidatesTotal_{0};
    std::atomic<int> loopAttempts_{0};
    std::atomic<int> loopAccepted_{0};
#endif

    std::vector<TrajEntry> trajectory_;

    std::thread backendThread_;
    std::mutex backendMutex_;
    std::condition_variable backendCv_;
    std::atomic<bool> backendStop_{false};
    std::atomic<int> backendRequests_{0};
    bool backendBusy_ = false;
    bool backendStarted_ = false;

    std::mutex mapMutex_;
};

SlamSystem::SlamSystem() : impl_(std::make_unique<Impl>(ORB::create(), BFMatcher::create(NORM_HAMMING))) {}
SlamSystem::SlamSystem(Ptr<Feature2D> det, Ptr<DescriptorMatcher> matcher)
    : impl_(std::make_unique<Impl>(std::move(det), std::move(matcher))) {}
SlamSystem::~SlamSystem() = default;

void SlamSystem::setFrontendOptions(const VisualOdometryOptions& o){ impl_->setFrontendOptions(o); }
void SlamSystem::setSystemOptions(const SlamSystemOptions& o){ impl_->setSystemOptions(o); }
void SlamSystem::setCameraIntrinsics(double fx, double fy, double cx, double cy){ impl_->setIntrinsics(fx, fy, cx, cy); }
void SlamSystem::setWorldScale(double scale_m){ impl_->setScale(scale_m); }
void SlamSystem::setMode(int mode){ impl_->setMode(mode); }
int SlamSystem::getMode() const { return impl_->getMode(); }
VisualOdometryOptions SlamSystem::getFrontendOptions() const { return impl_->frontendOptions(); }
SlamSystemOptions SlamSystem::getSystemOptions() const { return impl_->systemOptions(); }
void SlamSystem::setEnableBackend(bool enable){ auto opts = impl_->systemOptions(); opts.enableBackend = enable; impl_->setSystemOptions(opts); }
void SlamSystem::setBackendWindow(int w){ auto opts = impl_->systemOptions(); opts.backendWindow = std::max(1, w); impl_->setSystemOptions(opts); }
void SlamSystem::setBackendIterations(int it){ auto opts = impl_->systemOptions(); opts.backendIterations = std::max(1, it); impl_->setSystemOptions(opts); }
TrackingResult SlamSystem::track(InputArray frame, double ts){ return impl_->track(frame, ts); }
bool SlamSystem::saveTrajectoryTUM(const std::string& path) const { return impl_->saveTrajectoryTUM(path); }
bool SlamSystem::saveOptimizedTrajectoryTUM(const std::string& path) const { return impl_->saveOptimizedTrajectoryTUM(path); }
bool SlamSystem::saveMap(const std::string& path) const { return impl_->saveMap(path); }
bool SlamSystem::loadMap(const std::string& path){ return impl_->loadMap(path); }
bool SlamSystem::setLoopVocabulary(const std::string& path){ return impl_->setLoopVocabulary(path); }
void SlamSystem::reset(){ impl_->reset(); }
const MapManager& SlamSystem::getMap() const { return impl_->map(); }
MapManager& SlamSystem::getMapMutable(){ return impl_->mapMutable(); }

} // namespace vo
} // namespace cv
