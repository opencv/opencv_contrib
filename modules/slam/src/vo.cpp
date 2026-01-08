#include "opencv2/slam/data_loader.hpp"
#include "opencv2/slam/feature.hpp"
#include "opencv2/slam/initializer.hpp"
#include "opencv2/slam/keyframe.hpp"
#include "opencv2/slam/localizer.hpp"
#include "opencv2/slam/map.hpp"
#include "opencv2/slam/matcher.hpp"
#include "opencv2/slam/optimizer.hpp"
#include "opencv2/slam/pose.hpp"
#include "opencv2/slam/visualizer.hpp"
#include "opencv2/slam/vo.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace cv {
namespace vo {

VisualOdometry::VisualOdometry(Ptr<Feature2D> feature, Ptr<DescriptorMatcher> matcher)
    : feature_(std::move(feature)), matcher_(std::move(matcher)) {
}

int VisualOdometry::run(const std::string &imageDir, double scaleM){
    return run(imageDir, scaleM, options_);
}

void VisualOdometry::setEnableBackend(bool enable){
    options_.enableBackend = enable;
}

void VisualOdometry::setBackendWindow(int window){
    options_.backendWindow = std::max(1, window);
}

void VisualOdometry::setBackendIterations(int iterations){
    options_.backendIterations = std::max(1, iterations);
}

int VisualOdometry::run(const std::string &imageDir, double scaleM, const VisualOdometryOptions &options){
    const auto prevLogLevel = cv::utils::logging::getLogLevel();
    // Keep global OpenCV logs at WARNING to avoid noisy INFO logs from OpenCL/UI.
    // Module-specific verbosity is handled explicitly.
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    DataLoader loader(imageDir);
    if(loader.size() == 0){
        CV_LOG_WARNING(NULL, "VisualOdometry: no images found");
        cv::utils::logging::setLogLevel(prevLogLevel);
        return -1;
    }

    if(!feature_){
        feature_ = ORB::create(2000);
    }
    if(!matcher_){
        matcher_ = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    PoseEstimator poseEst;
    // Keep the topdown visualization scale consistent with the world scaling applied by scaleM.
    // When scaleM is large (e.g. 1.0), a larger meters_per_pixel zooms out to keep the trajectory visible.
    // When scaleM is small (e.g. 0.02), it zooms in accordingly.
    Visualizer vis(1000, 800, std::max(1e-9, scaleM));
    MapManager map;
    Initializer initializer;
    Localizer localizer(0.7f);

    // Configure map intrinsics and maintenance thresholds (milestone 2/3)
    map.setCameraIntrinsics(loader.fx(), loader.fy(), loader.cx(), loader.cy());
    map.setMapPointCullingParams(options.mapMinObservations,
                                 options.mapMinFoundRatio,
                                 options.mapMaxReprojErrorPx,
                                 options.mapMaxPointsKeep);
    map.setRedundantKeyframeCullingParams(options.redundantKeyframeMinObs,
                                          options.redundantKeyframeMinPointObs,
                                          options.redundantKeyframeRatio);

    // prepare per-run diagnostics folder
    std::string runTimestamp;
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_time_t);
    std::ostringstream ss; ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    runTimestamp = ss.str();
    std::string resultDirStr = imageDir;
    if(resultDirStr.empty()) resultDirStr = std::string(".");
    if(resultDirStr.back() == '/') resultDirStr.pop_back();
    resultDirStr += "/result";
    ensureDirectoryExists(resultDirStr);
    std::string runDirStr = resultDirStr + "/" + runTimestamp;
    ensureDirectoryExists(runDirStr);

    Mat R_g = Mat::eye(3,3,CV_64F);
    Mat t_g = Mat::zeros(3,1,CV_64F);

    // simple map structures
    std::vector<KeyFrame> keyframes;
    std::vector<MapPoint> mappoints;
    std::unordered_map<int,int> keyframeIdToIndex;

    // Trajectory recording: store timestamp + pose for each keyframe
    struct TrajectoryEntry {
        int frame_id;
        double timestamp;
        Mat R_w;
        Mat t_w;
    };
    std::vector<TrajectoryEntry> trajectory;

    // Backend (BA) thread primitives
    std::mutex mapMutex; // protects map and keyframe modifications and writeback
    std::condition_variable backendCv;
    std::atomic<bool> backendStop(false);
    std::atomic<int> backendRequests(0);
    bool backendBusy = false; // guarded by mapMutex
    // Expand BA window to optimize meaningful local map (not just last 5 frames)
    const int LOCAL_BA_WINDOW = std::max(60, options.backendWindow);

    // Start backend thread only if enabled
    std::thread backendThread;
    if(options.enableBackend){
        backendThread = std::thread([&]() {
            while(!backendStop.load()){
                std::unique_lock<std::mutex> lk(mapMutex);
                backendCv.wait(lk, [&]{ return backendStop.load() || backendRequests.load() > 0; });
                if(backendStop.load()) break;
                // snapshot map and keyframes
                auto kfs_snapshot = map.keyframes();
                auto mps_snapshot = map.mappoints();
                // reset requests and mark busy
                backendRequests.store(0);
                backendBusy = true;
                lk.unlock();

                // determine local window using covisibility with the latest keyframe
                int K = static_cast<int>(kfs_snapshot.size());
                if(K <= 0) continue;
                const int lastIdx = K - 1;
                const int lastId = kfs_snapshot[lastIdx].id;

                std::unordered_map<int,int> idToIdx;
                idToIdx.reserve(static_cast<size_t>(K));
                for(int i = 0; i < K; ++i) idToIdx[kfs_snapshot[i].id] = i;

                std::unordered_map<int,int> sharedCount;
                sharedCount.reserve(mps_snapshot.size());
                for(const auto &mp : mps_snapshot){
                    bool hasLast = false;
                    for(const auto &obs : mp.observations){
                        if(obs.first == lastId){ hasLast = true; break; }
                    }
                    if(!hasLast) continue;
                    for(const auto &obs : mp.observations){
                        if(obs.first == lastId) continue;
                        auto it = idToIdx.find(obs.first);
                        if(it == idToIdx.end()) continue;
                        sharedCount[it->second] += 1;
                    }
                }

                struct ScoredKf { int score; int idx; };
                std::vector<ScoredKf> scored;
                scored.reserve(sharedCount.size());
                for(const auto &kv : sharedCount){
                    scored.push_back({kv.second, kv.first});
                }
                std::sort(scored.begin(), scored.end(), [](const ScoredKf &a, const ScoredKf &b){ return a.score > b.score; });

                std::vector<int> localKfIndices;
                localKfIndices.reserve(static_cast<size_t>(LOCAL_BA_WINDOW));
                auto push_unique = [&](int idx){
                    if(idx < 0 || idx >= K) return;
                    if(idx == 0 || idx == 1) return;
                    if(std::find(localKfIndices.begin(), localKfIndices.end(), idx) != localKfIndices.end()) return;
                    localKfIndices.push_back(idx);
                };
                push_unique(lastIdx);
                for(const auto &s : scored){
                    if(static_cast<int>(localKfIndices.size()) >= LOCAL_BA_WINDOW) break;
                    push_unique(s.idx);
                }
                if(localKfIndices.empty()) localKfIndices.push_back(lastIdx);

                std::vector<int> fixedKfIndices;
                fixedKfIndices.push_back(0);
                if(K > 1) fixedKfIndices.push_back(1);
            #if defined(HAVE_SFM)
                CV_LOG_DEBUG(NULL, "Backend: Running BA");
                Optimizer::localBundleAdjustmentSFM(kfs_snapshot, mps_snapshot, localKfIndices, fixedKfIndices,
                                                loader.fx(), loader.fy(), loader.cx(), loader.cy(), options.backendIterations);
                CV_LOG_DEBUG(NULL, "Backend: BA completed");
            #else
                CV_LOG_DEBUG(NULL, "Backend: HAVE_SFM not defined, BA skipped");
            #endif
                // write back optimized poses/points into main map under lock using id-based lookup
                std::lock_guard<std::mutex> lk2(mapMutex);
                auto &kfs_ref = const_cast<std::vector<KeyFrame>&>(map.keyframes());
                auto &mps_ref = const_cast<std::vector<MapPoint>&>(map.mappoints());
                // copy back poses by id to ensure we update the authoritative containers
                for(const auto &kf_opt : kfs_snapshot){
                    int idx = map.keyframeIndex(kf_opt.id);
                    if(idx >= 0 && idx < static_cast<int>(kfs_ref.size())){
                        kfs_ref[idx].R_w = kf_opt.R_w.clone();
                        kfs_ref[idx].t_w = kf_opt.t_w.clone();
                    }
                }
                // copy back mappoint positions by id
                for(const auto &mp_opt : mps_snapshot){
                    if(mp_opt.id <= 0) continue;
                    int idx = map.mapPointIndex(mp_opt.id);
                    if(idx >= 0 && idx < static_cast<int>(mps_ref.size())){
                        mps_ref[idx].p = mp_opt.p;
                    }
                }
                backendBusy = false;
                backendCv.notify_all();
            }
        });
    }

    Mat frame;
    std::string imgPath;
    int frame_id = 0;

    // persistent previous-frame storage (declare outside loop so detectAndCompute can use them)
    static std::vector<KeyPoint> prevKp;
    static Mat prevGray, prevDesc;

    // Helper lambda to extract timestamp from EuRoC image filename (nanoseconds)
    auto extractTimestamp = [](const std::string &path, int fallbackId) -> double {
        try {
            std::string fname = path;
            auto slash = fname.find_last_of("/\\");
            if(slash != std::string::npos) fname = fname.substr(slash + 1);
            auto dot = fname.find_last_of('.');
            if(dot != std::string::npos) fname = fname.substr(0, dot);
            if(!fname.empty()){
                double ts = std::stod(fname);
                if(ts > 1e12) ts *= 1e-9; // likely nanoseconds -> seconds
                else if(ts > 1e9) ts *= 1e-6; // likely microseconds -> seconds
                return ts;
            }
        } catch(const std::exception &){}
        return static_cast<double>(fallbackId);
    };

    while(loader.getNextImage(frame, imgPath)){
        Mat gray = frame;
        if(gray.channels() > 1) cvtColor(gray, gray, COLOR_BGR2GRAY);

        std::vector<KeyPoint> kps;
        Mat desc;
        // Detect and compute descriptors using injected feature_ (e.g., ORB)
        feature_->detect(gray, kps);
        feature_->compute(gray, kps, desc);

    // (previous-frame storage declared outside loop)

        if(!prevGray.empty() && !prevDesc.empty() && !desc.empty()){
            // KNN match with ratio test and mutual cross-check
            std::vector<std::vector<DMatch>> knn12, knn21;
            matcher_->knnMatch(prevDesc, desc, knn12, 2);
            matcher_->knnMatch(desc, prevDesc, knn21, 2);
            // ratio test + mutual check (prev->curr)
            std::vector<DMatch> goodMatches;
            goodMatches.reserve(knn12.size());
            const float ratio = 0.75f;
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
                if(rbest.trainIdx == (int)qi) goodMatches.push_back(best);
            }

            Mat imgMatches;
            drawMatches(prevGray, prevKp, gray, kps, goodMatches, imgMatches,
                            Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

            std::vector<Point2f> pts1, pts2;
            pts1.reserve(goodMatches.size()); pts2.reserve(goodMatches.size());
            for(const auto &m: goodMatches){
                pts1.push_back(prevKp[m.queryIdx].pt);
                pts2.push_back(kps[m.trainIdx].pt);
            }

            // quick frame-diff to detect near-static frames
            double meanDiff = 0.0;
            if(!prevGray.empty()){
                Mat diff; absdiff(gray, prevGray, diff);
                meanDiff = mean(diff)[0];
            }

            // compute per-match flow magnitudes and median flow
            std::vector<double> flows; flows.reserve(pts1.size());
            double median_flow = 0.0;
            for(size_t i=0;i<pts1.size();++i){
                double dx = pts2[i].x - pts1[i].x;
                double dy = pts2[i].y - pts1[i].y;
                flows.push_back(std::sqrt(dx*dx + dy*dy));
            }
            if(!flows.empty()){
                std::vector<double> tmp = flows;
                size_t mid = tmp.size()/2;
                std::nth_element(tmp.begin(), tmp.begin()+mid, tmp.end());
                median_flow = tmp[mid];
            }

            int pre_matches = static_cast<int>(goodMatches.size());
            int post_matches = pre_matches;

            // Two-view initialization: if map is empty and this is the second frame, try to bootstrap
            if(map.keyframes().empty() && frame_id == 1){
                Mat R_init, t_init;
                std::vector<Point3d> pts3D;
                std::vector<bool> isTri;
                // call initializer using the matched keypoints between prev and current
                if(initializer.initialize(prevKp, kps, goodMatches, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R_init, t_init, pts3D, isTri)){
                    // build two keyframes: previous (id = frame_id-1) and current (id = frame_id)
                    Mat prevImg;
                    if(!prevGray.empty()){
                        if(prevGray.channels() == 1){ cvtColor(prevGray, prevImg, COLOR_GRAY2BGR); }
                        else prevImg = prevGray.clone();
                    }
                    KeyFrame kf0(frame_id - 1, prevImg, prevKp, prevDesc, Mat::eye(3,3,CV_64F), Mat::zeros(3,1,CV_64F));
                    // recoverPose returns R,t such that x2 = R*x1 + t (cam1 -> cam2).
                    // Our map uses camera->world rotation (R_w) and camera center in world (C_w).
                    Mat Rwc1 = R_init.t();
                    Mat Cw1 = (-Rwc1 * t_init) * scaleM;
                    KeyFrame kf1(frame_id, frame, kps, desc, Rwc1, Cw1);

                    // convert initializer 3D points (in first camera frame) to MapPoints in world coords (world==first)
                    std::vector<MapPoint> newMps;
                    newMps.reserve(pts3D.size());
                    for(size_t i=0;i<pts3D.size();++i){
                        if(!isTri[i]) continue;
                        MapPoint mp;
                        mp.p = Point3d(pts3D[i].x * scaleM, pts3D[i].y * scaleM, pts3D[i].z * scaleM);
                        // observation indices come from goodMatches order
                        if(i < goodMatches.size()){
                            const DMatch &m = goodMatches[i];
                            mp.observations.emplace_back(kf0.id, m.queryIdx);
                            mp.observations.emplace_back(kf1.id, m.trainIdx);
                        }
                        newMps.push_back(mp);
                    }

                    {
                        std::lock_guard<std::mutex> lk(mapMutex);
                        keyframes.push_back(std::move(kf0));
                        map.addKeyFrame(keyframes.back());
                        keyframes.push_back(std::move(kf1));
                        map.addKeyFrame(keyframes.back());
                        if(!newMps.empty()) map.addMapPoints(newMps);
                    }

                    // set global pose to second keyframe
                    Mat t_d; kf1.t_w.convertTo(t_d, CV_64F);
                    t_g = t_d;
                    R_g = kf1.R_w.clone();
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x, z);

                    // Record initial trajectory entries
                    double ts0 = extractTimestamp(imgPath, frame_id - 1);
                    double ts1 = extractTimestamp(imgPath, frame_id);
                    trajectory.push_back({frame_id - 1, ts0, Mat::eye(3,3,CV_64F), Mat::zeros(3,1,CV_64F)});
                    trajectory.push_back({frame_id, ts1, R_g.clone(), t_g.clone()});

                    // notify backend to run BA on initial map
                    backendRequests.fetch_add(1);
                    backendCv.notify_one();
                    // skip the usual PnP / poseEst path for this frame since we've initialized
                    prevGray = gray.clone(); prevKp = kps; prevDesc = desc.clone();
                    frame_id++;
                    continue;
                }
            }

            Mat R_est, t_est, mask_est;
            int inliers_est = 0;
            bool ok_est = false;
            bool ok_pnp = false;
            Mat R_pnp, t_pnp; int inliers_pnp = 0;
            int preMatches_pnp = 0, postMatches_pnp = 0; double meanReproj_pnp = 0.0;
            Mat R_use, t_use, mask_use;
            int inliers = 0;
            int matchCount = post_matches;
            bool integrate = false;

            if(pts1.size() >= 8){
                ok_est = poseEst.estimate(pts1, pts2, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R_est, t_est, mask_est, inliers_est);
            }
            if(!ok_est){
                ok_pnp = localizer.tryPnP(map, desc, kps, loader.fx(), loader.fy(), loader.cx(), loader.cy(), gray.cols, gray.rows,
                                options.minInliers, R_pnp, t_pnp, inliers_pnp, frame_id, &frame, runDirStr,
                                &preMatches_pnp, &postMatches_pnp, &meanReproj_pnp);
                (void)preMatches_pnp; (void)postMatches_pnp; (void)meanReproj_pnp;
            }
            if(ok_est || ok_pnp){
                R_use = ok_est ? R_est : R_pnp;
                t_use = ok_est ? t_est : t_pnp;
                mask_use = ok_est ? mask_est : Mat();
                inliers = ok_est ? inliers_est : inliers_pnp;
                matchCount = post_matches;

                integrate = true;
                if(inliers < options.minInliers || matchCount < options.minMatches){
                    integrate = false;
                }
                if(ok_est){
                    double t_norm = 0.0, rot_angle = 0.0;
                    Mat t_d; t_est.convertTo(t_d, CV_64F);
                    t_norm = norm(t_d);
                    Mat R_d; R_est.convertTo(R_d, CV_64F);
                    double trace = R_d.at<double>(0,0) + R_d.at<double>(1,1) + R_d.at<double>(2,2);
                    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
                    rot_angle = std::acos(cos_angle);
                    if(t_norm < options.minTranslationNorm && std::abs(rot_angle) < options.minRotationRad
                       && meanDiff < options.diffZeroThresh && median_flow < options.flowZeroThresh){
                        integrate = false;
                    }
                }
                if (inliers >= options.minInliers || (inliers >= 2 && matchCount > 50 && median_flow > 2.0)) {
                    integrate = true;
                }

                if(integrate){
                    // Pose convention: R_g camera->world, t_g camera center in world.
                    if(ok_est){
                        Mat R_d, t_d;
                        R_use.convertTo(R_d, CV_64F);
                        t_use.convertTo(t_d, CV_64F);
                        Mat C2_in_c1 = (-R_d.t() * t_d) * scaleM;
                        t_g = t_g + R_g * C2_in_c1;
                        R_g = R_g * R_d.t();
                    } else {
                        Mat R_abs, C_abs;
                        R_use.convertTo(R_abs, CV_64F);
                        t_use.convertTo(C_abs, CV_64F);
                        R_g = R_abs;
                        t_g = C_abs;
                    }
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x, z);

                    // Log per-frame pose for timestamps.
                    double ts = extractTimestamp(imgPath, frame_id);
                    trajectory.push_back({frame_id, ts, R_g.clone(), t_g.clone()});

                    // Keyframe insertion gating
                    bool insertKeyframe = false;
                    const int minGap = std::max(1, options.keyframeMinGap);
                    const int maxGap = std::max(minGap, options.keyframeMaxGap);
                    if(map.keyframes().empty()){
                        insertKeyframe = true;
                    } else {
                        const int lastKfId = map.keyframes().back().id;
                        const int gap = frame_id - lastKfId;
                        if(gap >= maxGap) insertKeyframe = true;
                        else if(gap >= minGap && median_flow >= options.keyframeMinParallaxPx) insertKeyframe = true;
                    }

                    if(insertKeyframe){
                        KeyFrame kf(frame_id, frame, kps, desc, R_g, t_g);

                        // Triangulate against last KF
                        bool didTriangulate = false;
                        std::vector<MapPoint> newPts;
                        if(!map.keyframes().empty() && !map.keyframes().back().desc.empty() && !kf.desc.empty()){
                            const KeyFrame &last = map.keyframes().back();

                            // Match last KF -> current KF
                            std::vector<std::vector<DMatch>> knn_kf12, knn_kf21;
                            matcher_->knnMatch(last.desc, kf.desc, knn_kf12, 2);
                            matcher_->knnMatch(kf.desc, last.desc, knn_kf21, 2);
                            std::vector<DMatch> kfMatches;
                            kfMatches.reserve(knn_kf12.size());
                            const float ratio_kf = 0.75f;
                            for(size_t qi = 0; qi < knn_kf12.size(); ++qi){
                                if(knn_kf12[qi].empty()) continue;
                                const DMatch &best = knn_kf12[qi][0];
                                if(knn_kf12[qi].size() >= 2){
                                    const DMatch &second = knn_kf12[qi][1];
                                    if(second.distance > 0.0f && best.distance / second.distance > ratio_kf) continue;
                                }
                                int trainIdx = best.trainIdx;
                                if(trainIdx < 0 || trainIdx >= (int)knn_kf21.size() || knn_kf21[trainIdx].empty()) continue;
                                const DMatch &rbest = knn_kf21[trainIdx][0];
                                if(rbest.trainIdx == (int)qi) kfMatches.push_back(best);
                            }

                            if(static_cast<int>(kfMatches.size()) > options.maxMatchesKeep){
                                std::nth_element(kfMatches.begin(), kfMatches.begin() + options.maxMatchesKeep, kfMatches.end(),
                                                 [](const DMatch &a, const DMatch &b){ return a.distance < b.distance; });
                                kfMatches.resize(options.maxMatchesKeep);
                            }

                            if(!kfMatches.empty()){
                                std::vector<Point2f> pts1n, pts2n;
                                pts1n.reserve(kfMatches.size());
                                pts2n.reserve(kfMatches.size());
                                std::vector<int> pts1_kp_idx; pts1_kp_idx.reserve(kfMatches.size());
                                std::vector<int> pts2_kp_idx; pts2_kp_idx.reserve(kfMatches.size());
                                double fx = loader.fx(), fy = loader.fy(), cx = loader.cx(), cy = loader.cy();
                                for(const auto &m : kfMatches){
                                    const Point2f &p1 = last.kps[m.queryIdx].pt;
                                    const Point2f &p2 = kf.kps[m.trainIdx].pt;
                                    pts1n.emplace_back(float((p1.x - cx)/fx), float((p1.y - cy)/fy));
                                    pts2n.emplace_back(float((p2.x - cx)/fx), float((p2.y - cy)/fy));
                                    pts1_kp_idx.push_back(m.queryIdx);
                                    pts2_kp_idx.push_back(m.trainIdx);
                                }
                                newPts = map.triangulateBetweenLastTwo(pts1n, pts2n, pts1_kp_idx, pts2_kp_idx, last, kf, fx, fy, cx, cy);
                                if(!newPts.empty()) didTriangulate = true;
                            }
                        }

                        // insert keyframe and new points under lock
                        {
                            std::lock_guard<std::mutex> lk(mapMutex);
                            keyframes.push_back(std::move(kf));
                            map.addKeyFrame(keyframes.back());
                            if(didTriangulate) map.addMapPoints(newPts);

                            if(options.enableMapMaintenance){
                                const int interval = std::max(1, options.maintenanceInterval);
                                if(static_cast<int>(map.keyframes().size()) % interval == 0){
                                    map.cullBadMapPoints();
                                    map.cullRedundantKeyFrames(std::max(1, options.maxKeyframeCullsPerMaintenance));
                                    auto &mps = map.mappointsMutable();
                                    for(auto &mp : mps){
                                        if(mp.isBad) continue;
                                        map.updateMapPointDescriptor(mp);
                                    }
                                }
                            }
                            // Notify backend thread to run local BA asynchronously every N inserted keyframes
                            if(options.enableBackend){
                                const int interval = std::max(1, options.backendTriggerInterval);
                                if(static_cast<int>(map.keyframes().size()) % interval == 0){
                                    backendRequests.fetch_add(1);
                                    backendCv.notify_one();
                                }
                            }
                        }
                    }
                }
            }

            // Always show a single image; if we have matches, draw small boxes around matched keypoints
            Mat visImg;
            if(frame.channels() > 1) visImg = frame.clone();
            else cvtColor(gray, visImg, COLOR_GRAY2BGR);
            std::string info = std::string("Frame ") + std::to_string(frame_id) + " matches=" + std::to_string(matchCount) + " inliers=" + std::to_string(inliers);
            if(!goodMatches.empty()){
                for(size_t mi=0; mi<goodMatches.size(); ++mi){
                    Point2f p2 = (mi < pts2.size()) ? pts2[mi] : kps[goodMatches[mi].trainIdx].pt;
                    // determine inlier status from mask_use
                    bool isInlier = false;
                    if(!mask_use.empty()){
                        if(mask_use.rows == static_cast<int>(goodMatches.size())){
                            isInlier = mask_use.at<uchar>(static_cast<int>(mi), 0) != 0;
                        } else if(mask_use.cols == static_cast<int>(goodMatches.size())){
                            isInlier = mask_use.at<uchar>(0, static_cast<int>(mi)) != 0;
                        }
                    }
                    Scalar col = isInlier ? Scalar(0,255,0) : Scalar(0,0,255);
                    Point ip(cvRound(p2.x), cvRound(p2.y));
                    Rect r(ip - Point(4,4), Size(8,8));
                    rectangle(visImg, r, col, 2, LINE_AA);
                }
            }
            putText(visImg, info, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0,255,0), 2);
            vis.showFrame(visImg);
        } else {
            Mat visFrame; drawKeypoints(gray, kps, visFrame, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            vis.showFrame(visFrame);
        }

        vis.showTopdown();
        // update prev
        prevGray = gray.clone(); prevKp = kps; prevDesc = desc.clone();
        frame_id++;
        char key = (char)waitKey(1);
        if(key == 27) break;
    }

    // Trigger final BA before saving trajectory and wait synchronously
    if(options.enableBackend){
        CV_LOG_WARNING(NULL, "Triggering final backend optimization before saving trajectory...");
        {
            std::unique_lock<std::mutex> lk(mapMutex);
            backendRequests.fetch_add(1);
            backendCv.notify_one();
            backendCv.wait(lk, [&]{ return backendRequests.load() == 0 && !backendBusy; });
        }

        // Run a final full-map BA synchronously (larger than sliding window)
        std::vector<KeyFrame> kfs_snapshot;
        std::vector<MapPoint> mps_snapshot;
        {
            std::lock_guard<std::mutex> lk(mapMutex);
            kfs_snapshot = map.keyframes();
            mps_snapshot = map.mappoints();
        }
        int K = static_cast<int>(kfs_snapshot.size());
        if(K > 0){
            std::vector<int> allIdx(K);
            std::iota(allIdx.begin(), allIdx.end(), 0);
            std::vector<int> fixed;
            fixed.push_back(0);
            if(K > 1) fixed.push_back(1);
        #if defined(HAVE_SFM)
            CV_LOG_WARNING(NULL, "Final BA: optimizing all keyframes");
            Optimizer::localBundleAdjustmentSFM(kfs_snapshot, mps_snapshot, allIdx, fixed,
                                                loader.fx(), loader.fy(), loader.cx(), loader.cy(), std::max(20, options.backendIterations * 2));
            CV_LOG_WARNING(NULL, "Final BA: completed");
            {
                std::lock_guard<std::mutex> lk(mapMutex);
                auto &kfs_ref = const_cast<std::vector<KeyFrame>&>(map.keyframes());
                auto &mps_ref = const_cast<std::vector<MapPoint>&>(map.mappoints());
                for(const auto &kf_opt : kfs_snapshot){
                    int idx = map.keyframeIndex(kf_opt.id);
                    if(idx >= 0 && idx < static_cast<int>(kfs_ref.size())){
                        kfs_ref[idx].R_w = kf_opt.R_w.clone();
                        kfs_ref[idx].t_w = kf_opt.t_w.clone();
                    }
                }
                for(const auto &mp_opt : mps_snapshot){
                    if(mp_opt.id <= 0) continue;
                    int idx = map.mapPointIndex(mp_opt.id);
                    if(idx >= 0 && idx < static_cast<int>(mps_ref.size())){
                        mps_ref[idx].p = mp_opt.p;
                    }
                }
            }
        #else
            CV_LOG_WARNING(NULL, "Final BA: HAVE_SFM not defined, skipped");
        #endif
        }
    }

    // save trajectory image into result/ folder
    try{
        // save trajectory into the per-run folder using a simple filename (no timestamp)
        std::string outDir = resultDirStr + "/" + runTimestamp;
        ensureDirectoryExists(outDir);
        std::string outPath = outDir + "/trajectory.png";
        if(options.enableBackend && !map.keyframes().empty()){
            std::vector<KeyFrame> kfs = map.keyframes();
            std::sort(kfs.begin(), kfs.end(), [](const KeyFrame &a, const KeyFrame &b){ return a.id < b.id; });
            std::vector<cv::Point2d> xz;
            xz.reserve(kfs.size());
            for(const auto &kf : kfs){
                if(kf.t_w.empty()) continue;
                xz.emplace_back(kf.t_w.at<double>(0,0), kf.t_w.at<double>(2,0));
            }
            vis.setTrajectoryXZ(xz);
        }
        (void)vis.saveTrajectory(outPath);
    } catch(const std::exception &e){
        CV_LOG_ERROR(NULL, cv::format("Error saving trajectory: %s", e.what()));
    }

    // Write trajectory to TUM-format CSV file
    // Extract final optimized poses from map keyframes (after backend BA)
    std::vector<TrajectoryEntry> finalTrajectory;
    if(options.enableBackend && !map.keyframes().empty()){
        // Build map from frame_id to timestamp
        std::unordered_map<int, double> idToTimestamp;
        for(const auto &trajEntry : trajectory){
            idToTimestamp[trajEntry.frame_id] = trajEntry.timestamp;
        }
        for(const auto &kf : map.keyframes()){
            if(kf.t_w.empty() || kf.R_w.empty()) continue;
            double ts = idToTimestamp.count(kf.id) ? idToTimestamp[kf.id] : static_cast<double>(kf.id);
            finalTrajectory.push_back({kf.id, ts, kf.R_w.clone(), kf.t_w.clone()});
        }
        if(options.verbose) CV_LOG_WARNING(NULL, "Backend enabled: using optimized poses");
    } else {
        finalTrajectory = trajectory; // use front-end trajectory if backend disabled
        if(options.verbose) CV_LOG_WARNING(NULL, "Backend disabled: using front-end poses");
    }
    
    try{
        std::string csvPath = resultDirStr + "/" + runTimestamp + "/trajectory_tum.csv";
        std::ofstream ofs(csvPath);
        if(!ofs.is_open()){
            CV_LOG_ERROR(NULL, cv::format("Failed to open trajectory CSV: %s", csvPath.c_str()));
        } else {
            ofs << "# timestamp,tx,ty,tz,qx,qy,qz,qw" << std::endl;
            for(const auto &entry : finalTrajectory){
                if(entry.R_w.empty() || entry.t_w.empty()) continue;
                Mat Rmat, tvec_w;
                entry.R_w.convertTo(Rmat, CV_64F);
                entry.t_w.convertTo(tvec_w, CV_64F);
                // Convert rotation matrix to quaternion
                cv::Matx33d Rm;
                Rmat.copyTo(Rm);
                double tr = Rm(0,0) + Rm(1,1) + Rm(2,2);
                double qw, qx, qy, qz;
                if(tr > 0){
                    double S = std::sqrt(tr + 1.0) * 2.0;
                    qw = 0.25 * S;
                    qx = (Rm(2,1) - Rm(1,2)) / S;
                    qy = (Rm(0,2) - Rm(2,0)) / S;
                    qz = (Rm(1,0) - Rm(0,1)) / S;
                } else if((Rm(0,0) > Rm(1,1)) && (Rm(0,0) > Rm(2,2))){
                    double S = std::sqrt(1.0 + Rm(0,0) - Rm(1,1) - Rm(2,2)) * 2.0;
                    qw = (Rm(2,1) - Rm(1,2)) / S;
                    qx = 0.25 * S;
                    qy = (Rm(0,1) + Rm(1,0)) / S;
                    qz = (Rm(0,2) + Rm(2,0)) / S;
                } else if(Rm(1,1) > Rm(2,2)){
                    double S = std::sqrt(1.0 + Rm(1,1) - Rm(0,0) - Rm(2,2)) * 2.0;
                    qw = (Rm(0,2) - Rm(2,0)) / S;
                    qx = (Rm(0,1) + Rm(1,0)) / S;
                    qy = 0.25 * S;
                    qz = (Rm(1,2) + Rm(2,1)) / S;
                } else {
                    double S = std::sqrt(1.0 + Rm(2,2) - Rm(0,0) - Rm(1,1)) * 2.0;
                    qw = (Rm(1,0) - Rm(0,1)) / S;
                    qx = (Rm(0,2) + Rm(2,0)) / S;
                    qy = (Rm(1,2) + Rm(2,1)) / S;
                    qz = 0.25 * S;
                }
                // Write in TUM format: timestamp,tx,ty,tz,qx,qy,qz,qw (integer nanoseconds)
                long long ts_ns = static_cast<long long>(std::llround(entry.timestamp * 1e9));
                ofs << ts_ns << ","
                    << std::setprecision(9) << tvec_w.at<double>(0,0) << "," << tvec_w.at<double>(1,0) << "," << tvec_w.at<double>(2,0) << ","
                    << qx << "," << qy << "," << qz << "," << qw << "\n";
            }
            ofs.close();
            if(options.verbose) CV_LOG_WARNING(NULL, "Saved trajectory CSV");
        }
    } catch(const std::exception &e){
        CV_LOG_ERROR(NULL, cv::format("Error writing trajectory CSV: %s", e.what()));
    }

    // Shutdown backend thread gracefully
    if(options.enableBackend){
        backendStop.store(true);
        backendCv.notify_one();
        if(backendThread.joinable()) backendThread.join();
    }

    cv::utils::logging::setLogLevel(prevLogLevel);
    return 0;
}

} // namespace vo
} // namespace cv