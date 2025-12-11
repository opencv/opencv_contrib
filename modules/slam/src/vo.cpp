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
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <fstream>
#include <algorithm>

namespace cv {
namespace vo {

VisualOdometry::VisualOdometry(Ptr<Feature2D> feature, Ptr<DescriptorMatcher> matcher)
    : feature_(std::move(feature)), matcher_(std::move(matcher)) {
}

int VisualOdometry::run(const std::string &imageDir, double scale_m, const VisualOdometryOptions &options){
    DataLoader loader(imageDir);
    std::cout << "VisualOdometry: loaded " << loader.size() << " images from " << imageDir << std::endl;
    if(loader.size() == 0){
        std::cerr << "VisualOdometry: no images found in " << imageDir << std::endl;
        return -1;
    }

    if(!feature_){
        feature_ = ORB::create(2000);
    }
    if(!matcher_){
        matcher_ = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    }
    // Remove internal FeatureExtractor/Matcher in favor of injected OpenCV components
    PoseEstimator poseEst;
    Visualizer vis;
    MapManager map;
    // two-view initializer
    Initializer initializer;
    // configure Localizer with a slightly stricter Lowe ratio (0.7)
    Localizer localizer(0.7f);

    // prepare per-run CSV diagnostics
    std::string runTimestamp;
    {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        std::ostringstream ss; ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        runTimestamp = ss.str();
    }
    // create a 'result' folder inside the provided imageDir (portable, avoids requiring <filesystem>)
    std::string resultDirStr = imageDir;
    if(resultDirStr.empty()) resultDirStr = std::string(".");
    if(resultDirStr.back() == '/') resultDirStr.pop_back();
    resultDirStr += "/result";
    ensureDirectoryExists(resultDirStr);
    // create a per-run folder under result/ named by timestamp
    std::string runDirStr = resultDirStr + "/" + runTimestamp;
    ensureDirectoryExists(runDirStr);
    std::string csvPath = runDirStr + "/run.csv";
    std::ofstream csv(csvPath);
    if(csv){
        csv << "frame_id,mean_diff,median_flow,pre_matches,post_matches,inliers,inlier_ratio,integrated\n";
        csv.flush();
        std::cout << "Writing diagnostics to " << csvPath << std::endl;
    } else {
        std::cerr << "Failed to open diagnostics CSV " << csvPath << std::endl;
    }

    Mat R_g = Mat::eye(3,3,CV_64F);
    Mat t_g = Mat::zeros(3,1,CV_64F);

    // simple map structures
    std::vector<KeyFrame> keyframes;
    std::vector<MapPoint> mappoints;
    std::unordered_map<int,int> keyframeIdToIndex;

    // Backend (BA) thread primitives
    std::mutex mapMutex; // protects map and keyframe modifications and writeback
    std::condition_variable backendCv;
    std::atomic<bool> backendStop(false);
    std::atomic<int> backendRequests(0);
    const int LOCAL_BA_WINDOW = 5; // window size for local BA (adjustable)

    // Start backend thread: waits for notifications and runs BA on a snapshot
    std::thread backendThread([&]() {
        while(!backendStop.load()){
            std::unique_lock<std::mutex> lk(mapMutex);
            backendCv.wait(lk, [&]{ return backendStop.load() || backendRequests.load() > 0; });
            if(backendStop.load()) break;
            // snapshot map and keyframes
            auto kfs_snapshot = map.keyframes();
            auto mps_snapshot = map.mappoints();
            // reset requests
            backendRequests.store(0);
            lk.unlock();

            // determine local window
            int K = static_cast<int>(kfs_snapshot.size());
            if(K <= 0) continue;
            int start = std::max(0, K - LOCAL_BA_WINDOW);
            std::vector<int> localKfIndices;
            for(int ii = start; ii < K; ++ii) localKfIndices.push_back(ii);
            std::vector<int> fixedKfIndices;
            if(start > 0) fixedKfIndices.push_back(0);
        #if defined(HAVE_SFM)
            // Run BA on snapshot (may take time) - uses Optimizer which will use g2o if enabled
            Optimizer::localBundleAdjustmentSFM(kfs_snapshot, mps_snapshot, localKfIndices, fixedKfIndices,
                                            loader.fx(), loader.fy(), loader.cx(), loader.cy(), 10);
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
        }
    });

    Mat frame;
    std::string imgPath;
    int frame_id = 0;

    // persistent previous-frame storage (declare outside loop so detectAndCompute can use them)
    static std::vector<KeyPoint> prevKp;
    static Mat prevGray, prevDesc;
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
            auto ratioKeep = [&](const std::vector<std::vector<DMatch>>& knn, bool forward) {
                std::vector<DMatch> filtered;
                for(size_t qi=0; qi<knn.size(); ++qi){
                    if(knn[qi].empty()) continue;
                    DMatch best = knn[qi][0];
                    float ratio = 0.75f;
                    if(knn[qi].size() >= 2){
                        if(knn[qi][1].distance > 0) {
                            if(best.distance / knn[qi][1].distance > ratio) continue;
                        }
                    }
                    // mutual check
                    int t = forward ? best.trainIdx : (int)qi;
                    // find reverse match for t
                    const auto &rev = forward ? knn21 : knn12;
                    if(t < 0 || t >= (int)rev.size() || rev[t].empty()) continue;
                    DMatch rbest = rev[t][0];
                    if((forward && rbest.trainIdx == (int)qi) || (!forward && rbest.trainIdx == best.queryIdx)){
                        filtered.push_back(best);
                    }
                }
                return filtered;
            };
            std::vector<DMatch> goodMatches = ratioKeep(knn12, true);

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
                std::cout << "Attempting two-view initialization (frame 0 + 1), matches=" << goodMatches.size() << std::endl;
                Mat R_init, t_init;
                std::vector<Point3d> pts3D;
                std::vector<bool> isTri;
                // call initializer using the matched keypoints between prev and current
                if(initializer.initialize(prevKp, kps, goodMatches, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R_init, t_init, pts3D, isTri)){
                    std::cout << "Initializer: success, creating initial keyframes and mappoints (" << pts3D.size() << ")" << std::endl;
                    // build two keyframes: previous (id = frame_id-1) and current (id = frame_id)
                    KeyFrame kf0, kf1;
                    kf0.id = frame_id - 1;
                    // prevGray/prevKp/prevDesc refer to previous frame
                    if(!prevGray.empty()){
                        if(prevGray.channels() == 1){ cvtColor(prevGray, kf0.image, COLOR_GRAY2BGR); }
                        else kf0.image = prevGray.clone();
                    }
                    kf0.kps = prevKp;
                    kf0.desc = prevDesc.clone();
                    kf0.R_w = Mat::eye(3,3,CV_64F);
                    kf0.t_w = Mat::zeros(3,1,CV_64F);

                    kf1.id = frame_id;
                    kf1.image = frame.clone();
                    kf1.kps = kps;
                    kf1.desc = desc.clone();
                    // initializer returns pose of second camera relative to first (world = first)
                    kf1.R_w = R_init.clone();
                    kf1.t_w = t_init.clone();

                    // convert initializer 3D points (in first camera frame) to MapPoints in world coords (world==first)
                    std::vector<MapPoint> newMps;
                    newMps.reserve(pts3D.size());
                    for(size_t i=0;i<pts3D.size();++i){
                        if(!isTri[i]) continue;
                        MapPoint mp;
                        mp.p = pts3D[i];
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

                    // set global pose to second keyframe (apply scale)
                    Mat t_d; kf1.t_w.convertTo(t_d, CV_64F);
                    t_g = t_d * scale_m;
                    R_g = kf1.R_w.clone();
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x,-z);

                    // write CSV entry for initialization frame
                    if(csv){
                        csv << frame_id << ",init,0,0,0,0,0,1\n";
                        csv.flush();
                    }

                    // notify backend to run BA on initial map
                    backendRequests.fetch_add(1);
                    backendCv.notify_one();
                    // skip the usual PnP / poseEst path for this frame since we've initialized
                    prevGray = gray.clone(); prevKp = kps; prevDesc = desc.clone();
                    frame_id++;
                    continue;
                } else {
                    std::cout << "Initializer: failed to initialize from first two frames" << std::endl;
                }
            }

            // Try PnP against map points first (via Localizer)
            bool solvedByPnP = false;
            Mat R_pnp, t_pnp; int inliers_pnp = 0;
            int preMatches_pnp = 0, postMatches_pnp = 0; double meanReproj_pnp = 0.0;
            if(localizer.tryPnP(map, desc, kps, loader.fx(), loader.fy(), loader.cx(), loader.cy(), gray.cols, gray.rows,
                                options.min_inliers, R_pnp, t_pnp, inliers_pnp, frame_id, &frame, runDirStr,
                                &preMatches_pnp, &postMatches_pnp, &meanReproj_pnp)){
                solvedByPnP = true;
                std::cout << "PnP solved: preMatches="<<preMatches_pnp<<" post="<<postMatches_pnp<<" inliers="<<inliers_pnp<<" meanReproj="<<meanReproj_pnp<<std::endl;
            }

            if(pts1.size() >= 8 && !solvedByPnP){
                Mat R, t, mask; int inliers = 0;
                bool ok = poseEst.estimate(pts1, pts2, loader.fx(), loader.fy(), loader.cx(), loader.cy(), R, t, mask, inliers);

                int matchCount = post_matches;
                double inlierRatio = matchCount > 0 ? double(inliers) / double(matchCount) : 0.0;

                // thresholds (tunable) -- relaxed and add absolute inlier guard
                const int MIN_MATCHES = 15;           // require at least this many matches (relative)
                const int MIN_INLIERS = 4;             // OR accept if at least this many absolute inliers
                double t_norm = 0.0, rot_angle = 0.0;
                if(ok){
                    Mat t_d; t.convertTo(t_d, CV_64F);
                    t_norm = norm(t_d);
                    Mat R_d; R.convertTo(R_d, CV_64F);
                    double trace = R_d.at<double>(0,0) + R_d.at<double>(1,1) + R_d.at<double>(2,2);
                    double cos_angle = std::min(1.0, std::max(-1.0, (trace - 1.0) * 0.5));
                    rot_angle = std::acos(cos_angle);
                }

                // Print per-frame diagnostics
                // std::cout << "F" << frame_id << " diff=" << meanDiff << " median_flow=" << median_flow
                //           << " pre_matches=" << pre_matches << " post_matches=" << matchCount << " inliers=" << inliers << " inlierRatio=" << inlierRatio
                //           << " t_norm=" << t_norm << " rot_rad=" << rot_angle << std::endl;

                // decide whether to integrate
                // Prefer geometry-based decision (absolute inliers OR matchCount + ratio). Use image-diff/flow
                // only to skip when geometry is weak or motion truly negligible.
                bool integrate = true;
                if(!ok){
                    integrate = false;
                    // std::cout << "  -> pose estimation failed, skipping integration." << std::endl;
                } else if(inliers < MIN_INLIERS || matchCount < MIN_MATCHES){
                    // Not enough geometric support -> skip (unless absolute inliers pass)
                    integrate = false;
                    // std::cout << "  -> insufficient matches/inliers (by both absolute and relative metrics), skipping integration." << std::endl;
                } else {
                    // We have sufficient geometric support. Only skip if motion is truly negligible
                    // (both translation and rotation tiny) AND the image/flow indicate near-identical frames.
                    const double MIN_TRANSLATION_NORM = 1e-4;
                    const double MIN_ROTATION_RAD = (0.5 * CV_PI / 180.0); // 0.5 degree
                    const double DIFF_ZERO_THRESH = 2.0;   // nearly identical image
                    const double FLOW_ZERO_THRESH = 0.3;   // nearly zero flow in pixels

                    if(t_norm < MIN_TRANSLATION_NORM && std::abs(rot_angle) < MIN_ROTATION_RAD
                       && meanDiff < DIFF_ZERO_THRESH && median_flow < FLOW_ZERO_THRESH){
                        integrate = false; // truly static
                        // std::cout << "  -> negligible motion and near-identical frames, skipping integration." << std::endl;
                    }
                }
                if (inliers >= options.min_inliers || (inliers >= 2 && matchCount > 50 && median_flow > 2.0)) {
                    integrate = true;
                }

                // integrate transform if allowed
                if(integrate){
                    Mat t_d; t.convertTo(t_d, CV_64F);
                    Mat t_scaled = t_d * scale_m;
                    Mat R_d; R.convertTo(R_d, CV_64F);
                    t_g = t_g + R_g * t_scaled;
                    R_g = R_g * R_d;
                    double x = t_g.at<double>(0);
                    double z = t_g.at<double>(2);
                    vis.addPose(x,-z);
                }

                    // if we integrated, create a keyframe and optionally triangulate new map points
                if(integrate){
                    KeyFrame kf;
                    kf.id = frame_id;
                    kf.image = frame.clone();
                    kf.kps = kps;
                    kf.desc = desc.clone();
                    kf.R_w = R_g.clone(); kf.t_w = t_g.clone();

                    bool didTriangulate = false;
                    if(!map.keyframes().empty() && map.keyframes().back().id == frame_id - 1){
                        // triangulate between last keyframe and this frame using normalized coordinates
                        const KeyFrame &last = map.keyframes().back();
                        std::vector<Point2f> pts1n, pts2n; pts1n.reserve(pts1.size()); pts2n.reserve(pts2.size());
                        double fx = loader.fx(), fy = loader.fy(), cx = loader.cx(), cy = loader.cy();
                        for(size_t i=0;i<pts1.size();++i){
                            pts1n.emplace_back(float((pts1[i].x - cx)/fx), float((pts1[i].y - cy)/fy));
                            pts2n.emplace_back(float((pts2[i].x - cx)/fx), float((pts2[i].y - cy)/fy));
                        }
                        // build kp index lists (matching goodMatches order)
                        std::vector<int> pts1_kp_idx; pts1_kp_idx.reserve(goodMatches.size());
                        std::vector<int> pts2_kp_idx; pts2_kp_idx.reserve(goodMatches.size());
                        for(const auto &m: goodMatches){ pts1_kp_idx.push_back(m.queryIdx); pts2_kp_idx.push_back(m.trainIdx); }
                        // triangulate between the last keyframe in the map and the CURRENT keyframe `kf`.
                        // previously the code passed `keyframes.back()` which refers to the previous local
                        // keyframe (or the same as `last`), resulting in triangulation between the same
                        // frame and thus no new points. Pass `kf` to triangulate between `last` and
                        // the current frame.
                        auto newPts = map.triangulateBetweenLastTwo(pts1n, pts2n, pts1_kp_idx, pts2_kp_idx, last, kf, fx, fy, cx, cy);
                        if(!newPts.empty()){
                            didTriangulate = true;
                            // already appended inside MapManager
                        }
                    }

                    {
                        // insert keyframe and map points under lock to keep consistent state
                        std::lock_guard<std::mutex> lk(mapMutex);
                        keyframes.push_back(std::move(kf));
                        map.addKeyFrame(keyframes.back());
                    }
                    if(didTriangulate){
                        std::cout << "Created keyframe " << frame_id << " and triangulated new map points (total=" << map.mappoints().size() << ")" << std::endl;
                    } else {
                        std::cout << "Created keyframe " << frame_id << " (no triangulation)" << std::endl;
                    }
                    // Notify backend thread to run local BA asynchronously
                    backendRequests.fetch_add(1);
                    backendCv.notify_one();
                }

                // write CSV line
                if(csv){
                    csv << frame_id << "," << meanDiff << "," << median_flow << "," << pre_matches << "," << post_matches << "," << inliers << "," << inlierRatio << "," << (integrate?1:0) << "\n";
                    csv.flush();
                }

                // Always show a single image; if we have matches, draw small boxes around matched keypoints
                Mat visImg;
                if(frame.channels() > 1) visImg = frame.clone();
                else cvtColor(gray, visImg, COLOR_GRAY2BGR);
                std::string info = std::string("Frame ") + std::to_string(frame_id) + " matches=" + std::to_string(matchCount) + " inliers=" + std::to_string(inliers);
                if(!goodMatches.empty()){
                    for(size_t mi=0; mi<goodMatches.size(); ++mi){
                        Point2f p2 = (mi < pts2.size()) ? pts2[mi] : kps[goodMatches[mi].trainIdx].pt;

                        // determine inlier status from mask (robust to mask shape)
                        bool isInlier = false;
                        if(!mask.empty()){
                            if(mask.rows == static_cast<int>(goodMatches.size())){
                                isInlier = mask.at<uchar>(static_cast<int>(mi), 0) != 0;
                            } else if(mask.cols == static_cast<int>(goodMatches.size())){
                                isInlier = mask.at<uchar>(0, static_cast<int>(mi)) != 0;
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
                vis.showFrame(gray);
            }
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

    // save trajectory with timestamp into result/ folder
    try{
        // save trajectory into the per-run folder using a simple filename (no timestamp)
        std::string outDir = resultDirStr + "/" + runTimestamp;
        ensureDirectoryExists(outDir);
        std::string outPath = outDir + "/trajectory.png";
        if(vis.saveTrajectory(outPath)){
            std::cout << "Saved trajectory to " << outPath << std::endl;
        } else {
            std::cerr << "Failed to save trajectory to " << outPath << std::endl;
        }
    } catch(const std::exception &e){
        std::cerr << "Error saving trajectory: " << e.what() << std::endl;
    }

    // Shutdown backend thread gracefully
    backendStop.store(true);
    backendCv.notify_one();
    if(backendThread.joinable()) backendThread.join();

    return 0;
}

} // namespace vo
} // namespace cv