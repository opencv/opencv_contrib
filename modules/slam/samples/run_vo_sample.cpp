// Minimal wrapper to exercise cv::vo::VisualOdometry on a folder of images (e.g., EuRoC MH01).
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>
#include <cmath>
#include <opencv2/slam.hpp>
#include <opencv2/slam/visualizer.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <string>

static std::string makeTimestampFolder(){
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return ss.str();
}

int main(int argc, char** argv) {
    if(argc < 2){
        std::cout << "Usage: " << argv[0] << " [image_dir] [scale_m=0.02] [output_dir] [test_mode] [map_path] [vocab_path]\n"
                  << "\nTest Modes (for comparison):\n"
                  << "  vo          - Visual Odometry only (no backend, no loop closure)\n"
                  << "  vo_backend  - VO + Backend BA (no loop closure)\n"
                  << "  slam        - Full SLAM (VO + Backend BA + Loop Closure, default)\n"
                  << "  localization - Localization mode (requires map_path)\n"
                  << "\nOther parameters:\n"
                  << "  - Default output: <dataset>/slam_output/<YYYYMMDD_HHMMSS>/\n"
                  << "  - Default map path: <default_output>/map.yml.gz\n"
                  << "  - vocab_path: ORB vocabulary file for DBoW3 (required for loop closure)\n"
                  << "\nExamples:\n"
                  << "  " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) vo\n"
                  << "  " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) vo_backend\n"
                  << "  " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) slam (auto) /root/orbvoc.dbow3\n"
                  << "  " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) localization <map_path>" << std::endl;
        return 0;
    }

    std::string imgDir = argv[1];
    // imgDir = "../../datasets/iphone/2025-11-05_170219";
    // imgDir = "../../datasets/EuRoC/MH01/mav0/cam0/data";
    // imgDir = "../../datasets/vivo/room";
    double scale_m = (argc >= 3) ? std::atof(argv[2]) : 0.02;
    std::string testMode = (argc >= 5) ? argv[4] : std::string("slam");

    // Default output root: dataset folder / slam_output / <timestamp>
    std::string imgDirAbs = cv::utils::fs::canonical(imgDir);
    std::string datasetRoot = cv::utils::fs::getParent(imgDirAbs);

    auto isAbsolutePath = [](const std::string& p){
        return !p.empty() && (p[0] == '/' || (p.size() > 1 && p[1] == ':'));
    };

    std::string userOut = (argc >= 4) ? argv[3] : std::string();
    std::string outRoot;
    if(!userOut.empty()){
        outRoot = isAbsolutePath(userOut) ? userOut : cv::utils::fs::join(datasetRoot, userOut);
    } else {
        outRoot = cv::utils::fs::join(datasetRoot, "slam_output");
    }

    std::string outDir = cv::utils::fs::join(outRoot, makeTimestampFolder());
    std::string defaultMapName = "map.yml.gz"; // gzip reduces map size significantly
    std::string defaultMapPath = cv::utils::fs::join(outDir, defaultMapName);
    std::string mapPath;
    if(argc >= 6){
        mapPath = argv[5];
        if(mapPath.empty() || mapPath == "(auto)"){
            mapPath = defaultMapPath;
        } else if(!isAbsolutePath(mapPath)){
            mapPath = cv::utils::fs::join(outDir, mapPath);
        }
    } else {
        mapPath = defaultMapPath;
    }
    std::string vocabPath = (argc >= 7) ? argv[6] : std::string();

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::vo::SlamSystem slam(feature, matcher);
    cv::vo::VisualOdometryOptions feOpts;
    cv::vo::SlamSystemOptions sysOpts;
    
    // Configure based on test mode
    bool enableBackend = false;
    bool enableLoopClosure = false;
    std::string modeDescription;
    
    if(testMode == "vo"){
        enableBackend = false;
        enableLoopClosure = false;
        modeDescription = "VO (Visual Odometry only, no backend, no loop closure)";
    } else if(testMode == "vo_backend"){
        enableBackend = true;
        enableLoopClosure = false;
        modeDescription = "VO_BACKEND (VO + Backend BA, no loop closure)";
    } else if(testMode == "slam"){
        enableBackend = true;
        enableLoopClosure = true;
        modeDescription = "SLAM (VO + Backend BA + Loop Closure)";
    } else if(testMode == "localization"){
        enableBackend = false;
        enableLoopClosure = false;
        modeDescription = "LOCALIZATION (map frozen, pose estimation only)";
    } else {
        std::cerr << "Unknown test mode: " << testMode << std::endl;
        std::cerr << "Valid modes: vo, vo_backend, slam, localization" << std::endl;
        return -1;
    }
    
    sysOpts.enableBackend = enableBackend;
    slam.setFrontendOptions(feOpts);
    slam.setSystemOptions(sysOpts);

    // Set loop closure vocabulary only if enabled
    if(enableLoopClosure){
        if(!vocabPath.empty() && vocabPath != "(auto)"){
            cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
            bool ok = slam.setLoopVocabulary(vocabPath);
            std::cout << "Loop vocabulary: " << vocabPath << " => " << (ok ? "OK" : "FAILED") << std::endl;
            if(!ok){
                std::cerr << "Warning: Failed to load vocabulary, loop closure will be disabled" << std::endl;
                enableLoopClosure = false;
            }
        } else {
            std::cout << "Loop vocabulary: (not set, loop closure disabled)" << std::endl;
            enableLoopClosure = false;
        }
    } else {
        std::cout << "Loop vocabulary: (disabled for this test mode)" << std::endl;
    }

    if(testMode == "localization"){
        if(!slam.loadMap(mapPath)){
            std::cerr << "Failed to load map for localization: " << mapPath << std::endl;
            return -4;
        }
        slam.setMode(cv::vo::MODE_LOCALIZATION);
        std::cout << "Mode: " << modeDescription << std::endl;
    } else {
        slam.setMode(cv::vo::MODE_SLAM);
        std::cout << "Mode: " << modeDescription << std::endl;
    }

    std::cout << "Running OpenCV SlamSystem on " << imgDir << std::endl;
    std::cout << "Controls: press 'q' or ESC to exit" << std::endl;

    if(!cv::vo::ensureDirectoryExists(outDir)){
        std::cerr << "Failed to create output directory: " << outDir << std::endl;
        return -2;
    }

    cv::vo::DataLoader loader(imgDir);
    if(loader.size() == 0){
        std::cerr << "No images found in: " << imgDir << std::endl;
        return -1;
    }

    slam.setCameraIntrinsics(loader.fx(), loader.fy(), loader.cx(), loader.cy());
    slam.setWorldScale(scale_m);

    // Reuse the module's Visualizer for the top-down trajectory view.
    // meters_per_pixel=0.02 => 50 px per meter (roughly matches previous drawScale=50).
    cv::vo::Visualizer viz(600, 600, 0.02);

    // Timing: measure total runtime using OpenCV TickMeter
    cv::TickMeter tm; tm.start();

    auto extractTimestamp = [](const std::string &p, int fallback)->double{
        try{
            std::string fname = p;
            auto slash = fname.find_last_of("/\\");
            if(slash != std::string::npos) fname = fname.substr(slash+1);
            auto dot = fname.find_last_of('.');
            if(dot != std::string::npos) fname = fname.substr(0, dot);
            if(!fname.empty()){
                double ts = std::stod(fname);
                if(ts > 1e12) ts *= 1e-9;
                else if(ts > 1e9) ts *= 1e-6;
                return ts;
            }
        } catch(...){}
        return static_cast<double>(fallback);
    };

    cv::Mat frame;
    std::string path;
    int fid = 0;
    while(loader.getNextImage(frame, path)){
        const double ts = extractTimestamp(path, fid);
        auto res = slam.track(frame, ts);

        cv::Mat vis;
        if(frame.channels() == 1) cv::cvtColor(frame, vis, cv::COLOR_GRAY2BGR);
        else vis = frame.clone();

        if(res.ok && !res.t_w.empty()){
            // res.t_w is camera center in world (per current implementation).
            const double x = res.t_w.at<double>(0);
            const double z = res.t_w.rows >= 3 ? res.t_w.at<double>(2) : 0.0;

            // Update trajectory view using existing Visualizer (for real-time display).
            viz.addPose(x, z);
        }

        // Use Visualizer to overlay tracking info and present frames.
        // Pass 1-based frame index to visualizer for display
        viz.drawTrackingInfo(vis, res, fid + 1);
        viz.showFrame(vis);
        viz.showTopdown();

        int key = cv::waitKey(1);
        if(key == 27 || key == 'q' || key == 'Q') break;
        fid++;
    }

    const bool mapSaved = slam.saveMap(mapPath);
    if(!mapSaved){
        std::cerr << "Failed to save map: " << mapPath << std::endl;
    }

    // Save trajectory as the final output (trajectory_tum.csv)
    // For VO mode, save online trajectory (no optimization)
    // For other modes, save optimized trajectory
    const std::string trajFinal = outDir + "/trajectory_tum.csv";
    if(testMode == "vo"){
        // VO only mode: save online trajectory (no backend optimization)
        slam.saveTrajectoryTUM(trajFinal);
        std::cout << "Online trajectory saved (VO only, no optimization): " << trajFinal << std::endl;
    } else {
        // Backend enabled: save optimized trajectory
        slam.saveOptimizedTrajectoryTUM(trajFinal);
        std::cout << "Optimized trajectory saved: " << trajFinal << std::endl;
    }

    // Update visualizer with optimized trajectory for final image
    // Read the optimized trajectory and set it to visualizer
    std::ifstream trajFile(trajFinal);
    if(trajFile.is_open()){
        std::string line;
        std::getline(trajFile, line); // Skip header
        std::vector<cv::Point2d> optimizedTraj;
        while(std::getline(trajFile, line)){
            if(line.empty()) continue;
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            while(std::getline(iss, token, ',')){
                tokens.push_back(token);
            }
            if(tokens.size() >= 4){
                try{
                    double tx = std::stod(tokens[1]);
                    double tz = std::stod(tokens[3]);
                    // addPose(x, z) now stores (x, z) directly for top-down view
                    // CSV tz is world z (forward = positive), store directly
                    optimizedTraj.emplace_back(tx, tz);
                } catch(...){
                    continue;
                }
            }
        }
        if(!optimizedTraj.empty()){
            viz.setTrajectoryXZ(optimizedTraj);
        }
    }

    const std::string trajPng = outDir + "/trajectory.png";
    if(!viz.saveTrajectory(trajPng)){
        std::cerr << "Failed to write trajectory image: " << trajPng << std::endl;
    } else {
        std::cout << "Trajectory visualization saved: " << trajPng << std::endl;
    }

    // Stop timer and print elapsed hours/minutes/seconds (seconds rounded)
    tm.stop();
    double elapsed = tm.getTimeSec();
    int totalSec = static_cast<int>(std::round(elapsed));
    int hrs = totalSec / 3600;
    int mins = (totalSec % 3600) / 60;
    int secs = totalSec % 60;
    std::ostringstream timess;
    timess << "Elapsed time: ";
    bool printed = false;
    if(hrs > 0){ timess << hrs << " hours "; printed = true; }
    if(mins > 0){ timess << mins << " minutes "; printed = true; }
    timess << secs << " seconds";
    std::cout << timess.str() << std::endl;

    return 0;
}
