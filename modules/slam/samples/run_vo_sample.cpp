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
#include <opencv2/slam.hpp>
#include <opencv2/slam/visualizer.hpp>
#include <opencv2/core/quaternion.hpp>
#include <opencv2/core/utils/filesystem.hpp>
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
        std::cout << "Usage: " << argv[0] << " [image_dir] [scale_m=0.02] [output_dir] [mode=slam|localization] [map_path]\n"
                  << "- Default output: <dataset>/slam_output/<YYYYMMDD_HHMMSS>/\n"
                  << "- Default map path: <default_output>/map.yml.gz (gzip to shrink size)\n"
                  << "Example (SLAM):  " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) slam\n"
                  << "Example (LOC):   " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02 (auto) localization <output>/map.yml.gz" << std::endl;
        return 0;
    }

    std::string imgDir = argv[1];
    // imgDir = "../../datasets/iphone/2025-11-05_170219";
    // std::string imgDir = "../../datasets/EuRoC/MH01/mav0/cam0/data";
    // imgDir = "../../datasets/vivo/room";
    double scale_m = (argc >= 3) ? std::atof(argv[2]) : 0.02;
    std::string modeStr = (argc >= 5) ? argv[4] : std::string("slam");

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
    std::string mapPath = (argc >= 6) ? argv[5] : cv::utils::fs::join(outDir, defaultMapName);

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::vo::SlamSystem slam(feature, matcher);
    cv::vo::VisualOdometryOptions feOpts;
    cv::vo::SlamSystemOptions sysOpts;
    // sysOpts.enableBackend = false;

    slam.setFrontendOptions(feOpts);
    slam.setSystemOptions(sysOpts);

    if(modeStr == "localization"){
        if(!slam.loadMap(mapPath)){
            std::cerr << "Failed to load map for localization: " << mapPath << std::endl;
            return -4;
        }
        slam.setMode(cv::vo::MODE_LOCALIZATION);
        std::cout << "Mode: LOCALIZATION (map frozen)" << std::endl;
    } else {
        slam.setMode(cv::vo::MODE_SLAM);
        std::cout << "Mode: SLAM" << std::endl;
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

    // TUM-like CSV: timestamp,tx,ty,tz,qx,qy,qz,qw
    std::ofstream trajCsv(outDir + "/trajectory_tum.csv", std::ios::out);
    if(!trajCsv.is_open()){
        std::cerr << "Failed to open trajectory output: " << (outDir + "/trajectory_tum.csv") << std::endl;
        return -3;
    }
    trajCsv << "timestamp,tx,ty,tz,qx,qy,qz,qw\n";

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

            // Update trajectory view using existing Visualizer.
            viz.addPose(x, z);

            if(!res.R_w.empty() && res.t_w.rows >= 3){
                cv::Quatd q = cv::Quatd::createFromRotMat(res.R_w);
                const cv::Vec4d qvec(q.at(1), q.at(2), q.at(3), q.at(0)); // [qx,qy,qz,qw]
                const double tx = res.t_w.at<double>(0);
                const double ty = res.t_w.at<double>(1);
                const double tz = res.t_w.at<double>(2);
                trajCsv << std::fixed << std::setprecision(9)
                        << ts << "," << tx << "," << ty << "," << tz << ","
                        << qvec[0] << "," << qvec[1] << "," << qvec[2] << "," << qvec[3] << "\n";
            }
        }

        // Use Visualizer to overlay tracking info and present frames.
        viz.drawTrackingInfo(vis, res);
        viz.showFrame(vis);
        viz.showTopdown();

        int key = cv::waitKey(1);
        if(key == 27 || key == 'q' || key == 'Q') break;
        fid++;
    }

    trajCsv.close();

    const std::string trajPng = outDir + "/trajectory.png";
    if(!viz.saveTrajectory(trajPng)){
        std::cerr << "Failed to write trajectory image: " << trajPng << std::endl;
    }

    if(!slam.saveMap(mapPath)){
        std::cerr << "Failed to save map: " << mapPath << std::endl;
    }

    return 0;
}
