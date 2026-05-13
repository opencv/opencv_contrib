/**
 * @brief Map save/load example
 *
 * Demonstrates how to save and load a SLAM map:
 * 1. Run SLAM and save the map
 * 2. Reload the map
 * 3. Verify map integrity
 *
 * Use cases:
 * - Save mapping results once and reuse multiple times
 * - Share one map across multiple robots
 * - Offline mapping + online localization
 *
 * Usage:
 *   ./example_map_save_load <config.yaml> <vocab.fbow> <image_dir> <output_dir>
 *
 * Example:
 *   ./example_map_save_load EuRoC_mono.yaml orb_vocab.fbow /path/to/images /tmp/output
 */

#include <opencv2/slam.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

/**
 * @brief Load images from a directory
 */
std::vector<std::pair<std::string, double>> load_images(const std::string& data_dir) {
    std::vector<std::pair<std::string, double>> images;

    std::string cmd = "ls -1 " + data_dir + "/*.png | sort";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return images;

    char buffer[256];
    size_t idx = 0;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string path(buffer);
        path.erase(path.find_last_not_of("\n\r") + 1);
        images.push_back({path, idx * 0.05});
        idx++;
    }
    pclose(pipe);

    return images;
}

/**
 * @brief Run SLAM and return the number of map points
 */
size_t run_slam(const std::string& config_file, const std::string& vocab_file,
                const std::vector<std::pair<std::string, double>>& images,
                const std::string& map_path, const std::string& traj_path) {

    auto orb = cv::ORB::create(1000, 1.2f, 8);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);


    std::cout << "Initializing SLAM system..." << std::endl;
    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);
    slam->setFeatureDetector(orb);
    slam->setMatcher(matcher);
    slam->setBackendEnabled(true, 10);
    slam->setLoopClosureEnabled(true);


    std::cout << "Processing " << images.size() << " frames..." << std::endl;
    int tracked = 0;

    for (size_t i = 0; i < images.size(); i++) {
        const auto& [img_path, timestamp] = images[i];

        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        auto pose = slam->processFrame(img, timestamp);
        if (pose.has_value()) {
            tracked++;
        }

        if (i % 100 == 0) {
            std::cout << "Frame " << i << "/" << images.size()
                      << " (tracked: " << tracked << ")" << std::endl;
        }
    }

    std::cout << "Tracked: " << tracked << "/" << images.size() << std::endl;


    auto map_points = slam->getMapPoints();
    std::cout << "Map contains " << map_points.size() << " 3D points" << std::endl;


    std::cout << "\nSaving map to: " << map_path << std::endl;
    if (!slam->saveMap(map_path)) {
        std::cerr << "Error: Failed to save map!" << std::endl;
        return 0;
    }
    std::cout << "Map saved successfully" << std::endl;


    std::cout << "Saving trajectory to: " << traj_path << std::endl;
    if (!slam->saveTrajectory(traj_path, "TUM")) {
        std::cerr << "Warning: Failed to save trajectory" << std::endl;
    }


    slam->release();

    return map_points.size();
}

/**
 * @brief Load and verify a map
 */
bool load_and_verify_map(const std::string& config_file, const std::string& vocab_file,
                         const std::string& map_path,
                         const std::vector<std::pair<std::string, double>>& images) {

    auto orb = cv::ORB::create(1000, 1.2f, 8);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);


    std::cout << "\nInitializing SLAM system for map loading..." << std::endl;
    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);
    slam->setFeatureDetector(orb);
    slam->setMatcher(matcher);
    slam->setBackendEnabled(true, 10);
    slam->setLoopClosureEnabled(true);


    std::cout << "Loading map from: " << map_path << std::endl;
    if (!slam->loadMap(map_path)) {
        std::cerr << "Error: Failed to load map!" << std::endl;
        return false;
    }


    auto map_points = slam->getMapPoints();
    std::cout << "Map loaded successfully (" << map_points.size() << " 3D points)" << std::endl;


    std::cout << "\nSwitching to LOCALIZATION mode..." << std::endl;
    slam->setMode(cv::vo::SLAMMode::LOCALIZATION);


    std::cout << "Testing localization on " << images.size() << " frames..." << std::endl;
    int tracked = 0;

    for (size_t i = 0; i < images.size(); i++) {
        const auto& [img_path, timestamp] = images[i];

        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        auto pose = slam->processFrame(img, timestamp);
        if (pose.has_value()) {
            tracked++;
        }
    }

    std::cout << "Localization tracked: " << tracked << "/" << images.size()
              << " (" << (100.0 * tracked / images.size()) << "%)" << std::endl;


    slam->release();

    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.yaml> <vocab.fbow> <image_dir> <output_dir>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0]
                  << " EuRoC_mono.yaml orb_vocab.fbow /path/to/images /tmp/output" << std::endl;
        return 1;
    }

    const std::string config_file = argv[1];
    const std::string vocab_file = argv[2];
    const std::string image_dir = argv[3];
    const std::string output_dir = argv[4];

    std::filesystem::create_directories(output_dir);

    std::cout << "Loading images from: " << image_dir << std::endl;
    auto images = load_images(image_dir);
    if (images.empty()) {
        std::cerr << "Error: No images found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << images.size() << " images" << std::endl;


    std::string map_path = output_dir + "/map.json";
    std::string traj_path = output_dir + "/trajectory.txt";


    std::cout << "\n=== Step 1: Run SLAM and Save Map ===" << std::endl;
    size_t original_map_size = run_slam(config_file, vocab_file, images, map_path, traj_path);

    if (original_map_size == 0) {
        std::cerr << "Error: SLAM failed!" << std::endl;
        return 1;
    }


    std::cout << "\n=== Step 2: Load Map and Verify ===" << std::endl;
    if (!load_and_verify_map(config_file, vocab_file, map_path, images)) {
        std::cerr << "Error: Map verification failed!" << std::endl;
        return 1;
    }

    std::cout << "\n=== Map Save/Load Test Complete ===" << std::endl;
    std::cout << "Original map size: " << original_map_size << " points" << std::endl;
    std::cout << "Map file: " << map_path << std::endl;
    std::cout << "Trajectory file: " << traj_path << std::endl;

    return 0;
}
