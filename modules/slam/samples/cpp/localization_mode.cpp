/**
 * @brief Pure localization mode example
 * 
 * Demonstrates how to use a prebuilt map for pure localization (without mapping):
 * 1. Load a prebuilt map
 * 2. Switch to LOCALIZATION mode
 * 3. Perform localization in a known environment
 * 
 * Use cases:
 * - Robot/AGV localization in a known environment
 * - AR/VR device tracking on a prebuilt map
 * - Repetitive inspection tasks (no need to rebuild the map)
 * 
 * Usage:
 *   ./example_localization_mode <config.yaml> <vocab.fbow> <map_path> <image_dir>
 * 
 * Example:
 *   ./example_localization_mode EuRoC_mono.yaml orb_vocab.fbow /tmp/map.msgpack /path/to/images
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

/**
 * @brief Load images from a directory (simplified version)
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
        images.push_back({path, idx * 0.05});  // Assume 20 FPS
        idx++;
    }
    pclose(pipe);
    
    return images;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] 
                  << " <config.yaml> <vocab.fbow> <map_path> <image_dir>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] 
                  << " EuRoC_mono.yaml orb_vocab.fbow /tmp/map.msgpack /path/to/images" << std::endl;
        return 1;
    }
    
    const std::string config_file = argv[1];
    const std::string vocab_file = argv[2];
    const std::string map_path = argv[3];
    const std::string image_dir = argv[4];
    
    
    std::cout << "Loading images from: " << image_dir << std::endl;
    auto images = load_images(image_dir);
    if (images.empty()) {
        std::cerr << "Error: No images found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << images.size() << " images" << std::endl;
    
    
    auto orb = cv::ORB::create(1000, 1.2f, 8);
    auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    
    
    std::cout << "\nInitializing SLAM system..." << std::endl;
    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);
    slam->setFeatureDetector(orb);
    slam->setMatcher(matcher);
    slam->setBackendEnabled(true, 10);
    slam->setLoopClosureEnabled(true);
    
    
    std::cout << "Loading map from: " << map_path << std::endl;
    if (!slam->loadMap(map_path)) {
        std::cerr << "Error: Failed to load map!" << std::endl;
        return 1;
    }
    
    
    auto map_points = slam->getMapPoints();
    std::cout << "Map loaded successfully (" << map_points.size() << " 3D points)" << std::endl;
    
    
    std::cout << "\nSwitching to LOCALIZATION mode..." << std::endl;
    slam->setMode(cv::vo::SLAMMode::LOCALIZATION);
    
    
    auto mode = slam->getMode();
    std::cout << "Current mode: " 
              << (mode == cv::vo::SLAMMode::LOCALIZATION ? "LOCALIZATION" : "SLAM") 
              << std::endl;
    
    
    std::cout << "\nProcessing " << images.size() << " frames in LOCALIZATION mode..." << std::endl;
    int tracked = 0;
    double total_time = 0;
    
    for (size_t i = 0; i < images.size(); i++) {
        const auto& [img_path, timestamp] = images[i];
        
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Warning: Failed to load " << img_path << std::endl;
            continue;
        }
        
        
        auto start = std::chrono::high_resolution_clock::now();
        auto pose = slam->processFrame(img, timestamp);
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double>(end - start).count() * 1000;
        total_time += elapsed;
        
        if (pose.has_value()) {
            tracked++;
            
            
            if (i % 100 == 0) {
                std::cout << "Frame " << i << "/" << images.size() 
                          << " (tracked: " << tracked 
                          << ", " << elapsed << " ms)" << std::endl;
            }
        }
    }
    
    
    std::cout << "\n=== Localization Statistics ===" << std::endl;
    std::cout << "Tracked frames: " << tracked << "/" << images.size() 
              << " (" << (100.0 * tracked / images.size()) << "%)" << std::endl;
    if (tracked > 0) {
        std::cout << "Average processing time: " << (total_time / images.size()) << " ms" << std::endl;
    }
    
    
    std::string traj_path = image_dir + "/../localization_trajectory.txt";
    std::cout << "\nSaving localization trajectory to: " << traj_path << std::endl;
    if (slam->saveTrajectory(traj_path, "TUM")) {
        std::cout << "Trajectory saved successfully" << std::endl;
    } else {
        std::cerr << "Warning: Failed to save trajectory" << std::endl;
    }
    
    
    std::cout << "\nShutting down SLAM system..." << std::endl;
    slam->release();
    
    std::cout << "Done!" << std::endl;
    return 0;
}
