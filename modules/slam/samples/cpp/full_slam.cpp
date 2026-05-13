/**
 * @brief Complete SLAM pipeline example
 *
 * Demonstrates how to use cv::vo::VisualOdometry to run a full SLAM pipeline:
 * 1. Initialize the SLAM system
 * 2. Process an image sequence
 * 3. Save the map and trajectory
 *
 * Usage:
 *   ./example_full_slam <config.yaml> <vocab.fbow> <image_dir> <output_dir>
 *
 * Example:
 *   ./example_full_slam EuRoC_mono.yaml orb_vocab.fbow /path/to/images /path/to/output
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
 * @brief Load image paths and timestamps from EuRoC-format data.csv
 */
std::vector<std::pair<std::string, double>> load_image_timestamps(const std::string& data_dir) {
    std::vector<std::pair<std::string, double>> images;
    std::string csv_path = data_dir + "/../data.csv";

    std::ifstream csv(csv_path);
    if (!csv.is_open()) {
        std::cerr << "Warning: Could not open " << csv_path << ", using file listing" << std::endl;

        // Fallback: list all PNG files
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

    std::string line;
    std::getline(csv, line);  // Skip header

    while (std::getline(csv, line)) {
        if (line.empty() || line[0] == '#') continue;

        auto comma = line.find(',');
        if (comma == std::string::npos) continue;

        long long ts_ns = std::stoll(line.substr(0, comma));
        std::string filename = line.substr(comma + 1);
        filename.erase(filename.find_last_not_of("\n\r ") + 1);

        double ts_s = static_cast<double>(ts_ns) / 1e9;
        images.push_back({data_dir + "/" + filename, ts_s});
    }

    return images;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.yaml> <vocab.fbow> <image_dir> <output_dir>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0]
                  << " modules/slam/testdata/config/euroc_mh01.yaml modules/slam/testdata/vocab/orb_vocab.fbow /datasets/EuRoC/MH01/mav0/cam0/data /tmp/output" << std::endl;
        return 1;
    }

    const std::string config_file = argv[1];
    const std::string vocab_file = argv[2];
    const std::string image_dir = argv[3];
    const std::string output_dir = argv[4];


    std::cout << "Loading images from: " << image_dir << std::endl;
    auto images = load_image_timestamps(image_dir);
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


    std::cout << "Processing " << images.size() << " frames..." << std::endl;
    int tracked = 0;
    double total_time = 0;

    auto start_total = std::chrono::high_resolution_clock::now();

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

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_total - start_total).count();


    std::cout << "\n=== SLAM Statistics ===" << std::endl;
    std::cout << "Tracked frames: " << tracked << "/" << images.size()
              << " (" << (100.0 * tracked / images.size()) << "%)" << std::endl;
    std::cout << "Average FPS: " << (images.size() / total_elapsed) << std::endl;
    std::cout << "Average processing time: " << (total_time / images.size()) << " ms" << std::endl;


    std::string map_path = output_dir + "/map.msgpack";
    std::cout << "\nSaving map to: " << map_path << std::endl;
    if (slam->saveMap(map_path)) {
        std::cout << "Map saved successfully" << std::endl;
    } else {
        std::cerr << "Warning: Failed to save map" << std::endl;
    }


    std::string traj_path = output_dir + "/trajectory.txt";
    std::cout << "Saving trajectory to: " << traj_path << std::endl;
    if (slam->saveTrajectory(traj_path, "TUM")) {
        std::cout << "Trajectory saved successfully" << std::endl;
    } else {
        std::cerr << "Warning: Failed to save trajectory" << std::endl;
    }


    auto map_points = slam->getMapPoints();
    std::cout << "Map contains " << map_points.size() << " 3D points" << std::endl;


    std::cout << "\nShutting down SLAM system..." << std::endl;
    slam->release();

    std::cout << "Done!" << std::endl;
    return 0;
}
