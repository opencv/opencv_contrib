#include <opencv2/slam.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::pair<std::string, double>> load_images(const std::string& image_dir) {
    std::vector<std::pair<std::string, double>> images;
    std::ifstream csv(image_dir + "/../data.csv");
    std::string line;
    std::getline(csv, line);
    while (std::getline(csv, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string ts, name;
        if (!std::getline(ss, ts, ',')) continue;
        if (!std::getline(ss, name)) continue;
        while (!name.empty() && (name.back() == '\r' || name.back() == '\n' || name.back() == ' ' || name.back() == '\t')) {
            name.pop_back();
        }
        images.push_back({image_dir + "/" + name, std::stod(ts) / 1e9});
    }
    return images;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: " << argv[0] << " <config.yaml> <vocab.fbow> <image_dir> <output_dir>\n";
        return 1;
    }

    const std::string config_file = argv[1];
    const std::string vocab_file = argv[2];
    const std::string image_dir = argv[3];
    const std::string output_dir = argv[4];

    auto images = load_images(image_dir);
    if (images.empty()) {
        std::cerr << "no images\n";
        return 1;
    }

    std::cout << "Creating SLAM system...\n";
    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);

    std::cout << "Enabling backend with local BA...\n";
    slam->setBackendEnabled(true, 10);

    std::cout << "Enabling loop closure...\n";
    slam->setLoopClosureEnabled(true);

    std::cout << "Setting SLAM mode...\n";
    slam->setMode(cv::vo::SLAMMode::SLAM);

    std::cout << "Processing " << images.size() << " images...\n";
    int processed = 0;
    for (const auto& item : images) {
        const auto& img_path = item.first;
        const auto timestamp = item.second;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        auto pose = slam->processFrame(img, timestamp);
        processed++;

        if (processed % 100 == 0) {
            std::cout << "Processed " << processed << "/" << images.size() << " frames\n";
        }
    }

    std::cout << "Saving trajectory to: " << output_dir << "/trajectory.txt\n";
    slam->saveTrajectory(output_dir + "/trajectory.txt", "TUM");

    std::cout << "Saving map to: " << output_dir << "/map.msgpack\n";
    slam->saveMap(output_dir + "/map.msgpack");

    std::cout << "Releasing SLAM system...\n";
    slam->release();

    std::cout << "Done! Processed " << processed << " frames.\n";
    return 0;
}
