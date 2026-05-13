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
        std::cerr << "usage: run_frontend_mh01 <config.yaml> <vocab.fbow> <image_dir> <output_dir>\n";
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

    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);
    slam->setBackendEnabled(true, 10);
    slam->setLoopClosureEnabled(true);  // 启用回环检测
    slam->setMode(cv::vo::SLAMMode::SLAM);

    for (const auto& item : images) {
        const auto& img_path = item.first;
        const auto timestamp = item.second;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        slam->processFrame(img, timestamp);
    }

    slam->saveTrajectory(output_dir + "/trajectory.txt", "TUM");
    slam->release();
    return 0;
}
