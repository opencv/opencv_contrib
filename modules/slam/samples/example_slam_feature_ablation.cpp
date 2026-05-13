#include <opencv2/slam.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

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

static cv::Ptr<cv::Feature2D> make_detector(const std::string& name) {
    if (name == "ORB") return cv::ORB::create(1000, 1.2f, 8);
    if (name == "SIFT") return cv::SIFT::create(1000);
    if (name == "AKAZE") return cv::AKAZE::create();
    if (name == "BRISK") return cv::BRISK::create();
    throw std::runtime_error("unknown detector: " + name);
}

static cv::Ptr<cv::DescriptorMatcher> make_matcher(const std::string& name) {
    if (name == "BF_HAMMING") return cv::BFMatcher::create(cv::NORM_HAMMING);
    if (name == "BF_L2") return cv::BFMatcher::create(cv::NORM_L2);
    if (name == "FLANN_LSH") {
        auto index_params = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
        auto search_params = cv::makePtr<cv::flann::SearchParams>(50);
        return cv::makePtr<cv::FlannBasedMatcher>(index_params, search_params);
    }
    throw std::runtime_error("unknown matcher: " + name);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "usage: run_feature_ablation_mh01 <config.yaml> <vocab.fbow> <image_dir> <output_dir> <detector> <matcher>\n";
        return 1;
    }

    const std::string config_file = argv[1];
    const std::string vocab_file = argv[2];
    const std::string image_dir = argv[3];
    const std::string output_dir = argv[4];
    const std::string detector_name = argv[5];
    const std::string matcher_name = argv[6];

    auto images = load_images(image_dir);
    if (images.empty()) {
        std::cerr << "no images\n";
        return 1;
    }

    auto slam = cv::vo::VisualOdometry::create(config_file, vocab_file);
    slam->setFeatureDetector(make_detector(detector_name));
    slam->setMatcher(make_matcher(matcher_name));
    slam->setBackendEnabled(true, 10);
    slam->setLoopClosureEnabled(false);
    slam->setMode(cv::vo::SLAMMode::SLAM);

    for (const auto& item : images) {
        cv::Mat img = cv::imread(item.first, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        slam->processFrame(img, item.second);
    }

    slam->saveTrajectory(output_dir + "/trajectory.txt", "TUM");
    slam->release();
    return 0;
}
