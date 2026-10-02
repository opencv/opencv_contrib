#include "system.hpp"
#include "tracking_module.hpp"
#include "config.hpp"

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
        std::cerr << "usage: run_frontend_only_mh01 <config.yaml> <vocab.fbow> <image_dir> <output_dir>\n";
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

    auto cfg = std::make_shared<cv::slam::config>(config_file);
    cv::slam::system sys(cfg, vocab_file);
    sys.startup(false);
    sys.set_enable_backend(false, 10);
    sys.set_enable_loop_closure(false);
    sys.set_allow_initialization(true);
    sys.pause_tracker();
    sys.resume_tracker();

    auto tracker = sys.get_tracking_module_for_frontend_only();
    if (!tracker) {
        std::cerr << "tracking module not available\n";
        return 1;
    }

    for (const auto& [img_path, timestamp] : images) {
        cv::Mat img = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        auto frm = sys.create_monocular_frame(img, timestamp);
        tracker->feed_frame(std::move(frm));
    }

    sys.save_frame_trajectory(output_dir + "/trajectory.txt", "TUM");
    sys.request_terminate();
    sys.shutdown();
    return 0;
}
