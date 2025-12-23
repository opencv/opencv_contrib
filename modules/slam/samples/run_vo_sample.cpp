// Minimal wrapper to exercise cv::vo::VisualOdometry on a folder of images (e.g., EuRoC MH01).
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/slam.hpp>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if(argc < 2){
        std::cout << "Usage: " << argv[0] << " [image_dir] [scale_m=0.02]\n"
                  << "Example (EuRoC): " << argv[0] << " /datasets/EuRoC/MH01/mav0/cam0/data 0.02" << std::endl;
        return 0;
    }

    std::string imgDir = argv[1];
    double scale_m = (argc >= 3) ? std::atof(argv[2]) : 0.02;

    cv::Ptr<cv::Feature2D> feature = cv::ORB::create(2000);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    cv::vo::VisualOdometry vo(feature, matcher);
    cv::vo::VisualOdometryOptions options;
    // Configure options if desired, e.g. disable backend or tweak thresholds
    // options.enableBackend = false; // purely front-end VO

    std::cout << "Running OpenCV VisualOdometry on " << imgDir << std::endl;
    int ret = vo.run(imgDir, scale_m, options);
    std::cout << "VisualOdometry finished with code " << ret << std::endl;
    return ret;
}
