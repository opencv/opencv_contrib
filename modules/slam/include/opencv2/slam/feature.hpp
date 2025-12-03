#pragma once
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace cv {
namespace vo {

class FeatureExtractor {
public:
    explicit FeatureExtractor(int nfeatures = 2000);
    // detectAndCompute: detect keypoints and compute descriptors.
    // If previous-frame data (prevGray, prevKp) is provided, a flow-aware grid allocation
    // will be used (score = response * (1 + flow_lambda * normalized_flow)). Otherwise a
    // simpler ANMS selection is used. The prev arguments have defaults so this function
    // replaces the two previous overloads.
    void detectAndCompute(const Mat &image, std::vector<KeyPoint> &kps, Mat &desc,
                          const Mat &prevGray = Mat(), const std::vector<KeyPoint> &prevKp = std::vector<KeyPoint>(),
                          double flow_lambda = 5.0);
private:
    Ptr<ORB> orb_;
    int nfeatures_;
};

// Function to detect and compute features in an image
inline void detectAndComputeFeatures(const Mat &image,
                                     std::vector<KeyPoint> &keypoints,
                                     Mat &descriptors) {
    // Create ORB detector and descriptor
    auto orb = ORB::create();
    // Detect keypoints
    orb->detect(image, keypoints);
    // Compute descriptors
    orb->compute(image, keypoints, descriptors);
}

} // namespace vo
} // namespace cv