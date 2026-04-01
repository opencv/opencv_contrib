#include "feature/feature_detector_wrapper.hpp"

#include <stdexcept>
#include <iostream>

namespace cv::slam::feature {

// ============================================================================

// ============================================================================

class ORBDetectorWrapper : public FeatureDetectorWrapper {
public:
    ORBDetectorWrapper(int nfeatures,
                       float scaleFactor,
                       int nlevels,
                       int iniFastThreshold,
                       int minFastThreshold)
        : nfeatures_(nfeatures)
    {
        
        orb_ = cv::ORB::create(
            nfeatures,
            scaleFactor,
            nlevels,
            31,     // edgeThreshold
            0,      // firstLevel
            2,      // WTA_K
            cv::ORB::HARRIS_SCORE,
            31,     // patchSize
            iniFastThreshold
        );
    }
    
    void detectAndCompute(
        const cv::Mat& image,
        const cv::Mat& mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) override
    {
        orb_->detectAndCompute(image, mask, keypoints, descriptors);
    }
    
    FeatureType getType() const override { return FeatureType::ORB; }
    int descriptorSize() const override { return 32; }      // 256 bits = 32 bytes
    int descriptorType() const override { return CV_8U; }
    int defaultNormType() const override { return cv::NORM_HAMMING; }
    
private:
    cv::Ptr<cv::ORB> orb_;
    int nfeatures_;
};

// ============================================================================

// ============================================================================

std::shared_ptr<FeatureDetectorWrapper> FeatureDetectorWrapper::create(
    const std::string& type,
    int nfeatures)
{
    if (type == "ORB") {
        return createORB(nfeatures);
    }
    
    // else if (type == "ALIKED") {
    //     return createALIKED(model_path, nfeatures);
    // }
    else {
        throw std::runtime_error("Unsupported feature type: " + type);
    }
}

std::shared_ptr<FeatureDetectorWrapper> FeatureDetectorWrapper::createORB(
    int nfeatures,
    float scaleFactor,
    int nlevels,
    int iniFastThreshold,
    int minFastThreshold)
{
    return std::make_shared<ORBDetectorWrapper>(
        nfeatures, scaleFactor, nlevels, iniFastThreshold, minFastThreshold
    );
}

} // namespace cv::slam::feature
