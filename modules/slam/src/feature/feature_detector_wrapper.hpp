#ifndef CV_SLAM_FEATURE_DETECTOR_WRAPPER_HPP
#define CV_SLAM_FEATURE_DETECTOR_WRAPPER_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

namespace cv::slam::feature {

/**
 * @brief Feature type enumeration
 */
enum class FeatureType {
    ORB,
    ALIKED,
    SuperPoint,
    SIFT,
    Custom
};

/**
 * @brief Base class for feature detector wrappers
 * 
 * Provides a unified feature extraction interface with pluggable detectors.
 */
class FeatureDetectorWrapper {
public:
    virtual ~FeatureDetectorWrapper() = default;
    
    // ========== Core interface ==========
    
    /**
     * @brief Detect keypoints and compute descriptors
     * @param image Input image
     * @param mask Mask
     * @param keypoints Output keypoints
     * @param descriptors Output descriptors
     */
    virtual void detectAndCompute(
        const cv::Mat& image,
        const cv::Mat& mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat& descriptors) = 0;
    
    // ========== Query information ==========
    
    virtual FeatureType getType() const = 0;
    virtual int descriptorSize() const = 0;     
    virtual int descriptorType() const = 0;     // CV_8U for ORB, CV_32F for ALIKED
    virtual int defaultNormType() const = 0;    // NORM_HAMMING or NORM_L2
    
    // ========== Factory methods ==========
    
    /**
     * @brief Create a suitable wrapper by type
     * @param type Feature type string ("ORB", "ALIKED", "SuperPoint")
     * @param nfeatures Maximum number of keypoints
     * @return Wrapper pointer
     */
    static std::shared_ptr<FeatureDetectorWrapper> create(
        const std::string& type,
        int nfeatures = 1000);
    
    /**
     * @brief Create an ORB feature detector
     * @param nfeatures Maximum number of keypoints
     * @param scaleFactor Pyramid scale factor
     * @param nlevels Number of pyramid levels
     * @param iniFastThreshold Initial FAST threshold
     * @param minFastThreshold Minimum FAST threshold
     * @return ORB wrapper pointer
     */
    static std::shared_ptr<FeatureDetectorWrapper> createORB(
        int nfeatures = 1000,
        float scaleFactor = 1.2f,
        int nlevels = 8,
        int iniFastThreshold = 20,
        int minFastThreshold = 7);
};

} // namespace cv::slam::feature

#endif // CV_SLAM_FEATURE_DETECTOR_WRAPPER_HPP
