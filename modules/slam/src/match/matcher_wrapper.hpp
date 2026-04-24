#ifndef CV_SLAM_MATCHER_WRAPPER_HPP
#define CV_SLAM_MATCHER_WRAPPER_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

namespace cv::slam::match {

/**
 * @brief Matcher type enumeration
 */
enum class MatcherType {
    BF,         // Brute-Force
    FLANN,      // FLANN-based
    LightGlue,  
    Custom
};

/**
 * @brief Base class for matcher wrappers
 * 
 * Provides a unified descriptor matching interface with pluggable matchers.
 */
class MatcherWrapper {
public:
    virtual ~MatcherWrapper() = default;
    
    // ========== Core interface ==========
    
    /**
     * @brief 2D-2D matching
     * @param descriptors1 Descriptor set 1
     * @param descriptors2 Descriptor set 2
     * @param matches Output match results
     */
    virtual void match(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& matches) = 0;
    
    /**
     * @brief KNN matching
     * @param descriptors1 Descriptor set 1
     * @param descriptors2 Descriptor set 2
     * @param matches Output match results
     * @param k KNN parameter
     */
    virtual void knnMatch(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<std::vector<cv::DMatch>>& matches,
        int k = 2) = 0;
    
    // ========== Query information ==========
    
    virtual MatcherType getType() const = 0;
    virtual int getNormType() const = 0;
    
    // ========== Factory methods ==========
    
    /**
     * @brief Automatically choose a suitable matcher by feature type
     * @param feature_type Feature type ("ORB", "ALIKED", "SuperPoint")
     * @return Matcher pointer
     */
    static std::shared_ptr<MatcherWrapper> create(
        const std::string& feature_type);
    
    /**
        * @brief Create a BFMatcher
        * @param normType Distance type (NORM_HAMMING, NORM_L2)
        * @return BFMatcher wrapper pointer
     */
    static std::shared_ptr<MatcherWrapper> createBF(int normType);
};

} // namespace cv::slam::match

#endif // CV_SLAM_MATCHER_WRAPPER_HPP
