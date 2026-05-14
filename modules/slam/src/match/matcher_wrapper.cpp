#include "match/matcher_wrapper.hpp"

#include <stdexcept>
#include <iostream>

namespace cv::slam::match {

// ============================================================================

// ============================================================================

class BFMatcherWrapper : public MatcherWrapper {
public:

    BFMatcherWrapper(const std::string& feature_type)
    {
        if (feature_type == "ORB") {
            normType_ = cv::NORM_HAMMING;
        }
        else {

            normType_ = cv::NORM_L2;
        }
        matcher_ = cv::BFMatcher::create(normType_);
    }


    BFMatcherWrapper(int normType)
        : normType_(normType)
    {
        matcher_ = cv::BFMatcher::create(normType_);
    }

    void match(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<cv::DMatch>& matches) override
    {
        matcher_->match(descriptors1, descriptors2, matches);
    }

    void knnMatch(
        const cv::Mat& descriptors1,
        const cv::Mat& descriptors2,
        std::vector<std::vector<cv::DMatch>>& matches,
        int k) override
    {
        matcher_->knnMatch(descriptors1, descriptors2, matches, k);
    }

    MatcherType getType() const override { return MatcherType::BF; }
    int getNormType() const override { return normType_; }

private:
    cv::Ptr<cv::BFMatcher> matcher_;
    int normType_;
};

// ============================================================================

// ============================================================================

std::shared_ptr<MatcherWrapper> MatcherWrapper::create(
    const std::string& feature_type)
{
    return std::make_shared<BFMatcherWrapper>(feature_type);
}

std::shared_ptr<MatcherWrapper> MatcherWrapper::createBF(int normType)
{
    return std::make_shared<BFMatcherWrapper>(normType);
}

} // namespace cv::slam::match
