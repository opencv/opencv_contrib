#ifndef _SAMPLES_UTILITY_HPP_
#define _SAMPLES_UTILITY_HPP_

#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>

inline cv::Ptr<cv::Tracker> createTrackerByName(const std::string& name)
{
    using namespace cv;

    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerTLD::create());
    else if (name == "BOOSTING")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerBoosting::create());
    else if (name == "MEDIAN_FLOW")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerMedianFlow::create());
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else if (name == "MOSSE")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerMOSSE::create());
    else if (name == "CSRT")
        tracker = cv::TrackerCSRT::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

inline cv::Ptr<cv::legacy::Tracker> createTrackerByName_legacy(const std::string& name)
{
    using namespace cv;

    cv::Ptr<cv::legacy::Tracker> tracker;

    if (name == "KCF")
        tracker = legacy::TrackerKCF::create();
    else if (name == "TLD")
        tracker = legacy::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = legacy::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = legacy::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = legacy::TrackerMIL::create();
    else if (name == "GOTURN")
        CV_Error(cv::Error::StsNotImplemented, "FIXIT: migration on new API is required");
    else if (name == "MOSSE")
        tracker = legacy::TrackerMOSSE::create();
    else if (name == "CSRT")
        tracker = legacy::TrackerCSRT::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

#endif
