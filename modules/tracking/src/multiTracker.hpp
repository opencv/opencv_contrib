#ifndef OPENCV_MULTITRACKER
#define OPENCV_MULTITRACKER

#include "tldTracker.hpp"
#include "tldUtils.hpp"
#include <math.h>

namespace cv
{
	void detect_all(const Mat& img, const Mat& imgBlurred, std::vector<Rect2d>& res, std::vector < std::vector < tld::TLDDetector::LabeledPatch >> &patches,
		std::vector<bool>& detect_flgs,	std::vector<Ptr<Tracker>>& trackers);
	void ocl_detect_all(const Mat& img, const Mat& imgBlurred, std::vector<Rect2d>& res, std::vector < std::vector < tld::TLDDetector::LabeledPatch >> &patches,
		std::vector<bool>& detect_flgs, std::vector<Ptr<Tracker>>& trackers);
	std::vector <Rect2d> debugStack[10];
}
#endif