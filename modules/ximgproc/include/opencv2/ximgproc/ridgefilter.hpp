#ifndef __OPENCV_XIMGPROC_RIDGEFILTER_HPP__
#define __OPENCV_XIMGPROC_RIDGEFILTER_HPP__

#include <opencv2/core.hpp>
//changed include statement to fix compile error
namespace cv {
    namespace ximgproc {
        namespace ridgefilter {
            class CV_EXPORTS_W RidgeFilter : public Algorithm {
            };
            class CV_EXPORTS_W RidgeDetectionFilter : public RidgeFilter {
            };
            CV_EXPORTS_W Ptr<RidgeDetectionFilter> createRidgeDetectionFilter();
        }
    }
}
#endif