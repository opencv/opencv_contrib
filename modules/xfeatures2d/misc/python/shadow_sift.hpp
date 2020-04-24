// Compatibility
// SIFT is moved to the main repository

namespace cv {
namespace xfeatures2d {

/** Use cv.SIFT_create() instead */
CV_WRAP static inline
Ptr<cv::SIFT> SIFT_create(int nfeatures = 0, int nOctaveLayers = 3,
        double contrastThreshold = 0.04, double edgeThreshold = 10,
        double sigma = 1.6)
{
    CV_LOG_ONCE_WARNING(NULL, "DEPRECATED: cv.xfeatures2d.SIFT_create() is deprecated due SIFT tranfer to the main repository. "
                              "https://github.com/opencv/opencv/issues/16736"
    );

    return SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
}

}}  // namespace
