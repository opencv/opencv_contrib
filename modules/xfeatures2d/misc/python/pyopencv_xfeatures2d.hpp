#ifdef HAVE_OPENCV_XFEATURES2D

#include "opencv2/xfeatures2d.hpp"
using cv::xfeatures2d::DAISY;

typedef DAISY::NormalizationType DAISY_NormalizationType;

CV_PY_FROM_ENUM(DAISY::NormalizationType);
CV_PY_TO_ENUM(DAISY::NormalizationType);
#endif
