#ifndef __OPENCV_XIMGPROC_LETTERBOXRESIZE_HPP__
#define __OPENCV_XIMGPROC_LETTERBOXRESIZE_HPP__

#include <opencv2/core.hpp>

namespace cv
{
  namespace ximgproc
  {

    void letterboxResize(InputArray _src, OutputArray _dst, Size dsize, int interpolation, int paddingMethod, int blueChannelValue, int greenChannelValue, int redChannelValue);

  }
}

#endif
