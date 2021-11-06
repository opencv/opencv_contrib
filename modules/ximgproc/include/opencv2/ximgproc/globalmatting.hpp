// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef __OPENCV_XIMGPROC_GLOBAL_MATTING_HPP__
#define __OPENCV_XIMGPROC_GLOBAL_MATTING_HPP__

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/edge_filter.hpp>


namespace cv { namespace ximgproc {

class CV_EXPORTS GlobalMatting
{
public:
    GlobalMatting();
    virtual ~GlobalMatting();

    virtual void globalMatting(InputArray image, InputArray trimap, OutputArray foreground, OutputArray alpha, OutputArray conf = noArray()) = 0;

    virtual void getMat(InputArray image, InputArray trimap, OutputArray foreground, OutputArray alpha, int niter=9) = 0;
};

CV_EXPORTS Ptr<GlobalMatting> createGlobalMatting();

}}  // namespace

#endif  // __OPENCV_XIMGPROC_GLOBAL_MATTING_HPP__
