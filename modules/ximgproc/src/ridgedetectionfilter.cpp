// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ximgproc/ridgefilter.hpp"

namespace cv { namespace ximgproc {

class RidgeDetectionFilterImpl : public RidgeDetectionFilter
{
public:
    ElemDepth _ddepth;
    int _dx, _dy, _ksize;
    double _scale, _delta;
    int _borderType;
    ElemType _out_dtype;
    RidgeDetectionFilterImpl(ElemDepth ddepth = CV_32F, int dx = 1, int dy = 1, int ksize = 3, ElemType out_dtype = CV_8UC1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    {
        CV_Assert((ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7));
        CV_Assert((ddepth == CV_32F || ddepth == CV_64F));
        _ddepth = ddepth;
        _dx = dx;
        _dy = dy;
        _ksize = ksize;
        _scale = scale;
        _delta = delta;
        _borderType = borderType;
        _out_dtype = out_dtype;
    }
    virtual void getRidgeFilteredImage(InputArray _img, OutputArray out) CV_OVERRIDE;
};

void RidgeDetectionFilterImpl::getRidgeFilteredImage(InputArray _img, OutputArray out)
{
    Mat img = _img.getMat();
    CV_Assert(img.channels() == 1 || img.channels() == 3);

    if(img.channels() == 3)
        cvtColor(img, img, COLOR_BGR2GRAY);

    Mat sbx, sby;
    Sobel(img, sbx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(img, sby, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Mat sbxx, sbyy, sbxy;
    Sobel(sbx, sbxx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(sby, sbyy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);
    Sobel(sbx, sbxy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Mat sb2xx, sb2yy, sb2xy;
    multiply(sbxx, sbxx, sb2xx);
    multiply(sbyy, sbyy, sb2yy);
    multiply(sbxy, sbxy, sb2xy);

    Mat sbxxyy;
    multiply(sbxx, sbyy, sbxxyy);

    Mat rootex;
    rootex = (sb2xx +  (sb2xy + sb2xy + sb2xy + sb2xy)  - (sbxxyy + sbxxyy) + sb2yy );
    Mat root;
    sqrt(rootex, root);
    Mat ridgexp;
    ridgexp = ( (sbxx + sbyy) + root );
    ridgexp.convertTo(out, CV_MAT_DEPTH(_out_dtype), 0.5);
}

Ptr<RidgeDetectionFilter> RidgeDetectionFilter::create(ElemDepth ddepth, int dx, int dy, int ksize, ElemType out_dtype, double scale, double delta, int borderType)
{
    return makePtr<RidgeDetectionFilterImpl>(ddepth, dx, dy, ksize, out_dtype, scale, delta, borderType);
}

}} // namespace
