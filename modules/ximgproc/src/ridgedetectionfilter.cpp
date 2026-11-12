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
    int _ddepth, _dx, _dy, _ksize;
    double _scale, _delta;
    int _borderType;
    int _out_dtype;
    RidgeDetectionFilterImpl(int ddepth=CV_32FC1, int dx=1, int dy=1, int ksize = 3, int out_dtype=CV_8UC1, double scale = 1, double delta = 0, int borderType = BORDER_DEFAULT)
    {
        CV_Assert((ksize == 1 || ksize == 3 || ksize == 5 || ksize == 7));
        CV_Assert((ddepth == CV_32FC1 || ddepth == CV_64FC1));
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

namespace
{
template <typename T>
struct RidgeEigenRowInvoker : public ParallelLoopBody
{
    RidgeEigenRowInvoker(const Mat& _sbxx, const Mat& _sbyy, Mat& _sbxy_out)
        : sbxx(_sbxx), sbyy(_sbyy), sbxy_out(_sbxy_out) {}

    void operator()(const Range& range) const CV_OVERRIDE
    {
        const int cols = sbxx.cols;
        for (int r = range.start; r < range.end; ++r)
        {
            const T* xxRow = sbxx.ptr<T>(r);
            const T* yyRow = sbyy.ptr<T>(r);
            T* xyRow = sbxy_out.ptr<T>(r);
            for (int c = 0; c < cols; ++c)
            {
                const T xx = xxRow[c];
                const T yy = yyRow[c];
                const T xy = xyRow[c];
                const T sb2xx = xx * xx;
                const T sb2yy = yy * yy;
                const T sb2xy = xy * xy;
                const T sbxxyy = xx * yy;
                const T four_sb2xy = ((sb2xy + sb2xy) + sb2xy) + sb2xy;
                const T two_sbxxyy = sbxxyy + sbxxyy;
                const T rootex = ((sb2xx + four_sb2xy) - two_sbxxyy) + sb2yy;
                xyRow[c] = (xx + yy) + std::sqrt(rootex);
            }
        }
    }

    const Mat& sbxx;
    const Mat& sbyy;
    Mat& sbxy_out;

private:
    RidgeEigenRowInvoker& operator=(const RidgeEigenRowInvoker&);
};
} // anonymous namespace

void RidgeDetectionFilterImpl::getRidgeFilteredImage(InputArray _img, OutputArray out)
{
    Mat img = _img.getMat();
    CV_Assert(img.channels() == 1 || img.channels() == 3);

    if(img.channels() == 3)
        cvtColor(img, img, COLOR_BGR2GRAY);

    Mat grad, sbxx, sbxy, sbyy;
    Sobel(img, grad, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(grad, sbxx, _ddepth, _dx, 0, _ksize, _scale, _delta, _borderType);
    Sobel(grad, sbxy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);

    Sobel(img, grad, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);
    img.release();
    Sobel(grad, sbyy, _ddepth, 0, _dy, _ksize, _scale, _delta, _borderType);
    grad.release();

    const int nstripes = std::max(1, getNumThreads()) * 4;
    if (_ddepth == CV_32FC1)
        parallel_for_(Range(0, sbxx.rows), RidgeEigenRowInvoker<float>(sbxx, sbyy, sbxy), nstripes);
    else
        parallel_for_(Range(0, sbxx.rows), RidgeEigenRowInvoker<double>(sbxx, sbyy, sbxy), nstripes);

    sbxx.release();
    sbyy.release();
    sbxy.convertTo(out, _out_dtype, 0.5);
}

Ptr<RidgeDetectionFilter> RidgeDetectionFilter::create(int ddepth, int dx, int dy, int ksize, int out_dtype, double scale, double delta, int borderType)
{
    return makePtr<RidgeDetectionFilterImpl>(ddepth, dx, dy, ksize, out_dtype, scale, delta, borderType);
}

}} // namespace
