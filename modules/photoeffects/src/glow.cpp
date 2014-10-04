#include "precomp.hpp"
#include <opencv2/core/utility.hpp>

namespace cv { namespace photoeffects {

namespace
{
    const int COUNT_CHANNEL = 3;

    class OverlayInvoker: public cv::ParallelLoopBody
    {
    public:
        OverlayInvoker(const Mat &foreground, const Mat &background, Mat &result)
            : foreImg(foreground),
              backImg(background),
              resImg(result),
              width(foreground.cols)
        {
            resImg.create(foreImg.size(), foreImg.type());
        }

        void operator()(const Range& rows) const
        {
            Mat foreStripe = foreImg.rowRange(rows.start, rows.end);
            Mat backStripe = backImg.rowRange(rows.start, rows.end);
            Mat resStripe = resImg.rowRange(rows.start, rows.end);
            int stripeHeight = foreStripe.rows;
            for (int i = 0; i < stripeHeight; i++)
            {
                uchar* foreRow = foreStripe.row(i).data;
                uchar* backRow = backStripe.row(i).data;
                uchar* resRow = resStripe.row(i).data;
                for (int j  = 0; j < width * COUNT_CHANNEL; j++)
                {
                    int intensResult = (2 * foreRow[j] * backRow[j]) / 255;

                    if (backRow[j] > 127)
                    {
                        intensResult = cv::min(-intensResult - 255 + 2 * (foreRow[j] + backRow[j]), 255);
                    }
                    resRow[j] = intensResult;
                }
            }
        }
    private:
        const Mat &foreImg;
        const Mat &backImg;
        Mat &resImg;
        int width;
        //OverlayInvoker& operator=(const OverlayInvoker&);
    };
}

int glow(InputArray src, OutputArray dst, int radius, float intensity)
{
    Mat srcImg = src.getMat();

    CV_Assert(srcImg.channels() == COUNT_CHANNEL);
    CV_Assert(radius >= 0);
    CV_Assert(intensity >= 0.0f && intensity <= 1.0f);

    int srcImgType = srcImg.type();
    if (srcImgType != CV_8UC3)
    {
        srcImg.convertTo(srcImg, CV_8UC3);
    }

    Mat blurImg;
    Size size(radius, radius);

    boxFilter(srcImg, blurImg, -1, size);

    Mat overlayImg;
    parallel_for_(Range(0, blurImg.rows), OverlayInvoker(blurImg, srcImg, overlayImg));

    uchar coeff = static_cast<uchar>(intensity * 255.0);
    Mat dstImg = (coeff * overlayImg + (255 - coeff) * srcImg) / 255;

    dstImg.convertTo(dst, srcImgType);

    return 0;
}

}}