#include "precomp.hpp"
#include <opencv2/core/utility.hpp>

namespace cv { namespace photoeffects {

namespace
{
    const int COUNT_CHANNEL = 3;
    void overlay(InputArray foreground, InputArray background, OutputArray result)
    {
        Mat foreImg = foreground.getMat();
        Mat backImg = background.getMat();

        result.create(backImg.size(), backImg.type());
        Mat resultImg = result.getMat();

        int width = foreImg.cols;
        int height = foreImg.rows;
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                for(int k = 0; k < COUNT_CHANNEL; k++)
                {
                    uchar intensFore = foreImg.at<Vec3b>(i, j)[k];
                    uchar intensBack = backImg.at<Vec3b>(i, j)[k];
                    int intensResult = (2 * intensFore * intensBack) / 255;

                    if (intensBack > 127)
                    {
                        intensResult = cv::min(-intensResult - 255 + 2 * (intensFore + intensBack), 255);
                    }
                    resultImg.at<Vec3b>(i, j)[k] = intensResult;
                }
            }
        }
    }
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
    overlay(blurImg, srcImg, overlayImg);

    uchar coeff = static_cast<uchar>(intensity * 255.0);
    Mat dstImg = (coeff * overlayImg + (255 - coeff) * srcImg) / 255;

    dstImg.convertTo(dst, srcImgType);

    return 0;
}

}}