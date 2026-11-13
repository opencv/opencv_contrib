// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace std;

namespace cv {
namespace intensity_transform {

void logTransform(const Mat input, Mat& output)
{
    CV_Assert(!input.empty());
    double maxVal = 0.0;
    minMaxLoc(input.reshape(1), NULL, &maxVal, NULL, NULL);
    if (maxVal <= 0.0)
    {
        output.create(input.size(), CV_MAKETYPE(CV_8U, input.channels()));
        output.setTo(Scalar::all(0));
        return;
    }
    const double c = 255.0 / std::log(1.0 + maxVal);
    if (input.depth() == CV_8U)
    {
        Mat lut(1, 256, CV_8U);
        uchar* p = lut.ptr<uchar>();
        for (int i = 0; i < 256; ++i)
            p[i] = saturate_cast<uchar>(c * std::log(1.0 + i));
        LUT(input, lut, output);
        return;
    }
    Mat add_one_64f;
    input.convertTo(add_one_64f, CV_64F, 1, 1.0f);
    Mat log_64f;
    cv::log(add_one_64f, log_64f);
    log_64f.convertTo(output, CV_8U, c, 0.0f);
}

void gammaCorrection(const Mat input, Mat& output, const float gamma)
{
    std::array<uchar, 256> table;
    for (int i = 0; i < 256; i++)
    {
        table[i] = saturate_cast<uchar>(pow((i / 255.0), gamma) * 255.0);
    }

    LUT(input, table, output);
}

void autoscaling(const Mat input, Mat& output)
{
    double minVal, maxVal;
    minMaxLoc(input, &minVal, &maxVal, NULL, NULL);
    output = 255 * (input - minVal) / (maxVal - minVal);
}

void contrastStretching(const Mat input, Mat& output, const int r1, const int s1, const int r2, const int s2)
{
    std::array<uchar, 256> table;
    for (int i = 0; i < 256; i++)
    {
        if (i <= r1)
        {
            table[i] = saturate_cast<uchar>(((float)s1 / (float)r1) * i);
        }
        else if (r1 < i && i <= r2)
        {
            table[i] = saturate_cast<uchar>(((float)(s2 - s1)/(float)(r2 - r1)) * (i - r1) + s1);
        }
        else // (r2 < i)
        {
            table[i] = saturate_cast<uchar>(((float)(255 - s2)/(float)(255 - r2)) * (i - r2) + s2);
        }
    }

    LUT(input, table, output);
}

}} // cv::intensity_transform::