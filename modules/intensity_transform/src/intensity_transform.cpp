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
    double maxVal;
    minMaxLoc(input, NULL, &maxVal, NULL, NULL);
    const double c = 255 / log(1 + maxVal);
    Mat add_one_64f;
    input.convertTo(add_one_64f, CV_64F, 1, 1.0f);
    Mat log_64f;
    cv::log(add_one_64f, log_64f);
    log_64f.convertTo(output, CV_8UC3, c, 0.0f);
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