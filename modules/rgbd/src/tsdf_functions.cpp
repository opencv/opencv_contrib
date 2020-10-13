// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv {

namespace kinfu {

v_float32x4 tsdfToFloat_INTR(const v_int32x4& num)
{
    v_float32x4 num128 = v_setall_f32(-1.f / 128.f);
    return v_cvt_f32(num) * num128;
}

TsdfType floatToTsdf(float num)
{
    //CV_Assert(-1 < num <= 1);
    int8_t res = int8_t(num * (-128.f));
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

float tsdfToFloat(TsdfType num)
{
    return float(num) * (-1.f / 128.f);
}


cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics)
{
    int height = depth.rows;
    int widht = depth.cols;
    Point2f fl(intrinsics.fx, intrinsics.fy);
    Point2f pp(intrinsics.cx, intrinsics.cy);
    Mat pixNorm(height, widht, CV_32F);
    std::vector<float> x(widht);
    std::vector<float> y(height);
    for (int i = 0; i < widht; i++)
        x[i] = (i - pp.x) / fl.x;
    for (int i = 0; i < height; i++)
        y[i] = (i - pp.y) / fl.y;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < widht; j++)
        {
            pixNorm.at<float>(i, j) = sqrtf(x[j] * x[j] + y[i] * y[i] + 1.0f);
        }
    }
    return pixNorm;
}

const bool fixMissingData = false;
depthType bilinearDepth(const Depth& m, cv::Point2f pt)
{
    const depthType defaultValue = qnan;
    if (pt.x < 0 || pt.x >= m.cols - 1 ||
        pt.y < 0 || pt.y >= m.rows - 1)
        return defaultValue;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

    const depthType* row0 = m[yi + 0];
    const depthType* row1 = m[yi + 1];

    depthType v00 = row0[xi + 0];
    depthType v01 = row0[xi + 1];
    depthType v10 = row1[xi + 0];
    depthType v11 = row1[xi + 1];

    // assume correct depth is positive
    bool b00 = v00 > 0;
    bool b01 = v01 > 0;
    bool b10 = v10 > 0;
    bool b11 = v11 > 0;

    if (!fixMissingData)
    {
        if (!(b00 && b01 && b10 && b11))
            return defaultValue;
        else
        {
            float tx = pt.x - xi, ty = pt.y - yi;
            depthType v0 = v00 + tx * (v01 - v00);
            depthType v1 = v10 + tx * (v11 - v10);
            return v0 + ty * (v1 - v0);
        }
    }
    else
    {
        int nz = b00 + b01 + b10 + b11;
        if (nz == 0)
        {
            return defaultValue;
        }
        if (nz == 1)
        {
            if (b00) return v00;
            if (b01) return v01;
            if (b10) return v10;
            if (b11) return v11;
        }
        if (nz == 2)
        {
            if (b00 && b10) v01 = v00, v11 = v10;
            if (b01 && b11) v00 = v01, v10 = v11;
            if (b00 && b01) v10 = v00, v11 = v01;
            if (b10 && b11) v00 = v10, v01 = v11;
            if (b00 && b11) v01 = v10 = (v00 + v11) * 0.5f;
            if (b01 && b10) v00 = v11 = (v01 + v10) * 0.5f;
        }
        if (nz == 3)
        {
            if (!b00) v00 = v10 + v01 - v11;
            if (!b01) v01 = v00 + v11 - v10;
            if (!b10) v10 = v00 + v11 - v01;
            if (!b11) v11 = v01 + v10 - v00;
        }

        float tx = pt.x - xi, ty = pt.y - yi;
        depthType v0 = v00 + tx * (v01 - v00);
        depthType v1 = v10 + tx * (v11 - v10);
        return v0 + ty * (v1 - v0);
    }
}


} // namespace kinfu
} // namespace cv
