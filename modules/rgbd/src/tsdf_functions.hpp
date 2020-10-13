// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_TSDF_FUNCTIONS_H__
#define __OPENCV_TSDF_FUNCTIONS_H__

#include "tsdf.hpp"

namespace cv
{
namespace kinfu
{
/*
typedef int8_t TsdfType;
typedef uchar WeightType;

struct TsdfVoxel
{
    TsdfType tsdf;
    WeightType weight;
};

typedef Vec<uchar, sizeof(TsdfVoxel)> VecTsdfVoxel;
*/

v_float32x4 tsdfToFloat_INTR(const v_int32x4& num);
TsdfType floatToTsdf(float num);
float tsdfToFloat(TsdfType num);

cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics);
depthType bilinearDepth(const Depth& m, cv::Point2f pt);

}  // namespace kinfu
}  // namespace cv
#endif
