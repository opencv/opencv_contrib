// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_TSDF_FUNCTIONS_H__
#define __OPENCV_TSDF_FUNCTIONS_H__

#include <opencv2/rgbd/volume.hpp>
#include "tsdf.hpp"

namespace cv
{
namespace kinfu
{

inline v_float32x4 tsdfToFloat_INTR(const v_int32x4& num)
{
    v_float32x4 num128 = v_setall_f32(-1.f / 128.f);
    return v_cvt_f32(num) * num128;
}

inline TsdfType floatToTsdf(float num)
{
    //CV_Assert(-1 < num <= 1);
    int8_t res = int8_t(num * (-128.f));
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

inline float tsdfToFloat(TsdfType num)
{
    return float(num) * (-1.f / 128.f);
}

cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics);
depthType bilinearDepth(const Depth& m, cv::Point2f pt);

void integrateVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::kinfu::Intr& intrinsics, InputArray _pixNorms, InputArray _volume);


struct Volume_NODE
{
    Vec3i idx = nan3;
    int row   = -1;
    int nextVolumeRow = -1;
    bool isActive = true;
    int lastVisibleIndex = -1;
    ocl::KernelArg pose;
};

size_t calc_hash(Vec3i x);

class VolumesTable
{
public:
    int hash_divisor = 32768;
    int list_size    = 4;
    int bufferNums  = 1;

    cv::Mat volumes;
    std::vector<Vec3i> indexes;

    VolumesTable();
    ~VolumesTable() {};


    void update(Vec3i indx);
    void update(Vec3i indx, int row);
    void update(Vec3i indx, bool isActive, int lastVisibleIndex, ocl::KernelArg pose);
    void expand();
    int getNextVolume(int hash, int& num, int i, int start);
    int find_Volume(Vec3i indx) const;
    bool isExist(Vec3i indx);
};



}  // namespace kinfu
}  // namespace cv
#endif
