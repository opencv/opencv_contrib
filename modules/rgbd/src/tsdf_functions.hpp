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

const int NAN_ELEMENT = -2147483647;

struct Volume_NODE
{
    Vec4i idx = Vec4i(NAN_ELEMENT);
    int32_t row   = -1;
    int32_t nextVolumeRow = -1;
    int32_t isActive = 0;
    int32_t lastVisibleIndex = -1;
};

const int _hash_divisor = 32768;
const int _list_size = 4;

class VolumesTable
{
public:
    const int hash_divisor = _hash_divisor;
    const int list_size    = _list_size;
    const int32_t free_row = -1;
    const int32_t free_isActive = 0;

    const cv::Vec4i nan4 = cv::Vec4i(NAN_ELEMENT);

    int bufferNums;
    std::vector<Vec3i> indexes;
    std::vector<Vec4i> indexesGPU;
    cv::Mat volumes;

    VolumesTable();
    const VolumesTable& operator=(const VolumesTable&);
    ~VolumesTable() {};

    Volume_NODE* insert(Vec3i idx, int row, bool isActive = false, int lastVisibleIndex = -1);
    const Volume_NODE* find(Vec3i idx) const;
    Volume_NODE* find(Vec3i idx);

    inline int getPos(Vec3i idx, int bufferNum) const
    {
        int hash = int(calc_hash(idx) % hash_divisor);
        return (bufferNum * hash_divisor + hash) * list_size;
    }

    inline size_t calc_hash(Vec3i x) const
    {
        uint32_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (int i = 0; i < 3; i++)
        {
            seed ^= x[i] + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};



}  // namespace kinfu
}  // namespace cv
#endif
