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
    int32_t dummy = 0;
    int32_t dummy2 = 0;
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
    cv::Mat volumes;

    VolumesTable();
    const VolumesTable& operator=(const VolumesTable&);
    ~VolumesTable() {};

    bool insert(Vec3i idx, int row);
    int findRow(Vec3i idx) const;

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


//TODO: hash set, not hash map
class ToyHashMap
{
public:
    static const int hashDivisor = 32768;
    static const int startCapacity = 1024; // 32768*4;

    std::vector<int> hashes;
    // 0-3 for key, 4th for internal use
    // don't keep keep value
    std::vector<Vec4i> data;
    int capacity;
    int last;

    ToyHashMap()
    {
        hashes.resize(hashDivisor);
        for (int i = 0; i < hashDivisor; i++)
            hashes[i] = -1;
        capacity = startCapacity;

        data.resize(capacity);
        for (int i = 0; i < capacity; i++)
            data[i] = { 0, 0, 0, -1 };

        last = 0;
    }

    ~ToyHashMap() { }

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

    // should work on existing elements too
    int insert(Vec3i idx)
    {
        if (last < capacity)
        {
            int hash = int(calc_hash(idx) % hashDivisor);
            int place = hashes[hash];
            if (place >= 0)
            {
                int oldPlace = place;
                while (place >= 0)
                {
                    if (data[place][0] == idx[0] &&
                        data[place][1] == idx[1] &&
                        data[place][2] == idx[2])
                        return 2;
                    else
                    {
                        oldPlace = place;
                        place = data[place][3];
                        //std::cout << "place=" << place << std::endl;
                    }
                }

                // found, create here
                data[oldPlace][3] = last;
            }
            else
            {
                // insert at last
                hashes[hash] = last;
            }

            data[last][0] = idx[0];
            data[last][1] = idx[1];
            data[last][2] = idx[2];
            data[last][3] = -1;
            last++;

            return 1;
        }
        else
            return 0;
    }

    int find(Vec3i idx) const
    {
        int hash = int(calc_hash(idx) % hashDivisor);
        int place = hashes[hash];
        // search a place
        while (place >= 0)
        {
            if (data[place][0] == idx[0] &&
                data[place][1] == idx[1] &&
                data[place][2] == idx[2])
                break;
            else
            {
                place = data[place][3];
            }
        }

        return place;
    }
};


}  // namespace kinfu
}  // namespace cv
#endif
