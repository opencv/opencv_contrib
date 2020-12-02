// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "precomp.hpp"
#include "hash_tsdf.hpp"

#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "kinfu_frame.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/utils/trace.hpp"
#include "utils.hpp"
#include "opencl_kernels_rgbd.hpp"

#define USE_INTERPOLATION_IN_GETNORMAL 1
#define VOLUMES_SIZE 1024

namespace cv
{
namespace kinfu
{

HashTSDFVolume::HashTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
    float _truncDist, int _maxWeight, float _truncateThreshold,
    int _volumeUnitRes, bool _zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
    maxWeight(_maxWeight),
    truncateThreshold(_truncateThreshold),
    volumeUnitResolution(_volumeUnitRes),
    volumeUnitSize(voxelSize* volumeUnitResolution),
    zFirstMemOrder(_zFirstMemOrder)
{
    truncDist = std::max(_truncDist, 4.0f * voxelSize);
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist,
                                     int _maxWeight, float _truncateThreshold, int _volumeUnitRes, bool _zFirstMemOrder)
    :HashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold, _volumeUnitRes,
           _zFirstMemOrder)
{
    int xdim, ydim, zdim;
    if (zFirstMemOrder)
    {
        xdim = volumeUnitResolution * volumeUnitResolution;
        ydim = volumeUnitResolution;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volumeUnitResolution;
        zdim = volumeUnitResolution * volumeUnitResolution;
    }
    volStrides = Vec4i(xdim, ydim, zdim);

    reset();
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(const VolumeParams& _params, bool _zFirstMemOrder)
    : HashTSDFVolume(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
           _params.depthTruncThreshold, _params.unitResolution, _zFirstMemOrder)
{
}
// zero volume, leave rest params the same
void HashTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();
    lastVolIndex = 0;
    volUnitsData = cv::Mat(VOLUMES_SIZE, volumeUnitResolution * volumeUnitResolution * volumeUnitResolution, rawType<TsdfVoxel>());
}

void HashTSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

    //! Compute volumes to be allocated
    const int depthStride = int(log2(volumeUnitResolution));
    const float invDepthFactor = 1.f / depthFactor;
    const Intr::Reprojector reproj(intrinsics.makeReprojector());
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Point3f truncPt(truncDist, truncDist, truncDist);
    VolumeUnitIndexSet newIndices;
    Mutex mutex;
    Range allocateRange(0, depth.rows);

    auto AllocateVolumeUnitsInvoker = [&](const Range& range) {
        VolumeUnitIndexSet localAccessVolUnits;
        for (int y = range.start; y < range.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = 0; x < depth.cols; x += depthStride)
            {
                depthType z = depthRow[x] * invDepthFactor;
                if (z <= 0 || z > this->truncateThreshold)
                    continue;
                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = this->volumeToVolumeUnitIdx(volPoint - truncPt);
                Vec3i upper_bound = this->volumeToVolumeUnitIdx(volPoint + truncPt);

                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= lower_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);
                            if (!localAccessVolUnits.count(tsdf_idx))
                            {
                                //! This volume unit will definitely be required for current integration
                                localAccessVolUnits.emplace(tsdf_idx);
                            }
                        }
            }
        }

        mutex.lock();
        for (const auto& tsdf_idx : localAccessVolUnits)
        {
            //! If the insert into the global set passes
            if (!this->volumeUnits.count(tsdf_idx))
            {
                // Volume allocation can be performed outside of the lock
                this->volumeUnits.emplace(tsdf_idx, VolumeUnit());
                newIndices.emplace(tsdf_idx);
            }
        }
        mutex.unlock();
    };
    parallel_for_(allocateRange, AllocateVolumeUnitsInvoker);

    //! Perform the allocation
    for (auto idx : newIndices)
    {
        VolumeUnit& vu = volumeUnits[idx];
        Matx44f subvolumePose = pose.translate(volumeUnitIdxToVolume(idx)).matrix;

        vu.pose = subvolumePose;
        vu.index = lastVolIndex; lastVolIndex++;
        if (lastVolIndex > VolumeIndex(volUnitsData.size().height))
        {
            volUnitsData.resize((lastVolIndex - 1) * 2);
        }
        volUnitsData.row(vu.index).forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int* /* position */)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        //! This volume unit will definitely be required for current integration
        vu.lastVisibleIndex = frameId;
        vu.isActive = true;
    }

    //! Get keys for all the allocated volume Units
    std::vector<Vec3i> totalVolUnits;
    for (const auto& keyvalue : volumeUnits)
    {
        totalVolUnits.push_back(keyvalue.first);
    }

    //! Mark volumes in the camera frustum as active
    Range inFrustumRange(0, (int)volumeUnits.size());
    parallel_for_(inFrustumRange, [&](const Range& range) {
        const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
        const Intr::Projector proj(intrinsics.makeProjector());

        for (int i = range.start; i < range.end; ++i)
        {
            Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            Point3f volumeUnitPos = volumeUnitIdxToVolume(it->first);
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;
            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
            {
                it->second.isActive = false;
                return;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols && cameraPoint.y < depth.rows)
            {
                assert(it != volumeUnits.end());
                it->second.lastVisibleIndex = frameId;
                it->second.isActive         = true;
            }
        }
        });

    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if ( !(frameParams==newParams) )
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, intrinsics);
    }

    //! Integrate the correct volumeUnits
    parallel_for_(Range(0, (int)totalVolUnits.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            Vec3i tsdf_idx = totalVolUnits[i];
            VolumeUnitIndexes::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            VolumeUnit& volumeUnit = it->second;
            if (volumeUnit.isActive)
            {
                //! The volume unit should already be added into the Volume from the allocator
                integrateVolumeUnit(truncDist, voxelSize, maxWeight, volumeUnit.pose,
                    Point3i(volumeUnitResolution, volumeUnitResolution, volumeUnitResolution), volStrides, depth,
                    depthFactor, cameraPose, intrinsics, pixNorms, volUnitsData.row(volumeUnit.index));

                //! Ensure all active volumeUnits are set to inactive for next integration
                volumeUnit.isActive = false;
            }
        }
        });
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVolumeUnitIdx(const cv::Point3f& p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize), cvFloor(p.y / volumeUnitSize),
                     cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeCPU::volumeUnitIdxToVolume(const cv::Vec3i& volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, volumeUnitIdx[1] * volumeUnitSize,
                       volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeCPU::voxelCoordToVolume(const cv::Vec3i& voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVoxelCoord(const cv::Point3f& point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv), cvFloor(point.y * voxelSizeInv),
                     cvFloor(point.z * voxelSizeInv));
}

inline TsdfVoxel HashTSDFVolumeCPU::_at(const cv::Vec3i& volumeIdx, VolumeIndex indx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    const TsdfVoxel* volData = volUnitsData.ptr<TsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] + volumeIdx[1] * volStrides[1] + volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

inline TsdfVoxel HashTSDFVolumeCPU::at(const cv::Vec3i& volumeIdx) const

{
    Vec3i volumeUnitIdx = Vec3i(cvFloor(volumeIdx[0] / volumeUnitResolution),
                                cvFloor(volumeIdx[1] / volumeUnitResolution),
                                cvFloor(volumeIdx[2] / volumeUnitResolution));

    VolumeUnitIndexes::const_iterator it = volumeUnits.find(volumeUnitIdx);

    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }

    cv::Vec3i volUnitLocalIdx = volumeIdx - cv::Vec3i(volumeUnitIdx[0] * volumeUnitResolution,
                                                      volumeUnitIdx[1] * volumeUnitResolution,
                                                      volumeUnitIdx[2] * volumeUnitResolution);

    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return _at(volUnitLocalIdx, it->second.index);

}

TsdfVoxel HashTSDFVolumeCPU::at(const Point3f& point) const
{
    cv::Vec3i volumeUnitIdx          = volumeToVolumeUnitIdx(point);
    VolumeUnitIndexes::const_iterator it = volumeUnits.find(volumeUnitIdx);

    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }

    cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return _at(volUnitLocalIdx, it->second.index);
}

static inline Vec3i voxelToVolumeUnitIdx(const Vec3i& pt, const int vuRes)
{
    if (!(vuRes & (vuRes - 1)))
    {
        // vuRes is a power of 2, let's get this power
        const int p2 = trailingZeros32(vuRes);
        return Vec3i(pt[0] >> p2, pt[1] >> p2, pt[2] >> p2);
    }
    else
    {
        return Vec3i(cvFloor(float(pt[0]) / vuRes),
            cvFloor(float(pt[1]) / vuRes),
            cvFloor(float(pt[2]) / vuRes));
    }
}

TsdfVoxel HashTSDFVolumeCPU::atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it) const
{
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    Vec3i volUnitLocalIdx = point - volumeUnitIdx * volumeUnitResolution;

    // expanding at(), removing bounds check
    const TsdfVoxel* volData = volUnitsData.ptr<TsdfVoxel>(it->second.index);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] + volUnitLocalIdx[1] * volStrides[1] + volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}



#if USE_INTRINSICS
inline float interpolate(float tx, float ty, float tz, float vx[8])
{
    v_float32x4 v0246, v1357;
    v_load_deinterleave(vx, v0246, v1357);

    v_float32x4 vxx = v0246 + v_setall_f32(tz) * (v1357 - v0246);

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v00_10 + v_setall_f32(ty) * (v01_11 - v00_10);
    float v0 = v0_1.get0();
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v0_1.get0();

    return v0 + tx * (v1 - v0);
}

#else
inline float interpolate(float tx, float ty, float tz, float vx[8])
{
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);
}
#endif

float HashTSDFVolumeCPU::interpolateVoxelPoint(const Point3f& point) const
{
    const Vec3i neighbourCoords[] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    VolumeUnitIndexes::const_iterator iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = volumeUnits.end();
        queried[i] = false;
    }

    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

    Vec3i iv(ix, iy, iz);
    float vx[8];
    for (int i = 0; i < 8; i++)
    {
        Vec3i pt = iv + neighbourCoords[i];

        Vec3i volumeUnitIdx = voxelToVolumeUnitIdx(pt, volumeUnitResolution);
        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            it = volumeUnits.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
            queried[dictIdx] = true;
        }

        vx[i] = atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
    }

    return interpolate(tx, ty, tz, vx);
}

inline float HashTSDFVolumeCPU::interpolateVoxel(const cv::Point3f& point) const
{
    return interpolateVoxelPoint(point * voxelSizeInv);
}


Point3f HashTSDFVolumeCPU::getNormalVoxel(const Point3f &point) const
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    VolumeUnitIndexes::const_iterator iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = volumeUnits.end();
        queried[i] = false;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    const Vec3i offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, // 0-3
                              { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}  // 4-7
    };
    const int nVals = 6;

#else
    const Vec3i offsets[] = { { 0,  0,  0}, { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1}, //  0-3
                              { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1}, //  4-7
                              {-1,  0,  0}, {-1,  0,  1}, {-1,  1,  0}, {-1,  1,  1}, //  8-11
                              { 2,  0,  0}, { 2,  0,  1}, { 2,  1,  0}, { 2,  1,  1}, // 12-15
                              { 0, -1,  0}, { 0, -1,  1}, { 1, -1,  0}, { 1, -1,  1}, // 16-19
                              { 0,  2,  0}, { 0,  2,  1}, { 1,  2,  0}, { 1,  2,  1}, // 20-23
                              { 0,  0, -1}, { 0,  1, -1}, { 1,  0, -1}, { 1,  1, -1}, // 24-27
                              { 0,  0,  2}, { 0,  1,  2}, { 1,  0,  2}, { 1,  1,  2}, // 28-31
    };
    const int nVals = 32;
#endif

    float vals[nVals];
    for (int i = 0; i < nVals; i++)
    {
        Vec3i pt = iptVox + offsets[i];

        Vec3i volumeUnitIdx = voxelToVolumeUnitIdx(pt, volumeUnitResolution);

        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            it = volumeUnits.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
            queried[dictIdx] = true;
        }

        vals[i] = tsdfToFloat(atVolumeUnit(pt, volumeUnitIdx, it).tsdf);
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    for (int c = 0; c < 3; c++)
    {
        normal[c] = vals[c * 2] - vals[c * 2 + 1];
    }
#else

    float cxv[8], cyv[8], czv[8];

    // How these numbers were obtained:
    // 1. Take the basic interpolation sequence:
    // 000, 001, 010, 011, 100, 101, 110, 111
    // where each digit corresponds to shift by x, y, z axis respectively.
    // 2. Add +1 for next or -1 for prev to each coordinate to corresponding axis
    // 3. Search corresponding values in offsets
    const int idxxp[8] = {  8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = {  4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = {  2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = {  1, 28,  3, 29,  5, 30,  7, 31 };

#if !USE_INTRINSICS
    for (int i = 0; i < 8; i++)
    {
        cxv[i] = vals[idxxn[i]] - vals[idxxp[i]];
        cyv[i] = vals[idxyn[i]] - vals[idxyp[i]];
        czv[i] = vals[idxzn[i]] - vals[idxzp[i]];
    }
#else

# if CV_SIMD >= 32
    v_float32x8 cxp = v_lut(vals, idxxp);
    v_float32x8 cxn = v_lut(vals, idxxn);

    v_float32x8 cyp = v_lut(vals, idxyp);
    v_float32x8 cyn = v_lut(vals, idxyn);

    v_float32x8 czp = v_lut(vals, idxzp);
    v_float32x8 czn = v_lut(vals, idxzn);

    v_float32x8 vcxv = cxn - cxp;
    v_float32x8 vcyv = cyn - cyp;
    v_float32x8 vczv = czn - czp;

    v_store(cxv, vcxv);
    v_store(cyv, vcyv);
    v_store(czv, vczv);
# else
    v_float32x4 cxp0 = v_lut(vals, idxxp + 0); v_float32x4 cxp1 = v_lut(vals, idxxp + 4);
    v_float32x4 cxn0 = v_lut(vals, idxxn + 0); v_float32x4 cxn1 = v_lut(vals, idxxn + 4);

    v_float32x4 cyp0 = v_lut(vals, idxyp + 0); v_float32x4 cyp1 = v_lut(vals, idxyp + 4);
    v_float32x4 cyn0 = v_lut(vals, idxyn + 0); v_float32x4 cyn1 = v_lut(vals, idxyn + 4);

    v_float32x4 czp0 = v_lut(vals, idxzp + 0); v_float32x4 czp1 = v_lut(vals, idxzp + 4);
    v_float32x4 czn0 = v_lut(vals, idxzn + 0); v_float32x4 czn1 = v_lut(vals, idxzn + 4);

    v_float32x4 cxv0 = cxn0 - cxp0; v_float32x4 cxv1 = cxn1 - cxp1;
    v_float32x4 cyv0 = cyn0 - cyp0; v_float32x4 cyv1 = cyn1 - cyp1;
    v_float32x4 czv0 = czn0 - czp0; v_float32x4 czv1 = czn1 - czp1;

    v_store(cxv + 0, cxv0); v_store(cxv + 4, cxv1);
    v_store(cyv + 0, cyv0); v_store(cyv + 4, cyv1);
    v_store(czv + 0, czv0); v_store(czv + 4, czv1);
#endif

#endif

    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    normal[0] = interpolate(tx, ty, tz, cxv);
    normal[1] = interpolate(tx, ty, tz, cyv);
    normal[2] = interpolate(tx, ty, tz, czv);
#endif

    float nv = sqrt(normal[0] * normal[0] +
                    normal[1] * normal[1] +
                    normal[2] * normal[2]);
    return nv < 0.0001f ? nan3 : normal / nv;
}

void HashTSDFVolumeCPU::raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
                                OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points1   = _points.getMat();
    Normals normals1 = _normals.getMat();

    Points& points(points1);
    Normals& normals(normals1);
    const HashTSDFVolumeCPU& volume(*this);
    const float tstep(volume.truncDist * volume.raycastStepFactor);
    const Affine3f cam2vol(volume.pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const int nstripes = -1;

    auto _HashRaycastInvoker = [&](const Range& range)
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot = cam2vol.rotation();
        const Matx33f vol2camRot = vol2cam.rotation();

        const float blockSize = volume.volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for (int x = 0; x < points.cols; x++)
            {
                //! Initialize default value
                Point3f point = nan3, normal = nan3;

                //! Ray origin and direction in the volume coordinate frame
                Point3f orig    = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin = 0;
                float tmax = volume.truncateThreshold;
                float tcurr = tmin;

                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                        std::numeric_limits<int>::min());


                float tprev = tcurr;
                float prevTsdf = volume.truncDist;
                Ptr<TSDFVolumeCPU> currVolumeUnit;
                while (tcurr < tmax)
                {
                    Point3f currRayPos = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volume.volumeToVolumeUnitIdx(currRayPos);


                    VolumeUnitIndexes::const_iterator it = volume.volumeUnits.find(currVolumeUnitIdx);

                    float currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;


                    //! The subvolume exists in hashtable
                    if (it != volume.volumeUnits.end())
                    {
                        cv::Point3f currVolUnitPos =
                            volume.volumeUnitIdxToVolume(currVolumeUnitIdx);
                        volUnitLocalIdx = volume.volumeToVoxelCoord(currRayPos - currVolUnitPos);


                        //! TODO: Figure out voxel interpolation
                        TsdfVoxel currVoxel = _at(volUnitLocalIdx, it->second.index);
                        currTsdf = tsdfToFloat(currVoxel.tsdf);
                        currWeight = currVoxel.weight;
                        stepSize = tstep;
                    }
                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = volume.getNormalVoxel(pv);

                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point = vol2cam * pv;
                            }
                        }
                        break;
                    }
                    prevVolumeUnitIdx = currVolumeUnitIdx;
                    prevTsdf = currTsdf;
                    tprev = tcurr;
                    tcurr += stepSize;
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    };

    parallel_for_(Range(0, points.rows), _HashRaycastInvoker, nstripes);
}

void HashTSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs;

        std::vector<Vec3i> totalVolUnits;
        for (const auto& keyvalue : volumeUnits)
        {
            totalVolUnits.push_back(keyvalue.first);
        }
        Range fetchRange(0, (int)totalVolUnits.size());
        const int nstripes = -1;

        const HashTSDFVolumeCPU& volume(*this);
        bool needNormals(_normals.needed());
        Mutex mutex;


        auto HashFetchPointsNormalsInvoker = [&](const Range& range)
        {


            std::vector<ptype> points, normals;
            for (int i = range.start; i < range.end; i++)
            {
                cv::Vec3i tsdf_idx = totalVolUnits[i];


                VolumeUnitIndexes::const_iterator it = volume.volumeUnits.find(tsdf_idx);
                Point3f base_point = volume.volumeUnitIdxToVolume(tsdf_idx);
                if (it != volume.volumeUnits.end())
                {
                    std::vector<ptype> localPoints;
                    std::vector<ptype> localNormals;
                    for (int x = 0; x < volume.volumeUnitResolution; x++)
                        for (int y = 0; y < volume.volumeUnitResolution; y++)
                            for (int z = 0; z < volume.volumeUnitResolution; z++)
                            {
                                cv::Vec3i voxelIdx(x, y, z);
                                TsdfVoxel voxel = _at(voxelIdx, it->second.index);

                                if (voxel.tsdf != -128 && voxel.weight != 0)
                                {
                                    Point3f point = base_point + volume.voxelCoordToVolume(voxelIdx);
                                    localPoints.push_back(toPtype(point));
                                    if (needNormals)
                                    {
                                        Point3f normal = volume.getNormalVoxel(point);
                                        localNormals.push_back(toPtype(normal));
                                    }
                                }
                            }

                    AutoLock al(mutex);
                    pVecs.push_back(localPoints);
                    nVecs.push_back(localNormals);
                }
            }
        };

        parallel_for_(fetchRange, HashFetchPointsNormalsInvoker, nstripes);



        std::vector<ptype> points, normals;
        for (size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
        }

        _points.create((int)points.size(), 1, POINT_TYPE);
        if (!points.empty())
            Mat((int)points.size(), 1, POINT_TYPE, &points[0]).copyTo(_points.getMat());

        if (_normals.needed())
        {
            _normals.create((int)normals.size(), 1, POINT_TYPE);
            if (!normals.empty())
                Mat((int)normals.size(), 1, POINT_TYPE, &normals[0]).copyTo(_normals.getMat());
        }
    }
}

void HashTSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();

        const HashTSDFVolumeCPU& _volume = *this;
        auto HashPushNormals             = [&](const ptype& point, const int* position) {
            const HashTSDFVolumeCPU& volume(_volume);
            Affine3f invPose(volume.pose.inv());
            Point3f p = fromPtype(point);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxelPoint = invPose * p;
                n                  = volume.pose.rotation() * volume.getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }
}

int HashTSDFVolumeCPU::getVisibleBlocks(int currFrameId, int frameThreshold) const
{
    int numVisibleBlocks = 0;
    //! TODO: Iterate over map parallely?
    for (const auto& keyvalue : volumeUnits)
    {
        const VolumeUnit& volumeUnit = keyvalue.second;
        if (volumeUnit.lastVisibleIndex > (currFrameId - frameThreshold))
            numVisibleBlocks++;
    }
    return numVisibleBlocks;
}


///////// GPU implementation /////////

#ifdef HAVE_OPENCL

HashTSDFVolumeGPU::HashTSDFVolumeGPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
    float _truncateThreshold, int _volumeUnitRes, bool _zFirstMemOrder)
    :HashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold, _volumeUnitRes, _zFirstMemOrder)
{
    int xdim, ydim, zdim;
    if (zFirstMemOrder)
    {
        xdim = volumeUnitResolution * volumeUnitResolution;
        ydim = volumeUnitResolution;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volumeUnitResolution;
        zdim = volumeUnitResolution * volumeUnitResolution;
    }
    volStrides = Vec4i(xdim, ydim, zdim);

    reset();
}

HashTSDFVolumeGPU::HashTSDFVolumeGPU(const VolumeParams & _params, bool _zFirstMemOrder)
    : HashTSDFVolume(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
        _params.depthTruncThreshold, _params.unitResolution, _zFirstMemOrder)
{
}
// zero volume, leave rest params the same
void HashTSDFVolumeGPU::reset()
{
    CV_TRACE_FUNCTION();
    _lastVolIndex = 0;
    degree = 15;
    buff_lvl = pow(2, degree);
    _volUnitsData = cv::Mat(buff_lvl, volumeUnitResolution * volumeUnitResolution * volumeUnitResolution, rawType<TsdfVoxel>());
    volUnitsData = cv::Mat(buff_lvl, volumeUnitResolution * volumeUnitResolution * volumeUnitResolution, rawType<TsdfVoxel>());
    //volUnitsData = cv::Mat(VOLUMES_SIZE, 1, rawType<UMat>());
    indexes = cv::Mat(buff_lvl, 1, rawType<Vec3i>());
    poses = cv::Mat(buff_lvl, 1, rawType<cv::Matx44f>());
    lastVisibleIndexes = cv::Mat(buff_lvl, 1, rawType<int>());
    _indexes = VolumesTable();
    posesGPU = cv::Mat(buff_lvl, 16, rawType<float>());
}

static inline bool _find(cv::Mat v, Vec3i tsdf_idx, int _lastVolIndex)
{
    for (int i = 0; i < _lastVolIndex+1; i++)
    {
        auto p = v.at<Vec3i>(i, 0);
        if (p == tsdf_idx)
        {
            return true;
        }
    }
    return false;
}

inline int HashTSDFVolumeGPU::find_idx(cv::Mat v, Vec3i tsdf_idx) const
{
    for (int i = 0; i < _lastVolIndex; i++)
    {
        Vec3i p = v.at<Vec3i>(i, 0);
        if (p == tsdf_idx)
        {
            return i;
        }
    }
    return -1;
}

static cv::UMat preCalculationPixNormGPU(int depth_rows, int depth_cols, Vec2f fxy, Vec2f cxy)
{
    Mat x(1, depth_cols, CV_32F);
    Mat y(1, depth_rows, CV_32F);
    Mat _pixNorm(1, depth_rows * depth_cols, CV_32F);

    for (int i = 0; i < depth_cols; i++)
        x.at<float>(0, i) = (i - cxy[0]) / fxy[0];
    for (int i = 0; i < depth_rows; i++)
        y.at<float>(0, i) = (i - cxy[1]) / fxy[1];

    cv::String errorStr;
    cv::String name = "preCalculationPixNorm";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    cv::String options = "-cl-mad-enable";
    ocl::Kernel kk;
    kk.create(name.c_str(), source, options, &errorStr);


    if (kk.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    AccessFlag af = ACCESS_READ;
    UMat pixNorm = _pixNorm.getUMat(af);
    UMat xx = x.getUMat(af);
    UMat yy = y.getUMat(af);

    kk.args(ocl::KernelArg::PtrReadWrite(pixNorm),
        ocl::KernelArg::PtrReadOnly(xx),
        ocl::KernelArg::PtrReadOnly(yy),
        depth_cols);

    size_t globalSize[2];
    globalSize[0] = depth_rows;
    globalSize[1] = depth_cols;

    if (!kk.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

    return pixNorm;
}
/*
void HashTSDFVolumeGPU::integrateVolumeUnitGPU( InputArray _depth, float depthFactor,
        const Matx44f& cameraPose, const Intr& intrinsics, VolumeIndex idx)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_depth.empty());

    UMat depth = _depth.getUMat();

    String errorStr;
    String name = "integrateVolumeUnit";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    float dfac = 1.f / depthFactor;
    Vec4i volResGpu(volumeUnitResolution, volumeUnitResolution, volumeUnitResolution);
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);

    // TODO: optimization possible
    // Use sampler for depth (mask needed)
    k.args(ocl::KernelArg::ReadOnly(depth),
        //ocl::KernelArg::PtrReadWrite(volUnitsData.row(idx).getUMat(ACCESS_RW)),
        ocl::KernelArg::PtrReadWrite(_volUnitsData.row(idx).getUMat(ACCESS_RW)),
        ocl::KernelArg::Constant(vol2cam.matrix.val,
            sizeof(vol2cam.matrix.val)),
        voxelSize,
        volResGpu.val,
        volStrides.val,
        fxy.val,
        cxy.val,
        dfac,
        truncDist,
        int(maxWeight),
        ocl::KernelArg::PtrReadOnly(_pixNorms)); 

    size_t globalSize[2];
    globalSize[0] = (size_t)volumeUnitResolution;
    globalSize[1] = (size_t)volumeUnitResolution;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}
*/

bool matIsEqual(const cv::Mat Mat1, const cv::Mat Mat2)
{
    if (Mat1.dims == Mat2.dims &&
        Mat1.size == Mat2.size &&
        Mat1.elemSize() == Mat2.elemSize())
    {
        if (Mat1.isContinuous() && Mat2.isContinuous())
        {
            return 0 == memcmp(Mat1.ptr(), Mat2.ptr(), Mat1.total() * Mat1.elemSize());
        }
        else
        {
            const cv::Mat* arrays[] = { &Mat1, &Mat2, 0 };
            uchar* ptrs[2];
            cv::NAryMatIterator it(arrays, ptrs, 2);
            for (unsigned int p = 0; p < it.nplanes; p++, ++it)
                if (0 != memcmp(it.ptrs[0], it.ptrs[1], it.size * Mat1.elemSize()))
                    return false;

            return true;
        }
    }

    return false;
}

void HashTSDFVolumeGPU::integrateAllVolumeUnitsGPU(InputArray _depth, float depthFactor, const Intr& intrinsics)
{
    //std::cout << "integrateAllVolumeUnitsGPU" << std::endl;
    CV_TRACE_FUNCTION();
    CV_Assert(!_depth.empty());

    UMat depth = _depth.getUMat();
    
    String errorStr;
    String name = "integrateAllVolumeUnits";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);
    
    //Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    float dfac = 1.f / depthFactor;
    Vec4i volResGpu(volumeUnitResolution, volumeUnitResolution, volumeUnitResolution);
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);

    int totalVolUnitsSize = _indexes.indexesGPU.size();
    Mat totalVolUnits(_indexes.indexesGPU, rawType<Vec4i>());

    //std::cout << Vec3i(7,7,1) << " = " << _indexes.find_Volume(Vec3i(7, 7, 1)) << 
    //    " | " << calc_hash(Vec4i(7, 7, 1, 0)) % _indexes.hash_divisor<< std::endl;
    
    //std::cout << calc_hash(Vec3i(7, 7, 1)) << std::endl;
    //std::cout << " lol =" << _indexes.list_size<<" "<< _indexes.bufferNums << " " << _indexes.hash_divisor << std::endl;
    //std::cout << "maxWeight == " << maxWeight << std::endl;
    Mat _tmp;
    _volUnitsData.copyTo(_tmp);
    UMat U_volUnitsData = _tmp.getUMat(ACCESS_RW);
    //UMat U_volUnitsData = _volUnitsData.getUMat(ACCESS_RW);

    k.args(ocl::KernelArg::ReadOnly(depth),
        ocl::KernelArg::PtrReadWrite(_indexes.volumes.getUMat(ACCESS_RW)),
        (int)_indexes.list_size,
        (int)_indexes.bufferNums,
        (int)_indexes.hash_divisor,
        ocl::KernelArg::PtrReadWrite(totalVolUnits.getUMat(ACCESS_RW)),
        //ocl::KernelArg::PtrReadWrite(_volUnitsData.getUMat(ACCESS_RW)),
        //ocl::KernelArg::ReadWrite(_volUnitsData.getUMat(ACCESS_RW)),
        ocl::KernelArg::ReadWrite(U_volUnitsData),
        ocl::KernelArg::PtrReadOnly(_pixNorms),
        //ocl::KernelArg::PtrReadOnly(pixNorms.getUMat(ACCESS_RW)),
        ocl::KernelArg::ReadOnly(posesGPU.getUMat(ACCESS_READ)),
        //ocl::KernelArg::ReadOnly(poses.getUMat(ACCESS_READ)),
        _lastVolIndex,
        voxelSize,
        volResGpu.val,
        volStrides.val,
        fxy.val,
        cxy.val,
        dfac,
        truncDist,
        int(maxWeight)
    );
    //std::cout << _indexes
    //int resol = 1;
    int resol = volumeUnitResolution;
    size_t globalSize[3];
    globalSize[0] = (size_t)resol; // volumeUnitResolution
    globalSize[1] = (size_t)resol; // volumeUnitResolution
    //globalSize[0] = (size_t)volResGpu.val[0]; // volResolution.x
    //globalSize[1] = (size_t)volResGpu.val[1]; // volResolution.y
    globalSize[2] = (size_t)totalVolUnitsSize; // num of voxels
    //printf("CPU: fxy = [%f, %f] | cxy = [%f, %f] \n", fxy[0], fxy[1], cxy[0], cxy[1]);
    //std::cout << "rmaxWeight == " << maxWeight << std::endl;
    if (!k.run(3, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
    
    Mat checking;
    _volUnitsData.copyTo(checking);

    U_volUnitsData.getMat(ACCESS_RW).copyTo(_volUnitsData);
    //_tmp = U_volUnitsData.getMat(ACCESS_RW);
    //_tmp.copyTo(_volUnitsData);
    //_tmp.release();

    //cv::Mat diff;
    //cv::compare(checking, _volUnitsData, diff, cv::CMP_NE);
    //int nz = cv::countNonZero(diff);

    //if (nz == 0) std::cout << "compare = " << true << std::endl;
    if (matIsEqual(checking, _volUnitsData)) 
        std::cout << "compare = " << true << std::endl;
    else
        std::cout << "compare = " << false << std::endl;

    //Mat _tmp = _volUnitsData;
    //(U_volUnitsData.getMat(ACCESS_RW)).copyTo(_volUnitsData);
    //Mat tmp = U_volUnitsData.getMat(ACCESS_RW);
    
    //tmp.copyTo(_volUnitsData);
    //U_volUnitsData.release();
    //tmp.release();
    //bool res = false;
    //if (_tmp == _volUnitsData) res = true;
    //std::cout << res << std::endl;
}

void HashTSDFVolumeGPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    //std::cout << "integrate" << std::endl;
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

    //! Compute volumes to be allocated
    const int depthStride = int(log2(volumeUnitResolution));
    const float invDepthFactor = 1.f / depthFactor;
    const Intr::Reprojector reproj(intrinsics.makeReprojector());
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Point3f truncPt(truncDist, truncDist, truncDist);
    _VolumeUnitIndexSet _newIndices = cv::Mat(VOLUMES_SIZE, 1, rawType<VolumeIndex>());
    cv::Mat newIndices = cv::Mat(VOLUMES_SIZE, 1, rawType<Vec3i>());
    Mutex mutex;
    Range allocateRange(0, depth.rows);
    int loc_vol_idx = 0;
    int vol_idx = 0;

    auto AllocateVolumeUnitsInvoker = [&](const Range& range) {
        _VolumeUnitIndexSet _localAccessVolUnits = cv::Mat(VOLUMES_SIZE, 1, rawType<Vec3i>());


        for (int y = range.start; y < range.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = 0; x < depth.cols; x += depthStride)
            {
                depthType z = depthRow[x] * invDepthFactor;
                if (z <= 0 || z > this->truncateThreshold)
                    continue;
                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = this->volumeToVolumeUnitIdx(volPoint - truncPt);
                Vec3i upper_bound = this->volumeToVolumeUnitIdx(volPoint + truncPt);

                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= lower_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);

                            if (!_find(_localAccessVolUnits, tsdf_idx, loc_vol_idx))
                            {
                                _localAccessVolUnits.at<Vec3i>(loc_vol_idx, 0) = tsdf_idx;
                                loc_vol_idx++;
                            }

                        }
            }
        }

        //mutex.lock();
        //std::cout << "lol\n";
        for (int i = 0; i < loc_vol_idx; i++)
        {
            Vec3i idx = _localAccessVolUnits.at<Vec3i>(i, 0);

            if (!_indexes.isExist(idx))
            {
                if (_lastVolIndex >= buff_lvl)
                {
                    //std::cout << "kek " << _lastVolIndex << " " << buff_lvl << std::endl;
                    degree++;
                    buff_lvl = pow(2, degree);
                    indexes.resize(buff_lvl);
                    _volUnitsData.resize(buff_lvl);
                    volUnitsData.resize(buff_lvl);
                    poses.resize(buff_lvl);
                    lastVisibleIndexes.resize(buff_lvl);
                    posesGPU.resize(buff_lvl);
                }
                this->indexes.at<Vec3i>(_lastVolIndex, 0) = idx;
                _indexes.update(idx, _lastVolIndex);
                _newIndices.at<VolumeIndex>(vol_idx, 0) = _lastVolIndex;
                newIndices.at<Vec3i>(vol_idx, 0) = idx;
                vol_idx++;
                _lastVolIndex++;
            }
        }
        //mutex.unlock();
    };

    //parallel_for_(allocateRange, AllocateVolumeUnitsInvoker);
    AllocateVolumeUnitsInvoker(allocateRange);

    //! Perform the allocation
    //std::cout << "lol\n";
    for (int i = 0; i < vol_idx; i++)
    {
        Vec3i tsdf_idx  = newIndices.at<Vec3i>(i, 0);
        VolumeIndex idx = _indexes.find_Volume(tsdf_idx);

        Matx44f subvolumePose = pose.translate(volumeUnitIdxToVolume(tsdf_idx)).matrix;

        poses.at<cv::Matx44f>(idx, 0) = subvolumePose;
        lastVisibleIndexes.at<int>(idx, 0) = frameId;
        //Affine3f cam2vol(Affine3f(subvolumePose) * Affine3f(cameraPose));
        Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
        _indexes.updateActive(tsdf_idx, 1);
 
        auto vol2camMatrix = vol2cam.matrix.val;
        for (int i = 0; i < 16; i++)
        {
            posesGPU.at<float>(idx, i) = vol2camMatrix[i];
        }

        /*
        printf(" CPU | %f %f %f %f | %f %f %f %f | %f %f %f %f | %f %f %f %f |\n",
            vol2camMatrix[0], vol2camMatrix[1], vol2camMatrix[2], vol2camMatrix[3],
            vol2camMatrix[4], vol2camMatrix[5], vol2camMatrix[6], vol2camMatrix[7],
            vol2camMatrix[8], vol2camMatrix[9], vol2camMatrix[10], vol2camMatrix[11],
            vol2camMatrix[12], vol2camMatrix[13], vol2camMatrix[14], vol2camMatrix[15]);
        */

        _volUnitsData.row(idx).forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int*)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        volUnitsData.row(idx).forEach<VecTsdfVoxel>([](VecTsdfVoxel& vv, const int*)
            {
                TsdfVoxel& v = reinterpret_cast<TsdfVoxel&>(vv);
                v.tsdf = floatToTsdf(0.0f); v.weight = 0;
            });
        //volUnitsData.at<UMat>(idx, 0) = cv::UMat(VOLUMES_SIZE, volumeUnitResolution * volumeUnitResolution * volumeUnitResolution, rawType<TsdfVoxel>());
    }

    //! Get keys for all the allocated volume Units
    std::vector<Vec3i> _totalVolUnits = _indexes.indexes;

    //! Mark volumes in the camera frustum as active
    Range _inFrustumRange(0, (int)_totalVolUnits.size());
    auto markActivities = [&](const Range& range) {
        const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
        const Intr::Projector proj(intrinsics.makeProjector());

        for (int i = range.start; i < range.end; ++i)
        {
            Vec3i tsdf_idx = _totalVolUnits[i];
            
            VolumeIndex idx = _indexes.find_Volume(tsdf_idx);
            //std::cout << tsdf_idx << " " << idx << std::endl;
            if (idx < 0 || idx == _lastVolIndex - 1) return;

            Point3f volumeUnitPos = volumeUnitIdxToVolume(poses.at<Vec3i>(idx, 0));
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;

            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
            {
                _indexes.updateActive(tsdf_idx, 0);
                return;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols && cameraPoint.y < depth.rows)
            {
                assert(idx == _lastVolIndex - 1);
                lastVisibleIndexes.at<int>(idx, 0) = frameId;
                std::cout << " - " << tsdf_idx << std::endl;
                _indexes.update(tsdf_idx, 1, frameId);
            }
        }
    };
    //parallel_for_(_inFrustumRange, markActivities);
    markActivities(_inFrustumRange);

    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, intrinsics);
        Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
        _pixNorms = preCalculationPixNormGPU(depth.rows, depth.cols, fxy, cxy);
    }

    //! Integrate the correct volumeUnits
    if (false)
    {
        auto Integrate = [&](const Range& range) {
            for (int i = range.start; i < range.end; i++)
            {
                Vec3i tsdf_idx = _totalVolUnits[i];
                VolumeIndex idx = _indexes.find_Volume(tsdf_idx);
                if (idx < 0 || idx == _lastVolIndex - 1) return;

                //bool& _isActive = isActive.at<bool>(idx, 0);
                bool _isActive = _indexes.getActive(tsdf_idx);
                //std::cout << _isActive;
                if (_isActive)
                {
                    //std::cout << " lol ";
                    //! The volume unit should already be added into the Volume from the allocator
                    Matx44f _pose = poses.at<Matx44f>(idx, 0);

                    integrateVolumeUnit(truncDist, voxelSize, maxWeight, _pose,
                        Point3i(volumeUnitResolution, volumeUnitResolution, volumeUnitResolution), volStrides, depth,
                        depthFactor, cameraPose, intrinsics, pixNorms, _volUnitsData.row(idx));

                    //integrateVolumeUnitGPU(depth, depthFactor, _pose, intrinsics, idx);
                    //! Ensure all active volumeUnits are set to inactive for next integration
                    //_isActive = false;
                    _indexes.updateActive(tsdf_idx, 0);
                }
            }
        };

        //parallel_for_(Range(0, (int)_totalVolUnits.size()), Integrate );
        Integrate(Range(0, (int)_totalVolUnits.size()));
    }
    //std::cout << "lol\n";
    else
    integrateAllVolumeUnitsGPU(depth, depthFactor, intrinsics);

    //std::cout << "lol\n";
}

cv::Vec3i HashTSDFVolumeGPU::volumeToVolumeUnitIdx(const cv::Point3f& p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize), cvFloor(p.y / volumeUnitSize),
        cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeGPU::volumeUnitIdxToVolume(const cv::Vec3i& volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, volumeUnitIdx[1] * volumeUnitSize,
        volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeGPU::voxelCoordToVolume(const cv::Vec3i& voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeGPU::volumeToVoxelCoord(const cv::Point3f& point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv), cvFloor(point.y * voxelSizeInv),
        cvFloor(point.z * voxelSizeInv));
}

inline TsdfVoxel HashTSDFVolumeGPU::new_at(const cv::Vec3i& volumeIdx, VolumeIndex indx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.0f);
        dummy.weight = 0;
        return dummy;
    }

    const TsdfVoxel* volData = _volUnitsData.ptr<TsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] +
        volumeIdx[1] * volStrides[1] +
        volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

TsdfVoxel HashTSDFVolumeGPU::new_atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeIndex indx) const
{
    if (indx < 0 || indx > _lastVolIndex - 1)
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    Vec3i volUnitLocalIdx = point - volumeUnitIdx * volumeUnitResolution;

    // expanding at(), removing bounds check
    const TsdfVoxel* volData = _volUnitsData.ptr<TsdfVoxel>(indx);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] +
        volUnitLocalIdx[1] * volStrides[1] +
        volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}

float HashTSDFVolumeGPU::interpolateVoxelPoint(const Point3f& point) const
{
    const Vec3i neighbourCoords[] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    //VolumeUnitIndexes::const_iterator iterMap[8];
    VolumeIndex iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        //iterMap[i] = volumeUnits.end();
        iterMap[i] = _lastVolIndex - 1;
        queried[i] = false;
    }

    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

    Vec3i iv(ix, iy, iz);
    float vx[8];
    for (int i = 0; i < 8; i++)
    {
        Vec3i pt = iv + neighbourCoords[i];

        Vec3i volumeUnitIdx = voxelToVolumeUnitIdx(pt, volumeUnitResolution);
        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (!queried[dictIdx])
        {
            //it = find_idx(indexes, volumeUnitIdx);
            it = _indexes.find_Volume(volumeUnitIdx);
            if (it >= 0 || it < _lastVolIndex)
            {
                iterMap[dictIdx] = it;
                queried[dictIdx] = true;
            }
            //it = volumeUnits.find(volumeUnitIdx);
            //iterMap[dictIdx] = it;
            //queried[dictIdx] = true;
        }

        //vx[i] = atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
        vx[i] = new_atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
    }

    return interpolate(tx, ty, tz, vx);
}

inline float HashTSDFVolumeGPU::interpolateVoxel(const cv::Point3f& point) const
{
    return interpolateVoxelPoint(point * voxelSizeInv);
}

Point3f HashTSDFVolumeGPU::_getNormalVoxel(const Point3f& point) const
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    // A small hash table to reduce a number of find() calls
    bool queried[8];
    VolumeIndex iterMap[8];

    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = _lastVolIndex - 1;
        queried[i] = false;
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    const Vec3i offsets[] = { { 1,  0,  0}, {-1,  0,  0}, { 0,  1,  0}, // 0-3
                              { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}  // 4-7
    };
    const int nVals = 6;

#else
    const Vec3i offsets[] = { { 0,  0,  0}, { 0,  0,  1}, { 0,  1,  0}, { 0,  1,  1}, //  0-3
                              { 1,  0,  0}, { 1,  0,  1}, { 1,  1,  0}, { 1,  1,  1}, //  4-7
                              {-1,  0,  0}, {-1,  0,  1}, {-1,  1,  0}, {-1,  1,  1}, //  8-11
                              { 2,  0,  0}, { 2,  0,  1}, { 2,  1,  0}, { 2,  1,  1}, // 12-15
                              { 0, -1,  0}, { 0, -1,  1}, { 1, -1,  0}, { 1, -1,  1}, // 16-19
                              { 0,  2,  0}, { 0,  2,  1}, { 1,  2,  0}, { 1,  2,  1}, // 20-23
                              { 0,  0, -1}, { 0,  1, -1}, { 1,  0, -1}, { 1,  1, -1}, // 24-27
                              { 0,  0,  2}, { 0,  1,  2}, { 1,  0,  2}, { 1,  1,  2}, // 28-31
    };
    const int nVals = 32;
#endif

    float vals[nVals];
    for (int i = 0; i < nVals; i++)
    {
        Vec3i pt = iptVox + offsets[i];

        Vec3i volumeUnitIdx = voxelToVolumeUnitIdx(pt, volumeUnitResolution);

        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];

        if (!queried[dictIdx])
        {
            //it = find_idx(indexes, volumeUnitIdx);
            it = _indexes.find_Volume(volumeUnitIdx);
            if (it >= 0 || it < _lastVolIndex)
            {
                iterMap[dictIdx] = it;
                queried[dictIdx] = true;
            }
        }

        vals[i] = tsdfToFloat(new_atVolumeUnit(pt, volumeUnitIdx, it).tsdf);
    }

#if !USE_INTERPOLATION_IN_GETNORMAL
    for (int c = 0; c < 3; c++)
    {
        normal[c] = vals[c * 2] - vals[c * 2 + 1];
    }
#else

    float cxv[8], cyv[8], czv[8];

    // How these numbers were obtained:
    // 1. Take the basic interpolation sequence:
    // 000, 001, 010, 011, 100, 101, 110, 111
    // where each digit corresponds to shift by x, y, z axis respectively.
    // 2. Add +1 for next or -1 for prev to each coordinate to corresponding axis
    // 3. Search corresponding values in offsets
    const int idxxp[8] = { 8,  9, 10, 11,  0,  1,  2,  3 };
    const int idxxn[8] = { 4,  5,  6,  7, 12, 13, 14, 15 };
    const int idxyp[8] = { 16, 17,  0,  1, 18, 19,  4,  5 };
    const int idxyn[8] = { 2,  3, 20, 21,  6,  7, 22, 23 };
    const int idxzp[8] = { 24,  0, 25,  2, 26,  4, 27,  6 };
    const int idxzn[8] = { 1, 28,  3, 29,  5, 30,  7, 31 };

#if !USE_INTRINSICS
    for (int i = 0; i < 8; i++)
    {
        cxv[i] = vals[idxxn[i]] - vals[idxxp[i]];
        cyv[i] = vals[idxyn[i]] - vals[idxyp[i]];
        czv[i] = vals[idxzn[i]] - vals[idxzp[i]];
    }
#else

# if CV_SIMD >= 32
    v_float32x8 cxp = v_lut(vals, idxxp);
    v_float32x8 cxn = v_lut(vals, idxxn);

    v_float32x8 cyp = v_lut(vals, idxyp);
    v_float32x8 cyn = v_lut(vals, idxyn);

    v_float32x8 czp = v_lut(vals, idxzp);
    v_float32x8 czn = v_lut(vals, idxzn);

    v_float32x8 vcxv = cxn - cxp;
    v_float32x8 vcyv = cyn - cyp;
    v_float32x8 vczv = czn - czp;

    v_store(cxv, vcxv);
    v_store(cyv, vcyv);
    v_store(czv, vczv);
# else
    v_float32x4 cxp0 = v_lut(vals, idxxp + 0); v_float32x4 cxp1 = v_lut(vals, idxxp + 4);
    v_float32x4 cxn0 = v_lut(vals, idxxn + 0); v_float32x4 cxn1 = v_lut(vals, idxxn + 4);

    v_float32x4 cyp0 = v_lut(vals, idxyp + 0); v_float32x4 cyp1 = v_lut(vals, idxyp + 4);
    v_float32x4 cyn0 = v_lut(vals, idxyn + 0); v_float32x4 cyn1 = v_lut(vals, idxyn + 4);

    v_float32x4 czp0 = v_lut(vals, idxzp + 0); v_float32x4 czp1 = v_lut(vals, idxzp + 4);
    v_float32x4 czn0 = v_lut(vals, idxzn + 0); v_float32x4 czn1 = v_lut(vals, idxzn + 4);

    v_float32x4 cxv0 = cxn0 - cxp0; v_float32x4 cxv1 = cxn1 - cxp1;
    v_float32x4 cyv0 = cyn0 - cyp0; v_float32x4 cyv1 = cyn1 - cyp1;
    v_float32x4 czv0 = czn0 - czp0; v_float32x4 czv1 = czn1 - czp1;

    v_store(cxv + 0, cxv0); v_store(cxv + 4, cxv1);
    v_store(cyv + 0, cyv0); v_store(cyv + 4, cyv1);
    v_store(czv + 0, czv0); v_store(czv + 4, czv1);
#endif

#endif

    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    normal[0] = interpolate(tx, ty, tz, cxv);
    normal[1] = interpolate(tx, ty, tz, cyv);
    normal[2] = interpolate(tx, ty, tz, czv);
#endif
    float nv = sqrt(normal[0] * normal[0] +
        normal[1] * normal[1] +
        normal[2] * normal[2]);
    return nv < 0.0001f ? nan3 : normal / nv;
}

void HashTSDFVolumeGPU::raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
    OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points1 = _points.getMat();
    Normals normals1 = _normals.getMat();

    Points& new_points(points1);
    Normals& new_normals(normals1);

    const HashTSDFVolumeGPU& volume(*this);
    const float tstep(volume.truncDist * volume.raycastStepFactor);
    const Affine3f cam2vol(volume.pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);
    const Intr::Reprojector reproj(intrinsics.makeReprojector());

    const int nstripes = -1;

    auto _HashRaycastInvoker = [&](const Range& range)
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot = cam2vol.rotation();
        const Matx33f vol2camRot = vol2cam.rotation();

        const float blockSize = volume.volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {
            ptype* _ptsRow = new_points[y];
            ptype* _nrmRow = new_normals[y];

            for (int x = 0; x < new_points.cols; x++)
            {
                //! Initialize default value
                Point3f point = nan3, normal = nan3;

                //! Ray origin and direction in the volume coordinate frame
                Point3f orig = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin = 0;
                float tmax = volume.truncateThreshold;
                float tcurr = tmin;

                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                        std::numeric_limits<int>::min());

                float tprev = tcurr;
                float prevTsdf = volume.truncDist;
                Ptr<TSDFVolumeCPU> currVolumeUnit;

                while (tcurr < tmax)
                {
                    Point3f currRayPos = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volume.volumeToVolumeUnitIdx(currRayPos);

                    //VolumeIndex idx = find_idx(indexes, currVolumeUnitIdx);
                    VolumeIndex idx = _indexes.find_Volume(currVolumeUnitIdx);
                    float currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;


                    //! The subvolume exists in hashtable
                    if (idx < _lastVolIndex && idx >= 0)
                    {
                        cv::Point3f currVolUnitPos =
                            volume.volumeUnitIdxToVolume(currVolumeUnitIdx);
                        volUnitLocalIdx = volume.volumeToVoxelCoord(currRayPos - currVolUnitPos);


                        //! TODO: Figure out voxel interpolation
                        TsdfVoxel currVoxel = new_at(volUnitLocalIdx, idx);
                        currTsdf = tsdfToFloat(currVoxel.tsdf);
                        currWeight = currVoxel.weight;
                        stepSize = tstep;
                    }

                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = volume._getNormalVoxel(pv);

                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point = vol2cam * pv;
                            }
                        }
                        break;
                    }

                    prevVolumeUnitIdx = currVolumeUnitIdx;
                    prevTsdf = currTsdf;
                    tprev = tcurr;
                    tcurr += stepSize;

                }

                _ptsRow[x] = toPtype(point);
                _nrmRow[x] = toPtype(normal);

            }

        }

    };

    parallel_for_(Range(0, new_points.rows), _HashRaycastInvoker, nstripes);
    //_HashRaycastInvoker(Range(0, new_points.rows));
}

void HashTSDFVolumeGPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs;

        //std::vector<Vec3i> _totalVolUnits;
        std::vector<Vec3i> _totalVolUnits = _indexes.indexes;
        //for (int i = 0; i < indexes.size().height; i++){_totalVolUnits.push_back(indexes.at<Vec3i>(i, 0));}

        Range _fetchRange(0, (int)_totalVolUnits.size());

        const int nstripes = -1;

        const HashTSDFVolumeGPU& volume(*this);
        bool needNormals(_normals.needed());
        Mutex mutex;


        auto _HashFetchPointsNormalsInvoker = [&](const Range& range)
        {
            std::vector<ptype> points, normals;
            for (int i = range.start; i < range.end; i++)
            {
                cv::Vec3i tsdf_idx = _totalVolUnits[i];

                //VolumeIndex idx = find_idx(indexes, tsdf_idx);
                VolumeIndex idx = _indexes.find_Volume(tsdf_idx);
                Point3f base_point = volume.volumeUnitIdxToVolume(tsdf_idx);

                if (idx >= 0 && idx < _lastVolIndex)
                {
                    std::vector<ptype> localPoints;
                    std::vector<ptype> localNormals;
                    for (int x = 0; x < volume.volumeUnitResolution; x++)
                        for (int y = 0; y < volume.volumeUnitResolution; y++)
                            for (int z = 0; z < volume.volumeUnitResolution; z++)
                            {
                                cv::Vec3i voxelIdx(x, y, z);
                                TsdfVoxel voxel = new_at(voxelIdx, idx);

                                if (voxel.tsdf != -128 && voxel.weight != 0)
                                {
                                    Point3f point = base_point + volume.voxelCoordToVolume(voxelIdx);

                                    localPoints.push_back(toPtype(point));
                                    if (needNormals)
                                    {
                                        Point3f normal = volume._getNormalVoxel(point);
                                        localNormals.push_back(toPtype(normal));
                                    }
                                }
                            }

                    AutoLock al(mutex);
                    pVecs.push_back(localPoints);
                    nVecs.push_back(localNormals);
                }
            }
        };

        parallel_for_(_fetchRange, _HashFetchPointsNormalsInvoker, nstripes);
        //_HashFetchPointsNormalsInvoker(_fetchRange);


        std::vector<ptype> points, normals;
        for (size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
        }

        _points.create((int)points.size(), 1, POINT_TYPE);
        if (!points.empty())
            Mat((int)points.size(), 1, POINT_TYPE, &points[0]).copyTo(_points.getMat());

        if (_normals.needed())
        {
            _normals.create((int)normals.size(), 1, POINT_TYPE);
            if (!normals.empty())
                Mat((int)normals.size(), 1, POINT_TYPE, &normals[0]).copyTo(_normals.getMat());
        }
    }
}

void HashTSDFVolumeGPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);
        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();
        const HashTSDFVolumeGPU& _volume = *this;
        auto HashPushNormals             = [&](const ptype& point, const int* position) {
            const HashTSDFVolumeGPU& volume(_volume);
            Affine3f invPose(volume.pose.inv());
            Point3f p = fromPtype(point);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxelPoint = invPose * p;
                n                  = volume.pose.rotation() * volume._getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }

}

int HashTSDFVolumeGPU::getVisibleBlocks(int currFrameId, int frameThreshold) const
{
    int numVisibleBlocks = 0;
    //! TODO: Iterate over map parallely?
    for (int i = 0; i < _lastVolIndex; i++)
    {
        if (lastVisibleIndexes.at<int>(i, 0) > (currFrameId - frameThreshold))
            numVisibleBlocks++;
    }
    return numVisibleBlocks;
}

#endif

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(const VolumeParams& _volumeParams)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<HashTSDFVolumeGPU>(_volumeParams);
#endif
    return makePtr<HashTSDFVolumeCPU>(_volumeParams);
}

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
    int _maxWeight, float truncateThreshold, int volumeUnitResolution)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<HashTSDFVolumeGPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
            volumeUnitResolution);
#endif
    return makePtr<HashTSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
        volumeUnitResolution);
}

}  // namespace kinfu
}  // namespace cv
