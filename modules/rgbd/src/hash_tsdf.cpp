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

#define USE_INTERPOLATION_IN_GETNORMAL 1

namespace cv
{
namespace kinfu
{

static inline TsdfType floatToTsdf(float num)
{
    //CV_Assert(-1 < num <= 1);
    int8_t res = int8_t(num * (-128.f));
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

static inline float tsdfToFloat(TsdfType num)
{
    return float(num) * (-1.f / 128.f);
}

HashTSDFVolume::HashTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                               float _truncDist, int _maxWeight, float _truncateThreshold,
                               int _volumeUnitRes, bool _zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
      maxWeight(_maxWeight),
      truncateThreshold(_truncateThreshold),
      volumeUnitResolution(_volumeUnitRes),
      volumeUnitSize(voxelSize * volumeUnitResolution),
      zFirstMemOrder(_zFirstMemOrder)
{
    truncDist = std::max(_truncDist, 4.0f * voxelSize);
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                                     float _truncDist, int _maxWeight, float _truncateThreshold,
                                     int _volumeUnitRes, bool _zFirstMemOrder)
    : HashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                     _truncateThreshold, _volumeUnitRes, _zFirstMemOrder)
{
}

// zero volume, leave rest params the same
void HashTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();
    volumeUnits.clear();
}

struct AllocateVolumeUnitsInvoker : ParallelLoopBody
{
    AllocateVolumeUnitsInvoker(HashTSDFVolumeCPU& _volume, const Depth& _depth, Intr intrinsics,
                               cv::Matx44f cameraPose, float _depthFactor, int _depthStride = 4)
        : ParallelLoopBody(),
          volume(_volume),
          depth(_depth),
          reproj(intrinsics.makeReprojector()),
          cam2vol(_volume.pose.inv() * Affine3f(cameraPose)),
          depthFactor(1.0f / _depthFactor),
          depthStride(_depthStride)
    {
    }

    virtual void operator()(const Range& range) const override
    {
        for (int y = range.start; y < range.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = 0; x < depth.cols; x += depthStride)
            {
                depthType z = depthRow[x] * depthFactor;
                if (z <= 0)
                    continue;

                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;

                //! Find accessed TSDF volume unit for valid 3D vertex
                cv::Vec3i lower_bound = volume.volumeToVolumeUnitIdx(
                    volPoint - cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));
                cv::Vec3i upper_bound = volume.volumeToVolumeUnitIdx(
                    volPoint + cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));
                VolumeUnitIndexSet localAccessVolUnits;
                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= lower_bound[2]; k++)
                        {
                            const cv::Vec3i tsdf_idx = cv::Vec3i(i, j, k);
                            if (!localAccessVolUnits.count(tsdf_idx))
                            {
                                localAccessVolUnits.emplace(tsdf_idx);
                            }
                        }
                AutoLock al(mutex);
                for (const auto& tsdf_idx : localAccessVolUnits)
                {
                    //! If the insert into the global set passes
                    if (!volume.volumeUnits.count(tsdf_idx))
                    {
                        VolumeUnit volumeUnit;
                        cv::Point3i volumeDims(volume.volumeUnitResolution,
                                               volume.volumeUnitResolution,
                                               volume.volumeUnitResolution);

                        cv::Matx44f subvolumePose =
                            volume.pose.translate(volume.volumeUnitIdxToVolume(tsdf_idx)).matrix;
                        volumeUnit.pVolume = cv::makePtr<TSDFVolumeCPU>(
                            volume.voxelSize, subvolumePose, volume.raycastStepFactor,
                            volume.truncDist, volume.maxWeight, volumeDims);
                        //! This volume unit will definitely be required for current integration
                        volumeUnit.index             = tsdf_idx;
                        volumeUnit.isActive          = true;
                        volume.volumeUnits[tsdf_idx] = volumeUnit;
                    }
                }
            }
        }
    }

    HashTSDFVolumeCPU& volume;
    const Depth& depth;
    const Intr::Reprojector reproj;
    const cv::Affine3f cam2vol;
    const float depthFactor;
    const int depthStride;
    mutable Mutex mutex;
};


void HashTSDFVolumeCPU::integrate(InputArray _depth, float depthFactor,
                                  const cv::Matx44f& cameraPose, const Intr& intrinsics)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

    //! Compute volumes to be allocated
    AllocateVolumeUnitsInvoker allocate_i(*this, depth, intrinsics, cameraPose, depthFactor);
    Range allocateRange(0, depth.rows);
    parallel_for_(allocateRange, allocate_i);

    //! Get volumes that are in the current camera frame
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
            cv::Vec3i tsdf_idx         = totalVolUnits[i];
            VolumeUnitMap::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            Point3f volumeUnitPos     = volumeUnitIdxToVolume(it->first);
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;
            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
            {
                it->second.isActive = false;
                return;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols &&
                cameraPoint.y < depth.rows)
            {
                assert(it != volumeUnits.end());
                it->second.isActive = true;
            }
        }
    });

    //! Integrate the correct volumeUnits
    parallel_for_(Range(0, (int)totalVolUnits.size()), [&](const Range& range) {
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3i tsdf_idx         = totalVolUnits[i];
            VolumeUnitMap::iterator it = volumeUnits.find(tsdf_idx);
            if (it == volumeUnits.end())
                return;

            VolumeUnit& volumeUnit = it->second;
            if (volumeUnit.isActive)
            {
                //! The volume unit should already be added into the Volume from the allocator
                volumeUnit.pVolume->integrate(depth, depthFactor, cameraPose, intrinsics);
                //! Ensure all active volumeUnits are set to inactive for next integration
                volumeUnit.isActive = false;
            }
        }
    });
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVolumeUnitIdx(cv::Point3f p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize), cvFloor(p.y / volumeUnitSize),
                     cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeCPU::volumeUnitIdxToVolume(cv::Vec3i volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, volumeUnitIdx[1] * volumeUnitSize,
                       volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeCPU::voxelCoordToVolume(cv::Vec3i voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVoxelCoord(cv::Point3f point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv), cvFloor(point.y * voxelSizeInv),
                     cvFloor(point.z * voxelSizeInv));
}

inline TsdfVoxel HashTSDFVolumeCPU::at(const cv::Vec3i& volumeIdx) const
{
    cv::Vec3i volumeUnitIdx = cv::Vec3i(cvFloor(volumeIdx[0] / volumeUnitResolution),
                                        cvFloor(volumeIdx[1] / volumeUnitResolution),
                                        cvFloor(volumeIdx[2] / volumeUnitResolution));

    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf   = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    cv::Ptr<TSDFVolumeCPU> volumeUnit =
        std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);

    cv::Vec3i volUnitLocalIdx = volumeIdx - cv::Vec3i(volumeUnitIdx[0] * volumeUnitResolution,
                                                      volumeUnitIdx[1] * volumeUnitResolution,
                                                      volumeUnitIdx[2] * volumeUnitResolution);

    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return volumeUnit->at(volUnitLocalIdx);
}

inline TsdfVoxel HashTSDFVolumeCPU::at(const cv::Point3f& point) const
{
    cv::Vec3i volumeUnitIdx          = volumeToVolumeUnitIdx(point);
    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf   = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    cv::Ptr<TSDFVolumeCPU> volumeUnit =
        std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);

    cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    volUnitLocalIdx =
        cv::Vec3i(abs(volUnitLocalIdx[0]), abs(volUnitLocalIdx[1]), abs(volUnitLocalIdx[2]));
    return volumeUnit->at(volUnitLocalIdx);
}

inline float HashTSDFVolumeCPU::interpolateVoxel(const cv::Point3f& point) const
{
    cv::Point3f neighbourCoords[] = {
                                Point3f(0, 0, 0),
                                Point3f(0, 0, 1),
                                Point3f(0, 1, 0),
                                Point3f(0, 1, 1),
                                Point3f(1, 0, 0),
                                Point3f(1, 0, 1),
                                Point3f(1, 1, 0),
                                Point3f(1, 1, 1) };

    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

    float vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = tsdfToFloat(at(neighbourCoords[i] * voxelSize + point).tsdf);

    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);
}

inline Point3f HashTSDFVolumeCPU::getNormalVoxel(Point3f point) const
{
    Vec3f pointVec(point);
    Vec3f normal = Vec3f(0, 0, 0);

    Vec3f pointPrev = point;
    Vec3f pointNext = point;

    for (int c = 0; c < 3; c++)
    {
        pointPrev[c] -= voxelSize;
        pointNext[c] += voxelSize;

#if USE_INTERPOLATION_IN_GETNORMAL
        normal[c] = interpolateVoxel(Point3f(pointNext)) - interpolateVoxel(Point3f(pointPrev));
#else
        normal[c] = tsdfToFloat(at(Point3f(pointNext)).tsdf - at(Point3f(pointPrev)).tsdf);
#endif
        pointPrev[c] = pointVec[c];
        pointNext[c] = pointVec[c];
    }
    
    float nv = sqrt(normal[0] * normal[0] +
                     normal[1] * normal[1] +
                     normal[2] * normal[2]);
    return nv < 0.0001f ? nan3 : normal/nv;
}

inline TsdfVoxel HashTSDFVolumeCPU::_at(const cv::Point3f& point) const
{
    //cv::Vec3i volumeUnitIdx = volumeToVolumeUnitIdx(point);
    cv::Vec3i volumeUnitIdx = cv::Vec3i(cvFloor(point.x / volumeUnitSize),
        cvFloor(point.y / volumeUnitSize),
        cvFloor(point.z / volumeUnitSize));

    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    
    cv::Ptr<TSDFVolumeCPU> volumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);
    //cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Point3f volumeUnitPos = cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, 
                                            volumeUnitIdx[1] * volumeUnitSize,
                                            volumeUnitIdx[2] * volumeUnitSize);
    //cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    cv::Point3f _point = point - volumeUnitPos;
    cv::Vec3i volUnitLocalIdx = cv::Vec3i(cvFloor(_point.x * voxelSizeInv), 
                                          cvFloor(_point.y * voxelSizeInv),
                                          cvFloor(_point.z * voxelSizeInv));
    
    volUnitLocalIdx = cv::Vec3i(abs(volUnitLocalIdx[0]), 
                                abs(volUnitLocalIdx[1]), 
                                abs(volUnitLocalIdx[2]));

    const TsdfVoxel* volData = volumeUnit->volume.ptr<TsdfVoxel>();
    Vec4i volDims = volumeUnit->volDims;
    int coordBase = volUnitLocalIdx[0] * volDims[0] + 
                    volUnitLocalIdx[1] * volDims[1] + 
                    volUnitLocalIdx[2] * volDims[2];
    return volData[coordBase];
}

inline TsdfVoxel HashTSDFVolumeCPU::__at(const cv::Point3f& point, cv::Vec3i volumeUnitIdx) const
{
    //cv::Vec3i volumeUnitIdx = volumeToVolumeUnitIdx(point);
    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    
    cv::Ptr<TSDFVolumeCPU> volumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);
    //cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Point3f volumeUnitPos = cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, 
                                            volumeUnitIdx[1] * volumeUnitSize,
                                            volumeUnitIdx[2] * volumeUnitSize);
    //cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    cv::Point3f _point = point - volumeUnitPos;
    cv::Vec3i volUnitLocalIdx = cv::Vec3i(cvFloor(_point.x * voxelSizeInv), 
                                          cvFloor(_point.y * voxelSizeInv),
                                          cvFloor(_point.z * voxelSizeInv));
    
    volUnitLocalIdx = cv::Vec3i(abs(volUnitLocalIdx[0]), 
                                abs(volUnitLocalIdx[1]), 
                                abs(volUnitLocalIdx[2]));

    const TsdfVoxel* volData = volumeUnit->volume.ptr<TsdfVoxel>();
    Vec4i volDims = volumeUnit->volDims;
    int coordBase = volUnitLocalIdx[0] * volDims[0] + 
                    volUnitLocalIdx[1] * volDims[1] + 
                    volUnitLocalIdx[2] * volDims[2];
    return volData[coordBase];
}

inline TsdfVoxel HashTSDFVolumeCPU::___at(const cv::Point3f& point, cv::Vec3i volumeUnitIdx, VolumeUnitMap::const_iterator it) const
{
    if (it == volumeUnits.end())
    {
        TsdfVoxel dummy;
        dummy.tsdf = floatToTsdf(1.f);
        dummy.weight = 0;
        return dummy;
    }
    
    cv::Ptr<TSDFVolumeCPU> volumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);
    //cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Point3f volumeUnitPos = cv::Point3f(volumeUnitIdx[0] * volumeUnitSize, 
                                            volumeUnitIdx[1] * volumeUnitSize,
                                            volumeUnitIdx[2] * volumeUnitSize);
    //cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    cv::Point3f _point = point - volumeUnitPos;
    cv::Vec3i volUnitLocalIdx = cv::Vec3i(cvFloor(_point.x * voxelSizeInv), 
                                          cvFloor(_point.y * voxelSizeInv),
                                          cvFloor(_point.z * voxelSizeInv));
    
    volUnitLocalIdx = cv::Vec3i(abs(volUnitLocalIdx[0]), 
                                abs(volUnitLocalIdx[1]), 
                                abs(volUnitLocalIdx[2]));

    const TsdfVoxel* volData = volumeUnit->volume.ptr<TsdfVoxel>();
    Vec4i volDims = volumeUnit->volDims;
    int coordBase = volUnitLocalIdx[0] * volDims[0] + 
                    volUnitLocalIdx[1] * volDims[1] + 
                    volUnitLocalIdx[2] * volDims[2];
    return volData[coordBase];
}

inline float HashTSDFVolumeCPU::_interpolate(const cv::Point3f& point, float vx[8]) const
{
    int ix = cvFloor(point.x);
    int iy = cvFloor(point.y);
    int iz = cvFloor(point.z);

    float tx = point.x - ix;
    float ty = point.y - iy;
    float tz = point.z - iz;

#if USE_INTRINSICS
    v_float32x4 v0246(vx[0], vx[2], vx[4], vx[6]);
    v_float32x4 v1357(vx[1], vx[3], vx[5], vx[7]);

    v_float32x4 vxx = v0246 + v_setall_f32(tz) * (v1357 - v0246);

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v00_10 + v_setall_f32(ty) * (v01_11 - v00_10);
    float v0 = v0_1.get0();
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v0_1.get0();
#else
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);
#endif

    return v0 + tx * (v1 - v0);
}

bool HashTSDFVolumeCPU::vuiCheck(cv::Vec3i vui[8]) const
{
    for (int i = 1; i < 8; i++)
        if (vui[i] != vui[0])
            return false;
    return true;
}


inline Point3f HashTSDFVolumeCPU::_getNormalVoxel(Point3f point) const
{
    cv::Point3f neighbourCoords[] = {
                                Point3f(0, 0, 0),
                                Point3f(0, 0, 1),
                                Point3f(0, 1, 0),
                                Point3f(0, 1, 1),
                                Point3f(1, 0, 0),
                                Point3f(1, 0, 1),
                                Point3f(1, 1, 0),
                                Point3f(1, 1, 1) };
    for (int i = 0; i < 8; i++)
        neighbourCoords[i] *= voxelSize;

    Point3f pointVec(point);
    Point3f normal = Point3f(0, 0, 0);

    Point3f c0_pointPrev = pointVec, c0_pointNext = pointVec;
    Point3f c1_pointPrev = pointVec, c1_pointNext = pointVec;
    Point3f c2_pointPrev = pointVec, c2_pointNext = pointVec;

    c0_pointPrev.x -= voxelSize; c0_pointNext.x += voxelSize;
    c1_pointPrev.y -= voxelSize; c1_pointNext.y += voxelSize;
    c2_pointPrev.z -= voxelSize; c2_pointNext.z += voxelSize;

    float c0_n_vx[8], c1_n_vx[8], c2_n_vx[8];
    float c0_p_vx[8], c1_p_vx[8], c2_p_vx[8];

    cv::Vec3i c0_n_vui[8], c0_p_vui[8];

    for (int i = 0; i < 8; i++)
        c0_n_vui[i] = volumeToVolumeUnitIdx(neighbourCoords[i] + c0_pointNext);
    if (vuiCheck(c0_n_vui))
    {
        VolumeUnitMap::const_iterator it = volumeUnits.find(c0_n_vui[0]);
        for (int i = 0; i < 8; i++)
            c0_n_vx[i] = tsdfToFloat(___at(neighbourCoords[i] + c0_pointNext, c0_n_vui[0], it).tsdf);
    }
    else {
        for (int i = 0; i < 8; i++)
            //c0_n_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c0_pointNext).tsdf);
            c0_n_vx[i] = tsdfToFloat(__at(neighbourCoords[i] + c0_pointNext, c0_n_vui[i]).tsdf);
    }

    for (int i = 0; i < 8; i++)
        c0_p_vui[i] = volumeToVolumeUnitIdx(neighbourCoords[i] + c0_pointPrev);
    if (vuiCheck(c0_p_vui))
    {
        VolumeUnitMap::const_iterator it = volumeUnits.find(c0_p_vui[0]);
        for (int i = 0; i < 8; i++)
            c0_p_vx[i] = tsdfToFloat(___at(neighbourCoords[i] + c0_pointPrev, c0_p_vui[0], it).tsdf);
    } else {
        for (int i = 0; i < 8; i++)
            //c0_p_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c0_pointPrev).tsdf);
            c0_p_vx[i] = tsdfToFloat(__at(neighbourCoords[i] + c0_pointPrev, c0_p_vui[i]).tsdf);
    }
    for (int i = 0; i < 8; i++)
        if (i % 4 > 1)
            //c1_n_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c1_pointNext).tsdf);
            c1_n_vx[i] = tsdfToFloat(_at(neighbourCoords[i] + c1_pointNext).tsdf);
        else 
            //c1_p_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c1_pointPrev).tsdf);
            c1_p_vx[i] = tsdfToFloat(_at(neighbourCoords[i] + c1_pointPrev).tsdf);

    for (int i = 0; i < 8; i++)
        if (i % 2 == 1)
            //c2_n_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c2_pointNext).tsdf);
            c2_n_vx[i] = tsdfToFloat(_at(neighbourCoords[i] + c2_pointNext).tsdf);
        else
            //c2_p_vx[i] = tsdfToFloat(at(neighbourCoords[i] + c2_pointPrev).tsdf);
            c2_p_vx[i] = tsdfToFloat(_at(neighbourCoords[i] + c2_pointPrev).tsdf);

    c1_n_vx[0] = c0_p_vx[6];
    c1_n_vx[1] = c0_p_vx[7];
    c1_p_vx[2] = c0_n_vx[2];
    c1_p_vx[3] = c0_n_vx[3];
    c1_n_vx[4] = c0_p_vx[4];
    c1_n_vx[5] = c0_p_vx[7];
    c1_p_vx[6] = c0_n_vx[0];
    c1_p_vx[7] = c0_n_vx[1];

    c2_n_vx[0] = c0_p_vx[5];
    c2_p_vx[1] = c0_p_vx[7];
    c2_n_vx[2] = c0_n_vx[1];
    c2_p_vx[3] = c0_n_vx[3];
    c2_n_vx[4] = c0_p_vx[4];
    c2_p_vx[5] = c0_p_vx[6];
    c2_n_vx[6] = c0_n_vx[0];
    c2_p_vx[7] = c0_n_vx[2];

    normal.x = _interpolate(c0_pointNext, c0_n_vx) - _interpolate(c0_pointPrev, c0_p_vx);
    normal.y = _interpolate(c1_pointNext, c1_n_vx) - _interpolate(c1_pointPrev, c1_p_vx);
    normal.z = _interpolate(c2_pointNext, c2_n_vx) - _interpolate(c2_pointPrev, c2_p_vx);

    float nv = sqrt(normal.x * normal.x +
        normal.y * normal.y +
        normal.z * normal.z);
    return nv < 0.0001f ? nan3 : normal / nv;
}

struct HashRaycastInvoker : ParallelLoopBody
{
    HashRaycastInvoker(Points& _points, Normals& _normals, const Matx44f& cameraPose,
                       const Intr& intrinsics, const HashTSDFVolumeCPU& _volume)
        : ParallelLoopBody(),
          points(_points),
          normals(_normals),
          volume(_volume),
          tstep(_volume.truncDist * _volume.raycastStepFactor),
          cam2vol(volume.pose.inv() * Affine3f(cameraPose)),
          vol2cam(Affine3f(cameraPose.inv()) * volume.pose),
          reproj(intrinsics.makeReprojector())
    {
    }

    virtual void operator()(const Range& range) const override
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot   = cam2vol.rotation();
        const Matx33f vol2camRot   = vol2cam.rotation();

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
                Point3f orig = cam2volTrans;
                Point3f rayDirV =
                    normalize(Vec3f(cam2volRot * reproj(Point3f(float(x), float(y), 1.f))));

                float tmin  = 0;
                float tmax  = volume.truncateThreshold;
                float tcurr = tmin;

                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::min());

                float tprev       = tcurr;
                float prevTsdf = volume.truncDist;
                cv::Ptr<TSDFVolumeCPU> currVolumeUnit;
                while (tcurr < tmax)
                {
                    Point3f currRayPos          = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volume.volumeToVolumeUnitIdx(currRayPos);

                    VolumeUnitMap::const_iterator it = volume.volumeUnits.find(currVolumeUnitIdx);

                    float currTsdf = prevTsdf;
                    int currWeight    = 0;
                    float stepSize    = 0.5f * blockSize;
                    cv::Vec3i volUnitLocalIdx;

                    //! Does the subvolume exist in hashtable
                    if (it != volume.volumeUnits.end())
                    {
                        currVolumeUnit =
                            std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);
                        cv::Point3f currVolUnitPos =
                            volume.volumeUnitIdxToVolume(currVolumeUnitIdx);
                        volUnitLocalIdx = volume.volumeToVoxelCoord(currRayPos - currVolUnitPos);

                        //! TODO: Figure out voxel interpolation
                        TsdfVoxel currVoxel = currVolumeUnit->at(volUnitLocalIdx);
                        currTsdf            = tsdfToFloat(currVoxel.tsdf);
                        currWeight          = currVoxel.weight;
                        stepSize            = tstep;
                    }
                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp =
                            (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            //Point3f nv = volume.getNormalVoxel(pv);
                            Point3f nv = volume._getNormalVoxel(pv);

                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point  = vol2cam * pv;
                            }
                        }
                        break;
                    }
                    prevVolumeUnitIdx = currVolumeUnitIdx;
                    prevTsdf          = currTsdf;
                    tprev             = tcurr;
                    tcurr += stepSize;
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    }

    Points& points;
    Normals& normals;
    const HashTSDFVolumeCPU& volume;
    const float tstep;
    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};

void HashTSDFVolumeCPU::raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
                                cv::Size frameSize, cv::OutputArray _points,
                                cv::OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points   = _points.getMat();
    Normals normals = _normals.getMat();

    HashRaycastInvoker ri(points, normals, cameraPose, intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}

struct HashFetchPointsNormalsInvoker : ParallelLoopBody
{
    HashFetchPointsNormalsInvoker(const HashTSDFVolumeCPU& _volume,
                              const std::vector<Vec3i>& _totalVolUnits,
                              std::vector<std::vector<ptype>>& _pVecs,
                              std::vector<std::vector<ptype>>& _nVecs, bool _needNormals)
        : ParallelLoopBody(),
          volume(_volume),
          totalVolUnits(_totalVolUnits),
          pVecs(_pVecs),
          nVecs(_nVecs),
          needNormals(_needNormals)
    {
    }

    virtual void operator()(const Range& range) const override
    {
        std::vector<ptype> points, normals;
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3i tsdf_idx = totalVolUnits[i];

            VolumeUnitMap::const_iterator it = volume.volumeUnits.find(tsdf_idx);
            Point3f base_point               = volume.volumeUnitIdxToVolume(tsdf_idx);
            if (it != volume.volumeUnits.end())
            {
                cv::Ptr<TSDFVolumeCPU> volumeUnit =
                    std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.pVolume);
                std::vector<ptype> localPoints;
                std::vector<ptype> localNormals;
                for (int x = 0; x < volume.volumeUnitResolution; x++)
                    for (int y = 0; y < volume.volumeUnitResolution; y++)
                        for (int z = 0; z < volume.volumeUnitResolution; z++)
                        {
                            cv::Vec3i voxelIdx(x, y, z);
                            TsdfVoxel voxel = volumeUnit->at(voxelIdx);

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
    }

    const HashTSDFVolumeCPU& volume;
    std::vector<cv::Vec3i> totalVolUnits;
    std::vector<std::vector<ptype>>& pVecs;
    std::vector<std::vector<ptype>>& nVecs;
    const TsdfVoxel* volDataStart;
    bool needNormals;
    mutable Mutex mutex;
};

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
        HashFetchPointsNormalsInvoker fi(*this, totalVolUnits, pVecs, nVecs, _normals.needed());
        Range range(0, (int)totalVolUnits.size());
        const int nstripes = -1;
        parallel_for_(range, fi, nstripes);
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

void HashTSDFVolumeCPU::fetchNormals(cv::InputArray _points, cv::OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();

        const HashTSDFVolumeCPU& _volume = *this;
        auto HashPushNormals = [&](const ptype& point, const int* position)
        {
            const HashTSDFVolumeCPU& volume(_volume);
            Affine3f invPose(volume.pose.inv());
            Point3f p = fromPtype(point);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxelPoint = invPose * p;
                n = volume.pose.rotation() * volume.getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }
}

cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Matx44f _pose,
                                           float _raycastStepFactor, float _truncDist,
                                           int _maxWeight, float _truncateThreshold,
                                           int _volumeUnitResolution)
{
    return cv::makePtr<HashTSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist,
                                          _maxWeight, _truncateThreshold, _volumeUnitResolution);
}

}  // namespace kinfu
}  // namespace cv
