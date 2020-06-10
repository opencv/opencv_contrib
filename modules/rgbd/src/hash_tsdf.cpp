// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory
#include "hash_tsdf.hpp"

#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/utils/trace.hpp"

namespace cv
{
namespace kinfu
{
HashTSDFVolume::HashTSDFVolume(float _voxelSize, int _volume_unit_res, cv::Affine3f _pose,
                               float _truncDist, int _maxWeight, float _raycastStepFactor,
                               bool _zFirstMemOrder)
    : voxelSize(_voxelSize),
      voxelSizeInv(1.0f / _voxelSize),
      pose(_pose),
      maxWeight(_maxWeight),
      raycastStepFactor(_raycastStepFactor),
      volumeUnitResolution(_volume_unit_res),
      volumeUnitSize(voxelSize * volumeUnitResolution),
      zFirstMemOrder(_zFirstMemOrder)
{
    truncDist = std::max(_truncDist, 2.1f * voxelSize);
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(float _voxelSize, int _volume_unit_res, cv::Affine3f _pose,
                                     float _truncDist, int _maxWeight, float _raycastStepFactor,
                                     bool _zFirstMemOrder)
    : HashTSDFVolume(_voxelSize, _volume_unit_res, _pose, _truncDist, _maxWeight,
                     _raycastStepFactor, _zFirstMemOrder)
{
}

// zero volume, leave rest params the same
void HashTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();
    volumeUnits.clear();
}

struct AccessedVolumeUnitsInvoker : ParallelLoopBody
{
    AccessedVolumeUnitsInvoker(HashTSDFVolumeCPU& _volume, VolumeUnitIndexSet& _accessVolUnits,
                               const Depth& _depth, Intr intrinsics, cv::Affine3f cameraPose,
                               float _depthFactor, int _depthStride = 4)
        : ParallelLoopBody(),
          volume(_volume),
          accessVolUnits(_accessVolUnits),
          depth(_depth),
          reproj(intrinsics.makeReprojector()),
          cam2vol(_volume.pose.inv() * cameraPose),
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
                            AutoLock al(mutex);
                            if (!localAccessVolUnits.count(tsdf_idx))
                            {
                                localAccessVolUnits.emplace(tsdf_idx);
                            }
                        }
                AutoLock al(mutex);
                for (const auto& tsdf_idx : localAccessVolUnits)
                {
                    if (accessVolUnits.emplace(tsdf_idx).second)
                    {
                        VolumeUnit volumeUnit;
                        cv::Point3i volumeDims(volume.volumeUnitResolution,
                                               volume.volumeUnitResolution,
                                               volume.volumeUnitResolution);

                        cv::Affine3f subvolumePose =
                            volume.pose.translate(volume.volumeUnitIdxToVolume(tsdf_idx));
                        volumeUnit.pVolume = cv::makePtr<TSDFVolumeCPU>(
                            volumeDims, volume.voxelSize, subvolumePose, volume.truncDist,
                            volume.maxWeight, volume.raycastStepFactor);
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
    VolumeUnitIndexSet& accessVolUnits;
    const Depth& depth;
    const Intr::Reprojector reproj;
    const cv::Affine3f cam2vol;
    const float depthFactor;
    const int depthStride;
    mutable Mutex mutex;
};

struct IntegrateSubvolumeInvoker : ParallelLoopBody
{
    IntegrateSubvolumeInvoker(HashTSDFVolumeCPU& _volume,
                              const std::vector<cv::Vec3i>& _totalVolUnits, const Depth& _depth,
                              Intr _intrinsics, cv::Affine3f _cameraPose, float _depthFactor)
        : ParallelLoopBody(),
          volume(_volume),
          totalVolUnits(_totalVolUnits),
          depth(_depth),
          depthFactor(_depthFactor),
          cameraPose(_cameraPose),
          intrinsics(_intrinsics)
    {
    }

    virtual void operator()(const Range& range) const override
    {
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3i tsdf_idx         = totalVolUnits[i];
            VolumeUnitMap::iterator it = volume.volumeUnits.find(tsdf_idx);
            if (it == volume.volumeUnits.end())
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
    }

    HashTSDFVolumeCPU& volume;
    std::vector<cv::Vec3i> totalVolUnits;
    const Depth& depth;
    float depthFactor;
    cv::Affine3f cameraPose;
    Intr intrinsics;
    mutable Mutex mutex;
};

struct VolumeUnitInFrustumInvoker : ParallelLoopBody
{
    VolumeUnitInFrustumInvoker(HashTSDFVolumeCPU& _volume,
                               const std::vector<cv::Vec3i>& _totalVolUnits, int& _noOfActiveVolumeUnits, const Depth& _depth,
                               Intr _intrinsics, cv::Affine3f _cameraPose, float _depthFactor)
        : ParallelLoopBody(),
          volume(_volume),
          totalVolUnits(_totalVolUnits),
          noOfActiveVolumeUnits(_noOfActiveVolumeUnits),
          depth(_depth),
          proj(_intrinsics.makeProjector()),
          depthFactor(_depthFactor),
          vol2cam(_cameraPose.inv() * _volume.pose)
    {
    }

    virtual void operator()(const Range& range) const override
    {
        for (int i = range.start; i < range.end; ++i)
        {
            cv::Vec3i tsdf_idx         = totalVolUnits[i];
            VolumeUnitMap::iterator it = volume.volumeUnits.find(tsdf_idx);
            if (it == volume.volumeUnits.end())
                return;

            Point3f volumeUnitPos     = volume.volumeUnitIdxToVolume(it->first);
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;
            if (volUnitInCamSpace.z < rgbd::Odometry::DEFAULT_MIN_DEPTH() ||
                volUnitInCamSpace.z > rgbd::Odometry::DEFAULT_MAX_DEPTH())
            {
                it->second.isActive = false;
                return;
            }
            Point2f cameraPoint = proj(volUnitInCamSpace);
            if (cameraPoint.x >= 0 && cameraPoint.y >= 0 && cameraPoint.x < depth.cols &&
                cameraPoint.y < depth.rows)
            {
                assert(it != volume.volumeUnits.end());
                it->second.isActive = true;
            }
            AutoLock al(mutex);
            if(it->second.isActive)
            {
                noOfActiveVolumeUnits++;
            }
        }
    }
    HashTSDFVolumeCPU& volume;
    const std::vector<cv::Vec3i> totalVolUnits;
    int& noOfActiveVolumeUnits;
    const Depth& depth;
    const Intr::Projector proj;
    const float depthFactor;
    const Affine3f vol2cam;
    mutable Mutex mutex;
};

void HashTSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose,
                                  Intr intrinsics)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();
    VolumeUnitIndexSet accessVolUnits;
    AccessedVolumeUnitsInvoker allocate_i(*this, accessVolUnits, depth, intrinsics, cameraPose,
                                          depthFactor);
    Range range(0, depth.rows);
    parallel_for_(range, allocate_i);

    std::vector<Vec3i> totalVolUnits;
    for (const auto& keyvalue : volumeUnits)
    {
        totalVolUnits.push_back(keyvalue.first);
    }
    int noOfActiveVolumeUnits = 0;
    VolumeUnitInFrustumInvoker infrustum_i(*this, totalVolUnits, noOfActiveVolumeUnits, depth, intrinsics, cameraPose,
                                           depthFactor);
    Range in_frustum_range(0, volumeUnits.size());
    parallel_for_(in_frustum_range, infrustum_i);

    std::cout << "Number of new volume units created: " << accessVolUnits.size() << "\n";
    std::cout << " Number of total units in volume: " << volumeUnits.size() << " " << totalVolUnits.size() << "\n";
    std::cout << " Number of Active volume units in frame: " << noOfActiveVolumeUnits << "\n";

    IntegrateSubvolumeInvoker integrate_i(*this, totalVolUnits, depth, intrinsics, cameraPose,
                                          depthFactor);
    Range accessed_units_range(0, totalVolUnits.size());
    parallel_for_(accessed_units_range, integrate_i);
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

inline Voxel HashTSDFVolumeCPU::at(const cv::Vec3i& volumeIdx) const
{
    cv::Vec3i volumeUnitIdx = cv::Vec3i(cvFloor(volumeIdx[0] / volumeUnitResolution),
                                        cvFloor(volumeIdx[1] / volumeUnitResolution),
                                        cvFloor(volumeIdx[2] / volumeUnitResolution));

    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    if (it == volumeUnits.end())
    {
        Voxel dummy;
        dummy.tsdf   = 1.f;
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
    /* std::cout << "VolumeIdx: " << volumeIdx << " VolumeUnitIdx: " << volumeUnitIdx << " Volume
     * Local unit idx: " << volUnitLocalIdx << "\n"; */
    return volumeUnit->at(volUnitLocalIdx);
}

inline Voxel HashTSDFVolumeCPU::at(const cv::Point3f& point) const
{
    cv::Vec3i volumeUnitIdx          = volumeToVolumeUnitIdx(point);
    VolumeUnitMap::const_iterator it = volumeUnits.find(volumeUnitIdx);
    if (it == volumeUnits.end())
    {
        Voxel dummy;
        dummy.tsdf   = 1.f;
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

inline TsdfType HashTSDFVolumeCPU::interpolateVoxel(const cv::Point3f& point) const
{
    cv::Vec3i voxelIdx = volumeToVoxelCoord(point);

    Point3f t = point - Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize);

    std::array<cv::Vec3i, 8> neighbourCoords = { cv::Vec3i(0, 0, 0), cv::Vec3i(0, 0, 1),
                                                 cv::Vec3i(0, 1, 0), cv::Vec3i(0, 1, 1),
                                                 cv::Vec3i(1, 0, 0), cv::Vec3i(1, 0, 1),
                                                 cv::Vec3i(1, 1, 0), cv::Vec3i(1, 1, 1) };

    std::array<TsdfType, 8> tsdf_array;
    for (int i = 0; i < 8; ++i)
    {
        Voxel voxel = at(neighbourCoords[i] + voxelIdx);
        /* std::cout << "VoxelIdx : " << voxelIdx << " voxel weight: " << voxel.weight << " voxel value: " << voxel.tsdf << "\n"; */
        /* if(voxel.weight != 0) */
            tsdf_array[i] = voxel.tsdf;
    }

    TsdfType v00 = tsdf_array[0] + t.z * (tsdf_array[1] - tsdf_array[0]);
    TsdfType v01 = tsdf_array[2] + t.z * (tsdf_array[3] - tsdf_array[2]);
    TsdfType v10 = tsdf_array[4] + t.z * (tsdf_array[5] - tsdf_array[4]);
    TsdfType v11 = tsdf_array[6] + t.z * (tsdf_array[7] - tsdf_array[6]);

    TsdfType v0 = v00 + t.y * (v01 - v00);
    TsdfType v1 = v10 + t.y * (v11 - v10);
    /* std::cout << " interpolation t: " << t << std::endl; */
    return v0 + t.x * (v1 - v0);
}

inline Point3f HashTSDFVolumeCPU::getNormalVoxel(Point3f point) const
{
    /* cv::Vec3i voxelIdx = volumeToVoxelCoord(point); */

    /* Vec3i prev   = voxelIdx; */
    /* Vec3i next   = voxelIdx; */
    /* Vec3f normal = Vec3f(0, 0, 0); */

    /* for (int c = 0; c < 3; c++) */
    /* { */
    /*     prev[c] -= 1; */
    /*     next[c] += 1; */

    /*     normal[c] = at(next).tsdf - at(prev).tsdf; */
    /*     normal[c] *= 0.5; */

    /*     prev[c] = voxelIdx[c]; */
    /*     next[c] = voxelIdx[c]; */
    /* } */
    /* return normalize(normal); */

    Vec3f pointVec(point);
    Vec3f normal = Vec3f(0, 0, 0);

    Vec3f pointPrev = point;
    Vec3f pointNext = point;

    for (int c = 0; c < 3; c++)
    {
        pointPrev[c] -= voxelSize * 0.5;
        pointNext[c] += voxelSize * 0.5;

        normal[c] = at(Point3f(pointNext)).tsdf - at(Point3f(pointPrev)).tsdf;
        /* std::cout << "pointPrev, pointNext: " << at(Point3f(pointPrev)).tsdf << ", " <<
         * at(Point3f(pointNext)).tsdf << std::endl; */
        normal[c] *= 0.5;
        /* std::cout << "TSDF diff: " << normal[c] << std::endl; */

        pointPrev[c] = pointVec[c];
        pointNext[c] = pointVec[c];
    }
    return normalize(normal);
}

/* inline Point3f HashTSDFVolumeCPU::getNormalVoxel(Point3f point) const */
/* { */
/*     cv::Vec3i voxelIdx = volumeToVoxelCoord(point); */
/*     cv::Vec3i next = voxelIdx; */
/*     cv::Vec3i prev = voxelIdx; */
/*     Point3f t = point - Point3f(voxelIdx[0] * voxelSize, voxelIdx[1] * voxelSize, voxelIdx[2] * voxelSize); */

/*     std::array<cv::Vec3i, 8> neighbourCoords = { */
/*         cv::Vec3i(0, 0, 0), */
/*         cv::Vec3i(0, 0, 1), */
/*         cv::Vec3i(0, 1, 0), */
/*         cv::Vec3i(0, 1, 1), */
/*         cv::Vec3i(1, 0, 0), */
/*         cv::Vec3i(1, 0, 1), */
/*         cv::Vec3i(1, 1, 0), */
/*         cv::Vec3i(1, 1, 1)}; */

/*     Vec3f normal; */
/*     for(int c = 0; c < 3; c++) */
/*     { */
/*         float& nv = normal[c]; */

/*         next[c] += 1; */
/*         prev[c] -= 1; */

/*         std::array<TsdfType, 8> vx; */
/*         for(int i = 0; i < 8; i++) */
/*             vx[i] = at(neighbourCoords[i] + next).tsdf - */
/*                     at(neighbourCoords[i] + prev).tsdf; */

/*         TsdfType v00 = vx[0] + t.z*(vx[1] - vx[0]); */
/*         TsdfType v01 = vx[2] + t.z*(vx[3] - vx[2]); */
/*         TsdfType v10 = vx[4] + t.z*(vx[5] - vx[4]); */
/*         TsdfType v11 = vx[6] + t.z*(vx[7] - vx[6]); */

/*         TsdfType v0 = v00 + t.y*(v01 - v00); */
/*         TsdfType v1 = v10 + t.y*(v11 - v10); */

/*         nv = v0 + t.x*(v1 - v0); */

/*         next[c] = voxelIdx[c]; */
/*         prev[c] = voxelIdx[c]; */
/*     } */

/*     return normalize(normal); */
/* } */

struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, Affine3f cameraPose, Intr intrinsics,
                   const HashTSDFVolumeCPU& _volume)
        : ParallelLoopBody(),
          points(_points),
          normals(_normals),
          volume(_volume),
          tstep(_volume.truncDist * _volume.raycastStepFactor),
          cam2vol(volume.pose.inv() * cameraPose),
          vol2cam(cameraPose.inv() * volume.pose),
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
                Point3f orig    = cam2volTrans;
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(x, y, 1.f))));

                float tmin = rgbd::Odometry::DEFAULT_MIN_DEPTH() / rayDirV.z;
                float tmax = rgbd::Odometry::DEFAULT_MAX_DEPTH() / rayDirV.z;

                float tprev = tmin;
                float tcurr = tmin;

                TsdfType prevTsdf = volume.truncDist;
                cv::Vec3i prevVolumeUnitIdx =
                    cv::Vec3i(std::numeric_limits<int>::min(), std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::min());
                cv::Ptr<TSDFVolumeCPU> currVolumeUnit;
                while (tcurr < tmax)
                {
                    Point3f currRayPos          = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx = volume.volumeToVolumeUnitIdx(currRayPos);

                    VolumeUnitMap::const_iterator it = volume.volumeUnits.find(currVolumeUnitIdx);

                    TsdfType currTsdf = prevTsdf;
                    int currWeight    = 0;
                    float stepSize    = blockSize;
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
                        Voxel currVoxel = currVolumeUnit->at(volUnitLocalIdx);
                        currTsdf        = currVoxel.tsdf;
                        /* currTsdf   = volume.interpolateVoxel(currRayPos); */
                        /* std::cout << "Current TSDF: " << currTsdf << " interpolated: " << interpolatedTsdf << "\n"; */
                        currWeight = currVoxel.weight;
                        stepSize   = tstep;
                    }
                    //! Surface crossing
                    if (prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        float tInterp =
                            (tcurr * prevTsdf - tprev * currTsdf) / (prevTsdf - currTsdf);
                        /* std::cout << "tcurr: " << tcurr << " tprev: " << tprev << " tInterp: " << tInterp << "\n"; */
                        if (!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = volume.getNormalVoxel(pv);

                            if (!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point  = vol2cam * pv;
                                break;
                            }
                        }
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

void HashTSDFVolumeCPU::raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics,
                                cv::Size frameSize, cv::OutputArray _points,
                                cv::OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create(frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points   = _points.getMat();
    Normals normals = _normals.getMat();

    RaycastInvoker ri(points, normals, cameraPose, intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}

struct FetchPointsNormalsInvoker : ParallelLoopBody
{
    FetchPointsNormalsInvoker(const HashTSDFVolumeCPU& _volume,
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
                            Voxel voxel = volumeUnit->at(voxelIdx);

                            if (voxel.tsdf != 1.f && voxel.weight != 0)
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
    const Voxel* volDataStart;
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
        std::cout << "Number of volumeUnits in volume(fetchPoints): " << totalVolUnits.size() << "\n";
        FetchPointsNormalsInvoker fi(*this, totalVolUnits, pVecs, nVecs, _normals.needed());
        Range range(0, totalVolUnits.size());
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

cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _truncDist,
                                           int _maxWeight, float _raycastStepFactor,
                                           int volumeUnitResolution)
{
    return cv::makePtr<HashTSDFVolumeCPU>(_voxelSize, volumeUnitResolution, _pose, _truncDist,
                                          _maxWeight, _raycastStepFactor);
}

}  // namespace kinfu
}  // namespace cv
