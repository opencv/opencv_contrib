// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd_wrapper.hpp"
#include "opencv2/core/fast_math.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/utils/trace.hpp"
#include "opencv2/rgbd/depth.hpp"
#include "precomp.hpp"
#include "hash_tsdf.hpp"
#include "opencl_kernels_rgbd.hpp"
#include <functional>
#include <limits>
#include "depth_to_3d.hpp"
#include "utils.hpp"

namespace cv {

namespace kinfu {


HashTSDFVolume::HashTSDFVolume(float _voxelSize, int _volume_unit_res, cv::Affine3f _pose, float _truncDist,
        int _maxWeight, float _raycastStepFactor, bool _zFirstMemOrder)
    : voxelSize(_voxelSize)
    , voxelSizeInv(1.0f / _voxelSize)
    , pose(_pose)
    , maxWeight(_maxWeight)
    , raycastStepFactor(_raycastStepFactor)
    , volumeUnitResolution(_volume_unit_res)
    , volumeUnitSize(voxelSize * volumeUnitResolution)
    , zFirstMemOrder(_zFirstMemOrder)
{
    truncDist = std::max(_truncDist, 2.1f * voxelSize);
}

HashTSDFVolumeCPU::HashTSDFVolumeCPU(float _voxelSize, int _volume_unit_res, cv::Affine3f _pose,
                      float _truncDist, int _maxWeight,
                      float _raycastStepFactor, bool _zFirstMemOrder)
    : HashTSDFVolume(_voxelSize, _volume_unit_res, _pose, _truncDist, _maxWeight, _raycastStepFactor, _zFirstMemOrder)
{
}

// zero volume, leave rest params the same
void HashTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();
    volume_units_.clear();
}

struct AccessedVolumeUnitsInvoker : ParallelLoopBody
{
    AccessedVolumeUnitsInvoker(HashTSDFVolumeCPU& _volume, VolumeUnitIndexSet& _accessVolUnits,
            const Depth& _depth, Intr intrinsics, cv::Affine3f cameraPose,
            float _depthFactor, int _depthStride = 4) :
        ParallelLoopBody(),
        volume(_volume),
        accessVolUnits(_accessVolUnits),
        depth(_depth),
        reproj(intrinsics.makeReprojector()),
        cam2vol(_volume.pose.inv() * cameraPose),
        dfac(1.0f/_depthFactor),
        depthStride(_depthStride)
    {
    }

    virtual void operator() (const Range& range) const override
    {
        for(int y = range.start; y < range.end; y += depthStride)
        {
            const depthType *depthRow = depth[y];
            for(int x = 0; x < depth.cols; x += depthStride)
            {
                depthType z = depthRow[x]*dfac;
                if (z <= 0)
                    continue;

                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;

                /* std::cout << "volPoint" << volPoint << "\n"; */

                //! Find accessed TSDF volume unit for valid 3D vertex
                cv::Vec3i lower_bound = volume.volumeToVolumeUnitIdx(
                        volPoint - cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));
                cv::Vec3i upper_bound = volume.volumeToVolumeUnitIdx(
                        volPoint + cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));

                //! TODO(Akash): Optimize this using differential analyzer algorithm
                for(int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for(int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for(int k = lower_bound[2]; k <= lower_bound[2]; k++)
                        {
                            const cv::Vec3i tsdf_idx = cv::Vec3i(i, j, k);
                            //! If the index does not exist
                            AutoLock al(mutex);
                            if(!accessVolUnits.count(tsdf_idx))
                            {
                                accessVolUnits.insert(tsdf_idx);
                                /* std::cout << "CamPoint: " << camPoint << " volPoint: " << volPoint << "\n"; */
                                /* std::cout << "Inserted tsdf_idx: (" << tsdf_idx[0] << ", " << tsdf_idx[1] << ", " << tsdf_idx[2] << ")\n"; */
                                //! Adds entry to unordered_map
                                //! and allocate memory for the volume unit
                                VolumeUnit volumeUnit;
                                /* std::cout << "Allocated volumeUnit in map" << std::endl; */
                                cv::Point3i volumeDims(volume.volumeUnitResolution,
                                                           volume.volumeUnitResolution,
                                                           volume.volumeUnitResolution);
                                    //! Translate the origin of the subvolume to the correct position in volume coordinate frame
                                cv::Affine3f subvolumePose = volume.pose.translate(volume.volumeUnitIdxToVolume(-tsdf_idx));
                                volumeUnit.p_volume = cv::makePtr<TSDFVolumeCPU>(volumeDims,
                                                            volume.volumeUnitSize,
                                                            subvolumePose,
                                                            volume.truncDist,
                                                            volume.maxWeight,
                                                            volume.raycastStepFactor);
                                volumeUnit.index = tsdf_idx;

                                volume.volume_units_[tsdf_idx] = volumeUnit;
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
    const float dfac;
    const int depthStride;
    mutable Mutex mutex;
};

struct IntegrateSubvolumeInvoker : ParallelLoopBody
{
    IntegrateSubvolumeInvoker(HashTSDFVolumeCPU& _volume, std::vector<cv::Vec3i> _accessVolUnitVec,
            const Depth& _depth, Intr _intrinsics, cv::Affine3f _cameraPose, float _depthFactor) :
        ParallelLoopBody(),
        volume(_volume),
        accessVolUnitsVec(_accessVolUnitVec),
        depth(_depth),
        depthFactor(_depthFactor),
        cameraPose(_cameraPose),
        intrinsics(_intrinsics)
    {
    }

    virtual void operator() (const Range& range) const override
    {
        for (int i = range.start; i < range.end; i++)
        {
            cv::Vec3i tsdf_idx = accessVolUnitsVec[i];

            VolumeUnitMap::iterator accessedVolUnit = volume.volume_units_.find(tsdf_idx);
            assert(accessedVolUnit != volume.volume_units_.end());

            VolumeUnit volumeUnit = accessedVolUnit->second;
            /* std::cout << "Integrating unit: " << accessedVolUnit->first << std::endl; */
            //! The volume unit should already be added into the Volume from the allocator
            volumeUnit.p_volume->integrate(depth, depthFactor, cameraPose, intrinsics);
        }
    }

    HashTSDFVolumeCPU& volume;
    std::vector<cv::Vec3i> accessVolUnitsVec;
    const Depth& depth;
    float depthFactor;
    cv::Affine3f cameraPose;
    Intr intrinsics;
};


void HashTSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose, Intr intrinsics)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();
    VolumeUnitIndexSet accessVolUnits;

    //TODO(Akash): Consider reusing pyrPoints and transform the points
    AccessedVolumeUnitsInvoker allocate_i(*this, accessVolUnits, depth, intrinsics, cameraPose, depthFactor);
    Range range(0, depth.rows);
    parallel_for_(range, allocate_i);

    std::vector<Vec3i> accessVolUnitsVec;
    accessVolUnitsVec.assign(accessVolUnits.begin(), accessVolUnits.end());
    std::cout << "Number of active subvolumes: " << accessVolUnitsVec.size() << "\n";
    IntegrateSubvolumeInvoker integrate_i(*this, accessVolUnitsVec, depth, intrinsics, cameraPose, depthFactor);
    Range accessed_units_range(0, accessVolUnitsVec.size());
    parallel_for_(accessed_units_range, integrate_i);
    std::cout << "Integration complete \n";
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVolumeUnitIdx(cv::Point3f p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize),
                     cvFloor(p.y / volumeUnitSize),
                     cvFloor(p.z / volumeUnitSize));
}

cv::Point3f HashTSDFVolumeCPU::volumeUnitIdxToVolume(cv::Vec3i volumeUnitIdx) const
{
    return cv::Point3f(volumeUnitIdx[0] * volumeUnitSize,
                       volumeUnitIdx[1] * volumeUnitSize,
                       volumeUnitIdx[2] * volumeUnitSize);
}

cv::Point3f HashTSDFVolumeCPU::voxelCoordToVolume(cv::Vec3i voxelIdx) const
{
    return cv::Point3f(voxelIdx[0] * voxelSize,
                       voxelIdx[1] * voxelSize,
                       voxelIdx[2] * voxelSize);
}

cv::Vec3i HashTSDFVolumeCPU::volumeToVoxelCoord(cv::Point3f point) const
{
    return cv::Vec3i(cvFloor(point.x * voxelSizeInv),
                     cvFloor(point.y * voxelSizeInv),
                     cvFloor(point.z * voxelSizeInv));
}

inline Voxel HashTSDFVolumeCPU::at(const cv::Vec3i &volumeIdx) const
{
    cv::Vec3i volumeUnitIdx   = cv::Vec3i(cvFloor(volumeIdx[0] / volumeUnitResolution),
                                          cvFloor(volumeIdx[1] / volumeUnitResolution),
                                          cvFloor(volumeIdx[2] / volumeUnitResolution));

    cv::Vec3i volUnitLocalIdx = volumeIdx - cv::Vec3i(volumeUnitIdx[0] * volumeUnitResolution,
                                                      volumeUnitIdx[1] * volumeUnitResolution,
                                                      volumeUnitIdx[2] * volumeUnitResolution);
    /* std::cout << "VolumeUnitIdx: " << volumeUnitIdx << "\n"; */
    /* std::cout << "subvolumeCoords: " << subVolumeCoords << "\n"; */
    VolumeUnitMap::const_iterator it = volume_units_.find(volumeUnitIdx);
    if(it == volume_units_.end())
    {
        Voxel dummy;
        dummy.tsdf = 0.f; dummy.weight = 0;
        return dummy;
    }
    cv::Ptr<TSDFVolumeCPU> volumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.p_volume);
    return volumeUnit->at(volUnitLocalIdx);
}

inline Voxel HashTSDFVolumeCPU::at(const cv::Point3f &point) const
{
    cv::Vec3i volumeUnitIdx = volumeToVolumeUnitIdx(point);
    cv::Point3f volumeUnitPos = volumeUnitIdxToVolume(volumeUnitIdx);
    cv::Vec3i volUnitLocalIdx = volumeToVoxelCoord(point - volumeUnitPos);
    VolumeUnitMap::const_iterator it = volume_units_.find(volumeUnitIdx);
    if(it == volume_units_.end())
    {
        Voxel dummy;
        dummy.tsdf = 0; dummy.weight = 0;
        return dummy;
    }
    cv::Ptr<TSDFVolumeCPU> volumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.p_volume);
    /* std::cout << "volumeUnitIdx: " << volumeUnitIdx << "volUnitLocalIdx: " << volUnitLocalIdx << std::endl; */
    /* std::cout << volumeUnit->at(volUnitLocalIdx).tsdf << std::endl; */
    return volumeUnit->at(volUnitLocalIdx);
}


inline TsdfType HashTSDFVolumeCPU::interpolateVoxel(cv::Point3f p) const
{
    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    TsdfType vx[8];
    //! This fetches the tsdf value from the correct subvolumes
    vx[0] = at(cv::Vec3i(0, 0, 0) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[1] = at(cv::Vec3i(0, 0, 1) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[2] = at(cv::Vec3i(0, 1, 0) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[3] = at(cv::Vec3i(0, 1, 1) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[4] = at(cv::Vec3i(1, 0, 0) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[5] = at(cv::Vec3i(1, 0, 1) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[6] = at(cv::Vec3i(0, 1, 0) + cv::Vec3i(ix, iy, iz)).tsdf;
    vx[7] = at(cv::Vec3i(1, 1, 1) + cv::Vec3i(ix, iy, iz)).tsdf;

    /* std::cout << "tsdf 7th: " << vx[7] << "\n"; */
    TsdfType v00 = vx[0] + tz*(vx[1] - vx[0]);
    TsdfType v01 = vx[2] + tz*(vx[3] - vx[2]);
    TsdfType v10 = vx[4] + tz*(vx[5] - vx[4]);
    TsdfType v11 = vx[6] + tz*(vx[7] - vx[6]);

    TsdfType v0 = v00 + ty*(v01 - v00);
    TsdfType v1 = v10 + ty*(v11 - v10);

    return v0 + tx*(v1 - v0);
}

inline Point3f HashTSDFVolumeCPU::getNormalVoxel(Point3f point) const
{

    Vec3f pointVec(point);
    Vec3f normal = Vec3f(0, 0, 0);

    Vec3f pointPrev = point;
    Vec3f pointNext = point;

    for(int c = 0; c < 3; c++)
    {
        pointPrev[c] -= voxelSize*0.5;
        pointNext[c] += voxelSize*0.5;

        normal[c] = at(Point3f(pointNext)).tsdf -  at(Point3f(pointPrev)).tsdf;
        /* std::cout << "pointPrev, pointNext: " << at(Point3f(pointPrev)).tsdf << ", " << at(Point3f(pointNext)).tsdf << std::endl; */
        normal[c] *= 0.5;
        /* std::cout << "TSDF diff: " << normal[c] << std::endl; */

        pointPrev[c] = pointVec[c];
        pointNext[c] = pointVec[c];
    }
    return normalize(normal);

    /* int ix = cvFloor(p.x); */
    /* int iy = cvFloor(p.y); */
    /* int iz = cvFloor(p.z); */

    /* float tx = p.x - ix; */
    /* float ty = p.y - iy; */
    /* float tz = p.z - iz; */

    /* Vec3i coordBase0 = cv::Vec3i(ix, iy, iz); */
    /* Vec3i coordBase1 = cv::Vec3i(ix, iy, iz); */
    /* Vec3f an; */
    /* for(int c = 0; c < 3; c++) */
    /* { */
    /*     float& nv = an[c]; */

    /*     TsdfType vx[8]; */
    /*     coordBase0[c] -= 1; */
    /*     coordBase1[c] += 1; */

    /*     vx[0] = at(cv::Vec3i(0, 0, 0) + coordBase1).tsdf - at(cv::Vec3i(0, 0, 0) + coordBase0).tsdf; */
    /*     vx[1] = at(cv::Vec3i(0, 0, 1) + coordBase1).tsdf - at(cv::Vec3i(0, 0, 1) + coordBase0).tsdf; */
    /*     vx[2] = at(cv::Vec3i(0, 1, 0) + coordBase1).tsdf - at(cv::Vec3i(0, 1, 0) + coordBase0).tsdf; */
    /*     vx[3] = at(cv::Vec3i(0, 1, 1) + coordBase1).tsdf - at(cv::Vec3i(0, 1, 1) + coordBase0).tsdf; */
    /*     vx[4] = at(cv::Vec3i(1, 0, 0) + coordBase1).tsdf - at(cv::Vec3i(1, 0, 0) + coordBase0).tsdf; */
    /*     vx[5] = at(cv::Vec3i(1, 0, 1) + coordBase1).tsdf - at(cv::Vec3i(1, 0, 1) + coordBase0).tsdf; */
    /*     vx[6] = at(cv::Vec3i(1, 1, 0) + coordBase1).tsdf - at(cv::Vec3i(1, 1, 0) + coordBase0).tsdf; */
    /*     vx[7] = at(cv::Vec3i(1, 1, 1) + coordBase1).tsdf - at(cv::Vec3i(1, 1, 1) + coordBase0).tsdf; */

    /*     TsdfType v00 = vx[0] + tz*(vx[1] - vx[0]); */
    /*     TsdfType v01 = vx[2] + tz*(vx[3] - vx[2]); */
    /*     TsdfType v10 = vx[4] + tz*(vx[5] - vx[4]); */
    /*     TsdfType v11 = vx[6] + tz*(vx[7] - vx[6]); */

    /*     TsdfType v0 = v00 + ty*(v01 - v00); */
    /*     TsdfType v1 = v10 + ty*(v11 - v10); */

    /*     nv = v0 + tx*(v1 - v0); */
    /* } */

    /* return normalize(an); */
}


struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, Affine3f cameraPose, Intr intrinsics, const HashTSDFVolumeCPU& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        volume(_volume),
        tstep(_volume.truncDist * _volume.raycastStepFactor),
        cam2vol(volume.pose.inv() * cameraPose),
        vol2cam(cameraPose.inv() * volume.pose),
        reproj(intrinsics.makeReprojector())
    {}

    virtual void operator() (const Range& range) const override
    {
        const Point3f cam2volTrans = cam2vol.translation();
        const Matx33f cam2volRot   = cam2vol.rotation();
        const Matx33f vol2camRot   = vol2cam.rotation();

        const float blockSize = volume.volumeUnitSize;

        for (int y = range.start; y < range.end; y++)
        {

            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3;

                //! Ray origin in the volume coordinate frame
                Point3f orig = cam2volTrans;
                //! Ray direction in the volume coordinate frame
                Point3f rayDirV = normalize(Vec3f(cam2volRot * reproj(Point3f(x, y, 1.f))));
                float tmin = rgbd::Odometry::DEFAULT_MIN_DEPTH()/rayDirV.z;
                float tmax = rgbd::Odometry::DEFAULT_MAX_DEPTH()/rayDirV.z;

                /* std::cout << "tmin, tmax :" << tmin << ", " << tmax << "\n"; */
                /* std::cout << "Origin: " << orig << " rayDirection: " << rayDirV << "\n"; */
                float tprev = tmin;
                float tcurr = tmin + tstep;
                //! Is this a reasonable initialization?
                TsdfType prevTsdf = volume.truncDist;
                cv::Vec3i prevVolumeUnitIdx = cv::Vec3i(std::numeric_limits<int>::min(),
                                                        std::numeric_limits<int>::min(),
                                                        std::numeric_limits<int>::min());
                cv::Ptr<TSDFVolumeCPU> currVolumeUnit;
                while(tcurr < tmax)
                {
                    Point3f currRayPos   = orig + tcurr * rayDirV;
                    cv::Vec3i currVolumeUnitIdx  = volume.volumeToVolumeUnitIdx(currRayPos);
                    /* std::cout << "tCurr:                      " << tcurr << "\n"; */
                    /* std::cout << "Current Ray cast position:  " << currRayPos << "\n"; */
                    /* std::cout << "Previous volume unit Index: " << prevVolumeUnitIdx << "\n"; */
                    /* std::cout << "Current volume unit Index:  " << currVolumeUnitIdx << "\n"; */

                    VolumeUnitMap::const_iterator it;
                    if(currVolumeUnitIdx != prevVolumeUnitIdx)
                        it = volume.volume_units_.find(currVolumeUnitIdx);

                    TsdfType currTsdf = prevTsdf;
                    int currWeight = 0;
                    float stepSize = blockSize;
                    cv::Vec3i volUnitLocalIdx;
                    //! Is the subvolume exists in hashtable
                    if(it != volume.volume_units_.end())
                    {
                        currVolumeUnit = std::dynamic_pointer_cast<TSDFVolumeCPU>(it->second.p_volume);
                        cv::Point3f currVolUnitPos = volume.volumeUnitIdxToVolume(currVolumeUnitIdx);
                        volUnitLocalIdx = volume.volumeToVoxelCoord(currRayPos - currVolUnitPos);

                        //! Figure out voxel interpolation
                        Voxel currVoxel = currVolumeUnit->at(volUnitLocalIdx);
                        currTsdf   = currVoxel.tsdf;
                        currWeight = currVoxel.weight;
                        stepSize   = max(currTsdf * volume.truncDist, tstep);

                    }
                    //! Surface crossing
                    if(prevTsdf > 0.f && currTsdf <= 0.f && currWeight > 0)
                    {
                        std::cout << "subvolume coords: " << volUnitLocalIdx << "\n";
                        std::cout << "current TSDF:     " << currTsdf << "\n";
                        std::cout << "current weight:   " << currWeight << "\n";
                        std::cout << "previous TSDF:    " << prevTsdf << "\n";
                        std::cout << "tcurr:            " << tcurr    << "\n";
                        float tInterp = (tcurr * prevTsdf - tprev * currTsdf)/(prevTsdf - currTsdf);

                        if(!cvIsNaN(tInterp) && !cvIsInf(tInterp))
                        {
                            Point3f pv = orig + tInterp * rayDirV;
                            Point3f nv = volume.getNormalVoxel(pv);
                            /* std::cout << "normal: " << nv << std::endl; */

                            if(!isNaN(nv))
                            {
                                normal = vol2camRot * nv;
                                point = vol2cam * pv;

                                /* std::cout << "Point:  " << point << "\n"; */
                                /* std::cout << "normal: " << normal << "\n"; */
                                break;
                            }
                        }
                    }

                    prevVolumeUnitIdx = currVolumeUnitIdx;
                    prevTsdf          = currTsdf;
                    tprev             = tcurr;
                    tcurr            += stepSize;
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
        cv::Size frameSize, cv::OutputArray _points, cv::OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(frameSize.area() > 0);

    _points.create (frameSize, POINT_TYPE);
    _normals.create (frameSize, POINT_TYPE);

    Points points = _points.getMat();
    Normals normals = _normals.getMat();
    RaycastInvoker ri(points, normals, cameraPose, intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);

}

cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                       float _raycastStepFactor, int volumeUnitResolution)
{
    return cv::makePtr<HashTSDFVolumeCPU>(_voxelSize, volumeUnitResolution, _pose, _truncDist, _maxWeight, _raycastStepFactor);
}

} // namespace kinfu
} // namespace cv

