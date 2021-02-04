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
#define VOLUMES_SIZE 8192

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

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const Vec3i& x) const noexcept
    {
        size_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (uint16_t i = 0; i < 3; i++)
        {
            seed ^= std::hash<int>()(x[i]) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct VolumeUnit
{
    cv::Vec3i coord;
    int index;
    cv::Matx44f pose;
    int lastVisibleIndex = 0;
    bool isActive;
};

typedef std::unordered_set<cv::Vec3i, tsdf_hash> VolumeUnitIndexSet;
typedef std::unordered_map<cv::Vec3i, VolumeUnit, tsdf_hash> VolumeUnitIndexes;

class HashTSDFVolumeCPU : public HashTSDFVolume
{
public:
    // dimension in voxels, size in meters
    HashTSDFVolumeCPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
        float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = true);

    HashTSDFVolumeCPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = true);

    void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
        const int frameId = 0) override;
    void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
        OutputArray normals) const override;

    void fetchNormals(InputArray points, OutputArray _normals) const override;
    void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    void reset() override;
    size_t getTotalVolumeUnits() const override { return volumeUnits.size(); }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const override;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    TsdfVoxel at(const Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel at(const cv::Point3f& point) const;
    virtual TsdfVoxel _at(const cv::Vec3i& volumeIdx, int indx) const;

    TsdfVoxel atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, VolumeUnitIndexes::const_iterator it) const;


    float interpolateVoxelPoint(const Point3f& point) const;
    float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord(const Point3f& point) const;

public:
    Vec4i volStrides;
    Vec6f frameParams;
    Mat pixNorms;
    VolumeUnitIndexes volumeUnits;
    cv::Mat volUnitsData;
    int lastVolIndex;
};


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
    frameParams = Vec6f();
    pixNorms = Mat();
    volumeUnits = VolumeUnitIndexes();
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
                        for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
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
        if (lastVolIndex > int(volUnitsData.size().height))
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
                continue;

            Point3f volumeUnitPos = volumeUnitIdxToVolume(it->first);
            Point3f volUnitInCamSpace = vol2cam * volumeUnitPos;
            if (volUnitInCamSpace.z < 0 || volUnitInCamSpace.z > truncateThreshold)
            {
                it->second.isActive = false;
                continue;
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

inline TsdfVoxel HashTSDFVolumeCPU::_at(const cv::Vec3i& volumeIdx, int indx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
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
        return TsdfVoxel(floatToTsdf(1.f), 0);
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
        return TsdfVoxel(floatToTsdf(1.f), 0);
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
        return TsdfVoxel(floatToTsdf(1.f), 0);
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



class HashTSDFVolumeGPU : public HashTSDFVolume
{
public:
    HashTSDFVolumeGPU(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
        float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = false);

    HashTSDFVolumeGPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = false);

    void reset() override;

    void integrateAllVolumeUnitsGPU(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics);

    void allocateVolumeUnits(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics);

    void markActive(const Matx44f& cameraPose, const Intr& intrinsics, const Size frameSz, const int frameId);

    void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
        const int frameId = 0) override;
    void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
        OutputArray normals) const override;


    void fetchNormals(InputArray points, OutputArray _normals) const override;
    void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    size_t getTotalVolumeUnits() const override { return size_t(hashMap.last); }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const override;



    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel new_at(const cv::Vec3i& volumeIdx, int indx) const;
    TsdfVoxel new_atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, int indx) const;


    float interpolateVoxelPoint(const Point3f& point) const;
    float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord(const Point3f& point) const;

public:
    Vec4i volStrides;
    Vec6f frameParams;
    int degree;
    int buff_lvl;

    // per-volume-unit data
    cv::UMat lastVisibleIndices;

    cv::UMat isActiveFlags;

    cv::UMat volUnitsData;
    //TODO: remove it when there's no CPU parts
    cv::Mat volUnitsDataCopy;

    cv::UMat pixNorms;

    //TODO: move indexes.volumes to GPU
    //VolumesTable indexes;
    ToyHashMap hashMap;


    
    Vec8i neighbourCoords;
};

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

    neighbourCoords = Vec8i(
        volStrides.dot(Vec4i(0, 0, 0)),
        volStrides.dot(Vec4i(0, 0, 1)),
        volStrides.dot(Vec4i(0, 1, 0)),
        volStrides.dot(Vec4i(0, 1, 1)),
        volStrides.dot(Vec4i(1, 0, 0)),
        volStrides.dot(Vec4i(1, 0, 1)),
        volStrides.dot(Vec4i(1, 1, 0)),
        volStrides.dot(Vec4i(1, 1, 1))
    );

    reset();
}

HashTSDFVolumeGPU::HashTSDFVolumeGPU(const VolumeParams & _params, bool _zFirstMemOrder)
    : HashTSDFVolume(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
                     _params.depthTruncThreshold, _params.unitResolution, _zFirstMemOrder)
{
    //TODO: move reset() contents here, reset() just clears the data
}
// zero volume, leave rest params the same
void HashTSDFVolumeGPU::reset()
{
    CV_TRACE_FUNCTION();
    degree = 15;
    buff_lvl = (int) pow(2, degree);

    int volCubed = volumeUnitResolution * volumeUnitResolution * volumeUnitResolution;
    volUnitsDataCopy = cv::Mat(buff_lvl, volCubed, rawType<TsdfVoxel>());

    volUnitsData = cv::UMat(buff_lvl, volCubed, CV_8UC2);

    lastVisibleIndices = cv::UMat(buff_lvl, 1, CV_32S);

    isActiveFlags = cv::UMat(buff_lvl, 1, CV_8U);

    //indexes = VolumesTable();
    hashMap = ToyHashMap();

    frameParams = Vec6f();
    pixNorms = UMat();
}


static cv::UMat preCalculationPixNormGPU(int depth_rows, int depth_cols, Vec2f fxy, Vec2f cxy)
{
    Mat x(1, depth_cols, CV_32FC1);
    Mat y(1, depth_rows, CV_32FC1);
    UMat pixNorm(depth_rows, depth_cols, CV_32F);

    for (int i = 0; i < depth_cols; i++)
        *x.ptr<float>(0, i) = (i - cxy[0]) / fxy[0];
    for (int i = 0; i < depth_rows; i++)
        *y.ptr<float>(0, i) = (i - cxy[1]) / fxy[1];

    cv::String errorStr;
    cv::String name = "preCalculationPixNorm";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    cv::String options = "-cl-mad-enable";
    ocl::Kernel kk;
    kk.create(name.c_str(), source, options, &errorStr);

    if (kk.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    AccessFlag af = ACCESS_READ;
    UMat xx = x.getUMat(af);
    UMat yy = y.getUMat(af);

    kk.args(ocl::KernelArg::WriteOnly(pixNorm),
            ocl::KernelArg::PtrReadOnly(xx),
            ocl::KernelArg::PtrReadOnly(yy));

    size_t globalSize[2];
    globalSize[0] = depth_rows;
    globalSize[1] = depth_cols;

    if (!kk.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

    return pixNorm;
}


void HashTSDFVolumeGPU::integrateAllVolumeUnitsGPU(const UMat& depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!depth.empty());

    String errorStr;
    String name = "integrateAllVolumeUnits";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    float dfac = 1.f / depthFactor;
    Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
    Matx44f vol2camMatrix = (Affine3f(cameraPose).inv() * pose).matrix;
    Matx44f camInvMatrix = Affine3f(cameraPose).inv().matrix;

    UMat hashesGpu(hashMap.hashDivisor, 1, CV_32S);
    Mat(hashMap.hashes, false).copyTo(hashesGpu);

    UMat hashDataGpu(hashMap.capacity, 1, CV_32SC4);
    Mat(hashMap.data, false).copyTo(hashDataGpu);

    k.args(ocl::KernelArg::ReadOnly(depth),
           ocl::KernelArg::PtrReadOnly(hashesGpu),
           ocl::KernelArg::PtrReadOnly(hashDataGpu),
           (int)hashMap.hashDivisor,
           ocl::KernelArg::ReadWrite(volUnitsData),
           ocl::KernelArg::ReadOnly(pixNorms),
           ocl::KernelArg::ReadOnly(isActiveFlags),
           vol2camMatrix,
           camInvMatrix,
           voxelSize,
           volumeUnitResolution,
           volStrides.val,
           fxy.val,
           cxy.val,
           dfac,
           truncDist,
           int(maxWeight)
    );

    int resol = volumeUnitResolution;
    size_t globalSize[3];
    globalSize[0] = (size_t)resol; // volumeUnitResolution
    globalSize[1] = (size_t)resol; // volumeUnitResolution
    globalSize[2] = (size_t)hashMap.last; // num of volume units

    if (!k.run(3, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}


void HashTSDFVolumeGPU::allocateVolumeUnits(const UMat& _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics)
{
    const int newIndicesCapacity = VOLUMES_SIZE;
    constexpr size_t pixCapacity = 16;
    typedef std::array<Vec3i, pixCapacity> LocalVolUnits;

    Depth depth = _depth.getMat(ACCESS_READ);

    //! Compute volumes to be allocated
    const int depthStride = int(log2(volumeUnitResolution));
    const float invDepthFactor = 1.f / depthFactor;
    const Intr::Reprojector reproj(intrinsics.makeReprojector());
    const Affine3f cam2vol(pose.inv() * Affine3f(cameraPose));
    const Point3f truncPt(truncDist, truncDist, truncDist);
    Mutex mutex;
    
    // for new indices
    ToyHashMap thm;

    /* -----------------------
        
    String errorStr;
    String name = "toy_alloc";

    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    ToyHashMap ths;
    UMat newSetHashes(ths.hashDivisor, 1, CV_32S);
    Mat(ths.hashes, false).copyTo(newSetHashes);
    int dt = CV_MAKETYPE(CV_32S, 8);
    UMat newSetData(ths.capacity, 1, dt);
    Mat(ths.data, false).copyTo(newSetData);
    UMat newSetLast(1, 1, CV_32S);
    newSetLast = ths.last;

    k.args(ocl::KernelArg::PtrReadWrite(newSetHashes),
           ocl::KernelArg::PtrReadWrite(newSetData),
           ocl::KernelArg::PtrReadWrite(newSetLast),
           ths.capacity,
           ths.hashDivisor,
           
           

    );

    size_t globalSize[2];
    globalSize[0] = divUp(depth.cols, depthStride);
    globalSize[1] = divUp(depth.rows, depthStride);

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

    newSetHashes.copyTo(Mat(ths.hashes, false));
    newSetData.copyTo(Mat(ths.data, false));

    //TODO HUGE: add to globalMap instead
            
    ------------------------------------------- */
    /*
                         __global int* newSetHashes,
                         __global int8* newSetData,
                         __global int* newSetLast,
                         const int newSetCapacity,
                         const int newSetHashDivisor,

                        __global const int* globalMapHashes,
                        __global const int8* globalMapData,
                        const int globalMapHashDivisor,

                        __global const char * depthPtr,
                        int depthStep, int depthOffset,
                        int depthRows, int depthCols,

                        const int depthStride,
                        const float invDepthFactor,
                        const float truncateThreshold,

                        const float2 fixy, const float2 cxy,

                        const float16 cam2vol,
                        const float volumeUnitSizeInv,
                        const float truncDist,

                        __global volatile int* hsMutex, // set to 0 initially
                        __global int* full // set to 0 initially
    
    */


    // -----------------------

    auto fillLocalAcessVolUnits = [&](const Range& xrange, const Range& yrange, ToyHashMap& ghm/*LocalVolUnits& localAccessVolUnits, int& locVolIdx*/)
    {
        for (int y = yrange.start; y < yrange.end; y += depthStride)
        {
            const depthType* depthRow = depth[y];
            for (int x = xrange.start; x < xrange.end; x += depthStride)
            {
                depthType z = depthRow[x] * invDepthFactor;
                if (z <= 0 || z > this->truncateThreshold)
                    continue;
                Point3f camPoint = reproj(Point3f((float)x, (float)y, z));
                Point3f volPoint = cam2vol * camPoint;
                //! Find accessed TSDF volume unit for valid 3D vertex
                Vec3i lower_bound = this->volumeToVolumeUnitIdx(volPoint - truncPt);
                Vec3i upper_bound = this->volumeToVolumeUnitIdx(volPoint + truncPt);
                
                int pixLocalCounter = 0;
                LocalVolUnits pixLocalVolUnits;
                for (int i = lower_bound[0]; i <= upper_bound[0]; i++)
                    for (int j = lower_bound[1]; j <= upper_bound[1]; j++)
                        for (int k = lower_bound[2]; k <= upper_bound[2]; k++)
                        {
                            const Vec3i tsdf_idx = Vec3i(i, j, k);

                            //if (indexes.findRow(tsdf_idx) < 0)
                            if (hashMap.find(tsdf_idx) < 0)
                            {
                                bool found = false;
                                for (int i = 0; i < pixLocalCounter; i++)
                                {
                                    if (pixLocalVolUnits[i] == tsdf_idx)
                                    {
                                        found = true; break;
                                    }
                                }
                                if (!found)
                                {
                                    pixLocalVolUnits[pixLocalCounter++] = tsdf_idx;
                                    if (pixLocalCounter >= pixCapacity)
                                    {
                                        //DEBUG
                                        std::cout << "allocate: pix capacity exhausted" << std::endl;
                                        return;
                                    }
                                }
                            }
                        }

                // lock localAccessVolUnits somehow
                for (int i = 0; i < pixLocalCounter; i++)
                {
                    Vec3i idx = pixLocalVolUnits[i];
                    if (!ghm.insert(idx))
                    {
                        //DEBUG
                        std::cout << "allocate: local capacity exhausted" << std::endl;
                        //return;
                    }
                }
                // unlock
            }
        }
    };

    Rect dim(0, 0, depth.cols, depth.rows);
    Size gsz(32, 32);
    Size gg(divUp(dim.width, gsz.width), divUp(dim.height, gsz.height));

    bool needReallocation = false;
    auto allocateLambda = [&](const Range& r)
    {

    for (int yg = r.start; yg < r.end; yg++)
    {
        for (int xg = 0; xg < gg.width; xg++)
        {
            Rect gr(xg * gsz.width, yg * gsz.height, (xg + 1) * gsz.width, (yg + 1) * gsz.height);
            gr = gr & dim;
            Range xr(gr.tl().x, gr.br().x), yr(gr.tl().y, gr.br().y);

            /*
            LocalVolUnits localAccessVolUnits;
            int loc_vol_idx = 0;
            */

            ToyHashMap ghm;

            fillLocalAcessVolUnits(xr, yr, ghm /*localAccessVolUnits, loc_vol_idx*/);

            if (ghm.last)
            {
                std::lock_guard<std::recursive_mutex> al(mutex);

                //mutex.lock();
                for (int i = 0; i < ghm.last; i++)
                {
                    Vec4i node = ghm.data[i];
                    Vec3i idx(node[0], node[1], node[2]);
                    
                    //TODO: 1. add to separate hash map instead, then merge on GPU side

                    int result = thm.insert(idx);
                    if (!result)
                    {
                        needReallocation = true;
                        //DEBUG
                        std::cout << "new indices: need reallocation, exiting" << std::endl;

                        return;
                    }

                    /*
                    // if not found
                    //if (indexes.findRow(idx) < 0)
                    if (hashMap.find(idx) < 0)
                    {
                        //bool extend = indexes.insert(idx, lastVolIndex);
                        int result = hashMap.insert(idx, lastVolIndex);
                        //TODO: replace 1 by enum
                        if (result == 1)
                        {
                            Vec4i idx4(idx[0], idx[1], idx[2], 0);
                            nodeIndices.push_back(idx4);
                            lastVolIndex++;
                        }
                        else if (result == 0)
                        {
                            needReallocation = true;
                            //DEBUG
                            std::cout << "need reallocation, exiting" << std::endl;

                            return;
                        }
                    }
                    */

                }
                //mutex.unlock();

            }

            /*
            if (loc_vol_idx > 0)
            {
                //DEBUG
                //std::cout << "loc_vol_idx: " << loc_vol_idx << std::endl;

                mutex.lock();
                for (int i = 0; i < loc_vol_idx; i++)
                {
                    Vec3i idx = localAccessVolUnits[i];

                    // if not found
                    if (indexes.findRow(idx) < 0)
                    {
                        Volume_NODE* node = indexes.insert(idx, lastVolIndex);
                        nodePtrs.push_back(node);
                        lastVolIndex++;
                    }
                }
                mutex.unlock();
            }
            */
        }
    }

    };

    do
    {
        if (needReallocation)
        {
            /*
            std::cout << "reallocation!! from: " << hashMap.capacity << " to x2: " << hashMap.capacity * 2 << std::endl;
            hashMap.capacity *= 2;
            hashMap.data.resize(hashMap.capacity);
            */
            std::cout << "reallocation group!! from: " << thm.capacity << " to x2: " << thm.capacity * 2 << std::endl;
            thm.capacity *= 2;
            thm.data.resize(thm.capacity);

            needReallocation = false;
        }

        parallel_for_(Range(0, gg.height), allocateLambda);
        //allocateLambda(Range(0, gg.height));
    } while (needReallocation);


    auto pushToGlobal = [](const ToyHashMap thm, ToyHashMap& globalHashMap,
                           bool& needReallocation, Mutex& mutex)
    {
        for (int i = 0; i < thm.last; i++)
        {
                Vec4i node = thm.data[i];
                Vec3i idx(node[0], node[1], node[2]);

                std::lock_guard<std::recursive_mutex> al(mutex);

                int result = globalHashMap.insert(idx);
                if (result == 0)
                {
                    needReallocation = true;
                    //DEBUG
                    std::cout << "need reallocation, exiting" << std::endl;
                    return;
                }
        }
    };
    /*
    String errorStr;
    String name = "push_to_global";

    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);
    
    ToyHashMap ths;
    UMat newSetHashes(ths.hashDivisor, 1, CV_32S);
    Mat(ths.hashes, false).copyTo(newSetHashes);
    int dt = CV_MAKETYPE(CV_32S, 8);
    UMat newSetData(ths.capacity, 1, dt);
    Mat(ths.data, false).copyTo(newSetData);
    UMat newSetLast(1, 1, CV_32S);
    newSetLast = ths.last;

    k.args(ocl::KernelArg::PtrReadWrite(newSetHashes),
        ocl::KernelArg::PtrReadWrite(newSetData),
        ocl::KernelArg::PtrReadWrite(newSetLast),
        ths.capacity,
        ths.hashDivisor,

        );

    size_t globalSize[1];
    globalSize[0] = thm.last;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

    newSetHashes.copyTo(Mat(ths.hashes, false));
    newSetData.copyTo(Mat(ths.data, false));
    */

    needReallocation = false;
    do
    {
        if (needReallocation)
        {
            std::cout << "reallocation global!! from: " << hashMap.capacity << " to x2: " << hashMap.capacity * 2 << std::endl;
            hashMap.capacity *= 2;
            hashMap.data.resize(hashMap.capacity);

            needReallocation = false;
        }

        pushToGlobal(thm, hashMap, needReallocation, mutex);
    } while (needReallocation);
        
    // ---------------------
}


void HashTSDFVolumeGPU::markActive(const Matx44f& cameraPose, const Intr& intrinsics, const Size frameSz, const int frameId)
{
    //! Mark volumes in the camera frustum as active
    String errorStr;
    String name = "markActive";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * pose);
    const Intr::Projector proj(intrinsics.makeProjector());
    Vec2f fxy(proj.fx, proj.fy), cxy(proj.cx, proj.cy);

    UMat hashesGpu(hashMap.hashDivisor, 1, CV_32S);
    Mat(hashMap.hashes, false).copyTo(hashesGpu);

    UMat hashDataGpu(hashMap.capacity, 1, CV_32SC4);
    Mat(hashMap.data, false).copyTo(hashDataGpu);

    k.args(
        ocl::KernelArg::PtrReadOnly(hashesGpu),
        ocl::KernelArg::PtrReadOnly(hashDataGpu),
        (int)hashMap.hashDivisor,
        ocl::KernelArg::WriteOnly(isActiveFlags),
        ocl::KernelArg::WriteOnly(lastVisibleIndices),
        vol2cam.matrix,
        fxy,
        cxy,
        frameSz,
        volumeUnitSize,
        hashMap.last,
        truncateThreshold,
        frameId
    );

    size_t globalSize[1] = { (size_t)hashMap.last };
    if (!k.run(1, globalSize, nullptr, true))
        throw std::runtime_error("Failed to run kernel");
}


void HashTSDFVolumeGPU::integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    UMat depth = _depth.getUMat();

    // Save length to fill new data in ranges
    int sizeBefore = hashMap.last;
    allocateVolumeUnits(depth, depthFactor, cameraPose, intrinsics);
    int sizeAfter = hashMap.last;
    //! Perform the allocation

    // Grow buffers
    if (sizeAfter >= buff_lvl)
    {
        degree = (int)(log2(sizeAfter) + 1); // clz() would be better
        int oldBuffSize = buff_lvl;
        buff_lvl = (int)pow(2, degree);

        volUnitsDataCopy.resize(buff_lvl);

        Range oldr(0, oldBuffSize);
        int volCubed = volumeUnitResolution * volumeUnitResolution * volumeUnitResolution;
        UMat newData(buff_lvl, volCubed, CV_8UC2);
        volUnitsData.copyTo(newData.rowRange(oldr));
        volUnitsData = newData;
        
        UMat newLastVisibleIndices(buff_lvl, 1, CV_32S);
        lastVisibleIndices.copyTo(newLastVisibleIndices.rowRange(oldr));
        lastVisibleIndices = newLastVisibleIndices;

        UMat newIsActiveFlags(buff_lvl, 1, CV_8U);
        isActiveFlags.copyTo(newIsActiveFlags.rowRange(oldr));
        isActiveFlags = newIsActiveFlags;
    }

    // Fill data for new volume units
    Range r(sizeBefore, sizeAfter);
    if (r.start < r.end)
    {
        lastVisibleIndices.rowRange(r) = frameId;
        isActiveFlags.rowRange(r) = 1;

        TsdfVoxel emptyVoxel(floatToTsdf(0.0f), 0);
        volUnitsData.rowRange(r) = Vec2b((uchar)(emptyVoxel.tsdf), (uchar)(emptyVoxel.weight));
    }

    //! Mark volumes in the camera frustum as active
    markActive(cameraPose, intrinsics, depth.size(), frameId);

    Vec6f newParams((float)depth.rows, (float)depth.cols,
                     intrinsics.fx, intrinsics.fy,
                     intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        Vec2f fxy(intrinsics.fx, intrinsics.fy), cxy(intrinsics.cx, intrinsics.cy);
        pixNorms = preCalculationPixNormGPU(depth.rows, depth.cols, fxy, cxy);
    }

    //! Integrate the correct volumeUnits
    integrateAllVolumeUnitsGPU(depth, depthFactor, cameraPose, intrinsics);
}

//TODO: replace .ptr<...> everywhere by .at<...>

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

inline TsdfVoxel HashTSDFVolumeGPU::new_at(const cv::Vec3i& volumeIdx, int indx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volumeUnitResolution || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volumeUnitResolution || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volumeUnitResolution || volumeIdx[2] < 0))
    {
        return TsdfVoxel(floatToTsdf(1.0f), 0);
    }

    const TsdfVoxel* volData = volUnitsDataCopy.ptr<TsdfVoxel>(indx);
    int coordBase =
        volumeIdx[0] * volStrides[0] +
        volumeIdx[1] * volStrides[1] +
        volumeIdx[2] * volStrides[2];
    return volData[coordBase];
}

TsdfVoxel HashTSDFVolumeGPU::new_atVolumeUnit(const Vec3i& point, const Vec3i& volumeUnitIdx, int indx) const
{
    if (indx < 0)
    {
        return TsdfVoxel(floatToTsdf(1.f), 0);
    }
    Vec3i volUnitLocalIdx = point - volumeUnitIdx * volumeUnitResolution;

    // expanding at(), removing bounds check
    const TsdfVoxel* volData = volUnitsDataCopy.ptr<TsdfVoxel>(indx);
    int coordBase = volUnitLocalIdx[0] * volStrides[0] +
        volUnitLocalIdx[1] * volStrides[1] +
        volUnitLocalIdx[2] * volStrides[2];
    return volData[coordBase];
}

float HashTSDFVolumeGPU::interpolateVoxelPoint(const Point3f& point) const
{
    const Vec3i local_neighbourCoords[] = { {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                                            {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1} };

    // A small hash table to reduce a number of find() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
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
        Vec3i pt = iv + local_neighbourCoords[i];

        Vec3i volumeUnitIdx = voxelToVolumeUnitIdx(pt, volumeUnitResolution);
        int dictIdx = (volumeUnitIdx[0] & 1) + (volumeUnitIdx[1] & 1) * 2 + (volumeUnitIdx[2] & 1) * 4;
        auto it = iterMap[dictIdx];
        if (it < -1)
        {
            //it = indexes.findRow(volumeUnitIdx);
            it = hashMap.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
        }

        vx[i] = new_atVolumeUnit(pt, volumeUnitIdx, it).tsdf;
    }

    return interpolate(tx, ty, tz, vx);
}

inline float HashTSDFVolumeGPU::interpolateVoxel(const cv::Point3f& point) const
{
    return interpolateVoxelPoint(point * voxelSizeInv);
}

Point3f HashTSDFVolumeGPU::getNormalVoxel(const Point3f& point) const
{
    Vec3f normal = Vec3f(0, 0, 0);

    Point3f ptVox = point * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));

    // A small hash table to reduce a number of find() calls
    // -2 and lower means not queried yet
    // -1 means not found
    // 0+ means found
    int iterMap[8];
    for (int i = 0; i < 8; i++)
    {
        iterMap[i] = -2;
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
        if (it < -1)
        {
            //it = indexes.findRow(volumeUnitIdx);
            it = hashMap.find(volumeUnitIdx);
            iterMap[dictIdx] = it;
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

    String errorStr;
    String name = "raycast";
    ocl::ProgramSource source = ocl::rgbd::hash_tsdf_oclsrc;
    String options = "-cl-mad-enable";
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if (k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    _points.create(frameSize, CV_32FC4);
    _normals.create(frameSize, CV_32FC4);

    UMat points = _points.getUMat();
    UMat normals = _normals.getUMat();

    Intr::Reprojector r = intrinsics.makeReprojector();
    Vec2f finv(r.fxinv, r.fyinv), cxy(r.cx, r.cy);
    
    Vec4f boxMin, boxMax(volumeUnitSize - voxelSize,
                         volumeUnitSize - voxelSize,
                         volumeUnitSize - voxelSize);

    float tstep = truncDist * raycastStepFactor;

    const HashTSDFVolumeGPU& volume(*this);
    const Affine3f cam2vol(volume.pose.inv() * Affine3f(cameraPose));
    const Affine3f vol2cam(Affine3f(cameraPose.inv()) * volume.pose);

    const Point3f cam2volTrans = cam2vol.translation();

    Vec4f cam2volTransGPU(cam2volTrans.x, cam2volTrans.y, cam2volTrans.z, 0);
    Matx44f cam2volRotGPU = cam2vol.matrix;
    Matx44f vol2camRotGPU = vol2cam.matrix;

    UMat volPoseGpu, invPoseGpu;
    Mat(pose.matrix).copyTo(volPoseGpu);
    Mat(pose.inv().matrix).copyTo(invPoseGpu);

    UMat hashesGpu(hashMap.hashDivisor, 1, CV_32S);
    Mat(hashMap.hashes, false).copyTo(hashesGpu);

    UMat hashDataGpu(hashMap.capacity, 1, CV_32SC4);
    Mat(hashMap.data, false).copyTo(hashDataGpu);

    k.args(
        ocl::KernelArg::PtrReadOnly(hashesGpu),
        ocl::KernelArg::PtrReadOnly(hashDataGpu),
        (int)hashMap.hashDivisor,
        /*
        ocl::KernelArg::PtrReadOnly(indexes.volumes.getUMat(ACCESS_RW)),
        (int)indexes.list_size,
        (int)indexes.bufferNums,
        (int)indexes.hash_divisor,
        */


        ocl::KernelArg::WriteOnlyNoSize(points),
        ocl::KernelArg::WriteOnlyNoSize(normals),
        frameSize,
        ocl::KernelArg::ReadOnly(volUnitsData),
        cam2volTransGPU,
        cam2volRotGPU,
        vol2camRotGPU,
        float(volume.truncateThreshold),
        finv.val, cxy.val,
        boxMin.val, boxMax.val,
        tstep,
        voxelSize,
        voxelSizeInv,
        volumeUnitSize,
        volume.truncDist,
        volumeUnitResolution,
        volStrides
    );

    size_t globalSize[2];
    globalSize[0] = (size_t)frameSize.width;
    globalSize[1] = (size_t)frameSize.height;

    if (!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

void HashTSDFVolumeGPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if (_points.needed())
    {
        //TODO: remove it when it works w/o CPU code
        volUnitsData.copyTo(volUnitsDataCopy);
        //TODO: remove it when it works w/o CPU code
        //TODO: enable it when it's on GPU
        //UMat hashDataGpu(hashMap.capacity, 1, CV_32SC4);
        //Mat(hashMap.data, false).copyTo(hashDataGpu);

        std::vector<std::vector<ptype>> pVecs, nVecs;

        Range _fetchRange(0, hashMap.last);

        const int nstripes = -1;

        const HashTSDFVolumeGPU& volume(*this);
        bool needNormals(_normals.needed());
        Mutex mutex;

        auto _HashFetchPointsNormalsInvoker = [&](const Range& range)
        {
            std::vector<ptype> points, normals;
            for (int row = range.start; row < range.end; row++)
            {
                cv::Vec4i idx4 = hashMap.data[row];
                cv::Vec3i idx(idx4[0], idx4[1], idx4[2]);

                Point3f base_point = volume.volumeUnitIdxToVolume(idx);

                std::vector<ptype> localPoints;
                std::vector<ptype> localNormals;
                for (int x = 0; x < volume.volumeUnitResolution; x++)
                    for (int y = 0; y < volume.volumeUnitResolution; y++)
                        for (int z = 0; z < volume.volumeUnitResolution; z++)
                        {
                            cv::Vec3i voxelIdx(x, y, z);
                            TsdfVoxel voxel = new_at(voxelIdx, row);

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
        };

        parallel_for_(_fetchRange, _HashFetchPointsNormalsInvoker, nstripes);

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
        //TODO: remove it when it works w/o CPU code
        volUnitsData.copyTo(volUnitsDataCopy);

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
                n                  = volume.pose.rotation() * volume.getNormalVoxel(voxelPoint);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(HashPushNormals);
    }

}

int HashTSDFVolumeGPU::getVisibleBlocks(int currFrameId, int frameThreshold) const
{
    Mat cpuIndices = lastVisibleIndices.getMat(ACCESS_READ);

    int numVisibleBlocks = 0;
    //! TODO: Iterate over map parallely?
    for (int i = 0; i < hashMap.last; i++)
    {
        if (*cpuIndices.ptr<int>(i) > (currFrameId - frameThreshold))
            numVisibleBlocks++;
    }
    return numVisibleBlocks;
}

#endif

//template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(const VolumeParams& _params)
{
#ifdef HAVE_OPENCL
    if (ocl::useOpenCL())
        return makePtr<HashTSDFVolumeGPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
            _params.depthTruncThreshold, _params.unitResolution);
#endif
    return makePtr<HashTSDFVolumeCPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor, _params.tsdfTruncDist, _params.maxWeight,
        _params.depthTruncThreshold, _params.unitResolution);
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
