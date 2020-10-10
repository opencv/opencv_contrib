// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_HASH_TSDF_H__
#define __OPENCV_HASH_TSDF_H__

#include <opencv2/rgbd/volume.hpp>
#include <unordered_map>
#include <unordered_set>

#include "tsdf.hpp"

namespace cv
{
namespace kinfu
{
struct VolumeUnit
{
    VolumeUnit() : pVolume(nullptr){};
    ~VolumeUnit() = default;

    Ptr<TSDFVolume> pVolume;
    int lastVisibleIndex = 0;
    bool isActive;
};

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const Vec3i& x) const noexcept
    {
        size_t seed                     = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (uint16_t i = 0; i < 3; i++)
        {
            seed ^= std::hash<int>()(x[i]) + GOLDEN_RATIO + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

typedef std::unordered_set<Vec3i, tsdf_hash> VolumeUnitIndexSet;
typedef std::unordered_map<Vec3i, VolumeUnit, tsdf_hash> VolumeUnitMap;

class HashTSDFVolume : public Volume
{
   public:
    // dimension in voxels, size in meters
    //! Use fixed volume cuboid
    HashTSDFVolume(float _voxelSize, const Matx44f& _pose, float _raycastStepFactor, float _truncDist,
                   int _maxWeight, float _truncateThreshold, int _volumeUnitRes,
                   bool zFirstMemOrder = true);

    virtual ~HashTSDFVolume() = default;

   public:
    int maxWeight;
    float truncDist;
    float truncateThreshold;
    int volumeUnitResolution;
    float volumeUnitSize;
    bool zFirstMemOrder;
};

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
    size_t getTotalVolumeUnits() const { return volumeUnits.size(); }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    TsdfVoxel at(const Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    TsdfVoxel at(const Point3f& point) const;

    float interpolateVoxelPoint(const Point3f& point) const;
    float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord(const Point3f& point) const;

   public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volumeUnits;
};

template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(const VolumeParams& _volumeParams)
{
    return makePtr<T>(_volumeParams);
}

template<typename T>
Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                                          int _maxWeight, float truncateThreshold, int volumeUnitResolution = 16)
{
    return makePtr<T>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
                      volumeUnitResolution);
}
}  // namespace kinfu
}  // namespace cv
#endif
