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
class HashTSDFVolume : public Volume
{
   public:
    // dimension in voxels, size in meters
    //! Use fixed volume cuboid
    HashTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
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

struct VolumeUnit
{
    VolumeUnit() : pVolume(nullptr){};
    ~VolumeUnit() = default;

    cv::Ptr<TSDFVolume> pVolume;
    cv::Vec3i index;
    bool isActive;
};

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const cv::Vec3i& x) const noexcept
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

typedef std::unordered_set<cv::Vec3i, tsdf_hash> VolumeUnitIndexSet;
typedef std::unordered_map<cv::Vec3i, VolumeUnit, tsdf_hash> VolumeUnitMap;

class HashTSDFVolumeCPU : public HashTSDFVolume
{
   public:
    // dimension in voxels, size in meters
    HashTSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                      float _truncDist, int _maxWeight, float _truncateThreshold,
                      int _volumeUnitRes, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
                           const cv::kinfu::Intr& intrinsics) override;
    virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
                         cv::Size frameSize, cv::OutputArray points,
                         cv::OutputArray normals) const override;

    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;

    virtual void reset() override;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    virtual TsdfVoxel at(const cv::Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel at(const cv::Point3f& point) const;

    float interpolateVoxelPoint(const Point3f& point) const;
    inline float interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

    //! Utility functions for coordinate transformations
    cv::Vec3i volumeToVolumeUnitIdx(cv::Point3f point) const;
    cv::Point3f volumeUnitIdxToVolume(cv::Vec3i volumeUnitIdx) const;

    cv::Point3f voxelCoordToVolume(cv::Vec3i voxelIdx) const;
    cv::Vec3i volumeToVoxelCoord(cv::Point3f point) const;

   public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volumeUnits;
};
cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Matx44f _pose,
                                           float _raycastStepFactor, float _truncDist,
                                           int _maxWeight, float truncateThreshold,
                                           int volumeUnitResolution = 16);
}  // namespace kinfu
}  // namespace cv
#endif
