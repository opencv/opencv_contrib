// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_HASH_TSDF_H__
#define __OPENCV_HASH_TSDF_H__

#include <unordered_map>
#include <unordered_set>

#include "tsdf.hpp"

namespace cv {
namespace kinfu {

class HashTSDFVolume
{
public:
    // dimension in voxels, size in meters
    //! Use fixed volume cuboid
    explicit HashTSDFVolume(float _voxelSize, int _voxel_unit_res, cv::Affine3f _pose,
                   float _truncDist, int _maxWeight,
                   float _raycastStepFactor, bool zFirstMemOrder = true);

    virtual ~HashTSDFVolume() = default;
    virtual void integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) = 0;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals) const = 0;

    /* virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0; */
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;

    virtual void reset() = 0;


public:
    float           voxelSize;
    float           voxelSizeInv;
    cv::Affine3f    pose;
    int             maxWeight;
    float           raycastStepFactor;
    float           truncDist;
    uint16_t        volumeUnitResolution;
    float           volumeUnitSize;
    bool            zFirstMemOrder;
};

struct VolumeUnit
{
    explicit VolumeUnit() : pVolume(nullptr) {};
    ~VolumeUnit() = default;

    cv::Ptr<TSDFVolume> pVolume;
    cv::Vec3i  index;
    bool isActive;
};

//! Spatial hashing
struct tsdf_hash
{
    size_t operator()(const cv::Vec3i & x) const noexcept
    {
        size_t seed = 0;
        constexpr uint32_t GOLDEN_RATIO = 0x9e3779b9;
        for (uint16_t i = 0; i < 3; i++) {
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
    HashTSDFVolumeCPU(float _voxelSize, int _volume_unit_res, cv::Affine3f _pose,
                      float _truncDist, int _maxWeight,
                      float _raycastStepFactor, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor,
            cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) override;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals) const override;

    /* virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override; */
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;

    virtual void reset() override;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    virtual Voxel at(const cv::Vec3i &volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit = 1m)
    virtual Voxel at(const cv::Point3f &point) const;

    inline TsdfType interpolateVoxel(const cv::Point3f& point) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

    //! Utility functions for coordinate transformations
    cv::Vec3i volumeToVolumeUnitIdx(cv::Point3f point) const;
    cv::Point3f volumeUnitIdxToVolume(cv::Vec3i volumeUnitIdx) const;

    cv::Point3f voxelCoordToVolume(cv::Vec3i voxelIdx) const;
    cv::Vec3i   volumeToVoxelCoord(cv::Point3f point) const;

public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volumeUnits;
};

cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                       float _raycastStepFactor, int volumeUnitResolution = 16);
} // namespace kinfu
} // namespace cv
#endif

