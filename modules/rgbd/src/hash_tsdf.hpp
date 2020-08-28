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
template<typename Derived>
class HashTSDFVolume : public Volume
{
   public:
    virtual ~HashTSDFVolume() = default;

    virtual void reset() override
    {
        Derived* derived = static_cast<Derived*>(this);
        derived->reset_();
    }

    virtual void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
                           const int frameId = 0) override
    {
        Derived* derived = static_cast<Derived*>(this);
        derived->integrate_(_depth, depthFactor, cameraPose, intrinsics, frameId);
    }
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
                         OutputArray normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->raycast_(cameraPose, intrinsics, frameSize, points, normals);
    }
    virtual void fetchNormals(InputArray points, OutputArray _normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->fetchNormals_(points, _normals);
    }
    virtual void fetchPointsNormals(OutputArray points, OutputArray normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->fetchPointsNormals_(points, normals);
    }
    inline size_t getTotalVolumeUnits() const { return static_cast<const Derived*>(this)->getTotalVolumeUnits_(); }
    inline int getVisibleBlocks(int currFrameId, int frameThreshold) const
    {
        return static_cast<const Derived*>(this)->getVisibleBlocks_(currFrameId, frameThreshold);
    }

    inline TsdfVoxel at(const Vec3i& volumeIdx) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->at_(volumeIdx);
    }
    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    inline TsdfVoxel at(const Point3f& point) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->at_(point);
    }

    inline TsdfType interpolateVoxel(const Point3f& point) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->interpolateVoxel_(point);
    }
    inline Point3f getNormalVoxel(const Point3f& point) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->getNormalVoxel_(point);
    }

    //! Utility functions for coordinate transformations
    inline Vec3i volumeToVolumeUnitIdx(const Point3f& point) const
    {
        return static_cast<const Derived*>(this)->volumeToVolumeUnitIdx_(point);
    }
    inline Point3f volumeUnitIdxToVolume(const Vec3i& volumeUnitIdx) const
    {
        return static_cast<const Derived*>(this)->volumeUnitIdxToVolume_(volumeUnitIdx);
    }

    inline Point3f voxelCoordToVolume(const Vec3i& voxelIdx) const
    {
        return static_cast<const Derived*>(this)->voxelCoordToVolume_(voxelIdx);
    }
    inline Vec3i volumeToVoxelCoord(const Point3f& point) const { return static_cast<const Derived*>(this)->volumeToVoxelCoord_(point); }

   public:
    int maxWeight;
    float truncDist;
    float truncateThreshold;
    int volumeUnitResolution;
    float volumeUnitSize;
    bool zFirstMemOrder;

   protected:
    //! dimension in voxels, size in meters
    //! Use fixed volume cuboid
    //! Can be only called by derived class
    HashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
                   float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = true);
    friend Derived;
};

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

class HashTSDFVolumeCPU : public HashTSDFVolume<HashTSDFVolumeCPU>
{
    typedef HashTSDFVolume<HashTSDFVolumeCPU> Base;

   public:
    // dimension in voxels, size in meters
    HashTSDFVolumeCPU(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist, int _maxWeight,
                      float _truncateThreshold, int _volumeUnitRes, bool zFirstMemOrder = true);

    HashTSDFVolumeCPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = true);

    void integrate_(InputArray _depth, float depthFactor, const Matx44f& cameraPose, const kinfu::Intr& intrinsics,
                    const int frameId = 0);
    void raycast_(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize, OutputArray points,
                  OutputArray normals) const;

    void fetchNormals_(InputArray points, OutputArray _normals) const;
    void fetchPointsNormals_(OutputArray points, OutputArray normals) const;

    void reset_();
    size_t getTotalVolumeUnits_() const { return volumeUnits.size(); }
    int getVisibleBlocks_(int currFrameId, int frameThreshold) const;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    TsdfVoxel at_(const Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    TsdfVoxel at_(const Point3f& point) const;

    TsdfType interpolateVoxel_(const Point3f& point) const;

    Point3f getNormalVoxel_(const Point3f& point) const;

    //! Utility functions for coordinate transformations
    Vec3i volumeToVolumeUnitIdx_(const Point3f& point) const;
    Point3f volumeUnitIdxToVolume_(const Vec3i& volumeUnitIdx) const;

    Point3f voxelCoordToVolume_(const Vec3i& voxelIdx) const;
    Vec3i volumeToVoxelCoord_(const Point3f& point) const;

   public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volumeUnits;
};

template<typename T>
Ptr<HashTSDFVolume<T>> makeHashTSDFVolume(const VolumeParams& _volumeParams)
{
    return makePtr<T>(_volumeParams);
}

template<typename T>
Ptr<HashTSDFVolume<T>> makeHashTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                                          int _maxWeight, float truncateThreshold, int volumeUnitResolution = 16)
{
    return makePtr<T>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, truncateThreshold,
                      volumeUnitResolution);
}
}  // namespace kinfu
}  // namespace cv
#endif
