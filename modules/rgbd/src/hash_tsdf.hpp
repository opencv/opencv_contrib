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

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Affine3f& cameraPose,
                           const cv::kinfu::Intr& intrinsics, const int frameId = 0) override
    {
        Derived* derived = static_cast<Derived*>(this);
        derived->integrate_(_depth, depthFactor, cameraPose, intrinsics, frameId);
    }
    virtual void raycast(const cv::Affine3f& cameraPose, const cv::kinfu::Intr& intrinsics,
                         cv::Size frameSize, cv::OutputArray points,
                         cv::OutputArray normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->raycast_(cameraPose, intrinsics, frameSize, points, normals);
    }
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->fetchNormals_(points, _normals);
    }
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override
    {
        const Derived* derived = static_cast<const Derived*>(this);
        derived->fetchPointsNormals_(points, normals);
    }
    size_t getTotalVolumeUnits() const
    {
        return static_cast<const Derived*>(this)->getTotalVolumeUnits_();
    }
    int getVisibleBlocks(int currFrameId, int frameThreshold) const
    {
        return static_cast<const Derived*>(this)->getVisibleBlocks_(currFrameId, frameThreshold);
    }

    TsdfVoxel at(const cv::Vec3i& volumeIdx) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->at_(volumeIdx);
    }
    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    virtual TsdfVoxel at(const cv::Point3f& point) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->at_(point);
    }

    inline TsdfType interpolateVoxel(const cv::Point3f& point) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->interpolateVoxel_(point);
    }
    Point3f getNormalVoxel(cv::Point3f p) const
    {
        const Derived* derived = static_cast<const Derived*>(this);
        return derived->getNormalVoxel_(p);
    }

    //! Utility functions for coordinate transformations
    cv::Vec3i volumeToVolumeUnitIdx(cv::Point3f point) const
    {
        return static_cast<const Derived*>(this)->volumeToVolumeUnitIdx_(point);
    }
    cv::Point3f volumeUnitIdxToVolume(cv::Vec3i volumeUnitIdx) const
    {
        return static_cast<const Derived*>(this)->volumeUnitIdxToVolume_(volumeUnitIdx);
    }

    cv::Point3f voxelCoordToVolume(cv::Vec3i voxelIdx) const
    {
        return static_cast<const Derived*>(this)->voxelCoordToVolume_(voxelIdx);
    }
    cv::Vec3i volumeToVoxelCoord(cv::Point3f point) const
    {
        return static_cast<const Derived*>(this)->volumeToVoxelCoord_(point);
    }

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
    HashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _raycastStepFactor, float _truncDist,
                   int _maxWeight, float _truncateThreshold, int _volumeUnitRes,
                   bool zFirstMemOrder = true);
    friend Derived;
};

struct VolumeUnit
{
    VolumeUnit() : pVolume(nullptr){};
    ~VolumeUnit() = default;

    cv::Ptr<TSDFVolume> pVolume;
    int lastVisibleIndex = 0;
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

class HashTSDFVolumeCPU : public HashTSDFVolume<HashTSDFVolumeCPU>
{
    typedef HashTSDFVolume<HashTSDFVolumeCPU> Base;

   public:
    // dimension in voxels, size in meters
    HashTSDFVolumeCPU(float _voxelSize, cv::Affine3f _pose, float _raycastStepFactor,
                      float _truncDist, int _maxWeight, float _truncateThreshold,
                      int _volumeUnitRes, bool zFirstMemOrder = true);

    HashTSDFVolumeCPU(const VolumeParams& _volumeParams, bool zFirstMemOrder = true);

    void integrate_(InputArray _depth, float depthFactor, const cv::Affine3f& cameraPose,
                    const cv::kinfu::Intr& intrinsics, const int frameId = 0);
    void raycast_(const cv::Affine3f& cameraPose, const cv::kinfu::Intr& intrinsics,
                  cv::Size frameSize, cv::OutputArray points, cv::OutputArray normals) const;

    void fetchNormals_(cv::InputArray points, cv::OutputArray _normals) const;
    void fetchPointsNormals_(cv::OutputArray points, cv::OutputArray normals) const;

    void reset_();
    size_t getTotalVolumeUnits_() const { return volumeUnits.size(); }
    int getVisibleBlocks_(int currFrameId, int frameThreshold) const;

    //! Return the voxel given the voxel index in the universal volume (1 unit = 1 voxel_length)
    TsdfVoxel at_(const cv::Vec3i& volumeIdx) const;

    //! Return the voxel given the point in volume coordinate system i.e., (metric scale 1 unit =
    //! 1m)
    TsdfVoxel at_(const cv::Point3f& point) const;

    inline TsdfType interpolateVoxel_(const cv::Point3f& point) const;
    Point3f getNormalVoxel_(cv::Point3f p) const;

    //! Utility functions for coordinate transformations
    cv::Vec3i volumeToVolumeUnitIdx_(cv::Point3f point) const;
    cv::Point3f volumeUnitIdxToVolume_(cv::Vec3i volumeUnitIdx) const;

    cv::Point3f voxelCoordToVolume_(cv::Vec3i voxelIdx) const;
    cv::Vec3i volumeToVoxelCoord_(cv::Point3f point) const;

   public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volumeUnits;
};
template<typename T>
cv::Ptr<HashTSDFVolume<T>> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose,
                                              float _raycastStepFactor, float _truncDist,
                                              int _maxWeight, float truncateThreshold,
                                              int volumeUnitResolution = 16)
{
    return cv::makePtr<T>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                          truncateThreshold, volumeUnitResolution);
}
}  // namespace kinfu
}  // namespace cv
#endif
