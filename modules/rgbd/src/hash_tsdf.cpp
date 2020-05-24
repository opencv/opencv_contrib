// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd_wrapper.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/utility.hpp"
#include "opencv2/rgbd/depth.hpp"
#include "precomp.hpp"
#include "tsdf.hpp"
#include "hash_tsdf.hpp"
#include "opencl_kernels_rgbd.hpp"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "depth_to_3d.hpp"
#include "utils.hpp"

namespace cv {

namespace kinfu {

// TODO: Optimization possible:
// * volumeType can be FP16
// * weight can be int16
typedef float volumeType;
struct Voxel
{
    volumeType v;
    int weight;
};
typedef Vec<uchar, sizeof(Voxel)> VecT;

struct VolumeUnit
{
    explicit VolumeUnit() : p_volume(nullptr) {};
    ~VolumeUnit() = default;

    cv::Ptr<TSDFVolume> p_volume;
    cv::Vec3i  index;
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
    /* virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, */
    /*                      cv::OutputArray points, cv::OutputArray normals) const override; */

    /* virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override; */
    /* virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override; */

    virtual void reset() override;

    cv::Vec3i findVoxelUnitIndex(cv::Point3f p) const;
    /* volumeType interpolateVoxel(cv::Point3f p) const; */
    /* Point3f getNormalVoxel(cv::Point3f p) const; */
//! TODO: Make this private
public:
    //! Hashtable of individual smaller volume units
    VolumeUnitMap volume_units_;
};

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

                //! Find accessed TSDF volume unit for valid 3D vertex
                cv::Vec3i lower_bound = volume.findVoxelUnitIndex(
                        volPoint - cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));
                cv::Vec3i upper_bound = volume.findVoxelUnitIndex(
                        volPoint + cv::Point3f(volume.truncDist, volume.truncDist, volume.truncDist));

                //! TODO(Akash): Optimize this using differential analyzer algorithm
                for(int i = lower_bound[0]; i < upper_bound[0]; i++)
                    for(int j = lower_bound[1]; j < upper_bound[1]; j++)
                        for(int k = lower_bound[2]; k < lower_bound[2]; k++)
                        {
                            const cv::Vec3i tsdf_idx = cv::Vec3i(i, j, k);
                            //! If the index has not already been accessed
                            if(accessVolUnits.find(tsdf_idx) == accessVolUnits.end())
                            {
                                accessVolUnits.insert(tsdf_idx);
                                //! Adds entry to unordered_map
                                //! and allocate memory for the volume unit
                                VolumeUnit volume_unit = volume.volume_units_[tsdf_idx];
                                if(!volume_unit.p_volume)
                                {
                                    cv::Point3i volumeDims(volume.volumeUnitResolution,
                                                           volume.volumeUnitResolution,
                                                           volume.volumeUnitResolution);
                                    cv::Point3f volume_unit_origin = cv::Point3f(tsdf_idx);
                                    volume_unit.p_volume = makeTSDFVolume(volumeDims,
                                                                volume.volumeUnitSize,
                                                                volume.pose,
                                                                volume.truncDist,
                                                                volume.maxWeight,
                                                                volume.raycastStepFactor, volume_unit_origin);
                                }
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

            //! The volume unit should already be added into the Volume from the allocator
            assert(accessedVolUnit != volume.volume_units_.end());
            accessedVolUnit->second.p_volume->integrate(depth, depthFactor, cameraPose, intrinsics);
        }
    }

    HashTSDFVolumeCPU& volume;
    std::vector<cv::Vec3i> accessVolUnitsVec;
    const Depth& depth;
    float depthFactor;
    cv::Affine3f cameraPose;
    Intr intrinsics;
};


// use depth instead of distance (optimization)
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

    IntegrateSubvolumeInvoker integrate_i(*this, accessVolUnitsVec, depth, intrinsics, cameraPose, depthFactor);
    Range accessed_units_range(0, accessVolUnitsVec.size());
    parallel_for_(accessed_units_range, integrate_i);
}

cv::Vec3i HashTSDFVolumeCPU::findVoxelUnitIndex(cv::Point3f p) const
{
    return cv::Vec3i(cvFloor(p.x / volumeUnitSize),
                     cvFloor(p.y / volumeUnitSize),
                     cvFloor(p.z / volumeUnitSize));
}

cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                       float _raycastStepFactor, int volumeUnitResolution)
{
    return cv::makePtr<HashTSDFVolumeCPU>(_voxelSize, volumeUnitResolution, _pose, _truncDist, _maxWeight, _raycastStepFactor);
}

} // namespace kinfu
} // namespace cv

