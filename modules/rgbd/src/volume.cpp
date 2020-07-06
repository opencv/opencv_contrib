// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <opencv2/rgbd/volume.hpp>

#include "hash_tsdf.hpp"
#include "tsdf.hpp"

namespace cv
{
namespace kinfu
{
cv::Ptr<VolumeParams> VolumeParams::defaultParams(VolumeType _volumeType)
{
    VolumeParams params;
    params.type        = _volumeType;
    params.maxWeight         = 64;
    params.raycastStepFactor = 0.25f;
    float volumeSize         = 3.0f;
    params.pose        = Affine3f().translate(Vec3f(-volumeSize/2.f, -volumeSize/2.f, 0.5f));
    switch (params.type)
    {
        case VolumeType::TSDF:
            params.resolution    = Vec3i::all(512);
            params.voxelSize           = volumeSize / 512.f;
            params.depthTruncThreshold = 0.f;  // depthTruncThreshold not required for TSDF
            break;
        case VolumeType::HASHTSDF:
            params.unitResolution = 16;
            params.voxelSize            = volumeSize / 512.f;
            params.depthTruncThreshold  = rgbd::Odometry::DEFAULT_MAX_DEPTH();
        default:
            //! TODO: Should throw some exception or error
            break;
    }
    params.tsdfTruncDist = 7 * params.voxelSize; //! About 0.04f in meters

    return makePtr<VolumeParams>(params);
}

cv::Ptr<VolumeParams> VolumeParams::coarseParams(VolumeType _volumeType)
{
    Ptr<VolumeParams> params = defaultParams(_volumeType);

    params->raycastStepFactor = 0.75f;
    float volumeSize = 3.0f;
    switch (params->type)
    {
        case VolumeType::TSDF:
            params->resolution = Vec3i::all(128);
            params->voxelSize = volumeSize/128.f;
            break;
        case VolumeType::HASHTSDF:
            params->voxelSize = volumeSize/128.f;
            break;
        default:
            break;
    }
    params->tsdfTruncDist = 2 * params->voxelSize; //! About 0.04f in meters
    return params;
}

cv::Ptr<Volume> makeVolume(const VolumeParams& _volumeParams)
{
    switch(_volumeParams.type)
    {
        case VolumeType::TSDF:
            return kinfu::makeTSDFVolume(_volumeParams);
            break;
        case VolumeType::HASHTSDF:
            return cv::makePtr<kinfu::HashTSDFVolumeCPU>(_volumeParams);
            break;
        default:
            return nullptr;
    }
}
cv::Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, cv::Affine3f _pose,
                           float _raycastStepFactor, float _truncDist, int _maxWeight,
                           float _truncateThreshold, Point3i _resolution)
{
    if (_volumeType == VolumeType::TSDF)
    {
        return kinfu::makeTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                              _resolution);
    }
    else if (_volumeType == VolumeType::HASHTSDF)
    {
/* #ifdef HAVE_OPENCL */
/*     if (cv::ocl::useOpenCL()) */
/*         return makeHashTSDFVolume<HashTSDFVolumeGPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, */
/*                                   _truncateThreshold); */
/* #endif */
        return kinfu::makeHashTSDFVolume<kinfu::HashTSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                                  _truncateThreshold);
    }
    else
        return nullptr;
}

}  // namespace kinfu
}  // namespace cv
