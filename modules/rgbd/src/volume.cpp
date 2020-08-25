// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include <opencv2/rgbd/volume.hpp>

#include "tsdf.hpp"
#include "hash_tsdf.hpp"

namespace cv
{
namespace kinfu
{
cv::Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, cv::Matx44f _pose,
                           float _raycastStepFactor, float _truncDist, int _maxWeight,
                           float _truncateThreshold, Vec3i _resolution)
{
    Point3i _presolution = _resolution;
    if (_volumeType == VolumeType::TSDF)
    {
        return makeTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                              _presolution);
    }
    else if (_volumeType == VolumeType::HASHTSDF)
    {
        return makeHashTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight,
                                  _truncateThreshold);
    }
    else
        return nullptr;
}

}  // namespace kinfu
}  // namespace cv
