// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_VOLUME_H__
#define __OPENCV_RGBD_VOLUME_H__

#include "intrinsics.hpp"
#include "opencv2/core/affine.hpp"

namespace cv
{
namespace kinfu
{
class CV_EXPORTS_W Volume
{
   public:
    Volume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor)
        : voxelSize(_voxelSize),
          voxelSizeInv(1.0f / voxelSize),
          pose(_pose),
          raycastStepFactor(_raycastStepFactor)
    {
    }

    virtual ~Volume(){};

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
                           const cv::kinfu::Intr& intrinsics)                        = 0;
    virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
                         cv::Size frameSize, cv::OutputArray points,
                         cv::OutputArray normals) const                                    = 0;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const       = 0;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
    virtual void reset()                                                                   = 0;

   public:
    const float voxelSize;
    const float voxelSizeInv;
    const cv::Affine3f pose;
    const float raycastStepFactor;
};

enum class VolumeType
{
    TSDF     = 0,
    HASHTSDF = 1
};

CV_EXPORTS_W cv::Ptr<Volume> makeVolume(VolumeType _volumeType, float _voxelSize, cv::Matx44f _pose,
                           float _raycastStepFactor, float _truncDist, int _maxWeight,
                           float _truncateThreshold, Vec3i _resolution);
}  // namespace kinfu
}  // namespace cv
#endif
