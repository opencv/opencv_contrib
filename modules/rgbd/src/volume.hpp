// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_VOLUME_H__
#define __OPENCV_VOLUME_H__

#include "precomp.hpp"
#include "kinfu_frame.hpp"
#include "opencv2/core/affine.hpp"

namespace cv
{
namespace kinfu
{
class Volume
{
   public:
    explicit Volume(float _voxelSize, cv::Affine3f _pose, float _raycastStepFactor)
        : voxelSize(_voxelSize),
          voxelSizeInv(1.0f / voxelSize),
          pose(_pose),
          raycastStepFactor(_raycastStepFactor)
    {
    }

    virtual ~Volume(){};

    virtual void integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose,
                           cv::kinfu::Intr intrinsics)                                     = 0;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals) const            = 0;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
    virtual void reset()                                                                   = 0;

   public:
    const float voxelSize;
    const float voxelSizeInv;
    const cv::Affine3f pose;
    const float raycastStepFactor;
};

// TODO: Optimization possible:
// * TsdfType can be FP16
// * weight can be uint16
typedef float TsdfType;
struct TsdfVoxel
{
    TsdfType tsdf;
    int weight;
};
typedef Vec<uchar, sizeof(TsdfVoxel)> VecTsdfVoxel;

}  // namespace kinfu
}  // namespace cv
#endif
