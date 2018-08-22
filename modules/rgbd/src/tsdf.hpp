// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "precomp.hpp"
#include "kinfu_frame.hpp"

namespace cv {
namespace kinfu {


class TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolume(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor);

    virtual void integrate(cv::Ptr<Frame> depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) = 0;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, int pyramidLevels,
                         cv::Ptr<FrameGenerator> frameGenerator, cv::Ptr<Frame> frame) const = 0;

    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;

    virtual void reset() = 0;

    virtual ~TSDFVolume() { }

    float edgeSize;
    int edgeResolution;
    int maxWeight;
    cv::Affine3f pose;
};

cv::Ptr<TSDFVolume> makeTSDFVolume(cv::kinfu::Params::PlatformType t,
                                   int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor);

} // namespace kinfu
} // namespace cv
#endif
