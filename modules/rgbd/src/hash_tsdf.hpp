// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_HASH_TSDF_H__
#define __OPENCV_HASH_TSDF_H__

#include "opencv2/core/affine.hpp"
#include "kinfu_frame.hpp"

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
    /* virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, */
    /*                      cv::OutputArray points, cv::OutputArray normals) const = 0; */

    /* virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0; */
    /* virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const = 0; */

    virtual void reset() = 0;


public:
    float           voxelSize;
    float           voxelSizeInv;
    cv::Affine3f    pose;
    int             maxWeight;
    float           raycastStepFactor;
    float           truncDist;
    uint16_t        volumeUnitResolution;
    uint16_t        volumeUnitSize;
    bool            zFirstMemOrder;

};
cv::Ptr<HashTSDFVolume> makeHashTSDFVolume(float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                       float _raycastStepFactor, int volumeUnitResolution = 16);
} // namespace kinfu
} // namespace cv
#endif

