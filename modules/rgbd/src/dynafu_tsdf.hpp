// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_DYNAFU_TSDF_H__
#define __OPENCV_DYNAFU_TSDF_H__

#include "kinfu_frame.hpp"
#include "warpfield.hpp"

namespace cv {
namespace dynafu {

class TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolume(Point3i _res, float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose,
                           cv::kinfu::Intr intrinsics, Ptr<WarpField> wf) = 0;

    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals) const = 0;

    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals,
                                    bool fetchVoxels=false) const = 0;

    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;

    virtual void marchCubes(OutputArray _vertices, OutputArray _edges) const = 0;

    virtual void reset() = 0;

    virtual nodeNeighboursType const& getVoxelNeighbours(Point3i v, int& n) const = 0;

    virtual ~TSDFVolume() { }

    float voxelSize;
    float voxelSizeInv;
    Point3i volResolution;
    float maxWeight;
    cv::Affine3f pose;
    float raycastStepFactor;

    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;
};

cv::Ptr<TSDFVolume> makeTSDFVolume(Point3i _res,  float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor);

} // namespace dynafu
} // namespace cv
#endif
