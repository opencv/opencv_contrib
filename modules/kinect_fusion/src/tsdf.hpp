//TODO: add license

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "precomp.hpp"

//TODO: put some members in parent class
//TODO: invent something with Points, Normals, Depth

class TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolume(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor, float _gradientDeltaFactor);

    virtual void integrate(Depth depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) = 0;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, Points points, Normals normals) const = 0;

    virtual void fetchPoints(cv::OutputArray points) const = 0;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const = 0;

    virtual void reset() = 0;

    virtual ~TSDFVolume() { }
};

cv::Ptr<TSDFVolume> makeTSDFVolume(cv::kinfu::KinFu::KinFuParams::PlatformType t,
                                   int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor, float _gradientDeltaFactor);

#endif


