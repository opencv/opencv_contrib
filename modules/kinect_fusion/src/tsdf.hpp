//TODO: add license

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "opencv2/kinect_fusion/utils.hpp"

class TSDFVolume
{
    typedef Points::value_type p3type;

public:
    // dimension in voxels, size in meters
    TSDFVolume(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor, float _gradientDeltaFactor);

    void integrate(Depth depth, float depthFactor, cv::Affine3f cameraPose, Intr intrinsics);
    void raycast(cv::Affine3f cameraPose, Intr intrinsics, Points points, Normals normals) const;

    kftype fetchVoxel(cv::Point3f p) const;
    kftype fetchi(cv::Point3i p) const;
    kftype interpolate(cv::Point3f p) const;
    p3type getNormalVoxel(cv::Point3f p) const;

    Points fetchCloud() const;

    void reset();

    // edgeResolution^3 array
    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    Volume volume;
    float edgeSize;
    int edgeResolution;
    float voxelSize;
    float voxelSizeInv;
    float truncDist;
    float raycastStepFactor;
    float gradientDeltaFactor;
    int maxWeight;
    cv::Affine3f pose;
};

#endif


