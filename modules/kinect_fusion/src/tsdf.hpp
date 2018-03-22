//TODO: add license

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include "precomp.hpp"

struct Voxel
{
    kftype v;
    int weight;
};

namespace cv
{

template<> class DataType<Voxel>
{
public:
    typedef Voxel       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_64F,
           channels     = 1,
           fmt          = (int)'v',
           type         = CV_MAKETYPE(depth, channels)
         };
};

}

typedef cv::Mat_<Voxel> Volume;

class TSDFVolume
{
    typedef Points::value_type p3type;

public:
    // dimension in voxels, size in meters
    TSDFVolume(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
               float _raycastStepFactor, float _gradientDeltaFactor);

    void integrate(Depth depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics);
    void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, Points points, Normals normals) const;

    kftype fetchVoxel(cv::Point3f p) const;
    kftype fetchi(cv::Point3i p) const;
    kftype interpolate(cv::Point3f p) const;
    p3type getNormalVoxel(cv::Point3f p) const;

    void fetchPoints(cv::OutputArray points) const;
    void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const;

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


