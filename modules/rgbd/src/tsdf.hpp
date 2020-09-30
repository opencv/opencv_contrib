// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_TSDF_H__
#define __OPENCV_KINFU_TSDF_H__

#include <opencv2/rgbd/volume.hpp>

#include "kinfu_frame.hpp"
#include "utils.hpp"

namespace cv
{
namespace kinfu
{
// TODO: Optimization possible:
// * TsdfType can be FP16
// * WeightType can be uint16

typedef int8_t TsdfType;
typedef uchar WeightType;

struct TsdfVoxel
{
    TsdfType tsdf;
    WeightType weight;
};

typedef Vec<uchar, sizeof(TsdfVoxel)> VecTsdfVoxel;

class TSDFVolume : public Volume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
               int _maxWeight, Point3i _resolution, bool zFirstMemOrder = true);
    virtual ~TSDFVolume() = default;

   public:

    Point3i volResolution;
    WeightType maxWeight;

    Point3f volSize;
    float truncDist;
    Vec4i volDims;
    Vec8i neighbourCoords;
};

class TSDFVolumeCPU : public TSDFVolume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
                  int _maxWeight, Vec3i _resolution, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
                           const cv::kinfu::Intr& intrinsics) override;
    virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
                         cv::Size frameSize, cv::OutputArray points,
                         cv::OutputArray normals) const override;

    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;

    virtual void reset() override;
    virtual TsdfVoxel at(const cv::Vec3i& volumeIdx) const;

    float interpolateVoxel(cv::Point3f p) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

#if USE_INTRINSICS
    float interpolateVoxel(const v_float32x4& p) const;
    v_float32x4 getNormalVoxel(const v_float32x4& p) const;
#endif
    Vec6f frameParams;
    Mat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Consist of Voxel elements
    Mat volume;
};

#ifdef HAVE_OPENCL
class TSDFVolumeGPU : public TSDFVolume
{
   public:
    // dimension in voxels, size in meters
    TSDFVolumeGPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
                  int _maxWeight, Point3i _resolution);

    virtual void integrate(InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
                           const cv::kinfu::Intr& intrinsics) override;
    virtual void raycast(const cv::Matx44f& cameraPose, const cv::kinfu::Intr& intrinsics,
                         cv::Size frameSize, cv::OutputArray _points,
                         cv::OutputArray _normals) const override;

    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray normals) const override;

    virtual void reset() override;

    Vec6f frameParams;
    UMat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Array elem is CV_32FC2, read as (float, int)
    // TODO: optimization possible to (fp16, int16), see Voxel definition
    UMat volume;
};
#endif
cv::Ptr<TSDFVolume> makeTSDFVolume(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                                   float _truncDist, int _maxWeight, Point3i _resolution);
}  // namespace kinfu
}  // namespace cv
#endif
