// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this
// module's directory

#ifndef __OPENCV_RGBD_LARGEKINFU_HPP__
#define __OPENCV_RGBD_LARGEKINFU_HPP__

#include <opencv2/3d.hpp>

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv
{
namespace large_kinfu
{

/** @brief Large Scale Dense Depth Fusion implementation

  This class implements a 3d reconstruction algorithm for larger environments using
  Spatially hashed TSDF volume "Submaps".
  It also runs a periodic posegraph optimization to minimize drift in tracking over long sequences.
  Currently the algorithm does not implement a relocalization or loop closure module.
  Potentially a Bag of words implementation or RGBD relocalization as described in
  Glocker et al. ISMAR 2013 will be implemented

  It takes a sequence of depth images taken from depth sensor
  (or any depth images source such as stereo camera matching algorithm or even raymarching
  renderer). The output can be obtained as a vector of points and their normals or can be
  Phong-rendered from given camera pose.

  An internal representation of a model is a spatially hashed voxel cube that stores TSDF values
  which represent the distance to the closest surface (for details read the @cite kinectfusion article
  about TSDF). There is no interface to that representation yet.

  For posegraph optimization, a Submap abstraction over the Volume class is created.
  New submaps are added to the model when there is low visibility overlap between current viewing frustrum
  and the existing volume/model. Multiple submaps are simultaneously tracked and a posegraph is created and
  optimized periodically.

  LargeKinfu does not use any OpenCL acceleration yet.
  To enable or disable it explicitly use cv::setUseOptimized() or cv::ocl::setUseOpenCL().

  This implementation is inspired from Kintinuous, InfiniTAM and other SOTA algorithms

  You need to set the OPENCV_ENABLE_NONFREE option in CMake to use KinectFusion.
*/
class CV_EXPORTS_W LargeKinfu
{
   public:
    CV_WRAP static Ptr<LargeKinfu> create();
    virtual ~LargeKinfu() = default;

    CV_WRAP virtual void render(OutputArray image) const = 0;
    CV_WRAP virtual void render(OutputArray image, const Matx44f& cameraPose) const = 0;

    CV_WRAP virtual void getCloud(OutputArray points, OutputArray normals) const = 0;

    CV_WRAP virtual void getPoints(OutputArray points) const = 0;

    CV_WRAP virtual void getNormals(InputArray points, OutputArray normals) const = 0;

    CV_WRAP virtual void reset() = 0;

    virtual const Affine3f getPose() const = 0;

    CV_WRAP virtual bool update(InputArray depth) = 0;
};

}  // namespace large_kinfu
}  // namespace cv
#endif
