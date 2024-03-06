// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_RGBD_DYNAFU_HPP__
#define __OPENCV_RGBD_DYNAFU_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

#include "kinfu.hpp"

namespace cv {

namespace dynafu {

/** @brief DynamicFusion implementation

  This class implements a 3d reconstruction algorithm as described in @cite dynamicfusion.

  It takes a sequence of depth images taken from depth sensor
  (or any depth images source such as stereo camera matching algorithm or even raymarching renderer).
  The output can be obtained as a vector of points and their normals
  or can be Phong-rendered from given camera pose.

  It extends the KinectFusion algorithm to handle non-rigidly deforming scenes by maintaining a sparse
  set of nodes covering the geometry such that each node contains a warp to transform it from a canonical
  space to the live frame.

  An internal representation of a model is a voxel cuboid that keeps TSDF values
  which are a sort of distances to the surface (for details read the @cite kinectfusion article about TSDF).
  There is no interface to that representation yet.

  Note that DynamicFusion is based on the KinectFusion algorithm which is patented and its use may be
  restricted by the list of patents mentioned in README.md file in this module directory.

  That's why you need to set the OPENCV_ENABLE_NONFREE option in CMake to use DynamicFusion.
*/


/** Backwards compatibility for old versions */
using Params = kinfu::Params;

class CV_EXPORTS_W DynaFu
{
public:
    CV_WRAP static Ptr<DynaFu> create(const Ptr<kinfu::Params>& _params);
    virtual ~DynaFu();

    /** @brief Get current parameters */
    virtual const kinfu::Params& getParams() const = 0;

    /** @brief Renders a volume into an image

      Renders a 0-surface of TSDF using Phong shading into a CV_8UC4 Mat.
      Light pose is fixed in DynaFu params.

        @param image resulting image
        @param cameraPose pose of camera to render from. If empty then render from current pose
        which is a last frame camera pose.
    */

    CV_WRAP virtual void render(OutputArray image, const Matx44f& cameraPose = Matx44f::eye()) const = 0;

    /** @brief Gets points and normals of current 3d mesh

      The order of normals corresponds to order of points.
      The order of points is undefined.

        @param points vector of points which are 4-float vectors
        @param normals vector of normals which are 4-float vectors
     */
    CV_WRAP virtual void getCloud(OutputArray points, OutputArray normals) const = 0;

    /** @brief Gets points of current 3d mesh

     The order of points is undefined.

        @param points vector of points which are 4-float vectors
     */
    CV_WRAP virtual void getPoints(OutputArray points) const = 0;

    /** @brief Calculates normals for given points
        @param points input vector of points which are 4-float vectors
        @param normals output vector of corresponding normals which are 4-float vectors
     */
    CV_WRAP virtual  void getNormals(InputArray points, OutputArray normals) const = 0;

    /** @brief Resets the algorithm

    Clears current model and resets a pose.
    */
    CV_WRAP virtual void reset() = 0;

    /** @brief Get current pose in voxel space */
    virtual Affine3f getPose() const = 0;

    /** @brief Process next depth frame

      Integrates depth into voxel space with respect to its ICP-calculated pose.
      Input image is converted to CV_32F internally if has another type.

    @param depth one-channel image which size and depth scale is described in algorithm's parameters
    @return true if succeeded to align new frame with current scene, false if opposite
    */
    CV_WRAP virtual bool update(InputArray depth) = 0;

    virtual std::vector<Point3f> getNodesPos() const = 0;

    virtual void marchCubes(OutputArray vertices, OutputArray edges) const = 0;

    virtual void renderSurface(OutputArray depthImage, OutputArray vertImage, OutputArray normImage, bool warp=true) = 0;
};

} // dynafu::
} // cv::
#endif // __OPENCV_RGBD_DYNAFU_HPP__
