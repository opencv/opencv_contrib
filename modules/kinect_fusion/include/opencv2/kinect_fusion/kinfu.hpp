// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE file found in this module's directory

#ifndef __OPENCV_KINECT_FUSION_HPP__
#define __OPENCV_KINECT_FUSION_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv {
namespace kinfu {
//! @addtogroup kinect_fusion
//! @{

/** @brief Camera intrinsics */
struct Intr
{
    /** Reprojects screen point to camera space given z coord. */
    struct Reprojector
    {
        Reprojector() {}
        inline Reprojector(Intr intr)
        {
            fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;
            cx = intr.cx, cy = intr.cy;
        }
        template<typename T>
        inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
        {
            T x = p.z * (p.x - cx) * fxinv;
            T y = p.z * (p.y - cy) * fyinv;
            return cv::Point3_<T>(x, y, p.z);
        }

        float fxinv, fyinv, cx, cy;
    };
    /** Projects camera space vector onto screen */
    struct Projector
    {
        inline Projector(Intr intr) : fx(intr.fx), fy(intr.fy), cx(intr.cx), cy(intr.cy) { }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p) const
        {
            T invz = T(1)/p.z;
            T x = fx*(p.x*invz) + cx;
            T y = fy*(p.y*invz) + cy;
            return cv::Point_<T>(x, y);
        }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p, cv::Point3_<T>& pixVec) const
        {
            T invz = T(1)/p.z;
            pixVec = cv::Point3_<T>(p.x*invz, p.y*invz, 1);
            T x = fx*pixVec.x + cx;
            T y = fy*pixVec.y + cy;
            return cv::Point_<T>(x, y);
        }
        float fx, fy, cx, cy;
    };
    Intr() : fx(), fy(), cx(), cy() { }
    Intr(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy) { }
    // scale intrinsics to pyramid level
    inline Intr scale(int pyr) const
    {
        float factor = (1.f /(1 << pyr));
        return Intr(fx*factor, fy*factor, cx*factor, cy*factor);
    }
    inline Reprojector makeReprojector() const { return Reprojector(*this); }
    inline Projector   makeProjector()   const { return Projector(*this);   }

    float fx, fy, cx, cy;
};

/** @brief KinectFusion implementation
  This class implements a 3d reconstruction algorithm described in
  @cite kinectfusion paper.

  It takes a sequence of depth images taken from depth sensor
  (or any depth images source such as stereo camera matching algorithm or even raymarching renderer).
  The output can be obtained as a vector of points and their normals
  or can be Phong-rendered from given camera pose.

  An internal representation of a model is a voxel cube that keeps TSDF values
  which are a sort of distances to the surface (for details read the @cite kinectfusion article about TSDF).
  There is no interface to that representation yet.
*/
class CV_EXPORTS KinFu
{
public:
    struct CV_EXPORTS Params
    {
        /** @brief Default parameters
        A set of parameters which provides better model quality, can be very slow.
        */
        static Params defaultParams();

        /** @brief Coarse parameters
        A set of parameters which provides better speed, can fail to match frames
        in case of rapid sensor motion.
        */
        static Params coarseParams();

        enum PlatformType
        {

            PLATFORM_CPU, PLATFORM_GPU
        };

        /** @brief A platform on which to run the algorithm.
         *
        Currently KinFu supports only one platform which is a CPU.
        GPU platform is to be implemented in the future.
        */
        PlatformType platform;

        /** @brief frame size in pixels */
        Size frameSize;

        /** @brief camera intrinsics */
        Intr intr;

        /** @brief pre-scale per 1 meter for input values

        Typical values are:
        * 5000 per 1 meter for the 16-bit PNG files of TUM database
        * 1 per 1 meter for the 32-bit float images in the ROS bag files
        */
        float depthFactor;

        /** @brief Depth sigma in meters for bilateral smooth */
        float bilateral_sigma_depth;
        /** @brief Spatial sigma in pixels for bilateral smooth */
        float bilateral_sigma_spatial;
        /** @brief Kernel size in pixels for bilateral smooth */
        int   bilateral_kernel_size;

        /** @brief Number of pyramid levels for ICP */
        int pyramidLevels;

        /** @brief Resolution of voxel cube

        Number of voxels in each cube edge.
        */
        int volumeDims;
        /** @brief Size of voxel cube side in meters */
        float volumeSize;

        /** @brief Minimal camera movement in meters

        Integrate new depth frame only if camera movement exceeds this value.
        */
        float tsdf_min_camera_movement;

        /** @brief initial volume pose in meters */
        Affine3f volumePose;

        /** @brief distance to truncate in meters

        Distances that exceed this value will be truncated in voxel cube values.
        */
        float tsdf_trunc_dist;

        /** @brief max number of frames per voxel

        Each voxel keeps running average of distances no longer than this value.
        */
        int tsdf_max_weight;

        /** @brief A length of one raycast step

        How much voxel sizes we skip each raycast step
        */
        float raycast_step_factor;

        // gradient delta in voxel sizes
        // fixed at 1.0f
        // float gradient_delta_factor;

        /** @brief light pose for rendering in meters */
        Vec3f lightPose;

        /** @brief distance theshold for ICP in meters */
        float icpDistThresh;
        /** angle threshold for ICP in radians */
        float icpAngleThresh;
        /** number of ICP iterations for each pyramid level */
        std::vector<int> icpIterations;

        // depth truncation is not used by default
        // float icp_truncate_depth_dist; //meters
    };

    KinFu(const Params& _params);
    virtual ~KinFu();

    /** @brief Get current parameters */
    const Params& getParams() const;
    Params& getParams();

    /** @brief Renders a volume into an image

      Renders a 0-surface of TSDF using Phong shading into a CV_8UC3 Mat.
      Light pose is fixed in KinFu params.

        @param image resulting image
        @param cameraPose pose of camera to render from. If empty then render from current pose
        which is a last frame camera pose.
    */

    void render(OutputArray image, const Affine3f cameraPose = Affine3f::Identity()) const;

    /** @brief Gets points and normals of current 3d mesh

      The order of normals corresponds to order of points.
      The order of points is undefined.

        @param points vector of points which are 4-float vectors
        @param normals vector of normals which are 4-float vectors
     */
    void fetchCloud(OutputArray points, OutputArray normals) const;

    /** @brief Gets points of current 3d mesh

     The order of points is undefined.

        @param points vector of points which are 4-float vectors
     */
    void fetchPoints(OutputArray points) const;

    /** @brief Calculates normals for given points
        @param points input vector of points which are 4-float vectors
        @param normals output vector of corresponding normals which are 4-float vectors
     */
    void fetchNormals(InputArray points, OutputArray normals) const;

    /** @brief Resets the algorithm

    Clears current model and resets a pose.
    */
    void reset();

    /** @brief Get current pose in voxel cube space */
    const Affine3f getPose() const;

    /** @brief Process next depth frame

      Integrates depth into voxel cube with respect to its ICP-calculated pose.
      Input image is converted to CV_32F internally if has another type.

    @param depth one-channel image which size and depth scale is described in algorithm's parameters
    @return true if succeded to align new frame with current scene, false if opposite
    */
    bool update(InputArray depth);

private:

    class KinFuImpl;
    cv::Ptr<KinFuImpl> impl;
};

//! @}
}
}
#endif
