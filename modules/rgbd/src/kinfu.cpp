// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"

namespace cv {
namespace kinfu {

void Params::setInitialVolumePose(Matx33f R, Vec3f t)
{
    setInitialVolumePose(Affine3f(R,t).matrix);
}

void Params::setInitialVolumePose(Matx44f homogen_tf)
{
    Params::volumePose = homogen_tf;
}

Ptr<Params> Params::defaultParams()
{
    Params p;

    p.frameSize = Size(640, 480);

    p.volumeKind = VolumeType::TSDF;

    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = p.frameSize.width/2 - 0.5f;
    cy = p.frameSize.height/2 - 0.5f;
    p.intr = Matx33f(fx,  0, cx,
                      0, fy, cy,
                      0,  0,  1);

    // 5000 for the 16-bit PNG files
    // 1 for the 32-bit float images in the ROS bag files
    p.depthFactor = 5000;

    // sigma_depth is scaled by depthFactor when calling bilateral filter
    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icpAngleThresh = (float)(30. * CV_PI / 180.); // radians
    p.icpDistThresh = 0.1f; // meters

    p.icpIterations = {10, 5, 4};
    p.pyramidLevels = (int)p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    p.volumeDims = Vec3i::all(512); //number of voxels

    float volSize = 3.f;
    p.voxelSize = volSize/512.f; //meters

    // default pose of volume cube
    p.volumePose = Affine3f().translate(Vec3f(-volSize/2.f, -volSize/2.f, 0.5f)).matrix;
    p.tsdf_trunc_dist = 7 * p.voxelSize; // about 0.04f in meters
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.25f;  //in voxel sizes
    // gradient delta factor is fixed at 1.0f and is not used
    //p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.lightPose = p.volume_pose.translation()/4; //meters
    p.lightPose = Vec3f::all(0.f); //meters

    // depth truncation is not used by default but can be useful in some scenes
    p.truncateThreshold = 0.f; //meters

    return makePtr<Params>(p);
}

Ptr<Params> Params::coarseParams()
{
    Ptr<Params> p = defaultParams();

    p->icpIterations = {5, 3, 2};
    p->pyramidLevels = (int)p->icpIterations.size();

    float volSize = 3.f;
    p->volumeDims = Vec3i::all(128); //number of voxels
    p->voxelSize  = volSize/128.f;
    p->tsdf_trunc_dist = 2 * p->voxelSize; // 0.04f in meters

    p->raycast_step_factor = 0.75f;  //in voxel sizes

    return p;
}
Ptr<Params> Params::hashTSDFParams(bool isCoarse)
{
    Ptr<Params> p;
    if(isCoarse)
        p = coarseParams();
    else
        p = defaultParams();
    p->volumeKind = VolumeType::HashTSDF;
    p->truncateThreshold = 4.f;
    return p;
}

Ptr<Params> Params::coloredTSDFParams(bool isCoarse)
{
    Ptr<Params> p;
    if (isCoarse)
        p = coarseParams();
    else
        p = defaultParams();
    p->volumeKind = VolumeType::ColorTSDF;

    return p;
}

// MatType should be Mat or UMat
template< typename MatType>
class KinFuImpl : public KinFu
{
public:
    KinFuImpl(const Params& _params);
    virtual ~KinFuImpl();

    static VolumeSettings paramsToSettings(const Params& params);
    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image) const CV_OVERRIDE;
    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth) CV_OVERRIDE;

    bool updateT(const MatType& depth);

private:
    Params params;
    VolumeSettings settings;

    Odometry icp;
    Volume volume;

    int frameCounter;
    Matx44f pose;
    OdometryFrame prevFrame;
    OdometryFrame renderFrame;
};

template< typename MatType >
VolumeSettings KinFuImpl<MatType>::paramsToSettings(const Params& params)
{
    VolumeSettings vs(VolumeType::TSDF);
    vs.setVoxelSize(params.voxelSize);
    vs.setVolumePose(params.volumePose);
    vs.setRaycastStepFactor(params.raycast_step_factor);
    vs.setTsdfTruncateDistance(params.tsdf_trunc_dist);
    vs.setMaxWeight(params.tsdf_max_weight);
    vs.setMaxDepth(params.truncateThreshold);
    vs.setCameraIntegrateIntrinsics(params.intr);
    vs.setDepthFactor(params.depthFactor);
    vs.setVolumeResolution(params.volumeDims);

    return vs;
}

template< typename MatType >
KinFuImpl<MatType>::KinFuImpl(const Params &_params) :
    params(_params),
    settings(paramsToSettings(params)),
    volume(VolumeType::TSDF, settings)
{
    OdometrySettings ods;
    ods.setMaxRotation(30.f);
    ods.setMaxTranslation(params.voxelSize * params.volumeDims[0] * 0.5f);
    ods.setCameraMatrix(Mat(params.intr));
    icp = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);

    reset();
}

template< typename MatType >
void KinFuImpl<MatType >::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity().matrix;
    volume.reset();
}

template< typename MatType >
KinFuImpl<MatType>::~KinFuImpl()
{ }

template< typename MatType >
const Params& KinFuImpl<MatType>::getParams() const
{
    return params;
}

template< typename MatType >
Affine3f KinFuImpl<MatType>::getPose() const
{
    return pose;
}


template<>
bool KinFuImpl<Mat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    Mat depth;
    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getMat());
    }
}


template<>
bool KinFuImpl<UMat>::update(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    UMat depth;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        return updateT(depth);
    }
    else
    {
        return updateT(_depth.getUMat());
    }
}


template< typename MatType >
bool KinFuImpl<MatType>::updateT(const MatType& _depth)
{
    CV_TRACE_FUNCTION();

    MatType depth;
    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    OdometryFrame newFrame(depth);

    if(frameCounter == 0)
    {
        icp.prepareFrame(newFrame);
        // use depth instead of distance
        volume.integrate(depth, pose);
    }
    else
    {
        Affine3f affine;
        Matx44d mrt;
        Mat Rt;
        icp.prepareFrames(newFrame, prevFrame);
        bool success = icp.compute(newFrame, prevFrame, Rt);
        if(!success)
            return false;
        affine.matrix = Matx44f(Rt);
        pose = (Affine3f(pose) * affine).matrix;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume.integrate(depth, pose);
        }

        MatType points, normals;
        volume.raycast(pose, points, normals);

        std::vector<MatType> pch(3);
        split(points, pch);
        newFrame = OdometryFrame(pch[2], noArray(), noArray(), normals);
    }

    renderFrame = newFrame;
    prevFrame = newFrame;

    frameCounter++;
    return true;
}


template< typename MatType >
void KinFuImpl<MatType>::render(OutputArray image) const
{
    CV_TRACE_FUNCTION();
    MatType pts, nrm;
    renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM,  0);
    detail::renderPointsNormals(pts, nrm, image, params.lightPose);
}


template< typename MatType >
void KinFuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);
    MatType points, normals;
    volume.raycast(_cameraPose, points, normals);
    detail::renderPointsNormals(points, normals, image, params.lightPose);
}


template< typename MatType >
void KinFuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    volume.fetchPointsNormals(p, n);
}


template< typename MatType >
void KinFuImpl<MatType>::getPoints(OutputArray points) const
{
    volume.fetchPointsNormals(points, noArray());
}


template< typename MatType >
void KinFuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    volume.fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<KinFu> KinFu::create(const Ptr<Params>& params)
{
    CV_Assert((int)params->icpIterations.size() == params->pyramidLevels);
    CV_Assert(params->intr(0,1) == 0 && params->intr(1,0) == 0 && params->intr(2,0) == 0 && params->intr(2,1) == 0 && params->intr(2,2) == 1);
#ifdef HAVE_OPENCL
    if(cv::ocl::useOpenCL())
        return makePtr< KinFuImpl<UMat> >(*params);
#endif
        return makePtr< KinFuImpl<Mat> >(*params);
}

#else
Ptr<KinFu> KinFu::create(const Ptr<Params>& /* params */)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

KinFu::~KinFu() {}

} // namespace kinfu
} // namespace cv
