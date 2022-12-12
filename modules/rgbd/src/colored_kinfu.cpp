// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "opencv2/3d.hpp"

namespace cv {
namespace colored_kinfu {
using namespace kinfu;

void Params::setInitialVolumePose(Matx33f R, Vec3f t)
{
    setInitialVolumePose(Affine3f(R,t).matrix);
}

void Params::setInitialVolumePose(Matx44f homogen_tf)
{
    volumePose = homogen_tf;
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

    float rgb_fx, rgb_fy, rgb_cx, rgb_cy;
    rgb_fx = 515.0f;
    rgb_fy = 550.0f;
    rgb_cx = 319.5f;
    rgb_cy = 239.5f;
    p.rgb_intr = Matx33f(rgb_fx,      0, rgb_cx,
                              0, rgb_fy, rgb_cy,
                              0,      0,      1);

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
    if (isCoarse)
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
    if(isCoarse)
        p = coarseParams();
    else
        p = defaultParams();
    p->volumeKind = VolumeType::ColorTSDF;

    return p;
}

// MatType should be Mat or UMat
template< typename MatType>
class ColoredKinFuImpl : public ColoredKinFu
{
public:
    ColoredKinFuImpl(const Params& _params);
    virtual ~ColoredKinFuImpl();

    static VolumeSettings paramsToSettings(const Params& params);
    const Params& getParams() const CV_OVERRIDE;

    void render(OutputArray image) const CV_OVERRIDE;
    void render(OutputArray image, const Matx44f& cameraPose) const CV_OVERRIDE;

    virtual void getCloud(OutputArray points, OutputArray normals) const CV_OVERRIDE;
    void getPoints(OutputArray points) const CV_OVERRIDE;
    void getNormals(InputArray points, OutputArray normals) const CV_OVERRIDE;

    void reset() CV_OVERRIDE;

    Affine3f getPose() const CV_OVERRIDE;

    bool update(InputArray depth, InputArray rgb) CV_OVERRIDE;

    bool updateT(const MatType& depth, const MatType& rgb);

private:
    Params params;
    VolumeSettings settings;

    Odometry icp;
    Volume volume;

    int frameCounter;
    Matx44f pose;
    // Mutable because it contains cache updated by icp.prepareFrame() during rendering
    mutable OdometryFrame renderFrame;
    OdometryFrame prevFrame;
};

template< typename MatType >
VolumeSettings ColoredKinFuImpl<MatType>::paramsToSettings(const Params& params)
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
ColoredKinFuImpl<MatType>::ColoredKinFuImpl(const Params &_params) :
    params(_params),
    settings(paramsToSettings(params)),
    volume(VolumeType::ColorTSDF, settings)
{
    OdometrySettings ods;
    ods.setCameraMatrix(Mat(params.intr));
    ods.setMaxRotation(30.f);
    ods.setMaxTranslation(params.voxelSize * (float)params.volumeDims[0] * 0.5f);
    ods.setIterCounts(params.icpIterations);

    icp = Odometry(OdometryType::DEPTH, ods, OdometryAlgoType::FAST);

    reset();
}

template< typename MatType >
void ColoredKinFuImpl<MatType >::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity().matrix;
    volume.reset();
}

template< typename MatType >
ColoredKinFuImpl<MatType>::~ColoredKinFuImpl()
{ }

template< typename MatType >
const Params& ColoredKinFuImpl<MatType>::getParams() const
{
    return params;
}

template< typename MatType >
Affine3f ColoredKinFuImpl<MatType>::getPose() const
{
    return pose;
}


template<>
bool ColoredKinFuImpl<Mat>::update(InputArray _depth, InputArray _rgb)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    Mat depth;
    Mat rgb;
    if(_depth.isUMat())
    {
        _depth.copyTo(depth);
        _rgb.copyTo(rgb);
        return updateT(depth, rgb);
    }
    else
    {
        return updateT(_depth.getMat(), _rgb.getMat());
    }
}


template<>
bool ColoredKinFuImpl<UMat>::update(InputArray _depth, InputArray _rgb)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    UMat depth;
    UMat rgb;
    if(!_depth.isUMat())
    {
        _depth.copyTo(depth);
        _rgb.copyTo(rgb);
        return updateT(depth, rgb);
    }
    else
    {
        return updateT(_depth.getUMat(), _rgb.getUMat());
    }
}


template< typename MatType >
bool ColoredKinFuImpl<MatType>::updateT(const MatType& _depth, const MatType& _rgb)
{
    CV_TRACE_FUNCTION();

    MatType depth, rgb;

    if(_depth.type() != DEPTH_TYPE)
        _depth.convertTo(depth, DEPTH_TYPE);
    else
        depth = _depth;

    if (_rgb.type() != COLOR_TYPE)
    {
        MatType rgb_tmp;
        std::vector<MatType> channels;
        _rgb.convertTo(rgb_tmp, COLOR_TYPE);
        cv::split(rgb_tmp, channels);
        // we use 4-channel RGB0 image
        // for vectorization simplicity
        channels.push_back(MatType::zeros(channels[0].size(), CV_32F));
        merge(channels, rgb);
    }
    else
        rgb = _rgb;

    OdometryFrame newFrame(depth, rgb);

    if(frameCounter == 0)
    {
        icp.prepareFrame(newFrame);

        // use depth instead of distance
        volume.integrate(depth, rgb, pose);
    }
    else
    {
        Affine3f affine;
        Matx44d mrt;
        Mat Rt;
        icp.prepareFrames(newFrame, prevFrame);
        bool success = icp.compute(newFrame, prevFrame, Rt);

        if (!success)
            return false;
        affine.matrix = Matx44f(Rt);
        pose = (Affine3f(pose) * affine).matrix;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume.integrate(depth, rgb, pose);
        }
        MatType points, normals, colors;

        volume.raycast(pose, points, normals, colors);
        std::vector<MatType> pch(3);
        split(points, pch);
        newFrame = OdometryFrame(pch[2], colors, noArray(), normals);
    }

    renderFrame = newFrame;
    prevFrame = newFrame;

    frameCounter++;
    return true;
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::render(OutputArray image) const
{
    CV_TRACE_FUNCTION();
    MatType pts, nrm, rgb;
    icp.prepareFrame(renderFrame);
    renderFrame.getPyramidAt(pts, OdometryFramePyramidType::PYR_CLOUD, 0);
    renderFrame.getPyramidAt(nrm, OdometryFramePyramidType::PYR_NORM, 0);
    renderFrame.getImage(rgb);

    detail::renderPointsNormalsColors(pts, nrm, rgb, image);
}

template< typename MatType >
void ColoredKinFuImpl<MatType>::render(OutputArray image, const Matx44f& _cameraPose) const
{
    CV_TRACE_FUNCTION();

    Affine3f cameraPose(_cameraPose);
    MatType points, normals, colors;
    volume.raycast(_cameraPose, points, normals, colors);
    detail::renderPointsNormalsColors(points, normals, colors, image);
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getCloud(OutputArray p, OutputArray n) const
{
    volume.fetchPointsNormals(p, n);
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getPoints(OutputArray points) const
{
    volume.fetchPointsNormals(points, noArray());
}


template< typename MatType >
void ColoredKinFuImpl<MatType>::getNormals(InputArray points, OutputArray normals) const
{
    volume.fetchNormals(points, normals);
}

// importing class

#ifdef OPENCV_ENABLE_NONFREE

Ptr<ColoredKinFu> ColoredKinFu::create(const Ptr<Params>& params)
{
    CV_Assert((int)params->icpIterations.size() == params->pyramidLevels);
    CV_Assert(params->intr(0,1) == 0 && params->intr(1,0) == 0 && params->intr(2,0) == 0 && params->intr(2,1) == 0 && params->intr(2,2) == 1);
    return makePtr< ColoredKinFuImpl<Mat> >(*params);
}

#else
Ptr<ColoredKinFu> ColoredKinFu::create(const Ptr<Params>& /* params */)
{
    CV_Error(Error::StsNotImplemented,
             "This algorithm is patented and is excluded in this configuration; "
             "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library");
}
#endif

ColoredKinFu::~ColoredKinFu() {}

} // namespace kinfu
} // namespace cv
