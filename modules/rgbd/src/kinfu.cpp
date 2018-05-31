// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "fast_icp.hpp"
#include "tsdf.hpp"
#include "kinfu_frame.hpp"

namespace cv {
namespace kinfu {

class KinFu::KinFuImpl
{
public:
    KinFuImpl(const KinFu::Params& _params);
    virtual ~KinFuImpl();

    const KinFu::Params& getParams() const;
    void setParams(const KinFu::Params&);

    void render(OutputArray image, const Affine3f cameraPose = Affine3f::Identity()) const;

    void getCloud(OutputArray points, OutputArray normals) const;
    void getPoints(OutputArray points) const;
    void getNormals(InputArray points, OutputArray normals) const;

    void reset();

    const Affine3f getPose() const;

    bool update(InputArray depth);

private:
    KinFu::Params params;

    cv::Ptr<FrameGenerator> frameGenerator;
    cv::Ptr<ICP> icp;
    cv::Ptr<TSDFVolume> volume;

    int frameCounter;
    Affine3f pose;
    cv::Ptr<Frame> frame;
};

KinFu::Params KinFu::Params::defaultParams()
{
    Params p;

    p.platform = PLATFORM_CPU;

    p.frameSize = Size(640, 480);

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
    // first non-zero numbers are accepted
    const int iters[] = {10, 5, 4, 0};

    for(size_t i = 0; i < sizeof(iters)/sizeof(int); i++)
    {
        if(iters[i])
        {
            p.icpIterations.push_back(iters[i]);
        }
        else
            break;
    }
    p.pyramidLevels = (int)p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    p.volumeDims = 512;  //number of voxels

    p.volumeSize = 3.f;  //meters

    // default pose of volume cube
    p.volumePose = Affine3f().translate(Vec3f(-p.volumeSize/2, -p.volumeSize/2, 0.5f));
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.25f;  //in voxel sizes
    // gradient delta factor is fixed at 1.0f and is not used
    //p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.lightPose = p.volume_pose.translation()/4; //meters
    p.lightPose = Vec3f::all(0.f); //meters

    // depth truncation is not used by default
    //p.icp_truncate_depth_dist = 0.f;        //meters, disabled

    return p;
}

KinFu::Params KinFu::Params::coarseParams()
{
    Params p = defaultParams();

    // first non-zero numbers are accepted
    const int iters[] = {5, 3, 2};

    p.icpIterations.clear();
    for(size_t i = 0; i < sizeof(iters)/sizeof(int); i++)
    {
        if(iters[i])
        {
            p.icpIterations.push_back(iters[i]);
        }
        else
            break;
    }
    p.pyramidLevels = (int)p.icpIterations.size();

    p.volumeDims = 128; //number of voxels

    p.raycast_step_factor = 0.75f;  //in voxel sizes

    return p;
}

KinFu::KinFuImpl::KinFuImpl(const KinFu::Params &_params) :
    params(_params),
    frameGenerator(makeFrameGenerator(params.platform)),
    icp(makeICP(params.platform, params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    volume(makeTSDFVolume(params.platform, params.volumeDims, params.volumeSize, params.volumePose,
                          params.tsdf_trunc_dist, params.tsdf_max_weight,
                          params.raycast_step_factor)),
    frame()
{
    reset();
}

void KinFu::KinFuImpl::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    volume->reset();
}

KinFu::KinFuImpl::~KinFuImpl()
{

}

const KinFu::Params& KinFu::KinFuImpl::getParams() const
{
    return params;
}

void KinFu::KinFuImpl::setParams(const KinFu::Params& p)
{
    params = p;
}

const Affine3f KinFu::KinFuImpl::getPose() const
{
    return pose;
}

bool KinFu::KinFuImpl::update(InputArray _depth)
{
    ScopeTime st("kinfu update");

    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    cv::Ptr<Frame> newFrame = (*frameGenerator)();
    (*frameGenerator)(newFrame, _depth,
                      params.intr,
                      params.pyramidLevels,
                      params.depthFactor,
                      params.bilateral_sigma_depth,
                      params.bilateral_sigma_spatial,
                      params.bilateral_kernel_size);

    if(frameCounter == 0)
    {
        // use depth instead of distance
        volume->integrate(newFrame, params.depthFactor, pose, params.intr);

        frame = newFrame;
    }
    else
    {
        Affine3f affine;
        bool success = icp->estimateTransform(affine, frame, newFrame);
        if(!success)
            return false;

        pose = pose * affine;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            // use depth instead of distance
            volume->integrate(newFrame, params.depthFactor, pose, params.intr);
        }

        // raycast and build a pyramid of points and normals
        volume->raycast(pose, params.intr, params.frameSize,
                        params.pyramidLevels, frameGenerator, frame);
    }

    frameCounter++;
    return true;
}


void KinFu::KinFuImpl::render(OutputArray image, const Affine3f cameraPose) const
{
    ScopeTime st("kinfu render");

    const Affine3f id = Affine3f::Identity();
    if((cameraPose.rotation() == pose.rotation() && cameraPose.translation() == pose.translation()) ||
       (cameraPose.rotation() == id.rotation()   && cameraPose.translation() == id.translation()))
    {
        frame->render(image, 0, params.lightPose);
    }
    else
    {
        // raycast and build a pyramid of points and normals
        cv::Ptr<Frame> f = (*frameGenerator)();
        volume->raycast(cameraPose, params.intr, params.frameSize,
                        params.pyramidLevels, frameGenerator, f);
        f->render(image, 0, params.lightPose);
    }
}


void KinFu::KinFuImpl::getCloud(OutputArray p, OutputArray n) const
{
    volume->fetchPointsNormals(p, n);
}

void KinFu::KinFuImpl::getPoints(OutputArray points) const
{
    volume->fetchPointsNormals(points, noArray());
}

void KinFu::KinFuImpl::getNormals(InputArray points, OutputArray normals) const
{
    volume->fetchNormals(points, normals);
}

// importing class

KinFu::KinFu(const Params& _params)
{
    impl = makePtr<KinFu::KinFuImpl>(_params);
}

KinFu::~KinFu() { }

const KinFu::Params& KinFu::getParams() const
{
    return impl->getParams();
}

void KinFu::setParams(const Params& p)
{
    impl->setParams(p);
}

const Affine3f KinFu::getPose() const
{
    return impl->getPose();
}

void KinFu::reset()
{
    impl->reset();
}

void KinFu::getCloud(cv::OutputArray points, cv::OutputArray normals) const
{
    impl->getCloud(points, normals);
}

void KinFu::getPoints(OutputArray points) const
{
    impl->getPoints(points);
}

void KinFu::getNormals(InputArray points, OutputArray normals) const
{
    impl->getNormals(points, normals);
}

bool KinFu::update(InputArray depth)
{
    return impl->update(depth);
}

void KinFu::render(cv::OutputArray image, const Affine3f cameraPose) const
{
    impl->render(image, cameraPose);
}

} // namespace kinfu
} // namespace cv
