//TODO: add license

#include "precomp.hpp"
#include "icp.hpp"
#include "tsdf.hpp"
#include "frame.hpp"

namespace cv {
namespace kinfu {

class KinFu::KinFuImpl
{
public:    
    KinFuImpl(const KinFu::KinFuParams& _params);
    virtual ~KinFuImpl();

    const KinFu::KinFuParams& getParams() const;
    KinFu::KinFuParams& getParams();

    void render(OutputArray image, const Affine3f cameraPose = Affine3f::Identity()) const;

    void fetchCloud(OutputArray points, OutputArray normals) const;
    void fetchPoints(OutputArray points) const;
    void fetchNormals(InputArray points, OutputArray normals) const;

    void reset();

    const Affine3f getPose() const;

    bool operator()(InputArray depth);

private:
    KinFu::KinFuParams params;

    int frameCounter;
    Affine3f pose;
    cv::Ptr<Frame> frame;

    cv::Ptr<FrameGenerator> frameGenerator;
    cv::Ptr<ICP> icp;
    cv::Ptr<TSDFVolume> volume;
};

KinFu::KinFuParams KinFu::KinFuParams::defaultParams()
{
    KinFuParams p;

    p.platform = PLATFORM_CPU;

    p.frameSize = Size(640, 480);

    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = p.frameSize.width/2 - 0.5f;
    cy = p.frameSize.height/2 - 0.5f;
    p.intr = Intr(fx, fy, cx, cy);

    // 5000 for the 16-bit PNG files
    // 1 for the 32-bit float images in the ROS bag files
    p.depthFactor = 5000;

    // sigma_depth is scaled by depthFactor when calling bilateral filter
    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icpAngleThresh = 30.f * CV_PI / 180.f; // radians
    p.icpDistThresh = 0.1f; // meters
    // default value
    // first non-zero numbers are accepted
    //const int iters[] = {10, 5, 4, 0};
    const int iters[] = {10, 5, 4, 1};

    for(size_t i = 0; i < sizeof(iters)/sizeof(int); i++)
    {
        if(iters[i])
        {
            p.icpIterations.push_back(iters[i]);
        }
        else
            break;
    }
    p.pyramidLevels = p.icpIterations.size();

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    // default value
    //p.volumeDims = 512;  //number of voxels
    p.volumeDims = 128;

    p.volumeSize = 3.f;  //meters

    // default pose of volume cube
    p.volumePose = Affine3f().translate(Vec3f(-p.volumeSize/2, -p.volumeSize/2, 0.5f));
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.lightPose = p.volume_pose.translation()/4; //meters
    p.lightPose = Vec3f::all(0.f); //meters

    // depth truncation is not used by default
    //p.icp_truncate_depth_dist = 0.f;        //meters, disabled

    return p;
}

KinFu::KinFuImpl::KinFuImpl(const KinFu::KinFuParams &_params) :
    params(_params),
    frame(),
    frameGenerator(makeFrameGenerator(params.platform)),
    icp(makeICP(params.platform, params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh)),
    volume(makeTSDFVolume(params.platform, params.volumeDims, params.volumeSize, params.volumePose,
                          params.tsdf_trunc_dist, params.tsdf_max_weight,
                          params.raycast_step_factor, params.gradient_delta_factor))
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

const KinFu::KinFuParams& KinFu::KinFuImpl::getParams() const
{
    return params;
}

KinFu::KinFuParams& KinFu::KinFuImpl::getParams()
{
    return params;
}

const Affine3f KinFu::KinFuImpl::getPose() const
{
    return pose;
}

bool KinFu::KinFuImpl::operator()(InputArray _depth)
{
    ScopeTime st("kinfu");

    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);

    cv::Ptr<Frame> newFrame = (*frameGenerator)(_depth, params.intr, params.pyramidLevels,
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
        {
            reset();
            return false;
        }

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
        frame = volume->raycast(pose, params.intr, params.frameSize,
                                params.pyramidLevels, frameGenerator);
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
        cv::Ptr<Frame> f = volume->raycast(cameraPose, params.intr, params.frameSize,
                            params.pyramidLevels, frameGenerator);
        f->render(image, 0, params.lightPose);
    }
}


void KinFu::KinFuImpl::fetchCloud(OutputArray p, OutputArray n) const
{
    volume->fetchPointsNormals(p, n);
}

void KinFu::KinFuImpl::fetchPoints(OutputArray points) const
{
    volume->fetchPointsNormals(points, noArray());
}

void KinFu::KinFuImpl::fetchNormals(InputArray points, OutputArray normals) const
{
    volume->fetchNormals(points, normals);
}

// importing class

KinFu::KinFu(const KinFuParams& _params)
{
    impl = makePtr<KinFu::KinFuImpl>(_params);
}

KinFu::~KinFu() { }

const KinFu::KinFuParams& KinFu::getParams() const
{
    return impl->getParams();
}

KinFu::KinFuParams& KinFu::getParams()
{
    return impl->getParams();
}

const Affine3f KinFu::getPose() const
{
    return impl->getPose();
}

void KinFu::reset()
{
    impl->reset();
}

void KinFu::fetchCloud(cv::OutputArray points, cv::OutputArray normals) const
{
    impl->fetchCloud(points, normals);
}

void KinFu::fetchPoints(OutputArray points) const
{
    impl->fetchPoints(points);
}

void KinFu::fetchNormals(InputArray points, OutputArray normals) const
{
    impl->fetchNormals(points, normals);
}

bool KinFu::operator()(InputArray depth)
{
    return impl->operator()(depth);
}

void KinFu::render(cv::OutputArray image, const Affine3f cameraPose) const
{
    impl->render(image, cameraPose);
}

}
}
