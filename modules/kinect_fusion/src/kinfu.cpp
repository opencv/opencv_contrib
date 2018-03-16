//TODO: add license

#include "precomp.hpp"

namespace cv {
namespace kinfu {

class KinFu::KinFuImpl
{
public:    
    KinFuImpl(const KinFu::KinFuParams& _params);
    virtual ~KinFuImpl();

    const KinFu::KinFuParams& getParams() const;
    KinFu::KinFuParams& getParams();

    Image render() const;
    void fetchCloud(Points&, Normals&) const;

    void reset();

    //TODO: enable this when (if) features are ready

    /*
    const TSDFVolume& tsdf() const;
    TSDFVolume& tsdf();

    void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

    Affine3f getCameraPose (int time = -1) const;
    */

    bool operator()(InputArray depth);

private:
    KinFu::KinFuParams params;

    int frameCounter;
    Affine3f pose;
    Frame frame;

    ICP icp;
    TSDFVolume volume;
};

KinFu::KinFuParams KinFu::KinFuParams::defaultParams()
{
    KinFuParams p;

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
    //const int iters[] = {10, 5, 4, 0};
    const int iters[] = {10, 5, 4, 1};
    p.icpIterations.assign(iters, iters + sizeof(iters)/sizeof(int));

    for(size_t i = 0; i < p.icpIterations.size(); i++)
    {
        if(p.icpIterations[i]) p.pyramidLevels = i+1;
    }

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
    icp(params.intr, params.icpIterations, params.icpAngleThresh, params.icpDistThresh),
    volume(params.volumeDims, params.volumeSize, params.volumePose,
           params.tsdf_trunc_dist, params.tsdf_max_weight,
           params.raycast_step_factor, params.gradient_delta_factor)
{
    reset();
}

void KinFu::KinFuImpl::reset()
{
    frameCounter = 0;
    pose = Affine3f::Identity();
    volume.reset();

    //TODO: enable if (when) needed
    /*
    volume_ = cv::Ptr<cuda::TsdfVolume>(new cuda::TsdfVolume(params_.volume_dims));

    volume_->setTruncDist(params_.tsdf_trunc_dist);
    volume_->setMaxWeight(params_.tsdf_max_weight);
    volume_->setSize(params_.volume_size);
    volume_->setPose(params_.volume_pose);
    volume_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

    icp_ = cv::Ptr<cuda::ProjectiveICP>(new cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    */
}

KinFu::KinFuImpl::~KinFuImpl()
{

}

const KinFu::KinFuParams& KinFu::KinFuImpl::getParams() const
{ return params; }

KinFu::KinFuParams& KinFu::KinFuImpl::getParams()
{ return params; }

bool KinFu::KinFuImpl::operator()(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);
    // CV_Assert(_depth.type() == CV_16S);

    // this should convert CV_16S to CV_32F
    // TODO: make it better
    Depth depth = toDepth(_depth);

    Frame newFrame(depth, params.intr, params.pyramidLevels,
                   params.depthFactor,
                   params.bilateral_sigma_depth,
                   params.bilateral_sigma_spatial,
                   params.bilateral_kernel_size);

    if(frameCounter == 0)
    {
        // use depth instead of distance
        //volume.integrate(newFrame.distance, pose, params.intr);
        volume.integrate(depth, params.depthFactor, pose, params.intr);

        frame = newFrame;
    }
    else
    {
        Affine3f affine;
        bool success = icp.estimateTransform(affine, frame.points, frame.normals, newFrame.points, newFrame.normals);
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
            volume.integrate(depth, params.depthFactor, pose, params.intr);
        }

        // points and normals are allocated for this raycast call
        Points p(params.frameSize);
        Normals n(params.frameSize);
        volume.raycast(pose, params.intr, p, n);
        // build a pyramid of points and normals
        frame = Frame(p, n, params.pyramidLevels);
    }

    frameCounter++;
    return true;
}


Image KinFu::KinFuImpl::render() const
{
    return frame.render(0, params.lightPose);
}


void KinFu::KinFuImpl::fetchCloud(Points& p, Normals& n) const
{
    return volume.fetchCloud(p, n);
}


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

void KinFu::fetchCloud(Points & p, Normals & n) const
{
    return impl->fetchCloud(p, n);
}

bool KinFu::operator()(InputArray depth)
{
    return impl->operator()(depth);
}

Image KinFu::render() const
{
    return impl->render();
}

}
}
