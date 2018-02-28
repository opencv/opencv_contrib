//TODO: add license

#include "precomp.hpp"

namespace cv {
namespace kinfu {

class KinFuImpl
{
public:    
    KinFuImpl(const KinFu::KinFuParams& _params);

    const KinFu::KinFuParams& getParams() const;
    KinFu::KinFuParams& getParams();

    Points fetchCloud() const;

    void reset();

    //TODO: enable this when (if) features are ready

    /*
    const TSDFVolume& tsdf() const;
    TSDFVolume& tsdf();

    void renderImage(cuda::Image& image, int flags = 0);
    void renderImage(cuda::Image& image, const Affine3f& pose, int flags = 0);

    Affine3f getCameraPose (int time = -1) const;
    */

    bool operator()(InputArray depth);

private:
    KinFu::KinFuParams params;

    TSDFVolume volume;
    ICP icp;

    int frameCounter;
    Frame frame;
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

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.pyramidLevels = 4;

    p.volumeDims = 512;  //number of voxels
    p.volumeSize = 3.f;  //meters

    p.tsdf_min_camera_movement = 0.f; //meters, disabled

    //TODO: enable when (if) needed
    /*

    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);


    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters
    */

    return p;
}

KinFuImpl::KinFuImpl(const KinFu::KinFuParams &_params) :
    params(_params)
{
    reset();
}

void KinFuImpl::reset()
{
    frameCounter = 0;
    //TODO: maybe better clear volume and frame instead of re-creating
    volume = TSDFVolume(params.volumeDims);
    frame = Frame(params.pyramidLevels, params.frameSize);
    icp = ICP(params.pyramidLevels);

    //TODO: enable if (when) needed
    /*
    CV_Assert(params.volume_dims[0] % 32 == 0);

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

const KinFu::KinFuParams& KinFuImpl::getParams() const
{ return params; }

KinFu::KinFuParams& KinFuImpl::getParams()
{ return params; }

bool KinFuImpl::operator()(InputArray _depth)
{
    CV_Assert(!_depth.empty() && _depth.size() == params.frameSize);
    // CV_Assert(_depth.type() == CV_16S);

    // this should convert CV_16S to CV_32F
    Depth depth = Depth(_depth);

    Distance distance = depthToDistance(depth, params.intr, params.depthFactor);

    // looks like OpenCV's bilateral filter works the same as KinFu's
    Depth smooth;
    bilateralFilter(depth, smooth, params.bilateral_kernel_size, params.bilateral_sigma_depth, params.bilateral_sigma_spatial);

    //TODO: enable it when/if needed
    /*
    if (p.icp_truncate_depth_dist > 0)
        kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);
    */

    //TODO: check KinFu's implementation of pyramid build with bilateral_sigma_depth
    std::vector<Depth> pyramid;
    buildPyramid(smooth, pyramid, params.pyramidLevels);

    Frame newFrame(params.pyramidLevels, params.frameSize);

    newFrame.computePointsNormals(pyramid, params.intr, params.depthFactor);

    if(frameCounter == 0)
    {
        volume.integrate(distance, Affine3f::Identity(), params.intr);

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

        frame.pose = frame.pose * affine;

        float rnorm = (float)cv::norm(affine.rvec());
        float tnorm = (float)cv::norm(affine.translation());
        // We do not integrate volume if camera does not move
        if((rnorm + tnorm)/2 >= params.tsdf_min_camera_movement)
        {
            volume.integrate(distance, frame.pose, params.intr);
        }

        volume.raycast(pose, params.intr, frame.points, frame.normals);
    }

    frameCounter++;
    return true;
}


Points KinFuImpl::fetchCloud() const
{
    return volume.fetchCloud();
}

// FYI: USE_DEPTH not defined

//TODO: enable when (if) needed

//void kfusion::KinFu::renderImage(cuda::Image& image, int flag)
//{
//    const KinFuParams& p = params_;
//    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

//#if defined USE_DEPTH
//    #define PASS1 prev_.depth_pyr
//#else
//    #define PASS1 prev_.points_pyr
//#endif

//    if (flag < 1 || flag > 3)
//        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
//    else if (flag == 2)
//        cuda::renderTangentColors(prev_.normals_pyr[0], image);
//    else /* if (flag == 3) */
//    {
//        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
//        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

//        cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
//        cuda::renderTangentColors(prev_.normals_pyr[0], i2);
//    }
//#undef PASS1
//}


//void kfusion::KinFu::renderImage(cuda::Image& image, const Affine3f& pose, int flag)
//{
//    const KinFuParams& p = params_;
//    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
//    depths_.create(p.rows, p.cols);
//    normals_.create(p.rows, p.cols);
//    points_.create(p.rows, p.cols);

//#if defined USE_DEPTH
//    #define PASS1 depths_
//#else
//    #define PASS1 points_
//#endif

//    volume_->raycast(pose, p.intr, PASS1, normals_);

//    if (flag < 1 || flag > 3)
//        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
//    else if (flag == 2)
//        cuda::renderTangentColors(normals_, image);
//    else /* if (flag == 3) */
//    {
//        DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
//        DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

//        cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
//        cuda::renderTangentColors(normals_, i2);
//    }
//#undef PASS1
//}

KinFu::KinFu(const KinFuParams& _params)
{
    impl = makePtr<KinFuImpl>(_params);
}

virtual KinFu::~KinFu() { }

const KinFu::KinFuParams& KinFu::getParams() const
{
    return impl->getParams();
}

KinFu::KinFuParams& KinFu::getParams()
{
    return impl->getParams();
}

Points KinFu::fetchCloud() const
{
    return impl->fetchCloud();
}

bool KinFu::operator()(InputArray depth)
{
    return impl->operator()(depth);
}

}
}
