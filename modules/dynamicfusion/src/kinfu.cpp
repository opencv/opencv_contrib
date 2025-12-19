#include <opencv2/dynamicfusion/cuda/precomp.hpp>
#include <opencv2/dynamicfusion/cuda/internal.hpp>
#include <tgmath.h>
#include <opencv2/dynamicfusion/utils/dual_quaternion.hpp>
#include <nanoflann/nanoflann.hpp>
#include <opencv2/dynamicfusion/utils/quaternion.hpp>
#include <opencv2/dynamicfusion/utils/knn_point_cloud.hpp>
#include <opencv2/dynamicfusion/warp_field.hpp>
#include <opencv2/dynamicfusion/cuda/tsdf_volume.hpp>
#include <opencv2/dynamicfusion/kinfu.hpp>
static inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

/**
 * \brief
 * \return
 */
cv::kfusion::KinFuParams cv::kfusion::KinFuParams::default_params_dynamicfusion()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;
// TODO: this should be coming from a calibration file / shouldn't be hardcoded
    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.intr = Intr(570.342f, 570.342f, 320.f, 240.f);

    p.volume_dims = Vec3i::all(512);  //number of voxels
    p.volume_size = Vec3f::all(1.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    return p;
}

/**
 * \brief
 * \return
 */
cv::kfusion::KinFuParams cv::kfusion::KinFuParams::default_params()
{
    const int iters[] = {10, 5, 4, 0};
    const int levels = sizeof(iters)/sizeof(iters[0]);

    KinFuParams p;
// TODO: this should be coming from a calibration file / shouldn't be hardcoded
    p.cols = 640;  //pixels
    p.rows = 480;  //pixels
    p.intr = Intr(525.f, 525.f, p.cols/2 - 0.5f, p.rows/2 - 0.5f);

    p.volume_dims = Vec3i::all(512);  //number of voxels
    p.volume_size = Vec3f::all(3.f);  //meters
    p.volume_pose = Affine3f().translate(Vec3f(-p.volume_size[0]/2, -p.volume_size[1]/2, 0.5f));

    p.bilateral_sigma_depth = 0.04f;  //meter
    p.bilateral_sigma_spatial = 4.5; //pixels
    p.bilateral_kernel_size = 7;     //pixels

    p.icp_truncate_depth_dist = 0.f;        //meters, disabled
    p.icp_dist_thres = 0.1f;                //meters
    p.icp_angle_thres = deg2rad(30.f); //radians
    p.icp_iter_num.assign(iters, iters + levels);

    p.tsdf_min_camera_movement = 0.f; //meters, disabled
    p.tsdf_trunc_dist = 0.04f; //meters;
    p.tsdf_max_weight = 64;   //frames

    p.raycast_step_factor = 0.75f;  //in voxel sizes
    p.gradient_delta_factor = 0.5f; //in voxel sizes

    //p.light_pose = p.volume_pose.translation()/4; //meters
    p.light_pose = Vec3f::all(0.f); //meters

    return p;
}

/**
 * \brief
 * \param params
 */
cv::kfusion::KinFu::KinFu(const KinFuParams& params) : frame_counter_(0), params_(params)
{
//    cv::CV_Assert(params.volume_dims[0] % 32 == 0);

    volume_ = cv::Ptr<cv::kfusion::cuda::TsdfVolume>(new cv::kfusion::cuda::TsdfVolume(params_.volume_dims));
    warp_ = cv::Ptr<WarpField>(new WarpField());

    volume_->setTruncDist(params_.tsdf_trunc_dist);
    volume_->setMaxWeight(params_.tsdf_max_weight);
    volume_->setSize(params_.volume_size);
    volume_->setPose(params_.volume_pose);
    volume_->setRaycastStepFactor(params_.raycast_step_factor);
    volume_->setGradientDeltaFactor(params_.gradient_delta_factor);

    icp_ = cv::Ptr<cv::kfusion::cuda::ProjectiveICP>(new cv::kfusion::cuda::ProjectiveICP());
    icp_->setDistThreshold(params_.icp_dist_thres);
    icp_->setAngleThreshold(params_.icp_angle_thres);
    icp_->setIterationsNum(params_.icp_iter_num);

    allocate_buffers();
    reset();
}

const cv::kfusion::KinFuParams& cv::kfusion::KinFu::params() const
{ return params_; }

cv::kfusion::KinFuParams& cv::kfusion::KinFu::params()
{ return params_; }

const cv::kfusion::cuda::TsdfVolume& cv::kfusion::KinFu::tsdf() const
{ return *volume_; }

cv::kfusion::cuda::TsdfVolume& cv::kfusion::KinFu::tsdf()
{ return *volume_; }

const cv::kfusion::cuda::ProjectiveICP& cv::kfusion::KinFu::icp() const
{ return *icp_; }

cv::kfusion::cuda::ProjectiveICP& cv::kfusion::KinFu::icp()
{ return *icp_; }

const cv::kfusion::WarpField& cv::kfusion::KinFu::getWarp() const
{ return *warp_; }

cv::kfusion::WarpField& cv::kfusion::KinFu::getWarp()
{ return *warp_; }

void cv::kfusion::KinFu::allocate_buffers()
{
    const int LEVELS = cv::kfusion::cuda::ProjectiveICP::MAX_PYRAMID_LEVELS;

    int cols = params_.cols;
    int rows = params_.rows;

    dists_.create(rows, cols);

    curr_.depth_pyr.resize(LEVELS);
    curr_.normals_pyr.resize(LEVELS);
    first_.normals_pyr.resize(LEVELS);
    first_.depth_pyr.resize(LEVELS);
    prev_.depth_pyr.resize(LEVELS);
    prev_.normals_pyr.resize(LEVELS);
    first_.normals_pyr.resize(LEVELS);

    curr_.points_pyr.resize(LEVELS);
    prev_.points_pyr.resize(LEVELS);
    first_.points_pyr.resize(LEVELS);

    for(int i = 0; i < LEVELS; ++i)
    {
        curr_.depth_pyr[i].create(rows, cols);
        curr_.normals_pyr[i].create(rows, cols);

        prev_.depth_pyr[i].create(rows, cols);
        prev_.normals_pyr[i].create(rows, cols);

        first_.depth_pyr[i].create(rows, cols);
        first_.normals_pyr[i].create(rows, cols);

        curr_.points_pyr[i].create(rows, cols);
        prev_.points_pyr[i].create(rows, cols);
        first_.points_pyr[i].create(rows, cols);

        cols /= 2;
        rows /= 2;
    }

    depths_.create(params_.rows, params_.cols);
    normals_.create(params_.rows, params_.cols);
    points_.create(params_.rows, params_.cols);
}

void cv::kfusion::KinFu::reset()
{
    if (frame_counter_)
        std::cout << "Reset" << std::endl;

    frame_counter_ = 0;
    poses_.clear();
    poses_.reserve(30000);
    poses_.push_back(Affine3f::Identity());
    volume_->clear();
    warp_->clear();
}

/**
 * \brief
 * \param time
 * \return
 */
cv::kfusion::Affine3f cv::kfusion::KinFu::getCameraPose (int time) const
{
    if (time > (int)poses_.size () || time < 0)
        time = (int)poses_.size () - 1;
    return poses_[time];
}

bool cv::kfusion::KinFu::operator()(const cv::kfusion::cuda::Depth& depth, const cv::kfusion::cuda::Image& /*image*/)
{
    const KinFuParams& p = params_;
    const int LEVELS = icp_->getUsedLevelsNum();

    cv::kfusion::cuda::computeDists(depth, dists_, p.intr);
    cv::kfusion::cuda::depthBilateralFilter(depth, curr_.depth_pyr[0], p.bilateral_kernel_size, p.bilateral_sigma_spatial, p.bilateral_sigma_depth);

    if (p.icp_truncate_depth_dist > 0)
        cv::kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    for (int i = 1; i < LEVELS; ++i)
        cv::kfusion::cuda::depthBuildPyramid(curr_.depth_pyr[i-1], curr_.depth_pyr[i], p.bilateral_sigma_depth);

    for (int i = 0; i < LEVELS; ++i)
#if defined USE_DEPTH
        cv::kfusion::cuda::computeNormalsAndMaskDepth(p.intr(i), curr_.depth_pyr[i], curr_.normals_pyr[i]);
#else
        cv::kfusion::cuda::computePointNormals(p.intr(i), curr_.depth_pyr[i], curr_.points_pyr[i], curr_.normals_pyr[i]);
#endif

    cv::kfusion::cuda::waitAllDefaultStream();

    //can't perform more on first frame
    if (frame_counter_ == 0)
    {

        volume_->integrate(dists_, poses_.back(), p.intr);
        volume_->compute_points();
        volume_->compute_normals();

        warp_->init(volume_->get_cloud_host(), volume_->get_normal_host());

        #if defined USE_DEPTH
        curr_.depth_pyr.swap(prev_.depth_pyr);
        curr_.depth_pyr.swap(first_.depth_pyr);
#else
        curr_.points_pyr.swap(prev_.points_pyr);
        curr_.points_pyr.swap(first_.points_pyr);
#endif
        curr_.normals_pyr.swap(prev_.normals_pyr);
        curr_.normals_pyr.swap(first_.normals_pyr);
        return ++frame_counter_, false;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // ICP
    Affine3f affine; // curr -> prev
    {
        //ScopeTime time("icp");
#if defined USE_DEPTH
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.depth_pyr, curr_.normals_pyr, prev_.depth_pyr, prev_.normals_pyr);
#else
        bool ok = icp_->estimateTransform(affine, p.intr, curr_.points_pyr, curr_.normals_pyr, prev_.points_pyr, prev_.normals_pyr);
#endif
        if (!ok)
            return reset(), false;
    }

    poses_.push_back(poses_.back() * affine); // curr -> global
//    auto d = depth;
    auto d = curr_.depth_pyr[0];
    auto pts = curr_.points_pyr[0];
    auto n = curr_.normals_pyr[0];
    dynamicfusion(d, pts, n);


    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting
    {
        //ScopeTime time("ray-cast-all");
#if defined USE_DEPTH
        volume_->raycast(poses_.back(), p.intr, prev_.depth_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizeDepthNormals(prev_.depth_pyr[i-1], prev_.normals_pyr[i-1], prev_.depth_pyr[i], prev_.normals_pyr[i]);
#else
        volume_->raycast(poses_.back(), p.intr, prev_.points_pyr[0], prev_.normals_pyr[0]);
        for (int i = 1; i < LEVELS; ++i)
            resizePointsNormals(prev_.points_pyr[i-1], prev_.normals_pyr[i-1], prev_.points_pyr[i], prev_.normals_pyr[i]);
#endif
        cv::kfusion::cuda::waitAllDefaultStream();
    }

    return ++frame_counter_, true;
}

/**
 * \brief
 * \param image
 * \param flag
 */
void cv::kfusion::KinFu::renderImage(cv::kfusion::cuda::Image& image, int flag)
{
    const KinFuParams& p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);

#if defined USE_DEPTH
    #define PASS1 prev_.depth_pyr
#else
    #define PASS1 prev_.points_pyr
#endif

    if (flag < 1 || flag > 3)
        cv::kfusion::cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cv::kfusion::cuda::renderTangentColors(prev_.normals_pyr[0], image);
    else /* if (flag == 3) */
    {
        cv::kfusion::cuda::DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        cv::kfusion::cuda::DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cv::kfusion::cuda::renderImage(PASS1[0], prev_.normals_pyr[0], params_.intr, params_.light_pose, i1);
        cv::kfusion::cuda::renderTangentColors(prev_.normals_pyr[0], i2);

    }
#undef PASS1
}

/**
 * \brief
 * \param image
 * \param flag
 */
void cv::kfusion::KinFu::dynamicfusion(cv::kfusion::cuda::Depth& depth, cv::kfusion::cuda::Cloud current_frame, cv::kfusion::cuda::Normals current_normals)
{
    cv::kfusion::cuda::Cloud cloud;
    cv::kfusion::cuda::Normals normals;
    cloud.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());
    auto camera_pose = poses_.back();
    tsdf().raycast(camera_pose, params_.intr, cloud, normals);


    cv::Mat cloud_host(depth.rows(), depth.cols(), CV_32FC4);
    cloud.download(cloud_host.ptr<Point>(), cloud_host.step);
    std::vector<Vec3f> warped(cloud_host.rows * cloud_host.cols);
    auto inverse_pose = camera_pose.inv(cv::DECOMP_SVD);
    for (int i = 0; i < cloud_host.rows; i++)
        for (int j = 0; j < cloud_host.cols; j++) {
            Point point = cloud_host.at<Point>(i, j);
            warped[i * cloud_host.cols + j][0] = point.x;
            warped[i * cloud_host.cols + j][1] = point.y;
            warped[i * cloud_host.cols + j][2] = point.z;
            warped[i * cloud_host.cols + j] = inverse_pose * warped[i * cloud_host.cols + j];
        }

    cv::Mat normal_host(cloud_host.rows, cloud_host.cols, CV_32FC4);
    normals.download(normal_host.ptr<Normal>(), normal_host.step);

    std::vector<Vec3f> warped_normals(normal_host.rows * normal_host.cols);
    for (int i = 0; i < normal_host.rows; i++)
        for (int j = 0; j < normal_host.cols; j++) {
            auto point = normal_host.at<Normal>(i, j);
            warped_normals[i * normal_host.cols + j][0] = point.x;
            warped_normals[i * normal_host.cols + j][1] = point.y;
            warped_normals[i * normal_host.cols + j][2] = point.z;
        }


//    current_frame.download(cloud_host.ptr<Point>(), cloud_host.step);
//    std::vector<Vec3f> live(cloud_host.rows * cloud_host.cols);
//    for (int i = 0; i < cloud_host.rows; i++)
//        for (int j = 0; j < cloud_host.cols; j++) {
//            Point point = cloud_host.at<Point>(i, j);
//            live[i * cloud_host.cols + j][0] = point.x;
//            live[i * cloud_host.cols + j][1] = point.y;
//            live[i * cloud_host.cols + j][2] = point.z;
//            live[i * cloud_host.cols + j] = inverse_pose * warped[i * cloud_host.cols + j];
//        }



    std::vector<Vec3f> canonical_visible(warped);
//    getWarp().energy_data(warped, warped_normals, warped, warped_normals); //crashes, leave out for now

//    getWarp().warp(warped, warped_normals);
//    //ScopeTime time("fusion");
    tsdf().surface_fusion(*warp_, warped, canonical_visible, depth, camera_pose, params_.intr);

    volume_->compute_points();
    volume_->compute_normals();
}

/**
 * \brief
 * \param image
 * \param pose
 * \param flag
 */
void cv::kfusion::KinFu::renderImage(cv::kfusion::cuda::Image& image, const Affine3f& pose, int flag) {
    const KinFuParams &p = params_;
    image.create(p.rows, flag != 3 ? p.cols : p.cols * 2);
    depths_.create(p.rows, p.cols);
    normals_.create(p.rows, p.cols);
    points_.create(p.rows, p.cols);

#if defined USE_DEPTH
#define PASS1 depths_
#else
#define PASS1 points_
#endif

    volume_->raycast(pose, p.intr, PASS1, normals_);

    if (flag < 1 || flag > 3)
        cv::kfusion::cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, image);
    else if (flag == 2)
        cv::kfusion::cuda::renderTangentColors(normals_, image);
    else /* if (flag == 3) */
    {
        cv::kfusion::cuda::DeviceArray2D<RGB> i1(p.rows, p.cols, image.ptr(), image.step());
        cv::kfusion::cuda::DeviceArray2D<RGB> i2(p.rows, p.cols, image.ptr() + p.cols, image.step());

        cv::kfusion::cuda::renderImage(PASS1, normals_, params_.intr, params_.light_pose, i1);
        cv::kfusion::cuda::renderTangentColors(normals_, i2);
    }
#undef PASS1
}
