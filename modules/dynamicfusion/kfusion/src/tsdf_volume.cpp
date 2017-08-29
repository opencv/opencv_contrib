#include "precomp.hpp"
#include "dual_quaternion.hpp"
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <kfusion/warp_field.hpp>
#include <knn_point_cloud.hpp>
#include <numeric>
#include <opencv/cv.h>
#include <opencv2/viz/vizcore.hpp>
#include <opencv/highgui.h>
using namespace kfusion;
using namespace kfusion::cuda;


kfusion::cuda::TsdfVolume::TsdfVolume(const Vec3i& dims) : data_(),
                                                           trunc_dist_(0.03f),
                                                           max_weight_(128),
                                                           dims_(dims),
                                                           size_(Vec3f::all(3.f)),
                                                           pose_(Affine3f::Identity()),
                                                           gradient_delta_factor_(0.75f),
                                                           raycast_step_factor_(0.75f)
{
    create(dims_);
}

kfusion::cuda::TsdfVolume::~TsdfVolume()
{
    delete cloud_host;
    delete cloud_buffer;
    delete cloud;
    delete normal_host;
    delete normal_buffer;
}

/**
 * \brief
 * \param dims
 */
void kfusion::cuda::TsdfVolume::create(const Vec3i& dims)
{
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    data_.create(voxels_number * sizeof(int));
    setTruncDist(trunc_dist_);
    clear();
}

/**
 * \brief
 * \return
 */
Vec3i kfusion::cuda::TsdfVolume::getDims() const
{
    return dims_;
}

/**
 * \brief
 * \return
 */
Vec3f kfusion::cuda::TsdfVolume::getVoxelSize() const
{
    return Vec3f(size_[0] / dims_[0], size_[1] / dims_[1], size_[2] / dims_[2]);
}

const CudaData kfusion::cuda::TsdfVolume::data() const { return data_; }
CudaData kfusion::cuda::TsdfVolume::data() {  return data_; }
Vec3f kfusion::cuda::TsdfVolume::getSize() const { return size_; }

void kfusion::cuda::TsdfVolume::setSize(const Vec3f& size)
{ size_ = size; setTruncDist(trunc_dist_); }

float kfusion::cuda::TsdfVolume::getTruncDist() const { return trunc_dist_; }

void kfusion::cuda::TsdfVolume::setTruncDist(float distance)
{
    Vec3f vsz = getVoxelSize();
    float max_coeff = std::max<float>(std::max<float>(vsz[0], vsz[1]), vsz[2]);
    trunc_dist_ = std::max (distance, 2.1f * max_coeff);
}
cv::Mat kfusion::cuda::TsdfVolume::get_cloud_host() const {return *cloud_host;};
cv::Mat kfusion::cuda::TsdfVolume::get_normal_host() const {return *normal_host;};
cv::Mat* kfusion::cuda::TsdfVolume::get_cloud_host_ptr() const {return cloud_host;};
cv::Mat* kfusion::cuda::TsdfVolume::get_normal_host_ptr() const {return normal_host;};

int kfusion::cuda::TsdfVolume::getMaxWeight() const { return max_weight_; }
void kfusion::cuda::TsdfVolume::setMaxWeight(int weight) { max_weight_ = weight; }
Affine3f kfusion::cuda::TsdfVolume::getPose() const  { return pose_; }
void kfusion::cuda::TsdfVolume::setPose(const Affine3f& pose) { pose_ = pose; }
float kfusion::cuda::TsdfVolume::getRaycastStepFactor() const { return raycast_step_factor_; }
void kfusion::cuda::TsdfVolume::setRaycastStepFactor(float factor) { raycast_step_factor_ = factor; }
float kfusion::cuda::TsdfVolume::getGradientDeltaFactor() const { return gradient_delta_factor_; }
void kfusion::cuda::TsdfVolume::setGradientDeltaFactor(float factor) { gradient_delta_factor_ = factor; }
void kfusion::cuda::TsdfVolume::swap(CudaData& data) { data_.swap(data); }
void kfusion::cuda::TsdfVolume::applyAffine(const Affine3f& affine) { pose_ = affine * pose_; }
void kfusion::cuda::TsdfVolume::clear()
{
    cloud_buffer = new cuda::DeviceArray<Point>();
    cloud = new cuda::DeviceArray<Point>();
    normal_buffer = new cuda::DeviceArray<Normal>();
    cloud_host = new cv::Mat();
    normal_host = new cv::Mat();

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::clear_volume(volume);
}

/**
 * \brief
 * \param dists
 * \param camera_pose
 * \param intr
 */
void kfusion::cuda::TsdfVolume::integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr)
{
    Affine3f vol2cam = camera_pose.inv() * pose_;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, aff, proj);
}

/**
 * \brief
 * \param camera_pose
 * \param intr
 * \param depth
 * \param normals
 */
void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals)
{
    DeviceArray2D<device::Normal>& n = (DeviceArray2D<device::Normal>&)normals;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);

}

/**
 * \brief
 * \param camera_pose
 * \param intr
 * \param points
 * \param normals
 */
void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals)
{
    device::Normals& n = (device::Normals&)normals;
    device::Points& p = (device::Points&)points;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
}

/**
 * \brief
 * \param cloud_buffer
 * \return
 */
DeviceArray<Point> kfusion::cuda::TsdfVolume::fetchCloud(DeviceArray<Point>& cloud_buffer) const
{
    //    enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };
    enum { DEFAULT_CLOUD_BUFFER_SIZE = 256 * 256 * 256 };

    if (cloud_buffer.empty ())
        cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

    DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    size_t size = extractCloud(volume, aff, b);

    return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}

/**
 *
 * @param cloud
 * @param normals
 */
void kfusion::cuda::TsdfVolume::fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const
{
    normals.create(cloud.size());
    DeviceArray<device::Point>& c = (DeviceArray<device::Point>&)cloud;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);
    device::Mat3f Rinv = device_cast<device::Mat3f>(pose_.rotation().inv(cv::DECOMP_SVD));

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::extractNormals(volume, c, aff, Rinv, gradient_delta_factor_, (float4*)normals.ptr());
}

//TODO: in order to make this more efficient, we can just pass in the already warped canonical points (x_t)
// and the canonical points
/**
 * \brief
 * \param warp_field
 * \param depth_img
 * \param camera_pose
 * \param intr
 */
void kfusion::cuda::TsdfVolume::surface_fusion(const WarpField& warp_field,
                                               std::vector<Vec3f> warped,
                                               std::vector<Vec3f> canonical,
                                               cuda::Depth& depth,
                                               const Affine3f& camera_pose,
                                               const Intr& intr)
{
    std::vector<float> ro = psdf(warped, depth, intr);

    cuda::Dists dists;
    cuda::computeDists(depth, dists, intr);
    integrate(dists, camera_pose, intr);

    for(size_t i = 0; i < ro.size(); i++)
    {
        if(ro[i] > -trunc_dist_)
        {
            warp_field.KNN(canonical[i]);
            float weight = weighting(warp_field.out_dist_sqr, KNN_NEIGHBOURS);
            float coeff = std::min(ro[i], trunc_dist_);

////            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_value * tsdf_entries[i].tsdf_weight + coeff * weight;
////            tsdf_entries[i].tsdf_value = tsdf_entries[i].tsdf_weight + weight;
////
////            tsdf_entries[i].tsdf_weight = std::min(tsdf_entries[i].tsdf_weight + weight, W_MAX);
        }
    }
}

/**
 * \fn TSDF::psdf (Mat3f K, Depth& depth, Vec3f voxel_center)
 * \brief return a quaternion that is the spherical linear interpolation between q1 and q2
 *        where percentage (from 0 to 1) defines the amount of interpolation
 * \param K: camera matrix
 * \param depth: a depth frame
 * \param voxel_center
 *
 */
std::vector<float> kfusion::cuda::TsdfVolume::psdf(const std::vector<Vec3f>& warped,
                                                   Dists& dists,
                                                   const Intr& intr)
{
    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);
    std::vector<float4, std::allocator<float4>> point_type(warped.size());
    for(int i = 0; i < warped.size(); i++)
    {
        point_type[i].x = warped[i][0];
        point_type[i].y = warped[i][1];
        point_type[i].z = warped[i][2];
        point_type[i].w = 0.f;
    }
    device::Points points;
    points.upload(point_type, dists.cols());
    device::project_and_remove(dists, points, proj);
    int size;
    points.download(point_type, size);
    Mat3f K = Mat3f(intr.fx, 0, intr.cx,
                    0, intr.fy, intr.cy,
                    0, 0, 1).inv();

    std::vector<float> distances(warped.size());
    for(int i = 0; i < warped.size(); i++)
        distances[i] = (K * Vec3f(point_type[i].x, point_type[i].y, point_type[i].z))[2] - warped[i][2];
    return distances;
}

/**
 * \brief
 * \param dist_sqr
 * \param k
 * \return
 */
float kfusion::cuda::TsdfVolume::weighting(const std::vector<float>& dist_sqr, int k) const
{
    float distances = 0;
    for(auto distance : dist_sqr)
        distances += sqrt(distance);
    return distances / k;
}
/**
 * \brief
 * \param dist_sqr
 * \param k
 * \return
 */
void kfusion::cuda::TsdfVolume::compute_points()
{
    *cloud = fetchCloud(*cloud_buffer);
    *cloud_host = cv::Mat(1, (int)cloud->size(), CV_32FC4);
    cloud->download(cloud_host->ptr<Point>());
}

void kfusion::cuda::TsdfVolume::compute_normals()
{
    fetchNormals(*cloud, *normal_buffer);
    *normal_host = cv::Mat(1, (int)cloud->size(), CV_32FC4);
    normal_buffer->download(normal_host->ptr<Normal>());
}
