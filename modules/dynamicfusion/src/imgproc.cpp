#include <opencv2/dynamicfusion/cuda/precomp.hpp>
#include <opencv2/dynamicfusion/types.hpp>
using namespace cv;
/**
 *
 * \param in
 * \param out
 * \param kernel_size
 * \param sigma_spatial
 * \param sigma_depth
 */
void kfusion::cuda::depthBilateralFilter(const Depth& in, Depth& out, int kernel_size, float sigma_spatial, float sigma_depth)
{ 
    out.create(in.rows(), in.cols());
    device::bilateralFilter(in, out, kernel_size, sigma_spatial, sigma_depth);
}

/**
 *
 * \param depth
 * \param threshold
 */
void kfusion::cuda::depthTruncation(Depth& depth, float threshold)
{
    device::truncateDepth(depth, threshold);
}

/**
 *
 * \param depth
 * \param pyramid
 * \param sigma_depth
 */
void kfusion::cuda::depthBuildPyramid(const Depth& depth, Depth& pyramid, float sigma_depth)
{ 
    pyramid.create(depth.rows () / 2, depth.cols () / 2);
    device::depthPyr(depth, pyramid, sigma_depth);
}

/**
 *
 */
void kfusion::cuda::waitAllDefaultStream()
{
    cudaSafeCall(cudaDeviceSynchronize());
}

/**
 *
 * \param intr
 * \param depth
 * \param normals
 */
void kfusion::cuda::computeNormalsAndMaskDepth(const Intr& intr, Depth& depth, Normals& normals)
{
    normals.create(depth.rows(), depth.cols());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Normals& n = (device::Normals&)normals;
    device::computeNormalsAndMaskDepth(reproj, depth, n);
}

/**
 *
 * \param intr
 * \param depth
 * \param points
 * \param normals
 */
void kfusion::cuda::computePointNormals(const Intr& intr, const Depth& depth, Cloud& points, Normals& normals)
{
    points.create(depth.rows(), depth.cols());
    normals.create(depth.rows(), depth.cols());

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Points& p = (device::Points&)points;
    device::Normals& n = (device::Normals&)normals;
    device::computePointNormals(reproj, depth, p, n);
}

/**
 *
 * \param depth
 * \param dists
 * \param intr
 */
void kfusion::cuda::computeDists(const Depth& depth, Dists& dists, const Intr& intr)
{
    dists.create(depth.rows(), depth.cols());
    device::compute_dists(depth, dists, make_float2(intr.fx, intr.fy), make_float2(intr.cx, intr.cy));
}

/**
 *
 * \param cloud
 * \param depth
 */
void kfusion::cuda::cloudToDepth(const Cloud& cloud, Depth& depth)
{
    depth.create(cloud.rows(), cloud.cols());
    device::Points points = (device::Points&)cloud;
    device::cloud_to_depth(points, depth);
}

/**
 *
 * \param depth
 * \param normals
 * \param depth_out
 * \param normals_out
 */
void kfusion::cuda::resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out)
{
    depth_out.create(depth.rows() / 2, depth.cols() / 2);
    normals_out.create(normals.rows() / 2, normals.cols() / 2);

    device::Normals& nsrc = (device::Normals&)normals;
    device::Normals& ndst = (device::Normals&)normals_out;

    device::resizeDepthNormals(depth, nsrc, depth_out, ndst);
}

/**
 *
 * \param points
 * \param normals
 * \param points_out
 * \param normals_out
 */
void kfusion::cuda::resizePointsNormals(const Cloud& points, const Normals& normals, Cloud& points_out, Normals& normals_out)
{
    points_out.create(points.rows() / 2, points.cols() / 2);
    normals_out.create(normals.rows() / 2, normals.cols() / 2);

    device::Points& pi = (device::Points&)points;
    device::Normals& ni= (device::Normals&)normals;

    device::Points& po = (device::Points&)points_out;
    device::Normals& no = (device::Normals&)normals_out;

    device::resizePointsNormals(pi, ni, po, no);
}

/**
 *
 * \param depth
 * \param normals
 * \param intr
 * \param light_pose
 * \param image
 */
void kfusion::cuda::renderImage(const Depth& depth, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image)
{
    image.create(depth.rows(), depth.cols());

    const device::Depth& d = (const device::Depth&)depth;
    const device::Normals& n = (const device::Normals&)normals;
    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
    device::Vec3f light = device_cast<device::Vec3f>(light_pose);

    device::Image& i = (device::Image&)image;
    device::renderImage(d, n, reproj, light, i);
    waitAllDefaultStream();
}

/**
 *
 * \param points
 * \param normals
 * \param intr
 * \param light_pose
 * \param image
 */
void kfusion::cuda::renderImage(const Cloud& points, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image)
{
    image.create(points.rows(), points.cols());

    const device::Points& p = (const device::Points&)points;
    const device::Normals& n = (const device::Normals&)normals;
    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);
    device::Vec3f light = device_cast<device::Vec3f>(light_pose);

    device::Image& i = (device::Image&)image;
    device::renderImage(p, n, reproj, light, i);
    waitAllDefaultStream();
}

/**
 * \brief
 * \param normals
 * \param image
 */
void kfusion::cuda::renderTangentColors(const Normals& normals, Image& image)
{
    image.create(normals.rows(), normals.cols());
    const device::Normals& n = (const device::Normals&)normals;
    device::Image& i = (device::Image&)image;

    device::renderTangentColors(n, i);
    waitAllDefaultStream();
}
