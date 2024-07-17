#include <host_defines.h>
#include <opencv2/dynamicfusion/cuda/device.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Depth bilateral filter

namespace kfusion
{
    namespace device
    {
        __global__ void bilateral_kernel(const PtrStepSz<ushort> src, PtrStep<ushort> dst, const int ksz, const float sigma_spatial2_inv_half, const float sigma_depth2_inv_half)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= src.cols || y >= src.rows)
                return;

            int value = src(y, x);

            int tx = min (x - ksz / 2 + ksz, src.cols - 1);
            int ty = min (y - ksz / 2 + ksz, src.rows - 1);

            float sum1 = 0;
            float sum2 = 0;

            for (int cy = max (y - ksz / 2, 0); cy < ty; ++cy)
            {
                for (int cx = max (x - ksz / 2, 0); cx < tx; ++cx)
                {
                    int depth = src(cy, cx);

                    float space2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                    float color2 = (value - depth) * (value - depth);

                    float weight = __expf (-(space2 * sigma_spatial2_inv_half + color2 * sigma_depth2_inv_half));

                    sum1 += depth * weight;
                    sum2 += weight;
                }
            }
            dst(y, x) = __float2int_rn (sum1 / sum2);
        }
    }
}

void kfusion::device::bilateralFilter (const Depth& src, Depth& dst, int kernel_size, float sigma_spatial, float sigma_depth)
{
    sigma_depth *= 1000; // meters -> mm

    dim3 block (32, 8);
    dim3 grid (divUp (src.cols (), block.x), divUp (src.rows (), block.y));

    cudaSafeCall( cudaFuncSetCacheConfig (bilateral_kernel, cudaFuncCachePreferL1) );
    bilateral_kernel<<<grid, block>>>(src, dst, kernel_size, 0.5f / (sigma_spatial * sigma_spatial), 0.5f / (sigma_depth * sigma_depth));
    cudaSafeCall ( cudaGetLastError () );
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Depth truncation

namespace kfusion
{
    namespace device
    {
        __global__ void truncate_depth_kernel(PtrStepSz<ushort> depth, ushort max_dist /*mm*/)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < depth.cols && y < depth.rows)
                if(depth(y, x) > max_dist)
                    depth(y, x) = 0;
        }
    }
}

void kfusion::device::truncateDepth(Depth& depth, float max_dist /*meters*/)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    truncate_depth_kernel<<<grid, block>>>(depth, static_cast<ushort>(max_dist * 1000.f));
    cudaSafeCall ( cudaGetLastError() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Build depth pyramid

namespace kfusion
{
    namespace device
    {
        __global__ void pyramid_kernel(const PtrStepSz<ushort> src, PtrStepSz<ushort> dst, float sigma_depth_mult3)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= dst.cols || y >= dst.rows)
                return;

            const int D = 5;
            int center = src(2 * y, 2 * x);

            int tx = min (2 * x - D / 2 + D, src.cols - 1);
            int ty = min (2 * y - D / 2 + D, src.rows - 1);
            int cy = max (0, 2 * y - D / 2);

            int sum = 0;
            int count = 0;

            for (; cy < ty; ++cy)
                for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
                {
                    int val = src(cy, cx);
                    if (abs (val - center) < sigma_depth_mult3)
                    {
                        sum += val;
                        ++count;
                    }
                }
            dst(y, x) = (count == 0) ? 0 : sum / count;
        }
    }
}

void kfusion::device::depthPyr(const Depth& source, Depth& pyramid, float sigma_depth)
{
    sigma_depth *= 1000; // meters -> mm

    dim3 block (32, 8);
    dim3 grid (divUp(pyramid.cols(), block.x), divUp(pyramid.rows(), block.y));

    pyramid_kernel<<<grid, block>>>(source, pyramid, sigma_depth * 3);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute normals

namespace kfusion
{
    namespace device
    {
        __global__ void compute_normals_kernel(const PtrStepSz<ushort> depth, const Reprojector reproj, PtrStep<Normal> normals)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= depth.cols || y >= depth.rows)
                return;

            const float qnan = numeric_limits<float>::quiet_NaN ();

            Normal n_out = make_float4(qnan, qnan, qnan, 0.f);

            if (x < depth.cols - 1 && y < depth.rows - 1)
            {
                //mm -> meters
                float z00 = depth(y,   x) * 0.001f;
                float z01 = depth(y, x+1) * 0.001f;
                float z10 = depth(y+1, x) * 0.001f;

                if (z00 * z01 * z10 != 0)
                {
                    float3 v00 = reproj(x,   y, z00);
                    float3 v01 = reproj(x+1, y, z01);
                    float3 v10 = reproj(x, y+1, z10);

                    float3 n = normalized( cross (v01 - v00, v10 - v00) );
                    n_out = make_float4(-n.x, -n.y, -n.z, 0.f);
                }
            }
            normals(y, x) = n_out;
        }

        __global__ void mask_depth_kernel(const PtrStep<Normal> normals, PtrStepSz<ushort> depth)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < depth.cols || y < depth.rows)
            {
                float4 n = normals(y, x);
                if (isnan(n.x))
                    depth(y, x) = 0;
            }
        }
    }
}

void kfusion::device::computeNormalsAndMaskDepth(const Reprojector& reproj, Depth& depth, Normals& normals)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    compute_normals_kernel<<<grid, block>>>(depth, reproj, normals);
    cudaSafeCall ( cudaGetLastError () );

    mask_depth_kernel<<<grid, block>>>(normals, depth);
    cudaSafeCall ( cudaGetLastError () );
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute computePointNormals

namespace kfusion
{
    namespace device
    {
        __global__ void points_normals_kernel(const Reprojector reproj, const PtrStepSz<ushort> depth, PtrStep<Point> points, PtrStep<Normal> normals)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= depth.cols || y >= depth.rows)
                return;

            const float qnan = numeric_limits<float>::quiet_NaN ();
            points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

            if (x >= depth.cols - 1 || y >= depth.rows - 1)
                return;

            //mm -> meters
            float z00 = depth(y,   x) * 0.001f;
            float z01 = depth(y, x+1) * 0.001f;
            float z10 = depth(y+1, x) * 0.001f;

            if (z00 * z01 * z10 != 0)
            {
                float3 v00 = reproj(x,   y, z00);
                float3 v01 = reproj(x+1, y, z01);
                float3 v10 = reproj(x, y+1, z10);

                float3 n = normalized( cross (v01 - v00, v10 - v00) );
                normals(y, x) = make_float4(-n.x, -n.y, -n.z, 0.f);
                points(y, x) = make_float4(v00.x, v00.y, v00.z, 0.f);
            }
        }
    }
}

void kfusion::device::computePointNormals(const Reprojector& reproj, const Depth& depth, Points& points, Normals& normals)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    points_normals_kernel<<<grid, block>>>(reproj, depth, points, normals);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute dists

namespace kfusion
{
    namespace device
    {
        __global__ void compute_dists_kernel(const PtrStepSz<ushort> depth, Dists dists, float2 finv, float2 c)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < depth.cols || y < depth.rows)
            {
                float xl = (x - c.x) * finv.x;
                float yl = (y - c.y) * finv.y;
                float lambda = sqrtf (xl * xl + yl * yl + 1);

                dists(y, x) = __float2half_rn(depth(y, x) * lambda * 0.001f); //meters
            }
        }

        __global__ void cloud_to_depth_kernel(const PtrStep<Point> cloud, PtrStepSz<ushort> depth)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < depth.cols || y < depth.rows)
            {
                depth(y, x) = cloud(y, x).z * 1000; //meters
            }
        }
    }
}

void kfusion::device::compute_dists(const Depth& depth, Dists dists, float2 f, float2 c)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols (), block.x), divUp (depth.rows (), block.y));

    compute_dists_kernel<<<grid, block>>>(depth, dists, make_float2(1.f/f.x, 1.f/f.y), c);
    cudaSafeCall ( cudaGetLastError () );
}

void kfusion::device::cloud_to_depth(const Points& cloud, Depth depth)
{
    dim3 block (32, 8);
    dim3 grid (divUp (cloud.cols (), block.x), divUp (cloud.rows (), block.y));

    cloud_to_depth_kernel<<<grid, block>>>(cloud, depth);
    cudaSafeCall ( cudaGetLastError () );
}

namespace kfusion
{
    namespace device
    {
        __global__ void resize_depth_normals_kernel(const PtrStep<ushort> dsrc, const PtrStep<float4> nsrc, PtrStepSz<ushort> ddst, PtrStep<float4> ndst)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= ddst.cols || y >= ddst.rows)
                return;

            const float qnan = numeric_limits<float>::quiet_NaN ();

            ushort d = 0;
            float4 n = make_float4(qnan, qnan, qnan, qnan);

            int xs = x * 2;
            int ys = y * 2;

            int d00 = dsrc(ys+0, xs+0);
            int d01 = dsrc(ys+0, xs+1);
            int d10 = dsrc(ys+1, xs+0);
            int d11 = dsrc(ys+1, xs+1);

            if (d00 * d01 != 0 && d10 * d11 != 0)
            {
                d = (d00 + d01 + d10 + d11)/4;

                float4 n00 = nsrc(ys+0, xs+0);
                float4 n01 = nsrc(ys+0, xs+1);
                float4 n10 = nsrc(ys+1, xs+0);
                float4 n11 = nsrc(ys+1, xs+1);

                n.x = (n00.x + n01.x + n10.x + n11.x)*0.25;
                n.y = (n00.y + n01.y + n10.y + n11.y)*0.25;
                n.z = (n00.z + n01.z + n10.z + n11.z)*0.25;
            }
            ddst(y, x) = d;
            ndst(y, x) = n;
        }
    }
}

void kfusion::device::resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out)
{
    int in_cols = depth.cols ();
    int in_rows = depth.rows ();

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    dim3 block (32, 8);
    dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));

    resize_depth_normals_kernel<<<grid, block>>>(depth, normals, depth_out, normals_out);
    cudaSafeCall ( cudaGetLastError () );
}

namespace kfusion
{
    namespace device
    {
        __global__ void resize_points_normals_kernel(const PtrStep<Point> vsrc, const PtrStep<Normal> nsrc, PtrStepSz<Point> vdst, PtrStep<Normal> ndst)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= vdst.cols || y >= vdst.rows)
                return;

            const float qnan = numeric_limits<float>::quiet_NaN ();
            vdst(y, x) = ndst(y, x) = make_float4(qnan, qnan, qnan, 0.f);

            int xs = x * 2;
            int ys = y * 2;

            float3 d00 = tr(vsrc(ys+0, xs+0));
            float3 d01 = tr(vsrc(ys+0, xs+1));
            float3 d10 = tr(vsrc(ys+1, xs+0));
            float3 d11 = tr(vsrc(ys+1, xs+1));

            if (!isnan(d00.x * d01.x * d10.x * d11.x))
            {
                float3 d = (d00 + d01 + d10 + d11) * 0.25f;
                vdst(y, x) = make_float4(d.x, d.y, d.z, 0.f);

                float3 n00 = tr(nsrc(ys+0, xs+0));
                float3 n01 = tr(nsrc(ys+0, xs+1));
                float3 n10 = tr(nsrc(ys+1, xs+0));
                float3 n11 = tr(nsrc(ys+1, xs+1));

                float3 n = (n00 + n01 + n10 + n11)*0.25f;
                ndst(y, x) = make_float4(n.x, n.y, n.z, 0.f);
            }
        }
    }
}

void kfusion::device::resizePointsNormals(const Points& points, const Normals& normals, Points& points_out, Normals& normals_out)
{
    int out_cols = points.cols () / 2;
    int out_rows = points.rows () / 2;

    dim3 block (32, 8);
    dim3 grid (divUp (out_cols, block.x), divUp (out_rows, block.y));

    resize_points_normals_kernel<<<grid, block>>>(points, normals, points_out, normals_out);
    cudaSafeCall ( cudaGetLastError () );
}

namespace kfusion
{
    namespace device
    {
        __global__ void render_image_kernel(const PtrStep<ushort> depth, const PtrStep<Normal> normals,
                                            const Reprojector reproj, const float3 light_pose, PtrStepSz<uchar4> dst)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= dst.cols || y >= dst.rows)
                return;

            float3 color;

            int d = depth(y,x);

            if (d == 0)
            {
                const float3 bgr1 = make_float3(4.f/255.f, 2.f/255.f, 2.f/255.f);
                const float3 bgr2 = make_float3(236.f/255.f, 120.f/255.f, 120.f/255.f);

                float w = static_cast<float>(y) / dst.rows;
                color = bgr1 * (1 - w) + bgr2 * w;
            }
            else
            {
                float3 P = reproj(x, y, d * 0.001f);
                float3 N = tr(normals(y,x));

                const float Ka = 0.3f;  //ambient coeff
                const float Kd = 0.5f;  //diffuse coeff
                const float Ks = 0.2f;  //specular coeff
                const float n = 20.f;  //specular power

                const float Ax = 1.f;   //ambient color,  can be RGB
                const float Dx = 1.f;   //diffuse color,  can be RGB
                const float Sx = 1.f;   //specular color, can be RGB
                const float Lx = 1.f;   //light color

                //Ix = Ax*Ka*Dx + Att*Lx [Kd*Dx*(N dot L) + Ks*Sx*(R dot V)^n]

                float3 L = normalized(light_pose - P);
                float3 V = normalized(make_float3(0.f, 0.f, 0.f) - P);
                float3 R = normalized(2 * N * dot(N, L) - L);

                float Ix = Ax*Ka*Dx + Lx * Kd * Dx * fmax(0.f, dot(N, L)) + Lx * Ks * Sx * __powf(fmax(0.f, dot(R, V)), n);
                color = make_float3(Ix, Ix, Ix);
            }

            uchar4 out;
            out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
            out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
            out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
            out.w = 0;
            dst(y, x) = out;
        }

        __global__ void render_image_kernel(const PtrStep<Point> points, const PtrStep<Normal> normals,
                                            const Reprojector reproj, const float3 light_pose, PtrStepSz<uchar4> dst)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= dst.cols || y >= dst.rows)
                return;

            float3 color;

            float3 p = tr(points(y,x));

            if (isnan(p.x))
            {
                const float3 bgr1 = make_float3(4.f/255.f, 2.f/255.f, 2.f/255.f);
                const float3 bgr2 = make_float3(236.f/255.f, 120.f/255.f, 120.f/255.f);

                float w = static_cast<float>(y) / dst.rows;
                color = bgr1 * (1 - w) + bgr2 * w;
            }
            else
            {
                float3 P = p;
                float3 N = tr(normals(y,x));

                const float Ka = 0.3f;  //ambient coeff
                const float Kd = 0.5f;  //diffuse coeff
                const float Ks = 0.2f;  //specular coeff
                const float n = 20.f;  //specular power

                const float Ax = 1.f;   //ambient color,  can be RGB
                const float Dx = 1.f;   //diffuse color,  can be RGB
                const float Sx = 1.f;   //specular color, can be RGB
                const float Lx = 1.f;   //light color

                //Ix = Ax*Ka*Dx + Att*Lx [Kd*Dx*(N dot L) + Ks*Sx*(R dot V)^n]

                float3 L = normalized(light_pose - P);
                float3 V = normalized(make_float3(0.f, 0.f, 0.f) - P);
                float3 R = normalized(2 * N * dot(N, L) - L);

                float Ix = Ax*Ka*Dx + Lx * Kd * Dx * fmax(0.f, dot(N, L)) + Lx * Ks * Sx * __powf(fmax(0.f, dot(R, V)), n);
                color = make_float3(Ix, Ix, Ix);
            }

            uchar4 out;
            out.x = static_cast<unsigned char>(__saturatef(color.x) * 255.f);
            out.y = static_cast<unsigned char>(__saturatef(color.y) * 255.f);
            out.z = static_cast<unsigned char>(__saturatef(color.z) * 255.f);
            out.w = 0;
            dst(y, x) = out;
        }
    }
}

void kfusion::device::renderImage(const Depth& depth, const Normals& normals, const Reprojector& reproj, const float3& light_pose, Image& image)
{
    dim3 block (32, 8);
    dim3 grid (divUp (depth.cols(), block.x), divUp (depth.rows(), block.y));

    render_image_kernel<<<grid, block>>>((PtrStep<ushort>)depth, normals, reproj, light_pose, image);
    cudaSafeCall ( cudaGetLastError () );
}

void kfusion::device::renderImage(const Points& points, const Normals& normals, const Reprojector& reproj, const Vec3f& light_pose, Image& image)
{
    dim3 block (32, 8);
    dim3 grid (divUp (points.cols(), block.x), divUp (points.rows(), block.y));

    render_image_kernel<<<grid, block>>>((PtrStep<Point>)points, normals, reproj, light_pose, image);
    cudaSafeCall ( cudaGetLastError () );
}

namespace kfusion
{
    namespace device
    {
        __global__ void tangent_colors_kernel(PtrStepSz<Normal> normals, PtrStep<uchar4> colors)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x >= normals.cols || y >= normals.rows)
                return;

            float4 n = normals(y, x);

        #if 0
            unsigned char r = static_cast<unsigned char>(__saturatef((-n.x + 1.f)/2.f) * 255.f);
            unsigned char g = static_cast<unsigned char>(__saturatef((-n.y + 1.f)/2.f) * 255.f);
            unsigned char b = static_cast<unsigned char>(__saturatef((-n.z + 1.f)/2.f) * 255.f);
        #else
            unsigned char r = static_cast<unsigned char>((5.f - n.x * 3.5f) * 25.5f);
            unsigned char g = static_cast<unsigned char>((5.f - n.y * 2.5f) * 25.5f);
            unsigned char b = static_cast<unsigned char>((5.f - n.z * 3.5f) * 25.5f);
        #endif
            colors(y, x) = make_uchar4(b, g, r, 0);
        }
    }
}

void kfusion::device::renderTangentColors(const Normals& normals, Image& image)
{
    dim3 block (32, 8);
    dim3 grid (divUp (normals.cols(), block.x), divUp (normals.rows(), block.y));

    tangent_colors_kernel<<<grid, block>>>(normals, image);
    cudaSafeCall ( cudaGetLastError () );
}


namespace kfusion
{
    namespace device
    {
        __global__ void mergePointNormalKernel (const Point* cloud, const float8* normals, PtrSz<float12> output)
        {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;

            if (idx < output.size)
            {
                float4 p = cloud[idx];
                float8 n = normals[idx];

                float12 o;
                o.x = p.x;
                o.y = p.y;
                o.z = p.z;

                o.normal_x = n.x;
                o.normal_y = n.y;
                o.normal_z = n.z;

                output.data[idx] = o;
            }
        }
    }
}

void kfusion::device::mergePointNormal (const DeviceArray<Point>& cloud, const DeviceArray<float8>& normals, const DeviceArray<float12>& output)
{
    const int block = 256;
    int total = (int)output.size ();

    mergePointNormalKernel<<<divUp (total, block), block>>>(cloud, normals, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}
