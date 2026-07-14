#include <vector_functions.h>
#include <opencv2/dynamicfusion/cuda/device.hpp>
#include <opencv2/dynamicfusion/cuda/texture_binder.hpp>
#include <opencv2/dynamicfusion/cuda/internal.hpp>
#include "math_constants.h"
using namespace kfusion::device;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume initialization

namespace kfusion
{
    namespace device
    {
        __global__ void clear_volume_kernel(TsdfVolume tsdf)
        {
            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;

            if (x < tsdf.dims.x && y < tsdf.dims.y)
            {
                ushort2 *beg = tsdf.beg(x, y);
                ushort2 *end = beg + tsdf.dims.x * tsdf.dims.y * tsdf.dims.z;

                for(ushort2* pos = beg; pos != end; pos = tsdf.zstep(pos))
                    *pos = pack_tsdf (0.f, 0);
            }
        }
    }
}

void kfusion::device::clear_volume(TsdfVolume volume)
{
    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = divUp (volume.dims.x, block.x);
    grid.y = divUp (volume.dims.y, block.y);

    clear_volume_kernel<<<grid, block>>>(volume);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume integration
float3 test;
namespace kfusion
{
    namespace device
    {
        texture<float, 2> dists_tex(0, cudaFilterModePoint, cudaAddressModeBorder, cudaCreateChannelDescHalf());
        struct TsdfIntegrator
        {
            Aff3f vol2cam;
            Projector proj;
            int2 dists_size;

            float tranc_dist_inv;

            __kf_device__
            void operator()(TsdfVolume& volume) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume.dims.x || y >= volume.dims.y)
                    return;

                //float3 zstep = vol2cam.R * make_float3(0.f, 0.f, volume.voxel_size.z);
                float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;

                float3 vx = make_float3(x * volume.voxel_size.x, y * volume.voxel_size.y, 0);
                float3 vc = vol2cam * vx; //tranform from volume coo frame to camera one

                TsdfVolume::elem_type* vptr = volume.beg(x, y);
                for(int i = 0; i < volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
                {
                    float2 coo = proj(vc);

                    //#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
                    // this is actually workaround for kepler. it doesn't return 0.f for texture
                    // fetches for out-of-border coordinates even for cudaaddressmodeborder mode
                    if (coo.x < 0 || coo.y < 0 || coo.x >= dists_size.x || coo.y >= dists_size.y)
                        continue;
                    //#endif
                    float Dp = tex2D(dists_tex, coo.x, coo.y);
                    if(Dp == 0 || vc.z <= 0)
                        continue;

                    float sdf = Dp - __fsqrt_rn(dot(vc, vc)); //Dp - norm(v)

                    if (sdf >= -volume.trunc_dist)
                    {
                        float tsdf = fmin(1.f, sdf * tranc_dist_inv);

                        //read and unpack
                        int weight_prev;
                        float tsdf_prev = unpack_tsdf (gmem::LdCs(vptr), weight_prev);

                        float tsdf_new = __fdividef(__fmaf_rn(tsdf_prev, weight_prev, tsdf), weight_prev + 1);
                        int weight_new = min (weight_prev + 1, volume.max_weight);

                        //pack and write
                        gmem::StCs(pack_tsdf (tsdf_new, weight_new), vptr);
                    }
                }  // for(;;)
            }


        };


        __global__ void integrate_kernel( const TsdfIntegrator integrator, TsdfVolume volume) { integrator(volume); };

        __global__
        void project_kernel(const Projector proj, PtrStep<Point> points, PtrStepSz<ushort> depth, int rows, int cols)
        {

            int x = threadIdx.x + blockIdx.x * blockDim.x;
            int y = threadIdx.y + blockIdx.y * blockDim.y;
            float qnan = numeric_limits<float>::quiet_NaN ();
            if (x < cols || y < rows) {
                Point pt = points(y, x);
                if(isnan(pt.x) || isnan(pt.y) || isnan(pt.z))
                    return;
                float3 point = make_float3(pt.x, pt.y, pt.z);
                float2 coo = proj(point);
                if (coo.x < 0 || coo.y < 0 || coo.y >= rows || coo.x >= cols)
                {
                    points(y, x) = make_float4(qnan, qnan, qnan, 0.f);
                    return;
                }

                float Dp = tex2D(dists_tex, coo.x, coo.y);
                depth(coo.y, coo.x) = 0;
                points(y, x) = make_float4(coo.x * Dp, coo.y * Dp, Dp, 0.f);
            }
        }
    }
}

void kfusion::device::integrate(const PtrStepSz<ushort>& dists, TsdfVolume& volume, const Aff3f& aff, const Projector& proj)
{
    TsdfIntegrator ti;
    ti.dists_size = make_int2(dists.cols, dists.rows);
    ti.vol2cam = aff;
    ti.proj = proj;
    ti.tranc_dist_inv = 1.f/volume.trunc_dist;

    dists_tex.filterMode = cudaFilterModePoint;
    dists_tex.addressMode[0] = cudaAddressModeBorder;
    dists_tex.addressMode[1] = cudaAddressModeBorder;
    dists_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder binder(dists, dists_tex, cudaCreateChannelDescHalf()); (void)binder;

    dim3 block(32, 8);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

    integrate_kernel<<<grid, block>>>(ti, volume);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall ( cudaDeviceSynchronize() );
}

//TODO: rename as now projecting + removing from depth
void kfusion::device::project_and_remove(const PtrStepSz<ushort>& dists, Points &vertices, const Projector &proj)
{
    dists_tex.filterMode = cudaFilterModePoint;
    dists_tex.addressMode[0] = cudaAddressModeBorder;
    dists_tex.addressMode[1] = cudaAddressModeBorder;
    dists_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder binder(dists, dists_tex, cudaCreateChannelDescHalf()); (void)binder;

    dim3 block(32, 8);
    dim3 grid(divUp(vertices.cols(), block.x), divUp(vertices.rows(), block.y));

    project_kernel <<<grid, block>>>(proj, vertices, dists, dists.rows, dists.cols);
    cudaSafeCall ( cudaGetLastError () );
}

//TODO: rename as now projecting + removing from depth
void kfusion::device::project(const PtrStepSz<ushort> &dists, Points &vertices, const Projector &proj)
{
    dists_tex.filterMode = cudaFilterModePoint;
    dists_tex.addressMode[0] = cudaAddressModeBorder;
    dists_tex.addressMode[1] = cudaAddressModeBorder;
    dists_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder binder(dists, dists_tex, cudaCreateChannelDescHalf()); (void)binder;

    dim3 block(32, 8);
    dim3 grid(divUp(vertices.cols(), block.x), divUp(vertices.rows(), block.y));

    project_kernel <<<grid, block>>>(proj, vertices, dists, dists.rows, dists.cols);
    cudaSafeCall ( cudaGetLastError () );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Volume ray casting

namespace kfusion
{
    namespace device
    {
        __kf_device__ void intersect(float3 ray_org, float3 ray_dir, /*float3 box_min,*/ float3 box_max, float &tnear, float &tfar)
        {
            const float3 box_min = make_float3(0.f, 0.f, 0.f);

            // compute intersection of ray with all six bbox planes
            float3 invR = make_float3(1.f/ray_dir.x, 1.f/ray_dir.y, 1.f/ray_dir.z);
            float3 tbot = invR * (box_min - ray_org);
            float3 ttop = invR * (box_max - ray_org);

            // re-order intersections to find smallest and largest on each axis
            float3 tmin = make_float3(fminf(ttop.x, tbot.x), fminf(ttop.y, tbot.y), fminf(ttop.z, tbot.z));
            float3 tmax = make_float3(fmaxf(ttop.x, tbot.x), fmaxf(ttop.y, tbot.y), fmaxf(ttop.z, tbot.z));

            // find the largest tmin and the smallest tmax
            tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
            tfar  = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));
        }

        template<typename Vol>
        __kf_device__ float interpolate(const Vol& volume, const float3& p_voxels)
        {
            float3 cf = p_voxels;

            //rounding to negative infinity
            int3 g = make_int3(__float2int_rd (cf.x), __float2int_rd (cf.y), __float2int_rd (cf.z));

            if (g.x < 0 || g.x >= volume.dims.x - 1 || g.y < 0 || g.y >= volume.dims.y - 1 || g.z < 0 || g.z >= volume.dims.z - 1)
                return numeric_limits<float>::quiet_NaN();

            float a = cf.x - g.x;
            float b = cf.y - g.y;
            float c = cf.z - g.z;

            float tsdf = 0.f;
            tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 0, g.z + 0)) * (1 - a) * (1 - b) * (1 - c);
            tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 0, g.z + 1)) * (1 - a) * (1 - b) *      c;
            tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 1, g.z + 0)) * (1 - a) *      b  * (1 - c);
            tsdf += unpack_tsdf(*volume(g.x + 0, g.y + 1, g.z + 1)) * (1 - a) *      b  *      c;
            tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 0, g.z + 0)) *      a  * (1 - b) * (1 - c);
            tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 0, g.z + 1)) *      a  * (1 - b) *      c;
            tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 1, g.z + 0)) *      a  *      b  * (1 - c);
            tsdf += unpack_tsdf(*volume(g.x + 1, g.y + 1, g.z + 1)) *      a  *      b  *      c;
            return tsdf;
        }

        struct TsdfRaycaster
        {
            TsdfVolume volume;

            Aff3f aff;
            Mat3f Rinv;

            Vec3f volume_size;
            Reprojector reproj;
            float time_step;
            float3 gradient_delta;
            float3 voxel_size_inv;

            TsdfRaycaster(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv, const Reprojector& _reproj);

            __kf_device__
            float fetch_tsdf(const float3& p) const
            {
                //rounding to nearest even
                int x = __float2int_rn (p.x * voxel_size_inv.x);
                int y = __float2int_rn (p.y * voxel_size_inv.y);
                int z = __float2int_rn (p.z * voxel_size_inv.z);
                return unpack_tsdf(*volume(x, y, z));
            }

            __kf_device__
            void operator()(PtrStepSz<ushort> depth, PtrStep<Normal> normals) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= depth.cols || y >= depth.rows)
                    return;

                const float qnan = numeric_limits<float>::quiet_NaN();

                depth(y, x) = 0;
                normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

                float3 ray_org = aff.t;
                float3 ray_dir = normalized( aff.R * reproj(x, y, 1.f) );

                // We do subtract voxel size to minimize checks after
                // Note: origin of volume coordinate is placed
                // in the center of voxel (0,0,0), not in the corner of the voxel!
                float3 box_max = volume_size - volume.voxel_size;

                float tmin, tmax;
                intersect(ray_org, ray_dir, box_max, tmin, tmax);

                const float min_dist = 0.f;
                tmin = fmax(min_dist, tmin);
                if (tmin >= tmax)
                    return;

                tmax -= time_step;
                float3 vstep = ray_dir * time_step;
                float3 next = ray_org + ray_dir * tmin;

                float tsdf_next = fetch_tsdf(next);
                for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
                {
                    float tsdf_curr = tsdf_next;
                    float3     curr = next;
                    next += vstep;

                    tsdf_next = fetch_tsdf(next);
                    if (tsdf_curr < 0.f && tsdf_next > 0.f)
                        break;

                    if (tsdf_curr > 0.f && tsdf_next < 0.f)
                    {
                        float Ft   = interpolate(volume, curr * voxel_size_inv);
                        float Ftdt = interpolate(volume, next * voxel_size_inv);

                        float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

                        float3 vertex = ray_org + ray_dir * Ts;
                        float3 normal = compute_normal(vertex);

                        if (!isnan(normal.x * normal.y * normal.z))
                        {
                            normal = Rinv * normal;
                            vertex = Rinv * (vertex - aff.t);

                            normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0);
                            depth(y, x) = static_cast<ushort>(vertex.z * 1000);
                        }
                        break;
                    }
                } /* for (;;) */
            }

            __kf_device__
            void operator()(PtrStepSz<Point> points, PtrStep<Normal> normals) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= points.cols || y >= points.rows)
                    return;

                const float qnan = numeric_limits<float>::quiet_NaN();

                points(y, x) = normals(y, x) = make_float4(qnan, qnan, qnan, qnan);

                float3 ray_org = aff.t;
                float3 ray_dir = normalized( aff.R * reproj(x, y, 1.f) );

                // We do subtract voxel size to minimize checks after
                // Note: origin of volume coordinate is placeed
                // in the center of voxel (0,0,0), not in the corener of the voxel!
                float3 box_max = volume_size - volume.voxel_size;

                float tmin, tmax;
                intersect(ray_org, ray_dir, box_max, tmin, tmax);

                const float min_dist = 0.f;
                tmin = fmax(min_dist, tmin);
                if (tmin >= tmax)
                    return;

                tmax -= time_step;
                float3 vstep = ray_dir * time_step;
                float3 next = ray_org + ray_dir * tmin;

                float tsdf_next = fetch_tsdf(next);
                for (float tcurr = tmin; tcurr < tmax; tcurr += time_step)
                {
                    float tsdf_curr = tsdf_next;
                    float3     curr = next;
                    next += vstep;

                    tsdf_next = fetch_tsdf(next);
                    if (tsdf_curr < 0.f && tsdf_next > 0.f)
                        break;

                    if (tsdf_curr > 0.f && tsdf_next < 0.f)
                    {
                        float Ft   = interpolate(volume, curr * voxel_size_inv);
                        float Ftdt = interpolate(volume, next * voxel_size_inv);

                        float Ts = tcurr - __fdividef(time_step * Ft, Ftdt - Ft);

                        float3 vertex = ray_org + ray_dir * Ts;
                        float3 normal = compute_normal(vertex);

                        if (!isnan(normal.x * normal.y * normal.z))
                        {
                            normal = Rinv * normal;
                            vertex = Rinv * (vertex - aff.t);

                            normals(y, x) = make_float4(normal.x, normal.y, normal.z, 0.f);
                            points(y, x) = make_float4(vertex.x, vertex.y, vertex.z, 0.f);
                        }
                        break;
                    }
                } /* for (;;) */
            }


            __kf_device__
            float3 compute_normal(const float3& p) const
            {
                float3 n;

                float Fx1 = interpolate(volume, make_float3(p.x + gradient_delta.x, p.y, p.z) * voxel_size_inv);
                float Fx2 = interpolate(volume, make_float3(p.x - gradient_delta.x, p.y, p.z) * voxel_size_inv);
                n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

                float Fy1 = interpolate(volume, make_float3(p.x, p.y + gradient_delta.y, p.z) * voxel_size_inv);
                float Fy2 = interpolate(volume, make_float3(p.x, p.y - gradient_delta.y, p.z) * voxel_size_inv);
                n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

                float Fz1 = interpolate(volume, make_float3(p.x, p.y, p.z + gradient_delta.z) * voxel_size_inv);
                float Fz2 = interpolate(volume, make_float3(p.x, p.y, p.z - gradient_delta.z) * voxel_size_inv);
                n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

                return normalized (n);
            }
        };

        inline TsdfRaycaster::TsdfRaycaster(const TsdfVolume& _volume, const Aff3f& _aff, const Mat3f& _Rinv, const Reprojector& _reproj)
                : volume(_volume), aff(_aff), Rinv(_Rinv), reproj(_reproj) {}

        __global__ void raycast_kernel(const TsdfRaycaster raycaster, PtrStepSz<ushort> depth, PtrStep<Normal> normals)
        { raycaster(depth, normals); };

        __global__ void raycast_kernel(const TsdfRaycaster raycaster, PtrStepSz<Point> points, PtrStep<Normal> normals)
        { raycaster(points, normals); };

    }
}

void kfusion::device::raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv, const Reprojector& reproj,
                              Depth& depth, Normals& normals, float raycaster_step_factor, float gradient_delta_factor)
{
    TsdfRaycaster rc(volume, aff, Rinv, reproj);

    rc.volume_size = volume.voxel_size * volume.dims;
    rc.time_step = volume.trunc_dist * raycaster_step_factor;
    rc.gradient_delta = volume.voxel_size * gradient_delta_factor;
    rc.voxel_size_inv = 1.f/volume.voxel_size;

    dim3 block(32, 8);
    dim3 grid (divUp (depth.cols(), block.x), divUp (depth.rows(), block.y));

    raycast_kernel<<<grid, block>>>(rc, (PtrStepSz<ushort>)depth, normals);
    cudaSafeCall (cudaGetLastError ());
}


void kfusion::device::raycast(const TsdfVolume& volume, const Aff3f& aff, const Mat3f& Rinv, const Reprojector& reproj,
                              Points& points, Normals& normals, float raycaster_step_factor, float gradient_delta_factor)
{
    TsdfRaycaster rc(volume, aff, Rinv, reproj);

    rc.volume_size = volume.voxel_size * volume.dims;
    rc.time_step = volume.trunc_dist * raycaster_step_factor;
    rc.gradient_delta = volume.voxel_size * gradient_delta_factor;
    rc.voxel_size_inv = 1.f/volume.voxel_size;

    dim3 block(32, 8);
    dim3 grid (divUp (points.cols(), block.x), divUp (points.rows(), block.y));

    raycast_kernel<<<grid, block>>>(rc, (PtrStepSz<Point>)points, normals);
    cudaSafeCall (cudaGetLastError ());
}

////////////////////////////////////////////////////////////////////////////////////////
/// Volume cloud exctraction

namespace kfusion
{
    namespace device
    {
        ////////////////////////////////////////////////////////////////////////////////////////
        ///// Prefix Scan utility

        enum ScanKind { exclusive, inclusive };

        template<ScanKind Kind, class T>
        __kf_device__ T scan_warp ( volatile T *ptr, const unsigned int idx = threadIdx.x )
        {
            const unsigned int lane = idx & 31;       // index of thread in warp (0..31)

            if (lane >=  1) ptr[idx] = ptr[idx -  1] + ptr[idx];
            if (lane >=  2) ptr[idx] = ptr[idx -  2] + ptr[idx];
            if (lane >=  4) ptr[idx] = ptr[idx -  4] + ptr[idx];
            if (lane >=  8) ptr[idx] = ptr[idx -  8] + ptr[idx];
            if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx];

            if (Kind == inclusive)
                return ptr[idx];
            else
                return (lane > 0) ? ptr[idx - 1] : 0;
        }


        __device__ int global_count = 0;
        __device__ int output_count;
        __device__ unsigned int blocks_done = 0;


        struct FullScan6
        {
            enum
            {
                CTA_SIZE_X = 32,
                CTA_SIZE_Y = 6,
                CTA_SIZE = CTA_SIZE_X * CTA_SIZE_Y,

                MAX_LOCAL_POINTS = 3
            };

            TsdfVolume volume;
            Aff3f aff;

            FullScan6(const TsdfVolume& vol) : volume(vol) {}

            __kf_device__ float fetch(int x, int y, int z, int& weight) const
            {
                return unpack_tsdf(*volume(x, y, z), weight);
            }

            __kf_device__ void operator () (PtrSz<Point> output) const
            {
                int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
                int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;
#if __CUDA_ARCH__ < 200
                __shared__ int cta_buffer[CTA_SIZE];
#endif

#if __CUDA_ARCH__ >= 120
                if (__all (x >= volume.dims.x) || __all (y >= volume.dims.y))
                    return;
#else
                if (Emulation::All(x >= volume.dims.x, cta_buffer) || Emulation::All(y >= volume.dims.y, cta_buffer))
                    return;
#endif

                float3 V;
                V.x = (x + 0.5f) * volume.voxel_size.x;
                V.y = (y + 0.5f) * volume.voxel_size.y;

                int ftid = Block::flattenedThreadId ();

                for (int z = 0; z < volume.dims.z - 1; ++z)
                {
                    float3 points[MAX_LOCAL_POINTS];
                    int local_count = 0;

                    if (x < volume.dims.x && y < volume.dims.y)
                    {
                        int W;
                        float F = fetch(x, y, z, W);

                        if (W != 0 && F != 1.f)
                        {
                            V.z = (z + 0.5f) * volume.voxel_size.z;

                            //process dx
                            if (x + 1 < volume.dims.x)
                            {
                                int Wn;
                                float Fn = fetch(x + 1, y, z, Wn);

                                if (Wn != 0 && Fn != 1.f)
                                    if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
                                    {
                                        float3 p;
                                        p.y = V.y;
                                        p.z = V.z;

                                        float Vnx = V.x + volume.voxel_size.x;

                                        float d_inv = 1.f / (fabs (F) + fabs (Fn));
                                        p.x = (V.x * fabs (Fn) + Vnx * fabs (F)) * d_inv;

                                        points[local_count++] = aff * p;
                                    }
                            }  /* if (x + 1 < volume.dims.x) */

                            //process dy
                            if (y + 1 < volume.dims.y)
                            {
                                int Wn;
                                float Fn = fetch (x, y + 1, z, Wn);

                                if (Wn != 0 && Fn != 1.f)
                                    if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
                                    {
                                        float3 p;
                                        p.x = V.x;
                                        p.z = V.z;

                                        float Vny = V.y + volume.voxel_size.y;

                                        float d_inv = 1.f / (fabs (F) + fabs (Fn));
                                        p.y = (V.y * fabs (Fn) + Vny * fabs (F)) * d_inv;

                                        points[local_count++] = aff * p;
                                    }
                            } /*  if (y + 1 < volume.dims.y) */

                            //process dz
                            //if (z + 1 < volume.dims.z) // guaranteed by loop
                            {
                                int Wn;
                                float Fn = fetch (x, y, z + 1, Wn);

                                if (Wn != 0 && Fn != 1.f)
                                    if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
                                    {
                                        float3 p;
                                        p.x = V.x;
                                        p.y = V.y;

                                        float Vnz = V.z + volume.voxel_size.z;

                                        float d_inv = 1.f / (fabs (F) + fabs (Fn));
                                        p.z = (V.z * fabs (Fn) + Vnz * fabs (F)) * d_inv;

                                        points[local_count++] = aff * p;
                                    }
                            } /* if (z + 1 < volume.dims.z) */
                        } /* if (W != 0 && F != 1.f) */
                    } /* if (x < volume.dims.x && y < volume.dims.y) */

#if __CUDA_ARCH__ >= 200
                    ///not we fulfilled points array at current iteration
                    int total_warp = __popc (__ballot (local_count > 0)) + __popc (__ballot (local_count > 1)) + __popc (__ballot (local_count > 2));
#else
                    int tid = Block::flattenedThreadId();
                    cta_buffer[tid] = local_count;
                    int total_warp = Emulation::warp_reduce(cta_buffer, tid);
#endif
                    __shared__ float storage_X[CTA_SIZE * MAX_LOCAL_POINTS];
                    __shared__ float storage_Y[CTA_SIZE * MAX_LOCAL_POINTS];
                    __shared__ float storage_Z[CTA_SIZE * MAX_LOCAL_POINTS];

                    if (total_warp > 0)
                    {
                        int lane = Warp::laneId ();
                        int storage_index = (ftid >> Warp::LOG_WARP_SIZE) * Warp::WARP_SIZE * MAX_LOCAL_POINTS;

                        volatile int* cta_buffer = (int*)(storage_X + storage_index);

                        cta_buffer[lane] = local_count;
                        int offset = scan_warp<exclusive>(cta_buffer, lane);

                        if (lane == 0)
                        {
                            int old_global_count = atomicAdd (&global_count, total_warp);
                            cta_buffer[0] = old_global_count;
                        }
                        int old_global_count = cta_buffer[0];

                        for (int l = 0; l < local_count; ++l)
                        {
                            storage_X[storage_index + offset + l] = points[l].x;
                            storage_Y[storage_index + offset + l] = points[l].y;
                            storage_Z[storage_index + offset + l] = points[l].z;
                        }

                        Point *pos = output.data + old_global_count + lane;
                        for (int idx = lane; idx < total_warp; idx += Warp::STRIDE, pos += Warp::STRIDE)
                        {
                            float x = storage_X[storage_index + idx];
                            float y = storage_Y[storage_index + idx];
                            float z = storage_Z[storage_index + idx];
                            *pos = make_float4(x, y, z, 0.f);
                        }

                        bool full = (old_global_count + total_warp) >= output.size;

                        if (full)
                            break;
                    }

                } /* for(int z = 0; z < volume.dims.z - 1; ++z) */


                ///////////////////////////
                // prepare for future scans
                if (ftid == 0)
                {
                    unsigned int total_blocks = gridDim.x * gridDim.y * gridDim.z;
                    unsigned int value = atomicInc (&blocks_done, total_blocks);

                    //last block
                    if (value == total_blocks - 1)
                    {
                        output_count = min ((int)output.size, global_count);
                        blocks_done = 0;
                        global_count = 0;
                    }
                }
            }
        };



        __global__ void extract_kernel(const FullScan6 fs, PtrSz<Point> output) { fs(output); }



        struct ExtractNormals
        {
            typedef float8 float8;

            TsdfVolume volume;
            PtrSz<Point> points;
            float3 voxel_size_inv;
            float3 gradient_delta;
            Aff3f aff;
            Mat3f Rinv;

            ExtractNormals(const TsdfVolume& vol) : volume(vol)
            {
                voxel_size_inv.x = 1.f/volume.voxel_size.x;
                voxel_size_inv.y = 1.f/volume.voxel_size.y;
                voxel_size_inv.z = 1.f/volume.voxel_size.z;
            }

            __kf_device__ int3 getVoxel (const float3& p) const
            {
                //rounding to nearest even
                int x = __float2int_rn (p.x * voxel_size_inv.x);
                int y = __float2int_rn (p.y * voxel_size_inv.y);
                int z = __float2int_rn (p.z * voxel_size_inv.z);
                return make_int3 (x, y, z);
            }

            __kf_device__ void operator () (float4* output) const
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;

                if (idx >= points.size)
                    return;

                const float qnan = numeric_limits<float>::quiet_NaN ();
                float3 n = make_float3 (qnan, qnan, qnan);

                float3 point = Rinv * (tr(points.data[idx]) - aff.t);
                int3 g = getVoxel (point);

                if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < volume.dims.x - 2 && g.y < volume.dims.y - 2 && g.z < volume.dims.z - 2)
                {
                    float3 t;

                    t = point;
                    t.x += gradient_delta.x;;
                    float Fx1 = interpolate(volume, t * voxel_size_inv);

                    t = point;
                    t.x -= gradient_delta.x;
                    float Fx2 = interpolate(volume, t * voxel_size_inv);

                    n.x = __fdividef(Fx1 - Fx2, gradient_delta.x);

                    t = point;
                    t.y += gradient_delta.y;
                    float Fy1 = interpolate(volume, t * voxel_size_inv);

                    t = point;
                    t.y -= gradient_delta.y;
                    float Fy2 = interpolate(volume, t * voxel_size_inv);

                    n.y = __fdividef(Fy1 - Fy2, gradient_delta.y);

                    t = point;
                    t.z += gradient_delta.z;
                    float Fz1 = interpolate(volume, t * voxel_size_inv);

                    t = point;
                    t.z -= gradient_delta.z;
                    float Fz2 = interpolate(volume, t * voxel_size_inv);

                    n.z = __fdividef(Fz1 - Fz2, gradient_delta.z);

                    n = normalized (aff.R * n);
                }

                output[idx] = make_float4(n.x, n.y, n.z, 0);
            }
        };

        __global__ void extract_normals_kernel (const ExtractNormals en, float4* output) { en(output); }
    }
}

size_t kfusion::device::extractCloud (const TsdfVolume& volume, const Aff3f& aff, PtrSz<Point> output)
{
    typedef FullScan6 FS;
    FS fs(volume);
    fs.aff = aff;

    dim3 block (FS::CTA_SIZE_X, FS::CTA_SIZE_Y);
    dim3 grid (divUp (volume.dims.x, block.x), divUp (volume.dims.y, block.y));

    extract_kernel<<<grid, block>>>(fs, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());

    int size;
    cudaSafeCall ( cudaMemcpyFromSymbol (&size, output_count, sizeof(size)) );
    return (size_t)size;
}

void kfusion::device::extractNormals (const TsdfVolume& volume, const PtrSz<Point>& points, const Aff3f& aff, const Mat3f& Rinv, float gradient_delta_factor, float4* output)
{
    ExtractNormals en(volume);
    en.points = points;
    en.gradient_delta = volume.voxel_size * gradient_delta_factor;
    en.aff = aff;
    en.Rinv = Rinv;

    dim3 block (256);
    dim3 grid (divUp ((int)points.size, block.x));

    extract_normals_kernel<<<grid, block>>>(en, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}
