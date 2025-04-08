/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

////////////////////////////////////////////////////////////////////////////////
//
// NVIDIA CUDA implementation of Brox et al Optical Flow algorithm
//
// Algorithm is explained in the original paper:
//      T. Brox, A. Bruhn, N. Papenberg, J. Weickert:
//      High accuracy optical flow estimation based on a theory for warping.
//      ECCV 2004.
//
// Implementation by Mikhail Smirnov
// email: msmirnov@nvidia.com, devsupport@nvidia.com
//
// Credits for help with the code to:
// Alexey Mendelenko, Anton Obukhov, and Alexander Kharlamov.
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <vector>
#include <memory>

#include "opencv2/core/cuda/utility.hpp"

#include "opencv2/cudalegacy/NPP_staging.hpp"
#include "opencv2/cudalegacy/NCVBroxOpticalFlow.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>


typedef NCVVectorAlloc<Ncv32f> FloatVector;
typedef cv::cudev::TexturePtr<float> Ptr2D;
typedef cv::cudev::Texture<float> Texture;

/////////////////////////////////////////////////////////////////////////////////////////
// Implementation specific constants
/////////////////////////////////////////////////////////////////////////////////////////
__device__ const float eps2 = 1e-6f;

/////////////////////////////////////////////////////////////////////////////////////////
// Additional defines
/////////////////////////////////////////////////////////////////////////////////////////

// rounded up division
inline int iDivUp(int a, int b)
{
    return (a + b - 1)/b;
}

/////////////////////////////////////////////////////////////////////////////////////////
// SUPPLEMENTARY FUNCTIONS
/////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// \brief performs pointwise summation of two vectors stored in device memory
/// \param d_res    - pointer to resulting vector (device memory)
/// \param d_op1    - term #1 (device memory)
/// \param d_op2    - term #2 (device memory)
/// \param len    - vector size
///////////////////////////////////////////////////////////////////////////////
__global__ void pointwise_add(float *d_res, const float *d_op1, const float *d_op2, const int len)
{
    const int pos = blockIdx.x*blockDim.x + threadIdx.x;

    if(pos >= len) return;

    d_res[pos] = d_op1[pos] + d_op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief wrapper for summation kernel.
///  Computes \b op1 + \b op2 and stores result to \b res
/// \param res   array, containing op1 + op2 (device memory)
/// \param op1   term #1 (device memory)
/// \param op2   term #2 (device memory)
/// \param count vector size
///////////////////////////////////////////////////////////////////////////////
static void add(float *res, const float *op1, const float *op2, const int count, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(count, threads.x));

    pointwise_add<<<blocks, threads, 0, stream>>>(res, op1, op2, count);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief wrapper for summation kernel.
/// Increments \b res by \b rhs
/// \param res   initial vector, will be replaced with result (device memory)
/// \param rhs   increment (device memory)
/// \param count vector size
///////////////////////////////////////////////////////////////////////////////
static void add(float *res, const float *rhs, const int count, cudaStream_t stream)
{
    add(res, res, rhs, count, stream);
}

///////////////////////////////////////////////////////////////////////////////
/// \brief kernel for scaling vector by scalar
/// \param d_res  scaled vector (device memory)
/// \param d_src  source vector (device memory)
/// \param scale  scalar to scale by
/// \param len    vector size (number of elements)
///////////////////////////////////////////////////////////////////////////////
__global__ void scaleVector(float *d_res, const float *d_src, float scale, const int len)
{
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= len) return;

    d_res[pos] = d_src[pos] * scale;
}

///////////////////////////////////////////////////////////////////////////////
/// \brief scale vector by scalar
///
/// kernel wrapper
/// \param d_res  scaled vector (device memory)
/// \param d_src  source vector (device memory)
/// \param scale  scalar to scale by
/// \param len    vector size (number of elements)
/// \param stream CUDA stream
///////////////////////////////////////////////////////////////////////////////
static void ScaleVector(float *d_res, const float *d_src, float scale, const int len, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(len, threads.x));

    scaleVector<<<blocks, threads, 0, stream>>>(d_res, d_src, scale, len);
}

const int SOR_TILE_WIDTH = 32;
const int SOR_TILE_HEIGHT = 6;
const int PSOR_TILE_WIDTH = 32;
const int PSOR_TILE_HEIGHT = 6;
const int PSOR_PITCH = PSOR_TILE_WIDTH + 4;
const int PSOR_HEIGHT = PSOR_TILE_HEIGHT + 4;

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Compute smooth term diffusivity along x axis
///\param s (out) pointer to memory location for result (diffusivity)
///\param pos (in) position within shared memory array containing \b u
///\param u (in) shared memory array containing \b u
///\param v (in) shared memory array containing \b v
///\param du (in) shared memory array containing \b du
///\param dv (in) shared memory array containing \b dv
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void diffusivity_along_x(float *s, int pos, const float *u, const float *v, const float *du, const float *dv)
{
    //x derivative between pixels (i,j) and (i-1,j)
    const int left = pos-1;
    float u_x = u[pos] + du[pos] - u[left] - du[left];
    float v_x = v[pos] + dv[pos] - v[left] - dv[left];
    const int up        = pos + PSOR_PITCH;
    const int down      = pos - PSOR_PITCH;
    const int up_left   = up - 1;
    const int down_left = down-1;
    //y derivative between pixels (i,j) and (i-1,j)
    float u_y = 0.25f*(u[up] + du[up] + u[up_left] + du[up_left] - u[down] - du[down] - u[down_left] - du[down_left]);
    float v_y = 0.25f*(v[up] + dv[up] + v[up_left] + dv[up_left] - v[down] - dv[down] - v[down_left] - dv[down_left]);
    *s = 0.5f / sqrtf(u_x*u_x + v_x*v_x + u_y*u_y + v_y*v_y + eps2);
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Compute smooth term diffusivity along y axis
///\param s (out) pointer to memory location for result (diffusivity)
///\param pos (in) position within shared memory array containing \b u
///\param u (in) shared memory array containing \b u
///\param v (in) shared memory array containing \b v
///\param du (in) shared memory array containing \b du
///\param dv (in) shared memory array containing \b dv
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void diffusivity_along_y(float *s, int pos, const float *u, const float *v, const float *du, const float *dv)
{
    //y derivative between pixels (i,j) and (i,j-1)
    const int down = pos-PSOR_PITCH;
    float u_y = u[pos] + du[pos] - u[down] - du[down];
    float v_y = v[pos] + dv[pos] - v[down] - dv[down];
    const int right      = pos + 1;
    const int left       = pos - 1;
    const int down_right = down + 1;
    const int down_left  = down - 1;
    //x derivative between pixels (i,j) and (i,j-1);
    float u_x = 0.25f*(u[right] + u[down_right] + du[right] + du[down_right] - u[left] - u[down_left] - du[left] - du[down_left]);
    float v_x = 0.25f*(v[right] + v[down_right] + dv[right] + dv[down_right] - v[left] - v[down_left] - dv[left] - dv[down_left]);
    *s = 0.5f/sqrtf(u_x*u_x + v_x*v_x + u_y*u_y + v_y*v_y + eps2);
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Load element of 2D global memory to shared memory
///\param smem pointer to shared memory array
///\param is shared memory array column
///\param js shared memory array row
///\param w number of columns in global memory array
///\param h number of rows in global memory array
///\param p global memory array pitch in floats
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void load_array_element(Ptr2D texSrc, float *smem, int is, int js, int i, int j, int w, int h, int p)
{
    //position within shared memory array
    const int ijs = js * PSOR_PITCH + is;
    //mirror reflection across borders
    i = max(i, -i-1);
    i = min(i, w-i+w-1);
    j = max(j, -j-1);
    j = min(j, h-j+h-1);
    const int pos = j * p + i;
    smem[ijs] = texSrc(pos);
}

///////////////////////////////////////////////////////////////////////////////
///\brief Utility function. Load part (tile) of 2D global memory to shared memory
///\param smem pointer to target shared memory array
///\param ig column number within source
///\param jg row number within source
///\param w number of columns in global memory array
///\param h number of rows in global memory array
///\param p global memory array pitch in floats
///////////////////////////////////////////////////////////////////////////////
__forceinline__ __device__ void load_array(Ptr2D texSrc, float *smem, int ig, int jg, int w, int h, int p)
{
    const int i = threadIdx.x + 2;
    const int j = threadIdx.y + 2;
    load_array_element(texSrc, smem, i, j, ig, jg, w, h, p);//load current pixel
    __syncthreads();
    if(threadIdx.y < 2)
    {
        //load bottom shadow elements
        load_array_element(texSrc, smem, i, j-2, ig, jg-2, w, h, p);
        if(threadIdx.x < 2)
        {
            //load bottom right shadow elements
            load_array_element(texSrc, smem, i+PSOR_TILE_WIDTH, j-2, ig+PSOR_TILE_WIDTH, jg-2, w, h, p);
            //load middle right shadow elements
            load_array_element(texSrc, smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load bottom left shadow elements
            load_array_element(texSrc, smem, i-PSOR_TILE_WIDTH, j-2, ig-PSOR_TILE_WIDTH, jg-2, w, h, p);
            //load middle left shadow elements
            load_array_element(texSrc, smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    else if(threadIdx.y >= PSOR_TILE_HEIGHT-2)
    {
        //load upper shadow elements
        load_array_element(texSrc, smem, i, j+2, ig, jg+2, w, h, p);
        if(threadIdx.x < 2)
        {
            //load upper right shadow elements
            load_array_element(texSrc, smem, i+PSOR_TILE_WIDTH, j+2, ig+PSOR_TILE_WIDTH, jg+2, w, h, p);
            //load middle right shadow elements
            load_array_element(texSrc, smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load upper left shadow elements
            load_array_element(texSrc, smem, i-PSOR_TILE_WIDTH, j+2, ig-PSOR_TILE_WIDTH, jg+2, w, h, p);
            //load middle left shadow elements
            load_array_element(texSrc, smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    else
    {
        //load middle shadow elements
        if(threadIdx.x < 2)
        {
            //load middle right shadow elements
            load_array_element(texSrc, smem, i+PSOR_TILE_WIDTH, j, ig+PSOR_TILE_WIDTH, jg, w, h, p);
        }
        else if(threadIdx.x >= PSOR_TILE_WIDTH-2)
        {
            //load middle left shadow elements
            load_array_element(texSrc, smem, i-PSOR_TILE_WIDTH, j, ig-PSOR_TILE_WIDTH, jg, w, h, p);
        }
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////
/// \brief computes matrix of linearised system for \c du, \c dv
/// Computed values reside in GPU memory. \n
/// Matrix computation is divided into two steps. This kernel performs first step\n
/// - compute smoothness term diffusivity between pixels - psi dash smooth
/// - compute robustness factor in the data term - psi dash data
/// \param diffusivity_x (in/out) diffusivity between pixels along x axis in smoothness term
/// \param diffusivity_y (in/out) diffusivity between pixels along y axis in smoothness term
/// \param denominator_u (in/out) precomputed part of expression for new du value in SOR iteration
/// \param denominator_v (in/out) precomputed part of expression for new dv value in SOR iteration
/// \param numerator_dudv (in/out) precomputed part of expression for new du and dv value in SOR iteration
/// \param numerator_u (in/out) precomputed part of expression for new du value in SOR iteration
/// \param numerator_v (in/out) precomputed part of expression for new dv value in SOR iteration
/// \param w (in) frame width
/// \param h (in) frame height
/// \param pitch (in) pitch in floats
/// \param alpha (in) alpha in Brox model (flow smoothness)
/// \param gamma (in) gamma in Brox model (edge importance)
///////////////////////////////////////////////////////////////////////////////
__global__ void prepare_sor_stage_1_tex(Ptr2D texU, Ptr2D texV, Ptr2D texDu, Ptr2D texDv, Ptr2D texI0, Ptr2D texI1, Ptr2D texIx, Ptr2D texIxx, Ptr2D texIx0, Ptr2D texIy, Ptr2D texIyy,
    Ptr2D texIy0, Ptr2D texIxy, float *diffusivity_x, float *diffusivity_y, float *denominator_u, float *denominator_v, float *numerator_dudv, float *numerator_u, float *numerator_v,
    int w, int h, int s, float alpha, float gamma)
{
    __shared__ float u[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float v[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float du[PSOR_PITCH * PSOR_HEIGHT];
    __shared__ float dv[PSOR_PITCH * PSOR_HEIGHT];

    //position within tile
    const int i = threadIdx.x;
    const int j = threadIdx.y;
    //position within smem arrays
    const int ijs = (j+2) * PSOR_PITCH + i + 2;
    //position within global memory
    const int ig  = blockIdx.x * blockDim.x + threadIdx.x;
    const int jg  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ijg = jg * s + ig;
    //position within texture
    float x = (float)ig + 0.5f;
    float y = (float)jg + 0.5f;
    //load u  and v to smem
    load_array(texU, u, ig, jg, w, h, s);
    load_array(texV, v, ig, jg, w, h, s);
    load_array(texDu, du, ig, jg, w, h, s);
    load_array(texDv, dv, ig, jg, w, h, s);
    //warped position
    float wx = (x + u[ijs])/(float)w;
    float wy = (y + v[ijs])/(float)h;
    x /= (float)w;
    y /= (float)h;
    //compute image derivatives
    const float Iz = texI1(wy, wx) - texI0(y,x);
    const float Ix  = texIx(wy, wx);
    const float Ixz = Ix - texIx0(y, x);
    const float Ixy = texIxy(wy, wx);
    const float Ixx = texIxx(wy, wx);
    const float Iy = texIy(wy, wx);
    const float Iyz = Iy - texIy0(y, x);
    const float Iyy = texIyy(wy, wx);
    //compute data term
    float q0, q1, q2;
    q0 = Iz  + Ix  * du[ijs] + Iy  * dv[ijs];
    q1 = Ixz + Ixx * du[ijs] + Ixy * dv[ijs];
    q2 = Iyz + Ixy * du[ijs] + Iyy * dv[ijs];
    float data_term = 0.5f * rsqrtf(q0*q0 + gamma*(q1*q1 + q2*q2) + eps2);
    //scale data term by 1/alpha
    data_term /= alpha;
    //compute smoothness term (diffusivity)
    float sx, sy;

    if(ig >= w || jg >= h) return;

    diffusivity_along_x(&sx, ijs, u, v, du, dv);
    diffusivity_along_y(&sy, ijs, u, v, du, dv);

    if(ig == 0) sx = 0.0f;
    if(jg == 0) sy = 0.0f;

    numerator_dudv[ijg] = data_term * (Ix*Iy + gamma * Ixy*(Ixx + Iyy));
    numerator_u[ijg]    = data_term * (Ix*Iz + gamma * (Ixx*Ixz + Ixy*Iyz));
    numerator_v[ijg]    = data_term * (Iy*Iz + gamma * (Iyy*Iyz + Ixy*Ixz));
    denominator_u[ijg]  = data_term * (Ix*Ix + gamma * (Ixy*Ixy + Ixx*Ixx));
    denominator_v[ijg]  = data_term * (Iy*Iy + gamma * (Ixy*Ixy + Iyy*Iyy));
    diffusivity_x[ijg]  = sx;
    diffusivity_y[ijg]  = sy;
}

///////////////////////////////////////////////////////////////////////////////
///\brief computes matrix of linearised system for \c du, \c dv
///\param inv_denominator_u
///\param inv_denominator_v
///\param w
///\param h
///\param s
///////////////////////////////////////////////////////////////////////////////
__global__ void prepare_sor_stage_2(Ptr2D texDiffX, Ptr2D texDiffY, float *inv_denominator_u, float *inv_denominator_v, int w, int h, int s)
{
    __shared__ float sx[(PSOR_TILE_WIDTH+1) * (PSOR_TILE_HEIGHT+1)];
    __shared__ float sy[(PSOR_TILE_WIDTH+1) * (PSOR_TILE_HEIGHT+1)];
    //position within tile
    const int i = threadIdx.x;
    const int j = threadIdx.y;
    //position within smem arrays
    const int ijs = j*(PSOR_TILE_WIDTH+1) + i;
    //position within global memory
    const int ig  = blockIdx.x * blockDim.x + threadIdx.x;
    const int jg  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ijg = jg*s + ig;
    int inside = ig < w && jg < h;
    float denom_u;
    float denom_v;
    if(inside)
    {
        denom_u = inv_denominator_u[ijg];
        denom_v = inv_denominator_v[ijg];
    }
    if(inside)
    {
        sx[ijs] = texDiffX(ijg);
        sy[ijs] = texDiffY(ijg);
    }
    else
    {
        sx[ijs] = 0.0f;
        sy[ijs] = 0.0f;
    }
    int up = ijs+PSOR_TILE_WIDTH+1;
    if(j == PSOR_TILE_HEIGHT-1)
    {
        if(jg < h-1 && inside)
            sy[up] = texDiffY(ijg + s);
        else
            sy[up] = 0.0f;
    }
    int right = ijs + 1;
    if(threadIdx.x == PSOR_TILE_WIDTH-1)
    {
        if(ig < w-1 && inside)
            sx[right] = texDiffX(ijg + 1);
        else
            sx[right] = 0.0f;
    }
    __syncthreads();
    float diffusivity_sum;
    diffusivity_sum = sx[ijs] + sx[ijs+1] + sy[ijs] + sy[ijs+PSOR_TILE_WIDTH+1];
    if(inside)
    {
        denom_u += diffusivity_sum;
        denom_v += diffusivity_sum;
        inv_denominator_u[ijg] = 1.0f/denom_u;
        inv_denominator_v[ijg] = 1.0f/denom_v;
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
// Red-Black SOR
/////////////////////////////////////////////////////////////////////////////////////////

template<int isBlack> __global__ void sor_pass(Ptr2D texU, Ptr2D texV, Ptr2D texDu, Ptr2D texDv, Ptr2D texDiffX, Ptr2D texDiffY, float *new_du, float *new_dv, const float *g_inv_denominator_u,
    const float *g_inv_denominator_v, const float *g_numerator_u, const float *g_numerator_v, const float *g_numerator_dudv, float omega, int width, int height, int stride)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= width || j >= height)
        return;

    const int pos = j * stride + i;
    const int pos_r = i < width - 1 ? pos + 1 : pos;
    const int pos_u = j < height - 1 ? pos + stride : pos;
    const int pos_d = j > 0 ? pos - stride : pos;
    const int pos_l = i > 0 ? pos - 1 : pos;

    //load smooth term
    float s_up, s_left, s_right, s_down;
    s_left = texDiffX(pos);
    s_down = texDiffY(pos);
    if(i < width-1)
        s_right = texDiffX(pos_r);
    else
        s_right = 0.0f; //Neumann BC
    if(j < height-1)
        s_up = texDiffY(pos_u);
    else
        s_up = 0.0f; //Neumann BC

    //load u, v and du, dv
    float u_up, u_left, u_right, u_down, u;
    float v_up, v_left, v_right, v_down, v;
    float du_up, du_left, du_right, du_down, du;
    float dv_up, dv_left, dv_right, dv_down, dv;

    u_left = texU(pos_l);
    u_right = texU(pos_r);
    u_down = texU(pos_d);
    u_up = texU(pos_u);
    u = texU(pos);

    v_left = texV(pos_l);
    v_right = texV(pos_r);
    v_down = texV(pos_d);
    v = texV(pos);
    v_up = texV(pos_u);

    du =  texDu(pos);
    du_left = texDu(pos_l);
    du_right = texDu(pos_r);
    du_down = texDu(pos_d);
    du_up = texDu(pos_u);

    dv = texDv(pos);
    dv_left = texDv(pos_l);
    dv_right = texDv(pos_r);
    dv_down = texDv(pos_d);
    dv_up = texDv(pos_u);
    float numerator_dudv    = g_numerator_dudv[pos];

    if((i+j)%2 == isBlack)
    {
        // update du
        float numerator_u = (s_left*(u_left + du_left) + s_up*(u_up + du_up) + s_right*(u_right + du_right) + s_down*(u_down + du_down) -
                             u * (s_left + s_right + s_up + s_down) - g_numerator_u[pos] - numerator_dudv*dv);

        du = (1.0f - omega) * du + omega * g_inv_denominator_u[pos] * numerator_u;

        // update dv
        float numerator_v = (s_left*(v_left + dv_left) + s_up*(v_up + dv_up) + s_right*(v_right + dv_right) + s_down*(v_down + dv_down) -
                             v * (s_left + s_right + s_up + s_down) - g_numerator_v[pos] - numerator_dudv*du);

        dv = (1.0f - omega) * dv + omega * g_inv_denominator_v[pos] * numerator_v;
    }
    new_du[pos] = du;
    new_dv[pos] = dv;
}

///////////////////////////////////////////////////////////////////////////////
// utility functions
///////////////////////////////////////////////////////////////////////////////
namespace
{
    struct ImagePyramid
    {
        std::vector<FloatVector*> img0;
        std::vector<FloatVector*> img1;

        std::vector<Ncv32u> w;
        std::vector<Ncv32u> h;

        explicit ImagePyramid(int outer_iterations)
        {
            img0.reserve(outer_iterations);
            img1.reserve(outer_iterations);

            w.reserve(outer_iterations);
            h.reserve(outer_iterations);
        }

        ~ImagePyramid()
        {
            w.clear();
            h.clear();

            for (int i = static_cast<int>(img0.size()) - 1; i >= 0; --i)
            {
                delete img1[i];
                delete img0[i];
            }

            img0.clear();
            img1.clear();
        }
    };
}

/////////////////////////////////////////////////////////////////////////////////////////
// MAIN FUNCTION
/////////////////////////////////////////////////////////////////////////////////////////
NCVStatus NCVBroxOpticalFlow(const NCVBroxOpticalFlowDescriptor desc,
                             INCVMemAllocator &gpu_mem_allocator,
                             const NCVMatrix<Ncv32f> &frame0,
                             const NCVMatrix<Ncv32f> &frame1,
                             NCVMatrix<Ncv32f> &uOut,
                             NCVMatrix<Ncv32f> &vOut,
                             cudaStream_t stream)
{
    ncvAssertPrintReturn(desc.alpha > 0.0f                   , "Invalid alpha"                      , NCV_INCONSISTENT_INPUT);
    ncvAssertPrintReturn(desc.gamma >= 0.0f                  , "Invalid gamma"                      , NCV_INCONSISTENT_INPUT);
    ncvAssertPrintReturn(desc.number_of_inner_iterations > 0 , "Invalid number of inner iterations" , NCV_INCONSISTENT_INPUT);
    ncvAssertPrintReturn(desc.number_of_outer_iterations > 0 , "Invalid number of outer iterations" , NCV_INCONSISTENT_INPUT);
    ncvAssertPrintReturn(desc.number_of_solver_iterations > 0, "Invalid number of solver iterations", NCV_INCONSISTENT_INPUT);

    const Ncv32u kSourceWidth  = frame0.width();
    const Ncv32u kSourceHeight = frame0.height();

    ncvAssertPrintReturn(frame1.width() == kSourceWidth && frame1.height() == kSourceHeight, "Frame dims do not match", NCV_INCONSISTENT_INPUT);
    ncvAssertReturn(uOut.width() == kSourceWidth && vOut.width() == kSourceWidth &&
        uOut.height() == kSourceHeight && vOut.height() == kSourceHeight, NCV_INCONSISTENT_INPUT);

    ncvAssertReturn(gpu_mem_allocator.isInitialized(), NCV_ALLOCATOR_NOT_INITIALIZED);

    bool kSkipProcessing = gpu_mem_allocator.isCounting();

    int cuda_device;
    ncvAssertCUDAReturn(cudaGetDevice(&cuda_device), NCV_CUDA_ERROR);

    cudaDeviceProp device_props;
    ncvAssertCUDAReturn(cudaGetDeviceProperties(&device_props, cuda_device), NCV_CUDA_ERROR);

    Ncv32u alignmentValue = gpu_mem_allocator.alignment ();

    const Ncv32u kStrideAlignmentFloat = alignmentValue / sizeof(float);
    const Ncv32u kSourcePitch = alignUp(kSourceWidth, kStrideAlignmentFloat) * sizeof(float);

    const Ncv32f scale_factor = desc.scale_factor;
    const Ncv32f alpha = desc.alpha;
    const Ncv32f gamma = desc.gamma;

    const Ncv32u kSizeInPixelsAligned = alignUp(kSourceWidth, kStrideAlignmentFloat)*kSourceHeight;

#if defined SAFE_VECTOR_DECL
#undef SAFE_VECTOR_DECL
#endif
#define SAFE_VECTOR_DECL(name, allocator, size) \
    FloatVector name((allocator), (size)); \
    ncvAssertReturn(name.isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    // matrix elements
    SAFE_VECTOR_DECL(diffusivity_x,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(diffusivity_y,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(denom_u,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(denom_v,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(num_dudv, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(num_u,    gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(num_v,    gpu_mem_allocator, kSizeInPixelsAligned);

    // flow components
    SAFE_VECTOR_DECL(u, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(v, gpu_mem_allocator, kSizeInPixelsAligned);

    SAFE_VECTOR_DECL(u_new, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(v_new, gpu_mem_allocator, kSizeInPixelsAligned);

    // flow increments
    SAFE_VECTOR_DECL(du, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(dv, gpu_mem_allocator, kSizeInPixelsAligned);

    SAFE_VECTOR_DECL(du_new, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(dv_new, gpu_mem_allocator, kSizeInPixelsAligned);

    // temporary storage
    SAFE_VECTOR_DECL(device_buffer, gpu_mem_allocator,
        alignUp(kSourceWidth, kStrideAlignmentFloat) * alignUp(kSourceHeight, kStrideAlignmentFloat));

    // image derivatives
    SAFE_VECTOR_DECL(Ix,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Ixx, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Ix0, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Iy,  gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Iyy, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Iy0, gpu_mem_allocator, kSizeInPixelsAligned);
    SAFE_VECTOR_DECL(Ixy, gpu_mem_allocator, kSizeInPixelsAligned);

    // spatial derivative filter size
    const int kDFilterSize = 5;
    SAFE_VECTOR_DECL(derivativeFilter, gpu_mem_allocator, kDFilterSize);

    if (!kSkipProcessing)
    {
        const float derivativeFilterHost[kDFilterSize] = {1.0f, -8.0f, 0.0f, 8.0f, -1.0f};

        ncvAssertCUDAReturn(cudaMemcpy(derivativeFilter.ptr(), derivativeFilterHost, sizeof(float) * kDFilterSize,
            cudaMemcpyHostToDevice), NCV_CUDA_ERROR);
    }

    //prepare image pyramid
    ImagePyramid pyr(desc.number_of_outer_iterations);

    float scale = 1.0f;

    //cuda arrays for frames
    std::unique_ptr<FloatVector> pI0(new FloatVector(gpu_mem_allocator, kSizeInPixelsAligned));
    ncvAssertReturn(pI0->isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    std::unique_ptr<FloatVector> pI1(new FloatVector(gpu_mem_allocator, kSizeInPixelsAligned));
    ncvAssertReturn(pI1->isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

    if (!kSkipProcessing)
    {
        //copy frame data to device
        size_t dst_width_in_bytes = alignUp(kSourceWidth, kStrideAlignmentFloat) * sizeof(float);
        size_t src_width_in_bytes = kSourceWidth * sizeof(float);
        size_t src_pitch_in_bytes = frame0.pitch();

        ncvAssertCUDAReturn( cudaMemcpy2DAsync(pI0->ptr(), dst_width_in_bytes, frame0.ptr(),
            src_pitch_in_bytes, src_width_in_bytes, kSourceHeight, cudaMemcpyDeviceToDevice, stream), NCV_CUDA_ERROR );

        ncvAssertCUDAReturn( cudaMemcpy2DAsync(pI1->ptr(), dst_width_in_bytes, frame1.ptr(),
            src_pitch_in_bytes, src_width_in_bytes, kSourceHeight, cudaMemcpyDeviceToDevice, stream), NCV_CUDA_ERROR );
    }

    FloatVector* I0 = pI0.release();
    FloatVector* I1 = pI1.release();

        //prepare pyramid
    pyr.img0.push_back(I0);
    pyr.img1.push_back(I1);

    pyr.w.push_back(kSourceWidth);
    pyr.h.push_back(kSourceHeight);

    scale *= scale_factor;

    Ncv32u prev_level_width  = kSourceWidth;
    Ncv32u prev_level_height = kSourceHeight;
    while((prev_level_width > 15) && (prev_level_height > 15) && (static_cast<Ncv32u>(pyr.img0.size()) < desc.number_of_outer_iterations))
    {
        //current resolution
        Ncv32u level_width  = static_cast<Ncv32u>(ceilf(kSourceWidth  * scale));
        Ncv32u level_height = static_cast<Ncv32u>(ceilf(kSourceHeight * scale));

        Ncv32u level_width_aligned  = alignUp(level_width,  kStrideAlignmentFloat);

        Ncv32u buffer_size = alignUp(level_width, kStrideAlignmentFloat) * level_height; // buffer size in floats

        Ncv32u prev_level_pitch = alignUp(prev_level_width, kStrideAlignmentFloat) * sizeof(float);

        std::unique_ptr<FloatVector> level_frame0(new FloatVector(gpu_mem_allocator, buffer_size));
        ncvAssertReturn(level_frame0->isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        std::unique_ptr<FloatVector> level_frame1(new FloatVector(gpu_mem_allocator, buffer_size));
        ncvAssertReturn(level_frame1->isMemAllocated(), NCV_ALLOCATOR_BAD_ALLOC);

        if (!kSkipProcessing)
        {
            ncvAssertCUDAReturn(cudaStreamSynchronize(stream), NCV_CUDA_ERROR);

            NcvSize32u srcSize (prev_level_width, prev_level_height);
            NcvSize32u dstSize (level_width, level_height);
            NcvRect32u srcROI (0, 0, prev_level_width, prev_level_height);
            NcvRect32u dstROI (0, 0, level_width, level_height);

            // frame 0
            ncvAssertReturnNcvStat( nppiStResize_32f_C1R (I0->ptr(), srcSize, prev_level_pitch, srcROI,
                level_frame0->ptr(), dstSize, level_width_aligned * sizeof (float), dstROI, scale_factor, scale_factor, nppStSupersample) );

            // frame 1
            ncvAssertReturnNcvStat( nppiStResize_32f_C1R (I1->ptr(), srcSize, prev_level_pitch, srcROI,
                level_frame1->ptr(), dstSize, level_width_aligned * sizeof (float), dstROI, scale_factor, scale_factor, nppStSupersample) );
        }

        I0 = level_frame0.release();
        I1 = level_frame1.release();

        //store pointers
        pyr.img0.push_back(I0);
        pyr.img1.push_back(I1);

        pyr.w.push_back(level_width);
        pyr.h.push_back(level_height);

        scale *= scale_factor;

        prev_level_width  = level_width;
        prev_level_height = level_height;
    }

    if (!kSkipProcessing)
    {
        //initial values for flow is 0
        ncvAssertCUDAReturn(cudaMemsetAsync(u.ptr(), 0, kSizeInPixelsAligned * sizeof(float), stream), NCV_CUDA_ERROR);
        ncvAssertCUDAReturn(cudaMemsetAsync(v.ptr(), 0, kSizeInPixelsAligned * sizeof(float), stream), NCV_CUDA_ERROR);

        //select images with lowest resolution
        ncvAssertCUDAReturn(cudaStreamSynchronize(stream), NCV_CUDA_ERROR);

        FloatVector* ptrU = &u;
        FloatVector* ptrV = &v;
        FloatVector* ptrUNew = &u_new;
        FloatVector* ptrVNew = &v_new;

        std::vector<FloatVector*>::const_reverse_iterator img0Iter = pyr.img0.rbegin();
        std::vector<FloatVector*>::const_reverse_iterator img1Iter = pyr.img1.rbegin();

        //outer loop
        //warping fixed point iteration
        while(!pyr.w.empty())
        {
            //current grid dimensions
            const Ncv32u kLevelWidth  = pyr.w.back();
            const Ncv32u kLevelHeight = pyr.h.back();
            const Ncv32u kLevelStride = alignUp(kLevelWidth, kStrideAlignmentFloat);

            //size of current image in bytes
            const int kLevelSizeInBytes = kLevelStride * kLevelHeight * sizeof(float);

            //number of points at current resolution
            const int kLevelSizeInPixels = kLevelStride * kLevelHeight;

            //initial guess for du and dv
            ncvAssertCUDAReturn(cudaMemsetAsync(du.ptr(), 0, kLevelSizeInBytes, stream), NCV_CUDA_ERROR);
            ncvAssertCUDAReturn(cudaMemsetAsync(dv.ptr(), 0, kLevelSizeInBytes, stream), NCV_CUDA_ERROR);

            I0 = *img0Iter;
            I1 = *img1Iter;

            ++img0Iter;
            ++img1Iter;

            Texture texI0(kLevelHeight, kLevelWidth, I0->ptr(), kLevelStride * sizeof(float), true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texI1(kLevelHeight, kLevelWidth, I1->ptr(), kLevelStride * sizeof(float), true, cudaFilterModeLinear, cudaAddressModeMirror);

            //compute derivatives
            dim3 dBlocks(iDivUp(kLevelWidth, 32), iDivUp(kLevelHeight, 6));
            //dim3 dThreads(32, 6);

            const int kPitchTex = kLevelStride * sizeof(float);

            NcvSize32u srcSize(kLevelWidth, kLevelHeight);
            Ncv32u nSrcStep = kLevelStride * sizeof(float);
            NcvRect32u oROI(0, 0, kLevelWidth, kLevelHeight);

            // Ix0
            ncvAssertReturnNcvStat( nppiStFilterRowBorder_32f_C1R (I0->ptr(), srcSize, nSrcStep, Ix0.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Iy0
            ncvAssertReturnNcvStat( nppiStFilterColumnBorder_32f_C1R (I0->ptr(), srcSize, nSrcStep, Iy0.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Ix
            ncvAssertReturnNcvStat( nppiStFilterRowBorder_32f_C1R (I1->ptr(), srcSize, nSrcStep, Ix.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Iy
            ncvAssertReturnNcvStat( nppiStFilterColumnBorder_32f_C1R (I1->ptr(), srcSize, nSrcStep, Iy.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Ixx
            ncvAssertReturnNcvStat( nppiStFilterRowBorder_32f_C1R (Ix.ptr(), srcSize, nSrcStep, Ixx.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Iyy
            ncvAssertReturnNcvStat( nppiStFilterColumnBorder_32f_C1R (Iy.ptr(), srcSize, nSrcStep, Iyy.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            // Ixy
            ncvAssertReturnNcvStat( nppiStFilterRowBorder_32f_C1R (Iy.ptr(), srcSize, nSrcStep, Ixy.ptr(), srcSize, nSrcStep, oROI,
                nppStBorderMirror, derivativeFilter.ptr(), kDFilterSize, kDFilterSize/2, 1.0f/12.0f) );

            Texture texIx(kLevelHeight, kLevelWidth, Ix.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIxx(kLevelHeight, kLevelWidth, Ixx.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIx0(kLevelHeight, kLevelWidth, Ix0.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIy(kLevelHeight, kLevelWidth, Iy.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIyy(kLevelHeight, kLevelWidth, Iyy.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIy0(kLevelHeight, kLevelWidth, Iy0.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texIxy(kLevelHeight, kLevelWidth, Ixy.ptr(), kPitchTex, true, cudaFilterModeLinear, cudaAddressModeMirror);
            Texture texDiffX(kLevelSizeInBytes, diffusivity_x.ptr());
            Texture texDiffY(kLevelSizeInBytes, diffusivity_y.ptr());

            //    flow
            Texture texU(kLevelSizeInBytes, ptrU->ptr());
            Texture texV(kLevelSizeInBytes, ptrV->ptr());
            //    flow increments
            Texture texDu(kLevelSizeInBytes, du.ptr());
            Texture texDv(kLevelSizeInBytes, dv.ptr());
            Texture texDuNew(kLevelSizeInBytes, du_new.ptr());
            Texture texDvNew(kLevelSizeInBytes, dv_new.ptr());

            dim3 psor_blocks(iDivUp(kLevelWidth, PSOR_TILE_WIDTH), iDivUp(kLevelHeight, PSOR_TILE_HEIGHT));
            dim3 psor_threads(PSOR_TILE_WIDTH, PSOR_TILE_HEIGHT);

            dim3 sor_blocks(iDivUp(kLevelWidth, SOR_TILE_WIDTH), iDivUp(kLevelHeight, SOR_TILE_HEIGHT));
            dim3 sor_threads(SOR_TILE_WIDTH, SOR_TILE_HEIGHT);

            // inner loop
            // lagged nonlinearity fixed point iteration
            ncvAssertCUDAReturn(cudaStreamSynchronize(stream), NCV_CUDA_ERROR);
            for (Ncv32u current_inner_iteration = 0; current_inner_iteration < desc.number_of_inner_iterations; ++current_inner_iteration)
            {
                //compute coefficients
                prepare_sor_stage_1_tex<<<psor_blocks, psor_threads, 0, stream>>> (texU, texV, texDu, texDv, texI0, texI1, texIx, texIxx, texIx0, texIy, texIyy, texIy0, texIxy,
                    diffusivity_x.ptr(), diffusivity_y.ptr(), denom_u.ptr(), denom_v.ptr(), num_dudv.ptr(), num_u.ptr(), num_v.ptr(), kLevelWidth, kLevelHeight, kLevelStride, alpha, gamma);

                ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);

                prepare_sor_stage_2<<<psor_blocks, psor_threads, 0, stream>>>(texDiffX, texDiffY, denom_u.ptr(), denom_v.ptr(), kLevelWidth, kLevelHeight, kLevelStride);

                ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);


                //solve linear system
                for (Ncv32u solver_iteration = 0; solver_iteration < desc.number_of_solver_iterations; ++solver_iteration)
                {
                    float omega = 1.99f;
                    sor_pass<0><<<sor_blocks, sor_threads, 0, stream>>>(texU, texV, texDu, texDv, texDiffX, texDiffY, du_new.ptr(), dv_new.ptr(), denom_u.ptr(), denom_v.ptr(),
                        num_u.ptr(), num_v.ptr(), num_dudv.ptr(), omega, kLevelWidth, kLevelHeight, kLevelStride);

                    ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);


                    sor_pass<1><<<sor_blocks, sor_threads, 0, stream>>>(texU, texV, texDuNew, texDvNew, texDiffX, texDiffY, du.ptr(), dv.ptr(), denom_u.ptr(), denom_v.ptr(), num_u.ptr(),
                        num_v.ptr(),num_dudv.ptr(), omega, kLevelWidth, kLevelHeight, kLevelStride);

                    ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);
                }//end of solver loop
            }// end of inner loop

            //update u and v
            add(ptrU->ptr(), du.ptr(), kLevelSizeInPixels, stream);
            ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);
            add(ptrV->ptr(), dv.ptr(), kLevelSizeInPixels, stream);
            ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);

            //prolongate using texture
            pyr.w.pop_back();
            pyr.h.pop_back();
            if (!pyr.w.empty())
            {
                //compute new image size
                Ncv32u nw = pyr.w.back();
                Ncv32u nh = pyr.h.back();
                Ncv32u ns = alignUp(nw, kStrideAlignmentFloat);

                dim3 p_blocks(iDivUp(nw, 32), iDivUp(nh, 8));
                //dim3 p_threads(32, 8);

                NcvSize32u inner_srcSize (kLevelWidth, kLevelHeight);
                NcvSize32u dstSize (nw, nh);
                NcvRect32u srcROI (0, 0, kLevelWidth, kLevelHeight);
                NcvRect32u dstROI (0, 0, nw, nh);

                ncvAssertReturnNcvStat( nppiStResize_32f_C1R (ptrU->ptr(), inner_srcSize, kLevelStride * sizeof (float), srcROI,
                    ptrUNew->ptr(), dstSize, ns * sizeof (float), dstROI, 1.0f/scale_factor, 1.0f/scale_factor, nppStBicubic) );

                ScaleVector(ptrUNew->ptr(), ptrUNew->ptr(), 1.0f/scale_factor, ns * nh, stream);
                ncvAssertCUDALastErrorReturn(NCV_CUDA_ERROR);

                ncvAssertReturnNcvStat( nppiStResize_32f_C1R (ptrV->ptr(), inner_srcSize, kLevelStride * sizeof (float), srcROI,
                    ptrVNew->ptr(), dstSize, ns * sizeof (float), dstROI, 1.0f/scale_factor, 1.0f/scale_factor, nppStBicubic) );

                ScaleVector(ptrVNew->ptr(), ptrVNew->ptr(), 1.0f/scale_factor, ns * nh, stream);
                ncvAssertCUDALastErrorReturn((int)NCV_CUDA_ERROR);

                cv::cuda::device::swap<FloatVector*>(ptrU, ptrUNew);
                cv::cuda::device::swap<FloatVector*>(ptrV, ptrVNew);
            }
            scale /= scale_factor;
        }

        // end of warping iterations
        ncvAssertCUDAReturn(cudaStreamSynchronize(stream), (int)NCV_CUDA_ERROR);

        ncvAssertCUDAReturn( cudaMemcpy2DAsync
            (uOut.ptr(), uOut.pitch(), ptrU->ptr(),
            kSourcePitch, kSourceWidth*sizeof(float), kSourceHeight, cudaMemcpyDeviceToDevice, stream), (int)NCV_CUDA_ERROR );

        ncvAssertCUDAReturn( cudaMemcpy2DAsync
            (vOut.ptr(), vOut.pitch(), ptrV->ptr(),
            kSourcePitch, kSourceWidth*sizeof(float), kSourceHeight, cudaMemcpyDeviceToDevice, stream), (int)NCV_CUDA_ERROR );

        ncvAssertCUDAReturn(cudaStreamSynchronize(stream), (int)NCV_CUDA_ERROR);
    }

    return NCV_SUCCESS;
}
