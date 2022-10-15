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

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include <opencv2/cudev/ptr2d/texture.hpp>
#include <limits.h>


namespace cv { namespace cuda { namespace device
{
    namespace stereobm
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////// Stereo BM ////////////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////

        #define ROWSperTHREAD 21     // the number of rows a thread will process

        #define BLOCK_W 128          // the thread block width (464)
        #define N_DISPARITIES 8

        #define STEREO_MIND 0                    // The minimum d range to check
        #define STEREO_DISP_STEP N_DISPARITIES   // the d step, must be <= 1 to avoid aliasing

        __device__ __forceinline__ int SQ(int a)
        {
            return a * a;
        }

        template<int RADIUS>
        __device__ unsigned int CalcSSD(volatile unsigned int *col_ssd_cache, volatile unsigned int *col_ssd, const int X, int cwidth)
        {
            unsigned int cache = 0;
            unsigned int cache2 = 0;

            if (X < cwidth - RADIUS)
            {
                for(int i = 1; i <= RADIUS; i++)
                    cache += col_ssd[i];
            }
            col_ssd_cache[0] = cache;

            __syncthreads();

            if (X < cwidth - RADIUS)
            {
                if (threadIdx.x < BLOCK_W - RADIUS)
                    cache2 = col_ssd_cache[RADIUS];
                else
                    for(int i = RADIUS + 1; i < (2 * RADIUS + 1); i++)
                        cache2 += col_ssd[i];
            }

            return col_ssd[0] + cache + cache2;
        }

        template<int RADIUS>
        __device__ uint2 MinSSD(volatile unsigned int *col_ssd_cache, volatile unsigned int *col_ssd, const int X, int cwidth, unsigned int* ssd)
        {
            //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * RADIUS)
            ssd[0] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 0 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[1] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 1 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[2] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 2 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[3] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 3 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[4] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 4 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[5] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 5 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[6] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 6 * (BLOCK_W + 2 * RADIUS), X, cwidth);
            __syncthreads();
            ssd[7] = CalcSSD<RADIUS>(col_ssd_cache, col_ssd + 7 * (BLOCK_W + 2 * RADIUS), X, cwidth);

            int mssd = ::min(::min(::min(ssd[0], ssd[1]), ::min(ssd[4], ssd[5])), ::min(::min(ssd[2], ssd[3]), ::min(ssd[6], ssd[7])));

            int bestIdx = 0;
            for (int i = 0; i < N_DISPARITIES; i++)
            {
                if (mssd == ssd[i])
                    bestIdx = i;
            }

            return make_uint2(mssd, bestIdx);
        }

        template<int RADIUS>
        __device__ void StepDown(int idx1, int idx2, unsigned char* imageL, unsigned char* imageR, int d, volatile unsigned int *col_ssd)
        {
            unsigned char leftPixel1;
            unsigned char leftPixel2;
            unsigned char rightPixel1[8];
            unsigned char rightPixel2[8];
            unsigned int diff1, diff2;

            leftPixel1 = imageL[idx1];
            leftPixel2 = imageL[idx2];

            idx1 = idx1 - d;
            idx2 = idx2 - d;

            rightPixel1[7] = imageR[idx1 - 7];
            rightPixel1[0] = imageR[idx1 - 0];
            rightPixel1[1] = imageR[idx1 - 1];
            rightPixel1[2] = imageR[idx1 - 2];
            rightPixel1[3] = imageR[idx1 - 3];
            rightPixel1[4] = imageR[idx1 - 4];
            rightPixel1[5] = imageR[idx1 - 5];
            rightPixel1[6] = imageR[idx1 - 6];

            rightPixel2[7] = imageR[idx2 - 7];
            rightPixel2[0] = imageR[idx2 - 0];
            rightPixel2[1] = imageR[idx2 - 1];
            rightPixel2[2] = imageR[idx2 - 2];
            rightPixel2[3] = imageR[idx2 - 3];
            rightPixel2[4] = imageR[idx2 - 4];
            rightPixel2[5] = imageR[idx2 - 5];
            rightPixel2[6] = imageR[idx2 - 6];

            //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * RADIUS)
            diff1 = leftPixel1 - rightPixel1[0];
            diff2 = leftPixel2 - rightPixel2[0];
            col_ssd[0 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[1];
            diff2 = leftPixel2 - rightPixel2[1];
            col_ssd[1 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[2];
            diff2 = leftPixel2 - rightPixel2[2];
            col_ssd[2 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[3];
            diff2 = leftPixel2 - rightPixel2[3];
            col_ssd[3 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[4];
            diff2 = leftPixel2 - rightPixel2[4];
            col_ssd[4 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[5];
            diff2 = leftPixel2 - rightPixel2[5];
            col_ssd[5 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[6];
            diff2 = leftPixel2 - rightPixel2[6];
            col_ssd[6 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);

            diff1 = leftPixel1 - rightPixel1[7];
            diff2 = leftPixel2 - rightPixel2[7];
            col_ssd[7 * (BLOCK_W + 2 * RADIUS)] += SQ(diff2) - SQ(diff1);
        }

        template<int RADIUS>
        __device__ void InitColSSD(int x_tex, int y_tex, int im_pitch, unsigned char* imageL, unsigned char* imageR, int d, volatile unsigned int *col_ssd)
        {
            unsigned char leftPixel1;
            int idx;
            unsigned int diffa[] = {0, 0, 0, 0, 0, 0, 0, 0};

            for(int i = 0; i < (2 * RADIUS + 1); i++)
            {
                idx = y_tex * im_pitch + x_tex;
                leftPixel1 = imageL[idx];
                idx = idx - d;

                diffa[0] += SQ(leftPixel1 - imageR[idx - 0]);
                diffa[1] += SQ(leftPixel1 - imageR[idx - 1]);
                diffa[2] += SQ(leftPixel1 - imageR[idx - 2]);
                diffa[3] += SQ(leftPixel1 - imageR[idx - 3]);
                diffa[4] += SQ(leftPixel1 - imageR[idx - 4]);
                diffa[5] += SQ(leftPixel1 - imageR[idx - 5]);
                diffa[6] += SQ(leftPixel1 - imageR[idx - 6]);
                diffa[7] += SQ(leftPixel1 - imageR[idx - 7]);

                y_tex += 1;
            }
            //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * RADIUS)
            col_ssd[0 * (BLOCK_W + 2 * RADIUS)] = diffa[0];
            col_ssd[1 * (BLOCK_W + 2 * RADIUS)] = diffa[1];
            col_ssd[2 * (BLOCK_W + 2 * RADIUS)] = diffa[2];
            col_ssd[3 * (BLOCK_W + 2 * RADIUS)] = diffa[3];
            col_ssd[4 * (BLOCK_W + 2 * RADIUS)] = diffa[4];
            col_ssd[5 * (BLOCK_W + 2 * RADIUS)] = diffa[5];
            col_ssd[6 * (BLOCK_W + 2 * RADIUS)] = diffa[6];
            col_ssd[7 * (BLOCK_W + 2 * RADIUS)] = diffa[7];
        }

        template<int RADIUS>
        __global__ void stereoKernel(unsigned char *left, unsigned char *right, size_t img_step, PtrStepb disp, int maxdisp,
                                     int uniquenessRatio, unsigned int* cminSSDImage, size_t cminSSD_step, int cwidth, int cheight)
        {
            extern __shared__ unsigned int col_ssd_cache[];
            uint line_ssds[2 + N_DISPARITIES]; // +2 - tail of previous batch for accurate uniquenessRatio check
            uint* batch_ssds = line_ssds + 2;

            uint line_ssd_tails[3*ROWSperTHREAD];
            uchar uniqueness_approved[ROWSperTHREAD];
            uchar local_disparity[ROWSperTHREAD];

            volatile unsigned int *col_ssd = col_ssd_cache + BLOCK_W + threadIdx.x;
            volatile unsigned int *col_ssd_extra = threadIdx.x < (2 * RADIUS) ? col_ssd + BLOCK_W : 0;

            const int X = (blockIdx.x * BLOCK_W + threadIdx.x + maxdisp + RADIUS);
            const int Y = (blockIdx.y * ROWSperTHREAD + RADIUS);

            unsigned int* minSSDImage = cminSSDImage + X + Y * cminSSD_step;
            unsigned char* disparImage = disp.data + X + Y * disp.step;
            float thresh_scale;

            int end_row = ::min(ROWSperTHREAD, cheight - Y - RADIUS);
            int y_tex;
            int x_tex = X - RADIUS;

            if (x_tex >= cwidth)
                return;

            for(int i = 0; i < ROWSperTHREAD; i++)
                local_disparity[i] = 0;

            for(int i = 0; i < 3*ROWSperTHREAD; i++)
            {
                line_ssd_tails[i] = UINT_MAX;
            }

            if (uniquenessRatio > 0)
            {
                batch_ssds[6] = UINT_MAX;
                batch_ssds[7] = UINT_MAX;
                thresh_scale = (1.0 + uniquenessRatio / 100.0f);
                for(int i = 0; i < ROWSperTHREAD; i++)
                {
                    uniqueness_approved[i] = 1;
                }
            }

            for(int d = STEREO_MIND; d < maxdisp; d += STEREO_DISP_STEP)
            {
                y_tex = Y - RADIUS;

                InitColSSD<RADIUS>(x_tex, y_tex, img_step, left, right, d, col_ssd);

                if (col_ssd_extra != nullptr)
                    if (x_tex + BLOCK_W < cwidth)
                        InitColSSD<RADIUS>(x_tex + BLOCK_W, y_tex, img_step, left, right, d, col_ssd_extra);

                __syncthreads(); //before MinSSD function

                if (Y < cheight - RADIUS)
                {
                    uint2 batch_opt = MinSSD<RADIUS>(col_ssd_cache + threadIdx.x, col_ssd, X, cwidth, batch_ssds);

                    // For threads that do not satisfy the if condition below("X < cwidth - RADIUS"), previously
                    // computed "batch_opt" value, which is the result of "MinSSD" function call, is not used at all.
                    //
                    // However, since the "MinSSD" function has "__syncthreads" call in its body, those threads
                    // must also call "MinSSD" to avoid deadlock. (#13850)
                    //
                    // From CUDA 9, using "__syncwarp" with proper mask value instead of using "__syncthreads"
                    // could be an option, but the shared memory access pattern does not allow this option,
                    // resulting in race condition. (Checked via "cuda-memcheck --tool racecheck")

                    if (X < cwidth - RADIUS)
                    {
                        unsigned int last_opt = line_ssd_tails[3*0 + 0];
                        unsigned int opt = ::min(last_opt, batch_opt.x);

                        if (uniquenessRatio > 0)
                        {
                            line_ssds[0] = line_ssd_tails[3*0 + 1];
                            line_ssds[1] = line_ssd_tails[3*0 + 2];

                            float thresh = thresh_scale * opt;
                            int dtest = local_disparity[0];

                            if(batch_opt.x < last_opt)
                            {
                                uniqueness_approved[0] = 1;
                                dtest = d + batch_opt.y;
                                if ((local_disparity[0] < dtest-1 || local_disparity[0] > dtest+1) && (last_opt <= thresh))
                                {
                                    uniqueness_approved[0] = 0;
                                }
                            }

                            if(uniqueness_approved[0])
                            {
                                // the trial to decompose the code on 2 loops without ld vs dtest makes
                                // uniqueness check dramatically slow. at least on gf 1080
                                for (int ld = d-2; ld < d + N_DISPARITIES; ld++)
                                {
                                    if ((ld < dtest-1 || ld > dtest+1) && (line_ssds[ld-d+2] <= thresh))
                                    {
                                        uniqueness_approved[0] = 0;
                                        break;
                                    }
                                }
                            }


                            line_ssd_tails[3*0 + 1] = batch_ssds[6];
                            line_ssd_tails[3*0 + 2] = batch_ssds[7];
                        }

                        line_ssd_tails[3*0 + 0] = opt;
                        if (batch_opt.x < last_opt)
                        {
                            local_disparity[0] = (unsigned char)(d + batch_opt.y);
                        }
                    }
                }

                for(int row = 1; row < end_row; row++)
                {
                    int idx1 = y_tex * img_step + x_tex;
                    int idx2 = (y_tex + (2 * RADIUS + 1)) * img_step + x_tex;

                    __syncthreads();

                    StepDown<RADIUS>(idx1, idx2, left, right, d, col_ssd);

                    if (col_ssd_extra)
                        if (x_tex + BLOCK_W < cwidth)
                            StepDown<RADIUS>(idx1, idx2, left + BLOCK_W, right + BLOCK_W, d, col_ssd_extra);

                    y_tex += 1;

                    __syncthreads();

                    if (row < cheight - RADIUS - Y)
                    {
                        uint2 batch_opt = MinSSD<RADIUS>(col_ssd_cache + threadIdx.x, col_ssd, X, cwidth, batch_ssds);
                        // For threads that do not satisfy the if condition below("X < cwidth - RADIUS"), previously
                        // computed "batch_opt" value, which is the result of "MinSSD" function call, is not used at all.
                        //
                        // However, since the "MinSSD" function has "__syncthreads" call in its body, those threads
                        // must also call "MinSSD" to avoid deadlock. (#13850)
                        //
                        // From CUDA 9, using "__syncwarp" with proper mask value instead of using "__syncthreads"
                        // could be an option, but the shared memory access pattern does not allow this option,
                        // resulting in race condition. (Checked via "cuda-memcheck --tool racecheck")

                        if (X < cwidth - RADIUS)
                        {
                            unsigned int last_opt = line_ssd_tails[3*row + 0];
                            unsigned int opt = ::min(last_opt, batch_opt.x);
                            if (uniquenessRatio > 0)
                            {
                                line_ssds[0] = line_ssd_tails[3*row + 1];
                                line_ssds[1] = line_ssd_tails[3*row + 2];

                                float thresh = thresh_scale * opt;
                                int dtest = local_disparity[row];

                                if(batch_opt.x < last_opt)
                                {
                                    uniqueness_approved[row] = 1;
                                    dtest = d + batch_opt.y;
                                    if ((local_disparity[row] < dtest-1 || local_disparity[row] > dtest+1) && (last_opt <= thresh))
                                    {
                                        uniqueness_approved[row] = 0;
                                    }
                                }

                                if(uniqueness_approved[row])
                                {
                                    for (int ld = 0; ld < N_DISPARITIES + 2; ld++)
                                    {
                                        if (((d+ld-2 < dtest-1) || (d+ld-2 > dtest+1)) && (line_ssds[ld] <= thresh))
                                        {
                                            uniqueness_approved[row] = 0;
                                            break;
                                        }
                                    }
                                }

                                line_ssd_tails[3*row + 1] = batch_ssds[6];
                                line_ssd_tails[3*row + 2] = batch_ssds[7];
                            }

                            line_ssd_tails[3*row + 0] = opt;

                            if (batch_opt.x < last_opt)
                            {
                                local_disparity[row] = (unsigned char)(d + batch_opt.y);
                            }
                        }
                    }
                } // for row loop

                __syncthreads(); // before initializing shared memory at the beginning of next loop

            } // for d loop

            for (int row = 0; row < end_row; row++)
            {
                minSSDImage[row * cminSSD_step] = line_ssd_tails[3*row + 0];
            }

            if (uniquenessRatio > 0)
            {
                for (int row = 0; row < end_row; row++)
                {
                    // drop disparity for pixel where uniqueness requirement was not satisfied (zero value)
                    disparImage[disp.step * row] = local_disparity[row] * uniqueness_approved[row];
                }
            }
            else
            {
                for (int row = 0; row < end_row; row++)
                {
                    disparImage[disp.step * row] = local_disparity[row];
                }
            }
        }

        template<int RADIUS> void kernel_caller(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp,
                                                int maxdisp, int uniquenessRatio, unsigned int* missd_buffer,
                                                size_t minssd_step, int cwidth, int cheight, cudaStream_t & stream)
        {
            dim3 grid(1,1,1);
            dim3 threads(BLOCK_W, 1, 1);

            grid.x = divUp(left.cols - maxdisp - 2 * RADIUS, BLOCK_W);
            grid.y = divUp(left.rows - 2 * RADIUS, ROWSperTHREAD);

            //See above:  #define COL_SSD_SIZE (BLOCK_W + 2 * RADIUS)
            size_t smem_size = (BLOCK_W + N_DISPARITIES * (BLOCK_W + 2 * RADIUS)) * sizeof(unsigned int);

            stereoKernel<RADIUS><<<grid, threads, smem_size, stream>>>(left.data, right.data, left.step, disp, maxdisp, uniquenessRatio,
                                                                       missd_buffer, minssd_step, cwidth, cheight);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        };

        typedef void (*kernel_caller_t)(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp,
                                        int maxdisp, int uniquenessRatio, unsigned int* missd_buffer,
                                        size_t minssd_step, int cwidth, int cheight, cudaStream_t & stream);

        const static kernel_caller_t callers[] =
        {
            0,
            kernel_caller< 1>, kernel_caller< 2>, kernel_caller< 3>, kernel_caller< 4>, kernel_caller< 5>,
            kernel_caller< 6>, kernel_caller< 7>, kernel_caller< 8>, kernel_caller< 9>, kernel_caller<10>,
            kernel_caller<11>, kernel_caller<12>, kernel_caller<13>, kernel_caller<14>, kernel_caller<15>,
            kernel_caller<16>, kernel_caller<17>, kernel_caller<18>, kernel_caller<19>, kernel_caller<20>,
            kernel_caller<21>, kernel_caller<22>, kernel_caller<23>, kernel_caller<24>, kernel_caller<25>

            //0,0,0, 0,0,0, 0,0,kernel_caller<9>
        };
        const int calles_num = sizeof(callers)/sizeof(callers[0]);

        void stereoBM_CUDA(const PtrStepSzb& left, const PtrStepSzb& right, const PtrStepSzb& disp, int maxdisp,
                           int winsz, int uniquenessRatio, const PtrStepSz<unsigned int>& minSSD_buf, cudaStream_t& stream)
        {
            int winsz2 = winsz >> 1;

            if (winsz2 == 0 || winsz2 >= calles_num)
                CV_Error(cv::Error::StsBadArg, "Unsupported window size");

            cudaSafeCall( cudaMemset2DAsync(disp.data, disp.step, 0, disp.cols, disp.rows, stream) );
            cudaSafeCall( cudaMemset2DAsync(minSSD_buf.data, minSSD_buf.step, 0xFF, minSSD_buf.cols * minSSD_buf.elemSize(), disp.rows, stream) );

            size_t minssd_step = minSSD_buf.step/minSSD_buf.elemSize();
            callers[winsz2](left, right, disp, maxdisp, uniquenessRatio, minSSD_buf.data, minssd_step, left.cols, left.rows, stream);
        }

        __device__ inline int clamp(int x, int a, int b)
        {
            return ::max(a, ::min(b, x));
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////// Sobel Prefiler ///////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////

        __global__ void prefilter_kernel_xsobel(PtrStepSzb input, PtrStepSzb output, int prefilterCap)
        {
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < output.cols && y < output.rows)
            {
                int conv = input.ptr(::max(0,y-1))[::max(0,x-1)] * (-1) + input.ptr(::max(0, y-1))[::min(x+1, input.cols-1)] * (1) +
                           input.ptr(y  )[::max(0,x-1)] * (-2) + input.ptr(y  )[::min(x+1, input.cols-1)] * (2) +
                           input.ptr(::min(y+1, input.rows-1))[::max(0,x-1)] * (-1) + input.ptr(::min(y+1, input.rows-1))[::min(x+1,input.cols-1)] * (1);

                conv = ::min(::min(::max(-prefilterCap, conv), prefilterCap) + prefilterCap, 255);
                output.ptr(y)[x] = conv & 0xFF;
            }
        }

        void prefilter_xsobel(const PtrStepSzb& input, const PtrStepSzb& output, int prefilterCap, cudaStream_t & stream)
        {
            dim3 threads(16, 16, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(input.cols, threads.x);
            grid.y = divUp(input.rows, threads.y);

            prefilter_kernel_xsobel<<<grid, threads, 0, stream>>>(input, output, prefilterCap);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////  Norm Prefiler ///////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////

        __global__ void prefilter_kernel_norm(PtrStepSzb input, PtrStepSzb output, int prefilterCap, int scale_g, int scale_s, int winsize)
        {
            // prefilterCap in range 1..63, checked in StereoBMImpl::compute
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            int cols = input.cols;
            int rows = input.rows;
            int WSZ2 = winsize / 2;

            if(x < cols && y < rows)
            {
                int cov1 =                               input.ptr(::max(y-1, 0))[x] * 1 +
                    input.ptr(y)[::min(x+1, cols-1)] * 1 + input.ptr(y  )[x] * 4 + input.ptr(y)[::min(x+1, cols-1)] * 1 +
                                                         input.ptr(::min(y+1, rows-1))[x] * 1;

                int cov2 = 0;
                for(int i = -WSZ2; i < WSZ2+1; i++)
                    for(int j = -WSZ2; j < WSZ2+1; j++)
                        cov2 += input.ptr(clamp(y+i, 0, rows-1))[clamp(x+j, 0, cols-1)];

                int res = (cov1*scale_g - cov2*scale_s)>>10;
                res = clamp(res, -prefilterCap, prefilterCap) + prefilterCap;
                output.ptr(y)[x] = res;
            }
        }

        void prefilter_norm(const PtrStepSzb& input, const PtrStepSzb& output, int prefilterCap, int winsize, cudaStream_t & stream)
        {
            dim3 threads(16, 16, 1);
            dim3 grid(1, 1, 1);

            grid.x = divUp(input.cols, threads.x);
            grid.y = divUp(input.rows, threads.y);

            int scale_g = winsize*winsize/8, scale_s = (1024 + scale_g)/(scale_g*2);
            scale_g *= scale_s;

            prefilter_kernel_norm<<<grid, threads, 0, stream>>>(input, output, prefilterCap, scale_g, scale_s, winsize);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }


        //////////////////////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////// Textureness filtering ////////////////////////////////////////
        //////////////////////////////////////////////////////////////////////////////////////////////////

        __device__ __forceinline__ float sobel(cv::cudev::TexturePtr<uchar, float> texSrc, int x, int y)
        {
            float conv = texSrc(y - 1, x - 1) * (-1) + texSrc(y - 1, x + 1) * (1) +
                texSrc(y, x - 1) * (-2) + texSrc(y, x + 1) * (2) +
                texSrc(y + 1, x - 1) * (-1) + texSrc(y + 1, x + 1) * (1);

            return fabs(conv);
        }

        __device__ float CalcSums(float *cols, float *cols_cache, int winsz)
        {
            float cache = 0;
            float cache2 = 0;
            int winsz2 = winsz/2;

            for(int i = 1; i <= winsz2; i++)
                cache += cols[i];

            cols_cache[0] = cache;

            __syncthreads();

            if (threadIdx.x < blockDim.x - winsz2)
                cache2 = cols_cache[winsz2];
            else
                for(int i = winsz2 + 1; i < winsz; i++)
                    cache2 += cols[i];

            return cols[0] + cache + cache2;
        }

        #define RpT (2 * ROWSperTHREAD)  // got experimentally

        __global__ void textureness_kernel(cv::cudev::TexturePtr<uchar,float> texSrc, PtrStepSzb disp, int winsz, float threshold)
        {
            int winsz2 = winsz/2;
            int n_dirty_pixels = (winsz2) * 2;

            extern __shared__ float cols_cache[];
            float *cols = cols_cache + blockDim.x + threadIdx.x;
            float *cols_extra = threadIdx.x < n_dirty_pixels ? cols + blockDim.x : 0;

            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int beg_row = blockIdx.y * RpT;
            int end_row = ::min(beg_row + RpT, disp.rows);

            if (x < disp.cols)
            {
                int y = beg_row;

                float sum = 0;
                float sum_extra = 0;

                for(int i = y - winsz2; i <= y + winsz2; ++i)
                {
                    sum += sobel(texSrc, x - winsz2, i);
                    if (cols_extra)
                        sum_extra += sobel(texSrc, x + blockDim.x - winsz2, i);
                }
                *cols = sum;
                if (cols_extra)
                    *cols_extra = sum_extra;

                __syncthreads();

                float sum_win = CalcSums(cols, cols_cache + threadIdx.x, winsz) * 255;
                if (sum_win < threshold)
                    disp.data[y * disp.step + x] = 0;

                __syncthreads();

                for(int y = beg_row + 1; y < end_row; ++y)
                {
                    sum = sum - sobel(texSrc, x - winsz2, y - winsz2 - 1) + sobel(texSrc, x - winsz2, y + winsz2);
                    *cols = sum;

                    if (cols_extra)
                    {
                        sum_extra = sum_extra - sobel(texSrc, x + blockDim.x - winsz2, y - winsz2 - 1) + sobel(texSrc, x + blockDim.x - winsz2, y + winsz2);
                        *cols_extra = sum_extra;
                    }

                    __syncthreads();
                    float sum_win = CalcSums(cols, cols_cache + threadIdx.x, winsz) * 255;
                    if (sum_win < threshold)
                        disp.data[y * disp.step + x] = 0;

                    __syncthreads();
                }
            }
        }

        void postfilter_textureness(const PtrStepSzb& input, int winsz, float avgTexturenessThreshold, const PtrStepSzb& disp, cudaStream_t & stream)
        {
            avgTexturenessThreshold *= winsz * winsz;
            cv::cudev::Texture<unsigned char, float> tex(input, false, cudaFilterModeLinear, cudaAddressModeWrap, cudaReadModeNormalizedFloat);
            dim3 threads(128, 1, 1);
            dim3 grid(1, 1, 1);
            grid.x = divUp(input.cols, threads.x);
            grid.y = divUp(input.rows, RpT);
            size_t smem_size = (threads.x + threads.x + (winsz/2) * 2 ) * sizeof(float);
            textureness_kernel<<<grid, threads, smem_size, stream>>>(tex, disp, winsz, avgTexturenessThreshold);
            cudaSafeCall( cudaGetLastError() );
            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }
    } // namespace stereobm
}}} // namespace cv { namespace cuda { namespace cudev


#endif /* CUDA_DISABLER */
