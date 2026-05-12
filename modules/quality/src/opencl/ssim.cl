// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// SSIM OpenCL kernel - computes the SSIM quality map from precomputed values
// Supports 1-4 channels via compile-time cn parameter

#if cn == 1
#define T float
#define loadpix(addr) *(__global const float *)(addr)
#define storepix(val, addr) *(__global float *)(addr) = val
#define PIXSIZE (int)sizeof(float)
#elif cn == 2
#define T float2
#define loadpix(addr) *(__global const float2 *)(addr)
#define storepix(val, addr) *(__global float2 *)(addr) = val
#define PIXSIZE (int)sizeof(float2)
#elif cn == 3
#define T float3
#define loadpix(addr) vload3(0, (__global const float *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global float *)(addr))
#define PIXSIZE (int)sizeof(float) * 3
#elif cn == 4
#define T float4
#define loadpix(addr) *(__global const float4 *)(addr)
#define storepix(val, addr) *(__global float4 *)(addr) = val
#define PIXSIZE (int)sizeof(float4)
#endif

// Compute SSIM quality map from precomputed mu, mu^2, and sigma^2 values
// SSIM = (2*mu1*mu2 + C1) * (2*sigma12 + C2) / ((mu1^2 + mu2^2 + C1) * (sigma1^2 + sigma2^2 + C2))
__kernel void ssim_map(
    __global const uchar * mu1_ptr, int mu1_step, int mu1_offset,
    __global const uchar * mu2_ptr, int mu2_step, int mu2_offset,
    __global const uchar * mu1_sq_ptr, int mu1_sq_step, int mu1_sq_offset,
    __global const uchar * mu2_sq_ptr, int mu2_sq_step, int mu2_sq_offset,
    __global const uchar * sigma1_sq_ptr, int sigma1_sq_step, int sigma1_sq_offset,
    __global const uchar * sigma2_sq_ptr, int sigma2_sq_step, int sigma2_sq_offset,
    __global const uchar * sigma12_ptr, int sigma12_step, int sigma12_offset,
    __global uchar * dst_ptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
    float C1, float C2)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int mu1_idx = mad24(y, mu1_step, mad24(x, PIXSIZE, mu1_offset));
        int mu2_idx = mad24(y, mu2_step, mad24(x, PIXSIZE, mu2_offset));
        int mu1_sq_idx = mad24(y, mu1_sq_step, mad24(x, PIXSIZE, mu1_sq_offset));
        int mu2_sq_idx = mad24(y, mu2_sq_step, mad24(x, PIXSIZE, mu2_sq_offset));
        int sigma1_sq_idx = mad24(y, sigma1_sq_step, mad24(x, PIXSIZE, sigma1_sq_offset));
        int sigma2_sq_idx = mad24(y, sigma2_sq_step, mad24(x, PIXSIZE, sigma2_sq_offset));
        int sigma12_idx = mad24(y, sigma12_step, mad24(x, PIXSIZE, sigma12_offset));
        int dst_idx = mad24(y, dst_step, mad24(x, PIXSIZE, dst_offset));

        T mu1_val = loadpix(mu1_ptr + mu1_idx);
        T mu2_val = loadpix(mu2_ptr + mu2_idx);
        T mu1_sq = loadpix(mu1_sq_ptr + mu1_sq_idx);
        T mu2_sq = loadpix(mu2_sq_ptr + mu2_sq_idx);
        T sigma1_sq = loadpix(sigma1_sq_ptr + sigma1_sq_idx);
        T sigma2_sq = loadpix(sigma2_sq_ptr + sigma2_sq_idx);
        T sigma12 = loadpix(sigma12_ptr + sigma12_idx);

        T mu1_mu2 = mu1_val * mu2_val;
        T num = (2.0f * mu1_mu2 + C1) * (2.0f * sigma12 + C2);
        T den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2);
        T ssim_val = num / den;

        storepix(ssim_val, dst_ptr + dst_idx);
    }
}
