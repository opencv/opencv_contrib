// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


// #if defined _HFS_CUDA_ON_

#include "../precomp.hpp"
#include "../magnitude/magnitude.hpp"

namespace cv { namespace hfs {

__global__ void derrivativeXYDevice(const uchar *gray_img,
    int *delta_x, int *delta_y, int *mag, Vector2i img_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > img_size.x - 1 || y > img_size.y - 1)
        return;

    int idx = y*img_size.x + x;

    if (x == 0)
        delta_x[idx] = gray_img[idx + 1] - gray_img[idx];
    else if (x == img_size.x - 1)
        delta_x[idx] = gray_img[idx] - gray_img[idx - 1];
    else
        delta_x[idx] = gray_img[idx + 1] - gray_img[idx - 1];

    if (y == 0)
        delta_y[idx] = gray_img[idx + img_size.x] - gray_img[idx];
    else if (y == img_size.y - 1)
        delta_y[idx] = gray_img[idx] - gray_img[idx - img_size.x];
    else
        delta_y[idx] = gray_img[idx + img_size.x] - gray_img[idx - img_size.x];

    mag[idx] = (int)(0.5 +
        sqrt((double)(delta_x[idx] * delta_x[idx] + delta_y[idx] * delta_y[idx])));
}

__device__ __forceinline__ int dmin(int a, int b)
{
    return a < b ? a : b;
}

__device__ __forceinline__ int dmax(int a, int b)
{
    return a > b ? a : b;
}

__global__ void nonMaxSuppDevice(uchar *nms_mag,
    int *delta_x, int *delta_y, int *mag, Vector2i img_size)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > img_size.x - 1 || y > img_size.y - 1) return;

    int idx = y*img_size.x + x;

    if (x == 0 || x == img_size.x - 1 || y == 0 || y == img_size.y - 1)
    {
        nms_mag[idx] = 0;
        return;
    }

    int m00, gx, gy, z1, z2;
    double mag1, mag2, xprep, yprep;

    m00 = mag[idx];
    if (m00 == 0)
    {
        nms_mag[idx] = 0;
        return;
    }
    else
    {
        xprep = -(gx = delta_x[idx]) / ((double)m00);
        yprep = (gy = delta_y[idx]) / ((double)m00);
    }

    if (gx >= 0)
    {
        if (gy >= 0)
        {
            if (gx >= gy)
            {
                z1 = mag[idx - 1];
                z2 = mag[idx - img_size.x - 1];
                mag1 = (m00 - z1)*xprep + (z2 - z1)*yprep;

                z1 = mag[idx + 1];
                z2 = mag[idx + img_size.x + 1];
                mag2 = (m00 - z1)*xprep + (z2 - z1)*yprep;
            }
            else
            {
                z1 = mag[idx - img_size.x];
                z2 = mag[idx - img_size.x - 1];
                mag1 = (z1 - z2)*xprep + (z1 - m00)*yprep;

                z1 = mag[idx + img_size.x];
                z2 = mag[idx + img_size.x + 1];
                mag2 = (z1 - z2)*xprep + (z1 - m00)*yprep;
            }
        }
        else
        {
            if (gx >= -gy)
            {
                z1 = mag[idx - 1];
                z2 = mag[idx + img_size.x - 1];
                mag1 = (m00 - z1)*xprep + (z1 - z2)*yprep;

                z1 = mag[idx + 1];
                z2 = mag[idx - img_size.x + 1];
                mag2 = (m00 - z1)*xprep + (z1 - z2)*yprep;
            }
            else
            {
                z1 = mag[idx + img_size.x];
                z2 = mag[idx + img_size.x - 1];
                mag1 = (z1 - z2)*xprep + (m00 - z1)*yprep;

                z1 = mag[idx - img_size.x];
                z2 = mag[idx - img_size.x + 1];
                mag2 = (z1 - z2)*xprep + (m00 - z1)*yprep;
            }
        }
    }
    else
    {
        if (gy >= 0)
        {
            if (-gx >= gy)
            {
                z1 = mag[idx + 1];
                z2 = mag[idx - img_size.x + 1];
                mag1 = (z1 - m00)*xprep + (z2 - z1)*yprep;

                z1 = mag[idx - 1];
                z2 = mag[idx + img_size.x - 1];
                mag2 = (z1 - m00)*xprep + (z2 - z1)*yprep;
            }
            else
            {
                z1 = mag[idx - img_size.x];
                z2 = mag[idx - img_size.x + 1];
                mag1 = (z2 - z1)*xprep + (z1 - m00)*yprep;

                z1 = mag[idx + img_size.x];
                z2 = mag[idx + img_size.x - 1];
                mag2 = (z2 - z1)*xprep + (z1 - m00)*yprep;
            }
        }
        else
        {
            if (-gx > -gy)
            {
                z1 = mag[idx + 1];
                z2 = mag[idx + img_size.x + 1];
                mag1 = (z1 - m00)*xprep + (z1 - z2)*yprep;

                z1 = mag[idx - 1];
                z2 = mag[idx - img_size.x - 1];
                mag2 = (z1 - m00)*xprep + (z1 - z2)*yprep;
            }
            else
            {
                z1 = mag[idx + img_size.x];
                z2 = mag[idx + img_size.x + 1];
                mag1 = (z2 - z1)*xprep + (m00 - z1)*yprep;

                z1 = mag[idx - img_size.x];
                z2 = mag[idx - img_size.x - 1];
                mag2 = (z2 - z1)*xprep + (m00 - z1)*yprep;
            }
        }
    }

    if (mag1 > 0 || mag2 >= 0)
        nms_mag[idx] = 0;
    else
        nms_mag[idx] = (uchar)dmin(dmax(m00, 0), 255);
}

void Magnitude::derrivativeXYGpu()
{
    uchar *gray_ptr = gray_img->getGpuData();
    int *dx_ptr = delta_x->getGpuData();
    int *dy_ptr = delta_y->getGpuData();
    int *mag_ptr = mag->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x),
        (int)ceil((float)img_size.y / (float)blockSize.y));

    derrivativeXYDevice << <gridSize, blockSize >> >
        (gray_ptr, dx_ptr, dy_ptr, mag_ptr, img_size);
}

void Magnitude::nonMaxSuppGpu()
{
    int *dx_ptr = delta_x->getGpuData();
    int *dy_ptr = delta_y->getGpuData();
    int *mag_ptr = mag->getGpuData();
    uchar *nms_ptr = nms_mag->getGpuData();

    dim3 blockSize(HFS_BLOCK_DIM, HFS_BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x),
        (int)ceil((float)img_size.y / (float)blockSize.y));

    nonMaxSuppDevice << <gridSize, blockSize >> >
        (nms_ptr, dx_ptr, dy_ptr, mag_ptr, img_size);
}

void Magnitude::processImgGpu(const Mat& bgr3u, Mat& mag1u)
{
    Mat gray, blur1u;
    cvtColor(bgr3u, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur1u, Size(7, 7), 1, 1);

    img_size.x = bgr3u.cols;
    img_size.y = bgr3u.rows;

    loadImage(blur1u, gray_img);
    gray_img->updateDeviceFromHost();
    derrivativeXYGpu();
    nonMaxSuppGpu();
    mag1u.create(bgr3u.rows, bgr3u.cols, CV_8UC1);
    nms_mag->updateHostFromDevice();
    loadImage(nms_mag, mag1u);
}

}}

// #endif
