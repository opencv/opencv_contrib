/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "../magnitude.hpp"


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

Magnitude::Magnitude(int height, int width)
{
    Vector2i size(height, width);
    delta_x = Ptr<IntImage>(new IntImage(size, true, true));
    delta_y = Ptr<IntImage>(new IntImage(size, true, true));
    mag = Ptr<IntImage>(new IntImage(size, true, true));
    gray_img = Ptr<UCharImage>(new UCharImage(size, true, true));
    nms_mag = Ptr<UCharImage>(new UCharImage(size, true, true));
    img_size = Vector2i(height, width);
}

Magnitude::~Magnitude()
{
}

void Magnitude::loadImage(const Mat& inimg, Ptr<UCharImage> outimg)
{
    const int _h = inimg.rows, _w = inimg.cols;
    uchar* outimg_ptr = outimg->getCpuData();
    for (int y = 0; y < _h; y++)
    {
        const uchar *ptr = inimg.ptr<uchar>(y);
        for (int x = 0; x < _w; x++)
        {
            int idx = x + y * _w;
            outimg_ptr[idx] = ptr[x];
        }
    }
}

void Magnitude::loadImage(const Ptr<UCharImage> inimg, Mat& outimg)
{
    const int _h = outimg.rows, _w = outimg.cols;
    const uchar* inimg_ptr = inimg->getCpuData();
    for (int y = 0; y < _h; y++)
    {
        uchar *ptr = outimg.ptr<uchar>(y);
        for (int x = 0; x < _w; x++)
        {
            int idx = x + y * outimg.cols;
            ptr[x] = inimg_ptr[idx];
        }
    }
}

void Magnitude::derrivativeXY()
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

void Magnitude::nonMaxSupp()
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

void Magnitude::processImg(const Mat& bgr3u, Mat& mag1u)
{
    Mat gray, blur1u;
    cvtColor(bgr3u, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur1u, Size(7, 7), 1, 1);

    img_size.x = bgr3u.cols;
    img_size.y = bgr3u.rows;

    loadImage(blur1u, gray_img);
    gray_img->updateDeviceFromHost();
    derrivativeXY();
    nonMaxSupp();
    mag1u.create(bgr3u.rows, bgr3u.cols, CV_8UC1);
    nms_mag->updateHostFromDevice();
    loadImage(nms_mag, mag1u);
}

}}