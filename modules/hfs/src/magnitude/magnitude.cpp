// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "../precomp.hpp"
#include "magnitude.hpp"

namespace cv { namespace hfs {

Magnitude::Magnitude(int height, int width)
{
    Vector2i size(height, width);
    delta_x = Ptr<IntImage>(new IntImage(size));
    delta_y = Ptr<IntImage>(new IntImage(size));
    mag = Ptr<IntImage>(new IntImage(size));
    gray_img = Ptr<UCharImage>(new UCharImage(size));
    nms_mag = Ptr<UCharImage>(new UCharImage(size));
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

void Magnitude::derrivativeXYCpu()
{
    uchar *gray_ptr = gray_img->getCpuData();
    int *dx_ptr = delta_x->getCpuData();
    int *dy_ptr = delta_y->getCpuData();
    int *mag_ptr = mag->getCpuData();

    for (int y = 0; y < img_size.y; ++y) {
        for (int x = 0; x < img_size.x; ++x) {
            int idx = y * img_size.x + x;
            if (x == 0)
                dx_ptr[idx] = gray_ptr[idx + 1] - gray_ptr[idx];
            else if (x == img_size.x - 1)
                dx_ptr[idx] = gray_ptr[idx] - gray_ptr[idx - 1];
            else
                dx_ptr[idx] = gray_ptr[idx + 1] - gray_ptr[idx - 1];

            if (y == 0)
                dy_ptr[idx] = gray_ptr[idx + img_size.x] - gray_ptr[idx];
            else if (y == img_size.y - 1)
                dy_ptr[idx] = gray_ptr[idx] - gray_ptr[idx - img_size.x];
            else
                dy_ptr[idx] = gray_ptr[idx + img_size.x] - gray_ptr[idx - img_size.x];

            mag_ptr[idx] = (int)(0.5 + sqrt((double)(dx_ptr[idx] * dx_ptr[idx] + dy_ptr[idx] * dy_ptr[idx])));

        }
    }
}

void Magnitude::nonMaxSuppCpu()
{
    int *dx_ptr = delta_x->getCpuData();
    int *dy_ptr = delta_y->getCpuData();
    int *mag_ptr = mag->getCpuData();
    uchar *nms_ptr = nms_mag->getCpuData();

    for (int y = 0; y < img_size.y; ++y) {
        for (int x = 0; x < img_size.x; ++x) {
            int idx = y*img_size.x + x;
            if (x == 0 || x == img_size.x - 1 || y == 0 || y == img_size.y - 1) {
                nms_ptr[idx] = 0;
                continue;
            }
            int m00, gx, gy, z1, z2;
            double mag1, mag2, xprep, yprep;

            m00 = mag_ptr[idx];
            if (m00 == 0) {
                nms_ptr[idx] = 0;
                continue;
            }
            else {
                xprep = -(gx = dx_ptr[idx]) / ((double)m00);
                yprep = (gy = dy_ptr[idx]) / ((double)m00);
            }

            if (gx >= 0) {
                if (gy >= 0) {
                    if (gx >= gy) {
                        z1 = mag_ptr[idx - 1];
                        z2 = mag_ptr[idx - img_size.x - 1];
                        mag1 = (m00 - z1)*xprep + (z2 - z1)*yprep;

                        z1 = mag_ptr[idx + 1];
                        z2 = mag_ptr[idx + img_size.x + 1];
                        mag2 = (m00 - z1)*xprep + (z2 - z1)*yprep;
                    }
                    else {
                        z1 = mag_ptr[idx - img_size.x];
                        z2 = mag_ptr[idx - img_size.x - 1];
                        mag1 = (z1 - z2)*xprep + (z1 - m00)*yprep;

                        z1 = mag_ptr[idx + img_size.x];
                        z2 = mag_ptr[idx + img_size.x + 1];
                        mag2 = (z1 - z2)*xprep + (z1 - m00)*yprep;
                    }
                }
                else {
                    if (gx >= -gy) {
                        z1 = mag_ptr[idx - 1];
                        z2 = mag_ptr[idx + img_size.x - 1];
                        mag1 = (m00 - z1)*xprep + (z1 - z2)*yprep;

                        z1 = mag_ptr[idx + 1];
                        z2 = mag_ptr[idx - img_size.x + 1];
                        mag2 = (m00 - z1)*xprep + (z1 - z2)*yprep;
                    }
                    else {
                        z1 = mag_ptr[idx + img_size.x];
                        z2 = mag_ptr[idx + img_size.x - 1];
                        mag1 = (z1 - z2)*xprep + (m00 - z1)*yprep;

                        z1 = mag_ptr[idx - img_size.x];
                        z2 = mag_ptr[idx - img_size.x + 1];
                        mag2 = (z1 - z2)*xprep + (m00 - z1)*yprep;
                    }
                }
            }
            else {
                if (gy >= 0) {
                    if (-gx >= gy) {
                        z1 = mag_ptr[idx + 1];
                        z2 = mag_ptr[idx - img_size.x + 1];
                        mag1 = (z1 - m00)*xprep + (z2 - z1)*yprep;

                        z1 = mag_ptr[idx - 1];
                        z2 = mag_ptr[idx + img_size.x - 1];
                        mag2 = (z1 - m00)*xprep + (z2 - z1)*yprep;
                    }
                    else {
                        z1 = mag_ptr[idx - img_size.x];
                        z2 = mag_ptr[idx - img_size.x + 1];
                        mag1 = (z2 - z1)*xprep + (z1 - m00)*yprep;

                        z1 = mag_ptr[idx + img_size.x];
                        z2 = mag_ptr[idx + img_size.x - 1];
                        mag2 = (z2 - z1)*xprep + (z1 - m00)*yprep;
                    }
                }
                else {
                    if (-gx > -gy) {
                        z1 = mag_ptr[idx + 1];
                        z2 = mag_ptr[idx + img_size.x + 1];
                        mag1 = (z1 - m00)*xprep + (z1 - z2)*yprep;

                        z1 = mag_ptr[idx - 1];
                        z2 = mag_ptr[idx - img_size.x - 1];
                        mag2 = (z1 - m00)*xprep + (z1 - z2)*yprep;
                    }
                    else {
                        z1 = mag_ptr[idx + img_size.x];
                        z2 = mag_ptr[idx + img_size.x + 1];
                        mag1 = (z2 - z1)*xprep + (m00 - z1)*yprep;

                        z1 = mag_ptr[idx - img_size.x];
                        z2 = mag_ptr[idx - img_size.x - 1];
                        mag2 = (z2 - z1)*xprep + (m00 - z1)*yprep;
                    }
                }
            }

            if (mag1 > 0 || mag2 >= 0)
                nms_ptr[idx] = 0;
            else
                nms_ptr[idx] = (uchar)min(max(m00, 0), 255);
        }
    }
}

void Magnitude::processImgCpu(const Mat &bgr3u, Mat &mag1u)
{
    Mat gray, blur1u;
    cvtColor(bgr3u, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blur1u, Size(7, 7), 1, 1);

    img_size.x = bgr3u.cols;
    img_size.y = bgr3u.rows;

    loadImage(blur1u, gray_img);
    derrivativeXYCpu();
    nonMaxSuppCpu();
    mag1u.create(bgr3u.rows, bgr3u.cols, CV_8UC1);
    loadImage(nms_mag, mag1u);
}

}}
