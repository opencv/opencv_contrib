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

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
using namespace std;
#define EPS 0.001F

namespace cv
{
namespace optflow
{

class DISOpticalFlowImpl : public DISOpticalFlow
{
  public:
    DISOpticalFlowImpl();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
    void collectGarbage();

  protected: // algorithm parameters
    int finest_scale, coarsest_scale;
    int patch_size;
    int patch_stride;
    int grad_descent_iter;

  public: // getters and setters
    int getFinestScale() const { return finest_scale; }
    void setFinestScale(int val) { finest_scale = val; }
    int getPatchSize() const { return patch_size; }
    void setPatchSize(int val) { patch_size = val; }
    int getPatchStride() const { return patch_stride; }
    void setPatchStride(int val) { patch_stride = val; }
    int getGradientDescentIterations() const { return grad_descent_iter; }
    void setGradientDescentIterations(int val) { grad_descent_iter = val; }

  private:                   // internal buffers
    vector< Mat_<uchar> > I0s; // gaussian pyramid for the current frame
    vector< Mat_<uchar> > I1s; // gaussian pyramid for the next frame

    vector< Mat_<short> > I0xs; // gaussian pyramid for the x gradient of the current frame
    vector< Mat_<short> > I0ys; // gaussian pyramid for the y gradient of the current frame

    vector< Mat_<float> > Ux; // x component of the flow vectors
    vector< Mat_<float> > Uy; // y component of the flow vectors

    Mat_<Vec2f> U; // buffers for the merged flow

    Mat_<float> Sx; // x component of the sparse flow vectors (before densification)
    Mat_<float> Sy; // y component of the sparse flow vectors (before densification)

    // structure tensor components and auxiliary buffers:
    Mat_<float> I0xx_buf; // sum of squares of x gradient values
    Mat_<float> I0yy_buf; // sum of squares of y gradient values
    Mat_<float> I0xy_buf; // sum of x and y gradient products

    Mat_<float> I0xx_buf_aux; // for computing sums using the summed area table
    Mat_<float> I0yy_buf_aux;
    Mat_<float> I0xy_buf_aux;
    ////////////////////////////////////////////////////////////

  private: // private methods
    void prepareBuffers(Mat &I0, Mat &I1);
    void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &I0x, Mat &I0y);
    void patchInverseSearch(Mat &dst_Sx, Mat &dst_Sy, Mat &src_Ux, Mat &src_Uy, Mat &I0, Mat &I0x, Mat &I0y, Mat &I1);
    void densification(Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx, Mat &src_Sy, Mat &I0, Mat &I1);
};

DISOpticalFlowImpl::DISOpticalFlowImpl()
{
    finest_scale = 3;
    patch_size = 9;
    patch_stride = 4;
    grad_descent_iter = 12;
}

void DISOpticalFlowImpl::prepareBuffers(Mat &I0, Mat &I1)
{
    I0s.resize(coarsest_scale + 1);
    I1s.resize(coarsest_scale + 1);
    I0xs.resize(coarsest_scale + 1);
    I0ys.resize(coarsest_scale + 1);
    Ux.resize(coarsest_scale + 1);
    Uy.resize(coarsest_scale + 1);
    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            Sx.create(cur_rows / patch_stride, cur_cols / patch_stride);
            Sy.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            U.create(cur_rows, cur_cols);
        }
        else if (i > finest_scale)
        {
            cur_rows = I0s[i - 1].rows / 2;
            cur_cols = I0s[i - 1].cols / 2;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0s[i - 1], I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1s[i - 1], I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);
        }

        fraction *= 2;

        if (i >= finest_scale)
        {
            I0xs[i].create(cur_rows, cur_cols);
            I0ys[i].create(cur_rows, cur_cols);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);
            Ux[i].create(cur_rows, cur_cols);
            Uy[i].create(cur_rows, cur_cols);
        }
    }
}

void DISOpticalFlowImpl::precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &I0x, Mat &I0y)
{
    short *I0x_ptr = I0x.ptr<short>();
    short *I0y_ptr = I0y.ptr<short>();

    float *I0xx_ptr = dst_I0xx.ptr<float>();
    float *I0yy_ptr = dst_I0yy.ptr<float>();
    float *I0xy_ptr = dst_I0xy.ptr<float>();

    float *I0xx_aux_ptr = I0xx_buf_aux.ptr<float>();
    float *I0yy_aux_ptr = I0yy_buf_aux.ptr<float>();
    float *I0xy_aux_ptr = I0xy_buf_aux.ptr<float>();

    int w = I0x.cols;
    int h = I0x.rows;
    int psz2 = patch_size / 2;
    int psz = 2 * psz2;
    // width of the sparse OF fields:
    int ws = 1 + (w - patch_size) / patch_stride;

    // separable box filter for computing patch sums on a sparse
    // grid (determined by patch_stride)
    for (int i = 0; i < h; i++)
    {
        float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f;
        short *x_row = I0x_ptr + i * w, *y_row = I0y_ptr + i * w;
        for (int j = 0; j < patch_size; j++)
        {
            sum_xx += x_row[j] * x_row[j];
            sum_yy += y_row[j] * y_row[j];
            sum_xy += x_row[j] * y_row[j];
        }
        I0xx_aux_ptr[i * ws] = sum_xx;
        I0yy_aux_ptr[i * ws] = sum_yy;
        I0xy_aux_ptr[i * ws] = sum_xy;
        int js = 1;
        for (int j = patch_size; j < w; j++)
        {
            sum_xx += (x_row[j] * x_row[j] - x_row[j - patch_size] * x_row[j - patch_size]);
            sum_yy += (y_row[j] * y_row[j] - y_row[j - patch_size] * y_row[j - patch_size]);
            sum_xy += (x_row[j] * y_row[j] - x_row[j - patch_size] * y_row[j - patch_size]);
            if ((j - psz) % patch_stride == 0)
            {
                I0xx_aux_ptr[i * ws + js] = sum_xx;
                I0yy_aux_ptr[i * ws + js] = sum_yy;
                I0xy_aux_ptr[i * ws + js] = sum_xy;
                js++;
            }
        }
    }

    AutoBuffer<float> sum_xx_buf(ws), sum_yy_buf(ws), sum_xy_buf(ws);
    float *sum_xx = (float *)sum_xx_buf;
    float *sum_yy = (float *)sum_yy_buf;
    float *sum_xy = (float *)sum_xy_buf;
    for (int j = 0; j < ws; j++)
    {
        sum_xx[j] = 0.0f;
        sum_yy[j] = 0.0f;
        sum_xy[j] = 0.0f;
    }

    for (int i = 0; i < patch_size; i++)
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += I0xx_aux_ptr[i * ws + j];
            sum_yy[j] += I0yy_aux_ptr[i * ws + j];
            sum_xy[j] += I0xy_aux_ptr[i * ws + j];
        }
    for (int j = 0; j < ws; j++)
    {
        I0xx_ptr[j] = sum_xx[j];
        I0yy_ptr[j] = sum_yy[j];
        I0xy_ptr[j] = sum_xy[j];
    }
    int is = 1;
    for (int i = patch_size; i < h; i++)
    {
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
            sum_yy[j] += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
            sum_xy[j] += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
        }
        if ((i - psz) % patch_stride == 0)
        {
            for (int j = 0; j < ws; j++)
            {
                I0xx_ptr[is * ws + j] = sum_xx[j];
                I0yy_ptr[is * ws + j] = sum_yy[j];
                I0xy_ptr[is * ws + j] = sum_xy[j];
            }
            is++;
        }
    }
}

void DISOpticalFlowImpl::patchInverseSearch(Mat &dst_Sx, Mat &dst_Sy, Mat &src_Ux, Mat &src_Uy, Mat &I0, Mat &I0x,
                                            Mat &I0y, Mat &I1)
{
    float *Ux_ptr = src_Ux.ptr<float>();
    float *Uy_ptr = src_Uy.ptr<float>();
    float *Sx_ptr = dst_Sx.ptr<float>();
    float *Sy_ptr = dst_Sy.ptr<float>();
    uchar *I0_ptr = I0.ptr<uchar>();
    uchar *I1_ptr = I1.ptr<uchar>();
    short *I0x_ptr = I0x.ptr<short>();
    short *I0y_ptr = I0y.ptr<short>();
    int w = I0.cols;
    int h = I1.rows;
    int psz2 = patch_size / 2;
    // width and height of the sparse OF fields:
    int ws = 1 + (w - patch_size) / patch_stride;
    int hs = 1 + (h - patch_size) / patch_stride;

    precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0x, I0y);
    float *xx_ptr = I0xx_buf.ptr<float>();
    float *yy_ptr = I0yy_buf.ptr<float>();
    float *xy_ptr = I0xy_buf.ptr<float>();

    // perform a fixed number of gradient descent iterations for each patch:
    int i = psz2;
    for (int is = 0; is < hs; is++)
    {
        int j = psz2;
        for (int js = 0; js < ws; js++)
        {
            float cur_Ux = Ux_ptr[i * w + j];
            float cur_Uy = Uy_ptr[i * w + j];
            float detH = xx_ptr[is * ws + js] * yy_ptr[is * ws + js] - xy_ptr[is * ws + js] * xy_ptr[is * ws + js];
            if (abs(detH) < EPS)
                detH = EPS;
            float invH11 = yy_ptr[is * ws + js] / detH;
            float invH12 = -xy_ptr[is * ws + js] / detH;
            float invH22 = xx_ptr[is * ws + js] / detH;
            float prev_sum_diff = 100000000.0f;
            for (int t = 0; t < grad_descent_iter; t++)
            {
                float dUx = 0, dUy = 0;
                float diff = 0;
                float sum_diff = 0.0f;
                for (int pos_y = i - psz2; pos_y <= i + psz2; pos_y++)
                    for (int pos_x = j - psz2; pos_x <= j + psz2; pos_x++)
                    {
                        float pos_x_shifted = min(max(pos_x + cur_Ux, 0.0f), w - 1.0f - EPS);
                        float pos_y_shifted = min(max(pos_y + cur_Uy, 0.0f), h - 1.0f - EPS);
                        int pos_x_lower = (int)pos_x_shifted;
                        int pos_x_upper = pos_x_lower + 1;
                        int pos_y_lower = (int)pos_y_shifted;
                        int pos_y_upper = pos_y_lower + 1;
                        diff = (pos_x_shifted - pos_x_lower) * (pos_y_shifted - pos_y_lower) *
                                 I1_ptr[pos_y_upper * w + pos_x_upper] +
                               (pos_x_upper - pos_x_shifted) * (pos_y_shifted - pos_y_lower) *
                                 I1_ptr[pos_y_upper * w + pos_x_lower] +
                               (pos_x_shifted - pos_x_lower) * (pos_y_upper - pos_y_shifted) *
                                 I1_ptr[pos_y_lower * w + pos_x_upper] +
                               (pos_x_upper - pos_x_shifted) * (pos_y_upper - pos_y_shifted) *
                                 I1_ptr[pos_y_lower * w + pos_x_lower] -
                               I0_ptr[pos_y * w + pos_x];
                        sum_diff += diff * diff;
                        dUx += I0x_ptr[pos_y * w + pos_x] * diff;
                        dUy += I0y_ptr[pos_y * w + pos_x] * diff;
                    }
                cur_Ux -= invH11 * dUx + invH12 * dUy;
                cur_Uy -= invH12 * dUx + invH22 * dUy;
                if (sum_diff > prev_sum_diff)
                    break;
                prev_sum_diff = sum_diff;
            }
            if (norm(Vec2f(cur_Ux - Ux_ptr[i * w + j], cur_Uy - Uy_ptr[i * w + j])) <= patch_size)
            {
                Sx_ptr[is * ws + js] = cur_Ux;
                Sy_ptr[is * ws + js] = cur_Uy;
            }
            else
            {
                Sx_ptr[is * ws + js] = Ux_ptr[i * w + j];
                Sy_ptr[is * ws + js] = Uy_ptr[i * w + j];
            }
            j += patch_stride;
        }
        i += patch_stride;
    }
}

void DISOpticalFlowImpl::densification(Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx, Mat &src_Sy, Mat &I0, Mat &I1)
{
    float *Ux_ptr = dst_Ux.ptr<float>();
    float *Uy_ptr = dst_Uy.ptr<float>();
    float *Sx_ptr = src_Sx.ptr<float>();
    float *Sy_ptr = src_Sy.ptr<float>();
    uchar *I0_ptr = I0.ptr<uchar>();
    uchar *I1_ptr = I1.ptr<uchar>();
    int w = I0.cols;
    int h = I0.rows;
    int psz2 = patch_size / 2;
    // width of the sparse OF fields:
    int ws = 1 + (w - patch_size) / patch_stride;

    int start_x;
    int start_y;

    start_y = psz2;
    for (int i = 0; i < h; i++)
    {
        if (i - psz2 > start_y && start_y + patch_stride < h - psz2)
            start_y += patch_stride;
        start_x = psz2;
        for (int j = 0; j < w; j++)
        {
            float coef, sum_coef = 0.0f;
            float sum_Ux = 0.0f;
            float sum_Uy = 0.0f;

            if (j - psz2 > start_x && start_x + patch_stride < w - psz2)
                start_x += patch_stride;

            for (int pos_y = start_y; pos_y <= min(i + psz2, h - psz2 - 1); pos_y += patch_stride)
                for (int pos_x = start_x; pos_x <= min(j + psz2, w - psz2 - 1); pos_x += patch_stride)
                {
                    float diff;
                    int is = (pos_y - psz2) / patch_stride;
                    int js = (pos_x - psz2) / patch_stride;
                    float j_shifted = min(max(j + Sx_ptr[is * ws + js], 0.0f), w - 1.0f - EPS);
                    float i_shifted = min(max(i + Sy_ptr[is * ws + js], 0.0f), h - 1.0f - EPS);
                    int j_lower = (int)j_shifted;
                    int j_upper = j_lower + 1;
                    int i_lower = (int)i_shifted;
                    int i_upper = i_lower + 1;
                    diff = (j_shifted - j_lower) * (i_shifted - i_lower) * I1_ptr[i_upper * w + j_upper] +
                           (j_upper - j_shifted) * (i_shifted - i_lower) * I1_ptr[i_upper * w + j_lower] +
                           (j_shifted - j_lower) * (i_upper - i_shifted) * I1_ptr[i_lower * w + j_upper] +
                           (j_upper - j_shifted) * (i_upper - i_shifted) * I1_ptr[i_lower * w + j_lower] -
                           I0_ptr[i * w + j];
                    coef = 1 / max(1.0f, diff * diff);
                    sum_Ux += coef * Sx_ptr[is * ws + js];
                    sum_Uy += coef * Sy_ptr[is * ws + js];
                    sum_coef += coef;
                }
            Ux_ptr[i * w + j] = sum_Ux / sum_coef;
            Uy_ptr[i * w + j] = sum_Uy / sum_coef;
        }
    }
}

void DISOpticalFlowImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));

    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    flow.create(I1Mat.size(), CV_32FC2);
    Mat &flowMat = flow.getMatRef();
    coarsest_scale = (int)(log((2 * I0Mat.cols) / (4.0 * patch_size))/log(2.0) + 0.5) - 1;

    prepareBuffers(I0Mat, I1Mat);
    Ux[coarsest_scale].setTo(0.0f);
    Uy[coarsest_scale].setTo(0.0f);

    for (int i = coarsest_scale; i >= finest_scale; i--)
    {
        patchInverseSearch(Sx, Sy, Ux[i], Uy[i], I0s[i], I0xs[i], I0ys[i], I1s[i]);
        densification(Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]);
        // TODO: variational refinement step

        if (i > finest_scale)
        {
            resize(Ux[i], Ux[i - 1], Ux[i - 1].size());
            resize(Uy[i], Uy[i - 1], Uy[i - 1].size());
            Ux[i - 1] *= 2;
            Uy[i - 1] *= 2;
        }
    }
    Mat uxy[] = {Ux[finest_scale], Uy[finest_scale]};
    merge(uxy, 2, U);
    resize(U, flowMat, flowMat.size());
    flowMat *= pow(2, finest_scale);
}

void DISOpticalFlowImpl::collectGarbage()
{
    I0s.clear();
    I1s.clear();
    I0xs.clear();
    I0ys.clear();
    Ux.clear();
    Uy.clear();
    U.release();
    Sx.release();
    Sy.release();
    I0xx_buf.release();
    I0yy_buf.release();
    I0xy_buf.release();
    I0xx_buf_aux.release();
    I0yy_buf_aux.release();
    I0xy_buf_aux.release();
}

Ptr<DISOpticalFlow> createOptFlow_DIS() { return makePtr<DISOpticalFlowImpl>(); }
}
}
