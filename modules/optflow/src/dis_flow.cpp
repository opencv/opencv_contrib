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

#include "opencv2/core/hal/intrin.hpp"
#include "precomp.hpp"
using namespace std;
#define EPS 0.001F
#define INF 1E+10F

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

  protected: //!< algorithm parameters
    int finest_scale, coarsest_scale;
    int patch_size;
    int patch_stride;
    int grad_descent_iter;
    int variational_refinement_iter;
    bool use_mean_normalization;
    bool use_spatial_propagation;

  protected: //!< some auxiliary variables
    int border_size;
    int w, h;   //!< flow buffer width and height on the current scale
    int ws, hs; //!< sparse flow buffer width and height on the current scale

  public:
    int getFinestScale() const { return finest_scale; }
    void setFinestScale(int val) { finest_scale = val; }
    int getPatchSize() const { return patch_size; }
    void setPatchSize(int val) { patch_size = val; }
    int getPatchStride() const { return patch_stride; }
    void setPatchStride(int val) { patch_stride = val; }
    int getGradientDescentIterations() const { return grad_descent_iter; }
    void setGradientDescentIterations(int val) { grad_descent_iter = val; }
    int getVariationalRefinementIterations() const { return variational_refinement_iter; }
    void setVariationalRefinementIterations(int val) { variational_refinement_iter = val; }
    bool getUseMeanNormalization() const { return use_mean_normalization; }
    void setUseMeanNormalization(bool val) { use_mean_normalization = val; }
    bool getUseSpatialPropagation() const { return use_spatial_propagation; }
    void setUseSpatialPropagation(bool val) { use_spatial_propagation = val; }

  protected:                      //!< internal buffers
    vector<Mat_<uchar> > I0s;     //!< Gaussian pyramid for the current frame
    vector<Mat_<uchar> > I1s;     //!< Gaussian pyramid for the next frame
    vector<Mat_<uchar> > I1s_ext; //!< I1s with borders

    vector<Mat_<short> > I0xs; //!< Gaussian pyramid for the x gradient of the current frame
    vector<Mat_<short> > I0ys; //!< Gaussian pyramid for the y gradient of the current frame

    vector<Mat_<float> > Ux; //!< x component of the flow vectors
    vector<Mat_<float> > Uy; //!< y component of the flow vectors

    Mat_<Vec2f> U; //!< a buffer for the merged flow

    Mat_<float> Sx; //!< intermediate sparse flow representation (x component)
    Mat_<float> Sy; //!< intermediate sparse flow representation (y component)

    /* Structure tensor components: */
    Mat_<float> I0xx_buf; //!< sum of squares of x gradient values
    Mat_<float> I0yy_buf; //!< sum of squares of y gradient values
    Mat_<float> I0xy_buf; //!< sum of x and y gradient products

    /* Extra buffers that are useful if patch mean-normalization is used: */
    Mat_<float> I0x_buf; //!< sum of of x gradient values
    Mat_<float> I0y_buf; //!< sum of of y gradient values

    /* Auxiliary buffers used in structure tensor computation: */
    Mat_<float> I0xx_buf_aux;
    Mat_<float> I0yy_buf_aux;
    Mat_<float> I0xy_buf_aux;
    Mat_<float> I0x_buf_aux;
    Mat_<float> I0y_buf_aux;

    vector<Ptr<VariationalRefinement> > variational_refinement_processors;

  private: //!< private methods and parallel sections
    void prepareBuffers(Mat &I0, Mat &I1);
    void precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x, Mat &dst_I0y, Mat &I0x,
                                   Mat &I0y);

    struct PatchInverseSearch_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        int nstripes, stripe_sz;
        int hs;
        Mat *Sx, *Sy, *Ux, *Uy, *I0, *I1, *I0x, *I0y;
        int num_iter;

        PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                   Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1, Mat &_I0x, Mat &_I0y, int _num_iter);
        void operator()(const Range &range) const;
    };

    struct Densification_ParBody : public ParallelLoopBody
    {
        DISOpticalFlowImpl *dis;
        int nstripes, stripe_sz;
        int h;
        Mat *Ux, *Uy, *Sx, *Sy, *I0, *I1;

        Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h, Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx,
                              Mat &src_Sy, Mat &_I0, Mat &_I1);
        void operator()(const Range &range) const;
    };
};

DISOpticalFlowImpl::DISOpticalFlowImpl()
{
    finest_scale = 2;
    patch_size = 8;
    patch_stride = 4;
    grad_descent_iter = 16;
    variational_refinement_iter = 5;
    border_size = 16;
    use_mean_normalization = true;
    use_spatial_propagation = true;

    /* Use separate variational refinement instances for different scales to avoid repeated memory allocation: */
    int max_possible_scales = 10;
    for (int i = 0; i < max_possible_scales; i++)
        variational_refinement_processors.push_back(createVariationalFlowRefinement());
}

void DISOpticalFlowImpl::prepareBuffers(Mat &I0, Mat &I1)
{
    I0s.resize(coarsest_scale + 1);
    I1s.resize(coarsest_scale + 1);
    I1s_ext.resize(coarsest_scale + 1);
    I0xs.resize(coarsest_scale + 1);
    I0ys.resize(coarsest_scale + 1);
    Ux.resize(coarsest_scale + 1);
    Uy.resize(coarsest_scale + 1);

    int fraction = 1;
    int cur_rows = 0, cur_cols = 0;

    for (int i = 0; i <= coarsest_scale; i++)
    {
        /* Avoid initializing the pyramid levels above the finest scale, as they won't be used anyway */
        if (i == finest_scale)
        {
            cur_rows = I0.rows / fraction;
            cur_cols = I0.cols / fraction;
            I0s[i].create(cur_rows, cur_cols);
            resize(I0, I0s[i], I0s[i].size(), 0.0, 0.0, INTER_AREA);
            I1s[i].create(cur_rows, cur_cols);
            resize(I1, I1s[i], I1s[i].size(), 0.0, 0.0, INTER_AREA);

            /* These buffers are reused in each scale so we initialize them once on the finest scale: */
            Sx.create(cur_rows / patch_stride, cur_cols / patch_stride);
            Sy.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xx_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0yy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0xy_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0x_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);
            I0y_buf.create(cur_rows / patch_stride, cur_cols / patch_stride);

            I0xx_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0yy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0xy_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0x_buf_aux.create(cur_rows, cur_cols / patch_stride);
            I0y_buf_aux.create(cur_rows, cur_cols / patch_stride);

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
            I1s_ext[i].create(cur_rows + 2 * border_size, cur_cols + 2 * border_size);
            copyMakeBorder(I1s[i], I1s_ext[i], border_size, border_size, border_size, border_size, BORDER_REPLICATE);
            I0xs[i].create(cur_rows, cur_cols);
            I0ys[i].create(cur_rows, cur_cols);
            spatialGradient(I0s[i], I0xs[i], I0ys[i]);
            Ux[i].create(cur_rows, cur_cols);
            Uy[i].create(cur_rows, cur_cols);
            variational_refinement_processors[i]->setAlpha(20.0f);
            variational_refinement_processors[i]->setDelta(5.0f);
            variational_refinement_processors[i]->setGamma(10.0f);
            variational_refinement_processors[i]->setSorIterations(5);
            variational_refinement_processors[i]->setFixedPointIterations(variational_refinement_iter);
        }
    }
}

/* This function computes the structure tensor elements (local sums of I0x^2, I0x*I0y and I0y^2).
 * A simple box filter is not used instead because we need to compute these sums on a sparse grid
 * and store them densely in the output buffers.
 */
void DISOpticalFlowImpl::precomputeStructureTensor(Mat &dst_I0xx, Mat &dst_I0yy, Mat &dst_I0xy, Mat &dst_I0x,
                                                   Mat &dst_I0y, Mat &I0x, Mat &I0y)
{
    float *I0xx_ptr = dst_I0xx.ptr<float>();
    float *I0yy_ptr = dst_I0yy.ptr<float>();
    float *I0xy_ptr = dst_I0xy.ptr<float>();
    float *I0x_ptr = dst_I0x.ptr<float>();
    float *I0y_ptr = dst_I0y.ptr<float>();

    float *I0xx_aux_ptr = I0xx_buf_aux.ptr<float>();
    float *I0yy_aux_ptr = I0yy_buf_aux.ptr<float>();
    float *I0xy_aux_ptr = I0xy_buf_aux.ptr<float>();
    float *I0x_aux_ptr = I0x_buf_aux.ptr<float>();
    float *I0y_aux_ptr = I0y_buf_aux.ptr<float>();

    /* Separable box filter: horizontal pass */
    for (int i = 0; i < h; i++)
    {
        float sum_xx = 0.0f, sum_yy = 0.0f, sum_xy = 0.0f, sum_x = 0.0f, sum_y = 0.0f;
        short *x_row = I0x.ptr<short>(i);
        short *y_row = I0y.ptr<short>(i);
        for (int j = 0; j < patch_size; j++)
        {
            sum_xx += x_row[j] * x_row[j];
            sum_yy += y_row[j] * y_row[j];
            sum_xy += x_row[j] * y_row[j];
            sum_x += x_row[j];
            sum_y += y_row[j];
        }
        I0xx_aux_ptr[i * ws] = sum_xx;
        I0yy_aux_ptr[i * ws] = sum_yy;
        I0xy_aux_ptr[i * ws] = sum_xy;
        I0x_aux_ptr[i * ws] = sum_x;
        I0y_aux_ptr[i * ws] = sum_y;
        int js = 1;
        for (int j = patch_size; j < w; j++)
        {
            sum_xx += (x_row[j] * x_row[j] - x_row[j - patch_size] * x_row[j - patch_size]);
            sum_yy += (y_row[j] * y_row[j] - y_row[j - patch_size] * y_row[j - patch_size]);
            sum_xy += (x_row[j] * y_row[j] - x_row[j - patch_size] * y_row[j - patch_size]);
            sum_x += (x_row[j] - x_row[j - patch_size]);
            sum_y += (y_row[j] - y_row[j - patch_size]);
            if ((j - patch_size + 1) % patch_stride == 0)
            {
                I0xx_aux_ptr[i * ws + js] = sum_xx;
                I0yy_aux_ptr[i * ws + js] = sum_yy;
                I0xy_aux_ptr[i * ws + js] = sum_xy;
                I0x_aux_ptr[i * ws + js] = sum_x;
                I0y_aux_ptr[i * ws + js] = sum_y;
                js++;
            }
        }
    }

    AutoBuffer<float> sum_xx_buf(ws), sum_yy_buf(ws), sum_xy_buf(ws), sum_x_buf(ws), sum_y_buf(ws);
    float *sum_xx = (float *)sum_xx_buf;
    float *sum_yy = (float *)sum_yy_buf;
    float *sum_xy = (float *)sum_xy_buf;
    float *sum_x = (float *)sum_x_buf;
    float *sum_y = (float *)sum_y_buf;
    for (int j = 0; j < ws; j++)
    {
        sum_xx[j] = 0.0f;
        sum_yy[j] = 0.0f;
        sum_xy[j] = 0.0f;
        sum_x[j] = 0.0f;
        sum_y[j] = 0.0f;
    }

    /* Separable box filter: vertical pass */
    for (int i = 0; i < patch_size; i++)
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += I0xx_aux_ptr[i * ws + j];
            sum_yy[j] += I0yy_aux_ptr[i * ws + j];
            sum_xy[j] += I0xy_aux_ptr[i * ws + j];
            sum_x[j] += I0x_aux_ptr[i * ws + j];
            sum_y[j] += I0y_aux_ptr[i * ws + j];
        }
    for (int j = 0; j < ws; j++)
    {
        I0xx_ptr[j] = sum_xx[j];
        I0yy_ptr[j] = sum_yy[j];
        I0xy_ptr[j] = sum_xy[j];
        I0x_ptr[j] = sum_x[j];
        I0y_ptr[j] = sum_y[j];
    }
    int is = 1;
    for (int i = patch_size; i < h; i++)
    {
        for (int j = 0; j < ws; j++)
        {
            sum_xx[j] += (I0xx_aux_ptr[i * ws + j] - I0xx_aux_ptr[(i - patch_size) * ws + j]);
            sum_yy[j] += (I0yy_aux_ptr[i * ws + j] - I0yy_aux_ptr[(i - patch_size) * ws + j]);
            sum_xy[j] += (I0xy_aux_ptr[i * ws + j] - I0xy_aux_ptr[(i - patch_size) * ws + j]);
            sum_x[j] += (I0x_aux_ptr[i * ws + j] - I0x_aux_ptr[(i - patch_size) * ws + j]);
            sum_y[j] += (I0y_aux_ptr[i * ws + j] - I0y_aux_ptr[(i - patch_size) * ws + j]);
        }
        if ((i - patch_size + 1) % patch_stride == 0)
        {
            for (int j = 0; j < ws; j++)
            {
                I0xx_ptr[is * ws + j] = sum_xx[j];
                I0yy_ptr[is * ws + j] = sum_yy[j];
                I0xy_ptr[is * ws + j] = sum_xy[j];
                I0x_ptr[is * ws + j] = sum_x[j];
                I0y_ptr[is * ws + j] = sum_y[j];
            }
            is++;
        }
    }
}

DISOpticalFlowImpl::PatchInverseSearch_ParBody::PatchInverseSearch_ParBody(DISOpticalFlowImpl &_dis, int _nstripes,
                                                                           int _hs, Mat &dst_Sx, Mat &dst_Sy,
                                                                           Mat &src_Ux, Mat &src_Uy, Mat &_I0, Mat &_I1,
                                                                           Mat &_I0x, Mat &_I0y, int _num_iter)
    : dis(&_dis), nstripes(_nstripes), hs(_hs), Sx(&dst_Sx), Sy(&dst_Sy), Ux(&src_Ux), Uy(&src_Uy), I0(&_I0), I1(&_I1),
      I0x(&_I0x), I0y(&_I0y), num_iter(_num_iter)
{
    stripe_sz = (int)ceil(hs / (double)nstripes);
}

/////////////////////////////////////////////* Patch processing functions */////////////////////////////////////////////

/* Some auxiliary macros */
#define HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION                                                                         \
    v_float32x4 w00v = v_setall_f32(w00);                                                                              \
    v_float32x4 w01v = v_setall_f32(w01);                                                                              \
    v_float32x4 w10v = v_setall_f32(w10);                                                                              \
    v_float32x4 w11v = v_setall_f32(w11);                                                                              \
                                                                                                                       \
    v_uint8x16 I0_row_16, I1_row_16, I1_row_shifted_16, I1_row_next_16, I1_row_next_shifted_16;                        \
    v_uint16x8 I0_row_8, I1_row_8, I1_row_shifted_8, I1_row_next_8, I1_row_next_shifted_8, tmp;                        \
    v_uint32x4 I0_row_4_left, I1_row_4_left, I1_row_shifted_4_left, I1_row_next_4_left, I1_row_next_shifted_4_left;    \
    v_uint32x4 I0_row_4_right, I1_row_4_right, I1_row_shifted_4_right, I1_row_next_4_right,                            \
      I1_row_next_shifted_4_right;                                                                                     \
    v_float32x4 I_diff_left, I_diff_right;                                                                             \
                                                                                                                       \
    /* Preload and expand the first row of I1: */                                                                      \
    I1_row_16 = v_load(I1_ptr);                                                                                        \
    I1_row_shifted_16 = v_extract<1>(I1_row_16, I1_row_16);                                                            \
    v_expand(I1_row_16, I1_row_8, tmp);                                                                                \
    v_expand(I1_row_shifted_16, I1_row_shifted_8, tmp);                                                                \
    v_expand(I1_row_8, I1_row_4_left, I1_row_4_right);                                                                 \
    v_expand(I1_row_shifted_8, I1_row_shifted_4_left, I1_row_shifted_4_right);                                         \
    I1_ptr += I1_stride;

#define HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION                                                                      \
    /* Load the next row of I1: */                                                                                     \
    I1_row_next_16 = v_load(I1_ptr);                                                                                   \
    /* Circular shift left by 1 element: */                                                                            \
    I1_row_next_shifted_16 = v_extract<1>(I1_row_next_16, I1_row_next_16);                                             \
    /* Expand to 8 ushorts (we only need the first 8 values): */                                                       \
    v_expand(I1_row_next_16, I1_row_next_8, tmp);                                                                      \
    v_expand(I1_row_next_shifted_16, I1_row_next_shifted_8, tmp);                                                      \
    /* Separate the left and right halves: */                                                                          \
    v_expand(I1_row_next_8, I1_row_next_4_left, I1_row_next_4_right);                                                  \
    v_expand(I1_row_next_shifted_8, I1_row_next_shifted_4_left, I1_row_next_shifted_4_right);                          \
                                                                                                                       \
    /* Load current row of I0: */                                                                                      \
    I0_row_16 = v_load(I0_ptr);                                                                                        \
    v_expand(I0_row_16, I0_row_8, tmp);                                                                                \
    v_expand(I0_row_8, I0_row_4_left, I0_row_4_right);                                                                 \
                                                                                                                       \
    /* Compute diffs between I0 and bilinearly interpolated I1: */                                                     \
    I_diff_left = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_left)) +                                              \
                  w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_left)) +                                      \
                  w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_left)) +                                         \
                  w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_left)) -                                 \
                  v_cvt_f32(v_reinterpret_as_s32(I0_row_4_left));                                                      \
    I_diff_right = w00v * v_cvt_f32(v_reinterpret_as_s32(I1_row_4_right)) +                                            \
                   w01v * v_cvt_f32(v_reinterpret_as_s32(I1_row_shifted_4_right)) +                                    \
                   w10v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_4_right)) +                                       \
                   w11v * v_cvt_f32(v_reinterpret_as_s32(I1_row_next_shifted_4_right)) -                               \
                   v_cvt_f32(v_reinterpret_as_s32(I0_row_4_right));

#define HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW                                                                     \
    I0_ptr += I0_stride;                                                                                               \
    I1_ptr += I1_stride;                                                                                               \
                                                                                                                       \
    I1_row_4_left = I1_row_next_4_left;                                                                                \
    I1_row_4_right = I1_row_next_4_right;                                                                              \
    I1_row_shifted_4_left = I1_row_next_shifted_4_left;                                                                \
    I1_row_shifted_4_right = I1_row_next_shifted_4_right;

/* This function essentially performs one iteration of gradient descent when finding the most similar patch in I1 for a
 * given one in I0. It assumes that I0_ptr and I1_ptr already point to the corresponding patches and w00, w01, w10, w11
 * are precomputed bilinear interpolation weights. It returns the SSD (sum of squared differences) between these patches
 * and computes the values (dst_dUx, dst_dUy) that are used in the flow vector update. HAL acceleration is implemented
 * only for the default patch size (8x8). Everything is processed in floats as using fixed-point approximations harms
 * the quality significantly.
 */
inline float processPatch(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr, short *I0y_ptr,
                          int I0_stride, int I1_stride, float w00, float w01, float w10, float w11, int patch_sz)
{
    float SSD = 0.0f;
#ifdef CV_SIMD128
    if (patch_sz == 8)
    {
        /* Variables to accumulate the sums */
        v_float32x4 Ux_vec = v_setall_f32(0);
        v_float32x4 Uy_vec = v_setall_f32(0);
        v_float32x4 SSD_vec = v_setall_f32(0);

        v_int16x8 I0x_row, I0y_row;
        v_int32x4 I0x_row_4_left, I0x_row_4_right, I0y_row_4_left, I0y_row_4_right;

        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            I0x_row = v_load(I0x_ptr);
            v_expand(I0x_row, I0x_row_4_left, I0x_row_4_right);
            I0y_row = v_load(I0y_ptr);
            v_expand(I0y_row, I0y_row_4_left, I0y_row_4_right);

            /* Update the sums: */
            Ux_vec += I_diff_left * v_cvt_f32(I0x_row_4_left) + I_diff_right * v_cvt_f32(I0x_row_4_right);
            Uy_vec += I_diff_left * v_cvt_f32(I0y_row_4_left) + I_diff_right * v_cvt_f32(I0y_row_4_right);
            SSD_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;

            I0x_ptr += I0_stride;
            I0y_ptr += I0_stride;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }

        /* Final reduce operations: */
        dst_dUx = v_reduce_sum(Ux_vec);
        dst_dUy = v_reduce_sum(Uy_vec);
        SSD = v_reduce_sum(SSD_vec);
    }
    else
    {
#endif
        dst_dUx = 0.0f;
        dst_dUy = 0.0f;
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                SSD += diff * diff;
                dst_dUx += diff * I0x_ptr[i * I0_stride + j];
                dst_dUy += diff * I0y_ptr[i * I0_stride + j];
            }
#ifdef CV_SIMD128
    }
#endif
    return SSD;
}

/* Same as processPatch, but with patch mean normalization, which improves robustness under changing
 * lighting conditions
 */
inline float processPatchMeanNorm(float &dst_dUx, float &dst_dUy, uchar *I0_ptr, uchar *I1_ptr, short *I0x_ptr,
                                  short *I0y_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                                  float w11, int patch_sz, float x_grad_sum, float y_grad_sum)
{
    float sum_diff = 0.0, sum_diff_sq = 0.0;
    float sum_I0x_mul = 0.0, sum_I0y_mul = 0.0;
    float n = (float)patch_sz * patch_sz;

#ifdef CV_SIMD128
    if (patch_sz == 8)
    {
        /* Variables to accumulate the sums */
        v_float32x4 sum_I0x_mul_vec = v_setall_f32(0);
        v_float32x4 sum_I0y_mul_vec = v_setall_f32(0);
        v_float32x4 sum_diff_vec = v_setall_f32(0);
        v_float32x4 sum_diff_sq_vec = v_setall_f32(0);

        v_int16x8 I0x_row, I0y_row;
        v_int32x4 I0x_row_4_left, I0x_row_4_right, I0y_row_4_left, I0y_row_4_right;

        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            I0x_row = v_load(I0x_ptr);
            v_expand(I0x_row, I0x_row_4_left, I0x_row_4_right);
            I0y_row = v_load(I0y_ptr);
            v_expand(I0y_row, I0y_row_4_left, I0y_row_4_right);

            /* Update the sums: */
            sum_I0x_mul_vec += I_diff_left * v_cvt_f32(I0x_row_4_left) + I_diff_right * v_cvt_f32(I0x_row_4_right);
            sum_I0y_mul_vec += I_diff_left * v_cvt_f32(I0y_row_4_left) + I_diff_right * v_cvt_f32(I0y_row_4_right);
            sum_diff_sq_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            sum_diff_vec += I_diff_left + I_diff_right;

            I0x_ptr += I0_stride;
            I0y_ptr += I0_stride;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }

        /* Final reduce operations: */
        sum_I0x_mul = v_reduce_sum(sum_I0x_mul_vec);
        sum_I0y_mul = v_reduce_sum(sum_I0y_mul_vec);
        sum_diff = v_reduce_sum(sum_diff_vec);
        sum_diff_sq = v_reduce_sum(sum_diff_sq_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                sum_diff += diff;
                sum_diff_sq += diff * diff;

                sum_I0x_mul += diff * I0x_ptr[i * I0_stride + j];
                sum_I0y_mul += diff * I0y_ptr[i * I0_stride + j];
            }
#ifdef CV_SIMD128
    }
#endif
    dst_dUx = sum_I0x_mul - sum_diff * x_grad_sum / n;
    dst_dUy = sum_I0y_mul - sum_diff * y_grad_sum / n;
    return sum_diff_sq - sum_diff * sum_diff / n;
}

/* Similar to processPatch, but compute only the sum of squared differences (SSD) between the patches */
inline float computeSSD(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01, float w10,
                        float w11, int patch_sz)
{
    float SSD = 0.0f;
#ifdef CV_SIMD128
    if (patch_sz == 8)
    {
        v_float32x4 SSD_vec = v_setall_f32(0);
        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            SSD_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }
        SSD = v_reduce_sum(SSD_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];
                SSD += diff * diff;
            }
#ifdef CV_SIMD128
    }
#endif
    return SSD;
}

/* Same as computeSSD, but with patch mean normalization */
inline float computeSSDMeanNorm(uchar *I0_ptr, uchar *I1_ptr, int I0_stride, int I1_stride, float w00, float w01,
                                float w10, float w11, int patch_sz)
{
    float sum_diff = 0.0f, sum_diff_sq = 0.0f;
    float n = (float)patch_sz * patch_sz;
#ifdef CV_SIMD128
    if (patch_sz == 8)
    {
        v_float32x4 sum_diff_vec = v_setall_f32(0);
        v_float32x4 sum_diff_sq_vec = v_setall_f32(0);
        HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION;
        for (int row = 0; row < 8; row++)
        {
            HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION;
            sum_diff_sq_vec += I_diff_left * I_diff_left + I_diff_right * I_diff_right;
            sum_diff_vec += I_diff_left + I_diff_right;
            HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW;
        }
        sum_diff = v_reduce_sum(sum_diff_vec);
        sum_diff_sq = v_reduce_sum(sum_diff_sq_vec);
    }
    else
    {
#endif
        float diff;
        for (int i = 0; i < patch_sz; i++)
            for (int j = 0; j < patch_sz; j++)
            {
                diff = w00 * I1_ptr[i * I1_stride + j] + w01 * I1_ptr[i * I1_stride + j + 1] +
                       w10 * I1_ptr[(i + 1) * I1_stride + j] + w11 * I1_ptr[(i + 1) * I1_stride + j + 1] -
                       I0_ptr[i * I0_stride + j];

                sum_diff += diff;
                sum_diff_sq += diff * diff;
            }
#ifdef CV_SIMD128
    }
#endif
    return sum_diff_sq - sum_diff * sum_diff / n;
}

#undef HAL_INIT_BILINEAR_8x8_PATCH_EXTRACTION
#undef HAL_PROCESS_BILINEAR_8x8_PATCH_EXTRACTION
#undef HAL_BILINEAR_8x8_PATCH_EXTRACTION_NEXT_ROW
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void DISOpticalFlowImpl::PatchInverseSearch_ParBody::operator()(const Range &range) const
{
    // force separate processing of stripes if we are using spatial propagation:
    if(dis->use_spatial_propagation && range.end>range.start+1)
    {
        for(int n=range.start;n<range.end;n++)
            (*this)(Range(n,n+1));
        return;
    }
    int psz = dis->patch_size;
    int psz2 = psz / 2;
    int w_ext = dis->w + 2 * dis->border_size; //!< width of I1_ext
    int bsz = dis->border_size;

    /* Input dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    /* Output sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();
    short *I0x_ptr = I0x->ptr<short>();
    short *I0y_ptr = I0y->ptr<short>();

    /* Precomputed structure tensor */
    float *xx_ptr = dis->I0xx_buf.ptr<float>();
    float *yy_ptr = dis->I0yy_buf.ptr<float>();
    float *xy_ptr = dis->I0xy_buf.ptr<float>();
    /* And extra buffers for mean-normalization: */
    float *x_ptr = dis->I0x_buf.ptr<float>();
    float *y_ptr = dis->I0y_buf.ptr<float>();

    int i, j, dir;
    int start_is, end_is, start_js, end_js;
    int start_i, start_j;
    float i_lower_limit = bsz - psz + 1.0f;
    float i_upper_limit = bsz + dis->h - 1.0f;
    float j_lower_limit = bsz - psz + 1.0f;
    float j_upper_limit = bsz + dis->w - 1.0f;
    float dUx, dUy, i_I1, j_I1, w00, w01, w10, w11, dx, dy;

#define INIT_BILINEAR_WEIGHTS(Ux, Uy)                                                                                  \
    i_I1 = min(max(i + Uy + bsz, i_lower_limit), i_upper_limit);                                                       \
    j_I1 = min(max(j + Ux + bsz, j_lower_limit), j_upper_limit);                                                       \
                                                                                                                       \
    w11 = (i_I1 - floor(i_I1)) * (j_I1 - floor(j_I1));                                                                 \
    w10 = (i_I1 - floor(i_I1)) * (floor(j_I1) + 1 - j_I1);                                                             \
    w01 = (floor(i_I1) + 1 - i_I1) * (j_I1 - floor(j_I1));                                                             \
    w00 = (floor(i_I1) + 1 - i_I1) * (floor(j_I1) + 1 - j_I1);

#define COMPUTE_SSD(dst, Ux, Uy)                                                                                       \
    INIT_BILINEAR_WEIGHTS(Ux, Uy);                                                                                     \
    if (dis->use_mean_normalization)                                                                                   \
        dst = computeSSDMeanNorm(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00,  \
                                 w01, w10, w11, psz);                                                                  \
    else                                                                                                               \
        dst = computeSSD(I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1, dis->w, w_ext, w00, w01,     \
                         w10, w11, psz);

    int num_inner_iter = (int)floor(dis->grad_descent_iter / (float)num_iter);
    for (int iter = 0; iter < num_iter; iter++)
    {
        if (iter % 2 == 0)
        {
            dir = 1;
            start_is = min(range.start * stripe_sz, hs);
            end_is = min(range.end * stripe_sz, hs);
            start_js = 0;
            end_js = dis->ws;
            start_i = start_is * dis->patch_stride;
            start_j = 0;
        }
        else
        {
            dir = -1;
            start_is = min(range.end * stripe_sz, hs) - 1;
            end_is = min(range.start * stripe_sz, hs) - 1;
            start_js = dis->ws - 1;
            end_js = -1;
            start_i = start_is * dis->patch_stride;
            start_j = (dis->ws - 1) * dis->patch_stride;
        }

        i = start_i;
        for (int is = start_is; dir * is < dir * end_is; is += dir)
        {
            j = start_j;
            for (int js = start_js; dir * js < dir * end_js; js += dir)
            {
                if (iter == 0)
                {
                    /* Using result form the previous pyramid level as the very first approximation: */
                    Sx_ptr[is * dis->ws + js] = Ux_ptr[(i + psz2) * dis->w + j + psz2];
                    Sy_ptr[is * dis->ws + js] = Uy_ptr[(i + psz2) * dis->w + j + psz2];
                }

                if (dis->use_spatial_propagation)
                {
                    /* Updating the current Sx_ptr, Sy_ptr to the best candidate: */
                    float min_SSD, cur_SSD;
                    COMPUTE_SSD(min_SSD, Sx_ptr[is * dis->ws + js], Sy_ptr[is * dis->ws + js]);
                    if (dir * js > dir * start_js)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[is * dis->ws + js - dir], Sy_ptr[is * dis->ws + js - dir]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[is * dis->ws + js - dir];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[is * dis->ws + js - dir];
                        }
                    }
                    /* Flow vectors won't actually propagate across different stripes, which is the reason for keeping
                     * the number of stripes constant. It works well enough in practice and doesn't introduce any
                     * visible seams.
                     */
                    if (dir * is > dir * start_is)
                    {
                        COMPUTE_SSD(cur_SSD, Sx_ptr[(is - dir) * dis->ws + js], Sy_ptr[(is - dir) * dis->ws + js]);
                        if (cur_SSD < min_SSD)
                        {
                            min_SSD = cur_SSD;
                            Sx_ptr[is * dis->ws + js] = Sx_ptr[(is - dir) * dis->ws + js];
                            Sy_ptr[is * dis->ws + js] = Sy_ptr[(is - dir) * dis->ws + js];
                        }
                    }
                }

                /* Use the best candidate as a starting point for the gradient descent: */
                float cur_Ux = Sx_ptr[is * dis->ws + js];
                float cur_Uy = Sy_ptr[is * dis->ws + js];

                /* Computing the inverse of the structure tensor: */
                float detH = xx_ptr[is * dis->ws + js] * yy_ptr[is * dis->ws + js] -
                             xy_ptr[is * dis->ws + js] * xy_ptr[is * dis->ws + js];
                if (abs(detH) < EPS)
                    detH = EPS;
                float invH11 = yy_ptr[is * dis->ws + js] / detH;
                float invH12 = -xy_ptr[is * dis->ws + js] / detH;
                float invH22 = xx_ptr[is * dis->ws + js] / detH;
                float prev_SSD = INF, SSD;
                float x_grad_sum = x_ptr[is * dis->ws + js];
                float y_grad_sum = y_ptr[is * dis->ws + js];

                for (int t = 0; t < num_inner_iter; t++)
                {
                    INIT_BILINEAR_WEIGHTS(cur_Ux, cur_Uy);
                    if (dis->use_mean_normalization)
                        SSD = processPatchMeanNorm(dUx, dUy, I0_ptr + i * dis->w + j,
                                                   I1_ptr + (int)i_I1 * w_ext + (int)j_I1, I0x_ptr + i * dis->w + j,
                                                   I0y_ptr + i * dis->w + j, dis->w, w_ext, w00, w01, w10, w11, psz,
                                                   x_grad_sum, y_grad_sum);
                    else
                        SSD = processPatch(dUx, dUy, I0_ptr + i * dis->w + j, I1_ptr + (int)i_I1 * w_ext + (int)j_I1,
                                           I0x_ptr + i * dis->w + j, I0y_ptr + i * dis->w + j, dis->w, w_ext, w00, w01,
                                           w10, w11, psz);

                    dx = invH11 * dUx + invH12 * dUy;
                    dy = invH12 * dUx + invH22 * dUy;
                    cur_Ux -= dx;
                    cur_Uy -= dy;

                    /* Break when patch distance stops decreasing */
                    if (SSD >= prev_SSD)
                        break;
                    prev_SSD = SSD;
                }

                /* If gradient descent converged to a flow vector that is very far from the initial approximation
                 * (more than patch size) then we don't use it. Noticeably improves the robustness.
                 */
                if (norm(Vec2f(cur_Ux - Sx_ptr[is * dis->ws + js], cur_Uy - Sy_ptr[is * dis->ws + js])) <= psz)
                {
                    Sx_ptr[is * dis->ws + js] = cur_Ux;
                    Sy_ptr[is * dis->ws + js] = cur_Uy;
                }
                j += dir * dis->patch_stride;
            }
            i += dir * dis->patch_stride;
        }
    }
#undef INIT_BILINEAR_WEIGHTS
#undef COMPUTE_SSD
}

DISOpticalFlowImpl::Densification_ParBody::Densification_ParBody(DISOpticalFlowImpl &_dis, int _nstripes, int _h,
                                                                 Mat &dst_Ux, Mat &dst_Uy, Mat &src_Sx, Mat &src_Sy,
                                                                 Mat &_I0, Mat &_I1)
    : dis(&_dis), nstripes(_nstripes), h(_h), Ux(&dst_Ux), Uy(&dst_Uy), Sx(&src_Sx), Sy(&src_Sy), I0(&_I0), I1(&_I1)
{
    stripe_sz = (int)ceil(h / (double)nstripes);
}

/* This function transforms a sparse optical flow field obtained by PatchInverseSearch (which computes flow values
 * on a sparse grid defined by patch_stride) into a dense optical flow field by weighted averaging of values from the
 * overlapping patches.
 */
void DISOpticalFlowImpl::Densification_ParBody::operator()(const Range &range) const
{
    int start_i = min(range.start * stripe_sz, h);
    int end_i = min(range.end * stripe_sz, h);

    /* Input sparse flow */
    float *Sx_ptr = Sx->ptr<float>();
    float *Sy_ptr = Sy->ptr<float>();

    /* Output dense flow */
    float *Ux_ptr = Ux->ptr<float>();
    float *Uy_ptr = Uy->ptr<float>();

    uchar *I0_ptr = I0->ptr<uchar>();
    uchar *I1_ptr = I1->ptr<uchar>();

    int psz = dis->patch_size;
    int pstr = dis->patch_stride;
    int i_l, i_u;
    int j_l, j_u;
    float i_m, j_m, diff;

    /* These values define the set of sparse grid locations that contain patches overlapping with the current dense flow
     * location */
    int start_is, end_is;
    int start_js, end_js;

/* Some helper macros for updating this set of sparse grid locations */
#define UPDATE_SPARSE_I_COORDINATES                                                                                    \
    if (i % pstr == 0 && i + psz <= h)                                                                                 \
        end_is++;                                                                                                      \
    if (i - psz >= 0 && (i - psz) % pstr == 0 && start_is < end_is)                                                    \
        start_is++;

#define UPDATE_SPARSE_J_COORDINATES                                                                                    \
    if (j % pstr == 0 && j + psz <= dis->w)                                                                            \
        end_js++;                                                                                                      \
    if (j - psz >= 0 && (j - psz) % pstr == 0 && start_js < end_js)                                                    \
        start_js++;

    start_is = 0;
    end_is = -1;
    for (int i = 0; i < start_i; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
    }
    for (int i = start_i; i < end_i; i++)
    {
        UPDATE_SPARSE_I_COORDINATES;
        start_js = 0;
        end_js = -1;
        for (int j = 0; j < dis->w; j++)
        {
            UPDATE_SPARSE_J_COORDINATES;
            float coef, sum_coef = 0.0f;
            float sum_Ux = 0.0f;
            float sum_Uy = 0.0f;

            /* Iterate through all the patches that overlap the current location (i,j) */
            for (int is = start_is; is <= end_is; is++)
                for (int js = start_js; js <= end_js; js++)
                {
                    j_m = min(max(j + Sx_ptr[is * dis->ws + js], 0.0f), dis->w - 1.0f - EPS);
                    i_m = min(max(i + Sy_ptr[is * dis->ws + js], 0.0f), dis->h - 1.0f - EPS);
                    j_l = (int)j_m;
                    j_u = j_l + 1;
                    i_l = (int)i_m;
                    i_u = i_l + 1;
                    diff = (j_m - j_l) * (i_m - i_l) * I1_ptr[i_u * dis->w + j_u] +
                           (j_u - j_m) * (i_m - i_l) * I1_ptr[i_u * dis->w + j_l] +
                           (j_m - j_l) * (i_u - i_m) * I1_ptr[i_l * dis->w + j_u] +
                           (j_u - j_m) * (i_u - i_m) * I1_ptr[i_l * dis->w + j_l] - I0_ptr[i * dis->w + j];
                    coef = 1 / max(1.0f, abs(diff));
                    sum_Ux += coef * Sx_ptr[is * dis->ws + js];
                    sum_Uy += coef * Sy_ptr[is * dis->ws + js];
                    sum_coef += coef;
                }
            Ux_ptr[i * dis->w + j] = sum_Ux / sum_coef;
            Uy_ptr[i * dis->w + j] = sum_Uy / sum_coef;
        }
    }
#undef UPDATE_SPARSE_I_COORDINATES
#undef UPDATE_SPARSE_J_COORDINATES
}

void DISOpticalFlowImpl::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    CV_Assert(!I0.empty() && I0.depth() == CV_8U && I0.channels() == 1);
    CV_Assert(!I1.empty() && I1.depth() == CV_8U && I1.channels() == 1);
    CV_Assert(I0.sameSize(I1));
    CV_Assert(I0.isContinuous());
    CV_Assert(I1.isContinuous());

    Mat I0Mat = I0.getMat();
    Mat I1Mat = I1.getMat();
    flow.create(I1Mat.size(), CV_32FC2);
    Mat &flowMat = flow.getMatRef();
    coarsest_scale = (int)(log((2 * I0Mat.cols) / (4.0 * patch_size)) / log(2.0) + 0.5) - 1;
    int num_stripes = getNumThreads();

    prepareBuffers(I0Mat, I1Mat);
    Ux[coarsest_scale].setTo(0.0f);
    Uy[coarsest_scale].setTo(0.0f);

    for (int i = coarsest_scale; i >= finest_scale; i--)
    {
        w = I0s[i].cols;
        h = I0s[i].rows;
        ws = 1 + (w - patch_size) / patch_stride;
        hs = 1 + (h - patch_size) / patch_stride;

        precomputeStructureTensor(I0xx_buf, I0yy_buf, I0xy_buf, I0x_buf, I0y_buf, I0xs[i], I0ys[i]);
        if (use_spatial_propagation)
        {
            /* Use a fixed number of stripes regardless the number of threads to make inverse search
             * with spatial propagation reproducible
             */
            parallel_for_(Range(0, 8), PatchInverseSearch_ParBody(*this, 8, hs, Sx, Sy, Ux[i], Uy[i], I0s[i],
                                                                  I1s_ext[i], I0xs[i], I0ys[i], 2));
        }
        else
        {
            parallel_for_(Range(0, num_stripes),
                          PatchInverseSearch_ParBody(*this, num_stripes, hs, Sx, Sy, Ux[i], Uy[i], I0s[i], I1s_ext[i],
                                                     I0xs[i], I0ys[i], 1));
        }

        parallel_for_(Range(0, num_stripes),
                      Densification_ParBody(*this, num_stripes, I0s[i].rows, Ux[i], Uy[i], Sx, Sy, I0s[i], I1s[i]));
        if (variational_refinement_iter > 0)
            variational_refinement_processors[i]->calcUV(I0s[i], I1s[i], Ux[i], Uy[i]);

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
    I1s_ext.clear();
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

    for (int i = finest_scale; i <= coarsest_scale; i++)
        variational_refinement_processors[i]->collectGarbage();
    variational_refinement_processors.clear();
}

Ptr<DISOpticalFlow> createOptFlow_DIS(int preset)
{
    Ptr<DISOpticalFlow> dis = makePtr<DISOpticalFlowImpl>();
    dis->setPatchSize(8);
    if (preset == DISOpticalFlow::PRESET_ULTRAFAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(12);
        dis->setVariationalRefinementIterations(0);
    }
    else if (preset == DISOpticalFlow::PRESET_FAST)
    {
        dis->setFinestScale(2);
        dis->setPatchStride(4);
        dis->setGradientDescentIterations(16);
        dis->setVariationalRefinementIterations(5);
    }
    else if (preset == DISOpticalFlow::PRESET_MEDIUM)
    {
        dis->setFinestScale(1);
        dis->setPatchStride(3);
        dis->setGradientDescentIterations(25);
        dis->setVariationalRefinementIterations(5);
    }

    return dis;
}
}
}
