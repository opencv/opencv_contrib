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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "learning_based_color_balance_model.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xphoto.hpp"

using namespace std;
#define EPS 0.00001f

namespace cv
{
namespace xphoto
{

inline void getChromaticity(Vec2f &dst, float R, float G, float B)
{
    dst[0] = R / (R + G + B + EPS);
    dst[1] = G / (R + G + B + EPS);
}

struct hist_elem
{
    float hist_val;
    float r, g;
    hist_elem(float _hist_val, Vec2f chromaticity) : hist_val(_hist_val), r(chromaticity[0]), g(chromaticity[1]) {}
};
bool operator<(const hist_elem &a, const hist_elem &b);
bool operator<(const hist_elem &a, const hist_elem &b) { return a.hist_val > b.hist_val; }

class LearningBasedWBImpl : public LearningBasedWB
{
  private:
    int range_max_val, hist_bin_num, palette_size;
    float saturation_thresh, palette_bandwidth, prediction_thresh;
    int num_trees, num_tree_nodes, tree_depth;
    uchar *feature_idx;
    float *thresh_vals, *leaf_vals;
    Mat feature_idx_Mat, thresh_vals_Mat, leaf_vals_Mat;
    Mat mask;
    int src_max_val;

    void preprocessing(Mat &src);
    void getAverageAndBrightestColorChromaticity(Vec2f &average_chromaticity, Vec2f &brightest_chromaticity, Mat &src);
    void getColorPaletteMode(Vec2f &dst, hist_elem *palette);
    void getHistogramBasedFeatures(Vec2f &dominant_chromaticity, Vec2f &chromaticity_palette_mode, Mat &src);

    float regressionTreePredict(Vec2f src, uchar *tree_feature_idx, float *tree_thresh_vals, float *tree_leaf_vals);
    Vec2f predictIlluminant(vector<Vec2f> features);

  public:
    LearningBasedWBImpl(String path_to_model)
    {
        range_max_val = 255;
        saturation_thresh = 0.98f;
        hist_bin_num = 64;
        palette_size = 300;
        palette_bandwidth = 0.1f;
        prediction_thresh = 0.025f;
        if (path_to_model.empty())
        {
            /* use the default model */
            num_trees = _num_trees;
            num_tree_nodes = _num_tree_nodes;
            feature_idx = _feature_idx;
            thresh_vals = _thresh_vals;
            leaf_vals = _leaf_vals;
        }
        else
        {
            /* load model from file */
            FileStorage fs(path_to_model, 0);
            num_trees = fs["num_trees"];
            num_tree_nodes = fs["num_tree_nodes"];
            fs["feature_idx"] >> feature_idx_Mat;
            fs["thresh_vals"] >> thresh_vals_Mat;
            fs["leaf_vals"] >> leaf_vals_Mat;
            feature_idx = feature_idx_Mat.ptr<uchar>();
            thresh_vals = thresh_vals_Mat.ptr<float>();
            leaf_vals = leaf_vals_Mat.ptr<float>();
        }
    }

    int getRangeMaxVal() const { return range_max_val; }
    void setRangeMaxVal(int val) { range_max_val = val; }

    float getSaturationThreshold() const { return saturation_thresh; }
    void setSaturationThreshold(float val) { saturation_thresh = val; }

    int getHistBinNum() const { return hist_bin_num; }
    void setHistBinNum(int val) { hist_bin_num = val; }

    void extractSimpleFeatures(InputArray _src, OutputArray _dst)
    {
        CV_Assert(!_src.empty());
        CV_Assert(_src.isContinuous());
        CV_Assert(_src.type() == CV_8UC3 || _src.type() == CV_16UC3);
        Mat src = _src.getMat();
        vector<Vec2f> dst(num_features);

        preprocessing(src);
        getAverageAndBrightestColorChromaticity(dst[0], dst[1], src);
        getHistogramBasedFeatures(dst[2], dst[3], src);
        Mat(dst).convertTo(_dst, CV_32F);
    }

    void balanceWhite(InputArray _src, OutputArray _dst)
    {
        CV_Assert(!_src.empty());
        CV_Assert(_src.isContinuous());
        CV_Assert(_src.type() == CV_8UC3 || _src.type() == CV_16UC3);
        Mat src = _src.getMat();

        vector<Vec2f> features;
        extractSimpleFeatures(src, features);
        Vec2f illuminant = predictIlluminant(features);

        float denom = 1 - illuminant[0] - illuminant[1];
        float gainB = 1.0f;
        float gainG = denom / illuminant[1];
        float gainR = denom / illuminant[0];
        applyChannelGains(src, _dst, gainB, gainG, gainR);
    }
};

/* Computes a mask for non-saturated pixels and maximum pixel value
 * which are then used for feature computation
 */
void LearningBasedWBImpl::preprocessing(Mat &src)
{
    mask.create(src.size(), CV_8U);
    uchar *mask_ptr = mask.ptr<uchar>();
    int src_len = src.rows * src.cols;
    int thresh = (int)(saturation_thresh * range_max_val);
    int i = 0;
    int local_max;
    src_max_val = -1;

    if (src.type() == CV_8UC3)
    {
        uchar *src_ptr = src.ptr<uchar>();
#if CV_SIMD128
        v_uint8x16 v_inB, v_inG, v_inR, v_local_max;
        v_uint8x16 v_global_max = v_setall_u8(0), v_mask, v_thresh = v_setall_u8((uchar)thresh);
        for (; i < src_len - 15; i += 16)
        {
            v_load_deinterleave(src_ptr + 3 * i, v_inB, v_inG, v_inR);
            v_local_max = v_max(v_inB, v_max(v_inG, v_inR));
            v_global_max = v_max(v_local_max, v_global_max);
            v_mask = (v_local_max < v_thresh);
            v_store(mask_ptr + i, v_mask);
        }
        uchar global_max[16];
        v_store(global_max, v_global_max);
        for (int j = 0; j < 16; j++)
        {
            if (global_max[j] > src_max_val)
                src_max_val = global_max[j];
        }
#endif
        for (; i < src_len; i++)
        {
            local_max = max(src_ptr[3 * i], max(src_ptr[3 * i + 1], src_ptr[3 * i + 2]));
            if (local_max > src_max_val)
                src_max_val = local_max;
            if (local_max < thresh)
                mask_ptr[i] = 255;
            else
                mask_ptr[i] = 0;
        }
    }
    else if (src.type() == CV_16UC3)
    {
        ushort *src_ptr = src.ptr<ushort>();
#if CV_SIMD128
        v_uint16x8 v_inB, v_inG, v_inR, v_local_max;
        v_uint16x8 v_global_max = v_setall_u16(0), v_mask, v_thresh = v_setall_u16((ushort)thresh);
        for (; i < src_len - 7; i += 8)
        {
            v_load_deinterleave(src_ptr + 3 * i, v_inB, v_inG, v_inR);
            v_local_max = v_max(v_inB, v_max(v_inG, v_inR));
            v_global_max = v_max(v_local_max, v_global_max);
            v_mask = (v_local_max < v_thresh);
            v_pack_store(mask_ptr + i, v_mask);
        }
        ushort global_max[8];
        v_store(global_max, v_global_max);
        for (int j = 0; j < 8; j++)
        {
            if (global_max[j] > src_max_val)
                src_max_val = global_max[j];
        }
#endif
        for (; i < src_len; i++)
        {
            local_max = max(src_ptr[3 * i], max(src_ptr[3 * i + 1], src_ptr[3 * i + 2]));
            if (local_max > src_max_val)
                src_max_val = local_max;
            if (local_max < thresh)
                mask_ptr[i] = 255;
            else
                mask_ptr[i] = 0;
        }
    }
}

void LearningBasedWBImpl::getAverageAndBrightestColorChromaticity(Vec2f &average_chromaticity,
                                                                  Vec2f &brightest_chromaticity, Mat &src)
{
    int i = 0;
    int src_len = src.rows * src.cols;
    uchar *mask_ptr = mask.ptr<uchar>();
    uint brightestB = 0, brightestG = 0, brightestR = 0;
    uint max_sum = 0;
    if (src.type() == CV_8UC3)
    {
        uint sumB = 0, sumG = 0, sumR = 0;
        uchar *src_ptr = src.ptr<uchar>();
#if CV_SIMD128
        v_uint8x16 v_inB, v_inG, v_inR, v_mask;
        v_uint16x8 v_sR1, v_sR2, v_sG1, v_sG2, v_sB1, v_sB2, v_sum;
        v_uint16x8 v_max_sum = v_setall_u16(0), v_max_mask, v_brightestR, v_brightestG, v_brightestB;
        v_uint32x4 v_uint1, v_uint2, v_SB = v_setzero_u32(), v_SG = v_setzero_u32(), v_SR = v_setzero_u32();
        for (; i < src_len - 15; i += 16)
        {
            v_load_deinterleave(src_ptr + 3 * i, v_inB, v_inG, v_inR);
            v_mask = v_load(mask_ptr + i);

            v_inB &= v_mask;
            v_inG &= v_mask;
            v_inR &= v_mask;

            v_expand(v_inB, v_sB1, v_sB2);
            v_expand(v_inG, v_sG1, v_sG2);
            v_expand(v_inR, v_sR1, v_sR2);

            // update the brightest (R,G,B) tuple (process left half):
            v_sum = v_sB1 + v_sG1 + v_sR1;
            v_max_mask = (v_sum > v_max_sum);
            v_max_sum = v_max(v_sum, v_max_sum);
            v_brightestB = (v_sB1 & v_max_mask) + (v_brightestB & (~v_max_mask));
            v_brightestG = (v_sG1 & v_max_mask) + (v_brightestG & (~v_max_mask));
            v_brightestR = (v_sR1 & v_max_mask) + (v_brightestR & (~v_max_mask));

            // update the brightest (R,G,B) tuple (process right half):
            v_sum = v_sB2 + v_sG2 + v_sR2;
            v_max_mask = (v_sum > v_max_sum);
            v_max_sum = v_max(v_sum, v_max_sum);
            v_brightestB = (v_sB2 & v_max_mask) + (v_brightestB & (~v_max_mask));
            v_brightestG = (v_sG2 & v_max_mask) + (v_brightestG & (~v_max_mask));
            v_brightestR = (v_sR2 & v_max_mask) + (v_brightestR & (~v_max_mask));

            // update sums:
            v_sB1 = v_sB1 + v_sB2;
            v_sG1 = v_sG1 + v_sG2;
            v_sR1 = v_sR1 + v_sR2;
            v_expand(v_sB1, v_uint1, v_uint2);
            v_SB += v_uint1 + v_uint2;
            v_expand(v_sG1, v_uint1, v_uint2);
            v_SG += v_uint1 + v_uint2;
            v_expand(v_sR1, v_uint1, v_uint2);
            v_SR += v_uint1 + v_uint2;
        }
        sumB = v_reduce_sum(v_SB);
        sumG = v_reduce_sum(v_SG);
        sumR = v_reduce_sum(v_SR);
        ushort brightestB_arr[8], brightestG_arr[8], brightestR_arr[8], max_sum_arr[8];
        v_store(brightestB_arr, v_brightestB);
        v_store(brightestG_arr, v_brightestG);
        v_store(brightestR_arr, v_brightestR);
        v_store(max_sum_arr, v_max_sum);
        for (int j = 0; j < 8; j++)
        {
            if (max_sum_arr[j] > max_sum)
            {
                max_sum = max_sum_arr[j];
                brightestB = brightestB_arr[j];
                brightestG = brightestG_arr[j];
                brightestR = brightestR_arr[j];
            }
        }
#endif
        for (; i < src_len; i++)
        {
            uint sum_val = src_ptr[3 * i] + src_ptr[3 * i + 1] + src_ptr[3 * i + 2];
            if (mask_ptr[i])
            {
                sumB += src_ptr[3 * i];
                sumG += src_ptr[3 * i + 1];
                sumR += src_ptr[3 * i + 2];
                if (sum_val > max_sum)
                {
                    max_sum = sum_val;
                    brightestB = src_ptr[3 * i];
                    brightestG = src_ptr[3 * i + 1];
                    brightestR = src_ptr[3 * i + 2];
                }
            }
        }
        double maxRGB = (double)max(sumR, max(sumG, sumB));
        getChromaticity(average_chromaticity, (float)(sumR / maxRGB), (float)(sumG / maxRGB), (float)(sumB / maxRGB));
        getChromaticity(brightest_chromaticity, (float)brightestR, (float)brightestG, (float)brightestB);
    }
    else if (src.type() == CV_16UC3)
    {
        uint64 sumB = 0, sumG = 0, sumR = 0;
        ushort *src_ptr = src.ptr<ushort>();
#if CV_SIMD128
        v_uint16x8 v_inB, v_inG, v_inR, v_mask, v_mask_lower = v_setall_u16(255);
        v_uint32x4 v_iR1, v_iR2, v_iG1, v_iG2, v_iB1, v_iB2, v_sum;
        v_uint32x4 v_max_sum = v_setall_u32(0), v_max_mask, v_brightestR, v_brightestG, v_brightestB;
        v_uint64x2 v_uint64_1, v_uint64_2, v_SB = v_setzero_u64(), v_SG = v_setzero_u64(), v_SR = v_setzero_u64();
        for (; i < src_len - 7; i += 8)
        {
            v_load_deinterleave(src_ptr + 3 * i, v_inB, v_inG, v_inR);
            v_mask = v_load_expand(mask_ptr + i);
            v_mask = v_mask | ((v_mask & v_mask_lower) << 8);

            v_inB &= v_mask;
            v_inG &= v_mask;
            v_inR &= v_mask;

            v_expand(v_inB, v_iB1, v_iB2);
            v_expand(v_inG, v_iG1, v_iG2);
            v_expand(v_inR, v_iR1, v_iR2);

            // update the brightest (R,G,B) tuple (process left half):
            v_sum = v_iB1 + v_iG1 + v_iR1;
            v_max_mask = (v_sum > v_max_sum);
            v_max_sum = v_max(v_sum, v_max_sum);
            v_brightestB = (v_iB1 & v_max_mask) + (v_brightestB & (~v_max_mask));
            v_brightestG = (v_iG1 & v_max_mask) + (v_brightestG & (~v_max_mask));
            v_brightestR = (v_iR1 & v_max_mask) + (v_brightestR & (~v_max_mask));

            // update the brightest (R,G,B) tuple (process right half):
            v_sum = v_iB2 + v_iG2 + v_iR2;
            v_max_mask = (v_sum > v_max_sum);
            v_max_sum = v_max(v_sum, v_max_sum);
            v_brightestB = (v_iB2 & v_max_mask) + (v_brightestB & (~v_max_mask));
            v_brightestG = (v_iG2 & v_max_mask) + (v_brightestG & (~v_max_mask));
            v_brightestR = (v_iR2 & v_max_mask) + (v_brightestR & (~v_max_mask));

            // update sums:
            v_iB1 = v_iB1 + v_iB2;
            v_iG1 = v_iG1 + v_iG2;
            v_iR1 = v_iR1 + v_iR2;
            v_expand(v_iB1, v_uint64_1, v_uint64_2);
            v_SB += v_uint64_1 + v_uint64_2;
            v_expand(v_iG1, v_uint64_1, v_uint64_2);
            v_SG += v_uint64_1 + v_uint64_2;
            v_expand(v_iR1, v_uint64_1, v_uint64_2);
            v_SR += v_uint64_1 + v_uint64_2;
        }
        uint64 sum_arr[2];
        v_store(sum_arr, v_SB);
        sumB = sum_arr[0] + sum_arr[1];
        v_store(sum_arr, v_SG);
        sumG = sum_arr[0] + sum_arr[1];
        v_store(sum_arr, v_SR);
        sumR = sum_arr[0] + sum_arr[1];
        uint brightestB_arr[4], brightestG_arr[4], brightestR_arr[4], max_sum_arr[4];
        v_store(brightestB_arr, v_brightestB);
        v_store(brightestG_arr, v_brightestG);
        v_store(brightestR_arr, v_brightestR);
        v_store(max_sum_arr, v_max_sum);
        for (int j = 0; j < 4; j++)
        {
            if (max_sum_arr[j] > max_sum)
            {
                max_sum = max_sum_arr[j];
                brightestB = brightestB_arr[j];
                brightestG = brightestG_arr[j];
                brightestR = brightestR_arr[j];
            }
        }
#endif
        for (; i < src_len; i++)
        {
            uint sum_val = src_ptr[3 * i] + src_ptr[3 * i + 1] + src_ptr[3 * i + 2];
            if (mask_ptr[i])
            {
                sumB += src_ptr[3 * i];
                sumG += src_ptr[3 * i + 1];
                sumR += src_ptr[3 * i + 2];
                if (sum_val > max_sum)
                {
                    max_sum = sum_val;
                    brightestB = src_ptr[3 * i];
                    brightestG = src_ptr[3 * i + 1];
                    brightestR = src_ptr[3 * i + 2];
                }
            }
        }
        double maxRGB = (double)max(sumR, max(sumG, sumB));
        getChromaticity(average_chromaticity, (float)(sumR / maxRGB), (float)(sumG / maxRGB), (float)(sumB / maxRGB));
        getChromaticity(brightest_chromaticity, (float)brightestR, (float)brightestG, (float)brightestB);
    }
}

/* Returns the most high-density point (i.e. mode) of the color palette.
 * Uses a simplistic kernel density estimator with a Epanechnikov kernel and
 * fixed bandwidth.
 */
void LearningBasedWBImpl::getColorPaletteMode(Vec2f &dst, hist_elem *palette)
{
    float max_density = -1.0f;
    float denom = palette_bandwidth * palette_bandwidth;
    for (int i = 0; i < palette_size; i++)
    {
        float cur_density = 0.0f;
        float cur_dist_sq;

        for (int j = 0; j < palette_size; j++)
        {
            cur_dist_sq = (palette[i].r - palette[j].r) * (palette[i].r - palette[j].r) +
                          (palette[i].g - palette[j].g) * (palette[i].g - palette[j].g);
            cur_density += max((1.0f - (cur_dist_sq / denom)), 0.0f);
        }

        if (cur_density > max_density)
        {
            max_density = cur_density;
            dst[0] = palette[i].r;
            dst[1] = palette[i].g;
        }
    }
}

void LearningBasedWBImpl::getHistogramBasedFeatures(Vec2f &dominant_chromaticity, Vec2f &chromaticity_palette_mode,
                                                    Mat &src)
{
    MatND hist;
    int channels[] = {0, 1, 2};
    int histSize[] = {hist_bin_num, hist_bin_num, hist_bin_num};
    float range[] = {0, (float)max(hist_bin_num, src_max_val)};
    const float *ranges[] = {range, range, range};
    calcHist(&src, 1, channels, mask, hist, 3, histSize, ranges);

    int dominant_B = 0, dominant_G = 0, dominant_R = 0;
    double max_hist_val = 0;
    float *hist_ptr = hist.ptr<float>();
    for (int i = 0; i < hist_bin_num; i++)
        for (int j = 0; j < hist_bin_num; j++)
            for (int k = 0; k < hist_bin_num; k++)
            {
                if (*hist_ptr > max_hist_val)
                {
                    max_hist_val = *hist_ptr;
                    dominant_B = i;
                    dominant_G = j;
                    dominant_R = k;
                }
                hist_ptr++;
            }
    getChromaticity(dominant_chromaticity, (float)dominant_R, (float)dominant_G, (float)dominant_B);

    vector<hist_elem> palette;
    palette.reserve(palette_size);
    hist_ptr = hist.ptr<float>();
    // extract top palette_size most common colors and add them to the palette:
    for (int i = 0; i < hist_bin_num; i++)
        for (int j = 0; j < hist_bin_num; j++)
            for (int k = 0; k < hist_bin_num; k++)
            {
                float bin_count = *hist_ptr;
                if (bin_count < EPS)
                {
                    hist_ptr++;
                    continue;
                }
                Vec2f chromaticity;
                getChromaticity(chromaticity, (float)k, (float)j, (float)i);
                hist_elem el(bin_count, chromaticity);

                if (palette.size() < (uint)palette_size)
                {
                    palette.push_back(el);
                    if (palette.size() == (uint)palette_size)
                        make_heap(palette.begin(), palette.end());
                }
                else if (bin_count > palette.front().hist_val)
                {
                    pop_heap(palette.begin(), palette.end());
                    palette.back() = el;
                    push_heap(palette.begin(), palette.end());
                }
                hist_ptr++;
            }
    getColorPaletteMode(chromaticity_palette_mode, (hist_elem *)(&palette[0]));
}

float LearningBasedWBImpl::regressionTreePredict(Vec2f src, uchar *tree_feature_idx, float *tree_thresh_vals,
                                                 float *tree_leaf_vals)
{
    int node_idx = 0;
    for (int i = 0; i < tree_depth; i++)
    {
        if (src[tree_feature_idx[node_idx]] <= tree_thresh_vals[node_idx])
            node_idx = 2 * node_idx + 1;
        else
            node_idx = 2 * node_idx + 2;
    }
    return tree_leaf_vals[node_idx - num_tree_nodes + 1];
}

Vec2f LearningBasedWBImpl::predictIlluminant(vector<Vec2f> features)
{
    int feature_model_size = 2 * (num_tree_nodes - 1);
    int local_model_size = num_features * feature_model_size;
    int feature_model_size_leaf = 2 * num_tree_nodes;
    int local_model_size_leaf = num_features * feature_model_size_leaf;
    tree_depth = cvRound( (log(static_cast<float>(num_tree_nodes)) / log(2.0f)) );

    vector<float> consensus_r, consensus_g;
    vector<float> all_r, all_g;
    for (int i = 0; i < num_trees; i++)
    {
        Vec2f local_predictions[num_features];
        for (int j = 0; j < num_features; j++)
        {
            float r = regressionTreePredict(features[j], feature_idx + local_model_size * i + feature_model_size * j,
                                            thresh_vals + local_model_size * i + feature_model_size * j,
                                            leaf_vals + local_model_size_leaf * i + feature_model_size_leaf * j);
            float g = regressionTreePredict(
              features[j], feature_idx + local_model_size * i + feature_model_size * j + feature_model_size / 2,
              thresh_vals + local_model_size * i + feature_model_size * j + feature_model_size / 2,
              leaf_vals + local_model_size_leaf * i + feature_model_size_leaf * j + feature_model_size_leaf / 2);
            local_predictions[j] = Vec2f(r, g);
            all_r.push_back(r);
            all_g.push_back(g);
        }
        int agreement_degree = 0;
        for (int j = 0; j < num_features - 1; j++)
            for (int k = j + 1; k < num_features; k++)
            {
                if (norm(local_predictions[j] - local_predictions[k]) < prediction_thresh)
                    agreement_degree++;
            }
        if (agreement_degree >= 3)
        {
            for (int j = 0; j < num_features; j++)
            {
                consensus_r.push_back(local_predictions[j][0]);
                consensus_g.push_back(local_predictions[j][1]);
            }
        }
    }

    float illuminant_r, illuminant_g;
    if (consensus_r.size() == 0)
    {
        nth_element(all_r.begin(), all_r.begin() + all_r.size() / 2, all_r.end());
        illuminant_r = all_r[all_r.size() / 2];
        nth_element(all_g.begin(), all_g.begin() + all_g.size() / 2, all_g.end());
        illuminant_g = all_g[all_g.size() / 2];
    }
    else
    {
        nth_element(consensus_r.begin(), consensus_r.begin() + consensus_r.size() / 2, consensus_r.end());
        illuminant_r = consensus_r[consensus_r.size() / 2];
        nth_element(consensus_g.begin(), consensus_g.begin() + consensus_g.size() / 2, consensus_g.end());
        illuminant_g = consensus_g[consensus_g.size() / 2];
    }
    return Vec2f(illuminant_r, illuminant_g);
}

Ptr<LearningBasedWB> createLearningBasedWB(const String& path_to_model)
{
    Ptr<LearningBasedWB> inst = makePtr<LearningBasedWBImpl>(path_to_model);
    return inst;
}
}
}
