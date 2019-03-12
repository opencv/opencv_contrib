/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#include "precomp.hpp"
#include "edgeaware_filters_common.hpp"
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace
{

using std::numeric_limits;
using std::vector;
using namespace cv;
using namespace cv::ximgproc;
using namespace cv::ximgproc::intrinsics;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

inline double Log2(double n)
{
    return log(n) / log(2.0);
}

inline double floor_to_power_of_two(double r)
{
    return pow(2.0, floor(Log2(r)));
}

inline int computeManifoldTreeHeight(double sigma_s, double sigma_r)
{
    const double Hs = floor(Log2(sigma_s)) - 1.0;
    const double Lr = 1.0 - sigma_r;
    return max(2, static_cast<int>(ceil(Hs * Lr)));
}

static void splitChannels(InputArrayOfArrays src, vector<Mat>& dst)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if ( src.isMat() || src.isUMat() )
    {
        split(src, dst);
    }
    else
    {
        Size sz;
        int depth, totalCnNum;

        checkSameSizeAndDepth(src, sz, depth);
        totalCnNum = getTotalNumberOfChannels(src);

        dst.resize(totalCnNum);
        vector<int> fromTo(2 * totalCnNum);
        for (int i = 0; i < totalCnNum; i++)
        {
            fromTo[i * 2 + 0] = i;
            fromTo[i * 2 + 1] = i;

            dst[i].create(sz, CV_MAKE_TYPE(depth, 1));
        }

        mixChannels(src, dst, fromTo);
    }
}

class AdaptiveManifoldFilterN : public AdaptiveManifoldFilter
{
public:
    AdaptiveManifoldFilterN();

    void filter(InputArray src, OutputArray dst, InputArray joint) CV_OVERRIDE;

    void collectGarbage() CV_OVERRIDE;

    inline double getSigmaS() const CV_OVERRIDE { return sigma_s_; }
    inline void setSigmaS(double val) CV_OVERRIDE { sigma_s_ = val; }
    inline double getSigmaR() const CV_OVERRIDE { return sigma_r_; }
    inline void setSigmaR(double val) CV_OVERRIDE { sigma_r_ = val; }
    inline int getTreeHeight() const CV_OVERRIDE { return tree_height_; }
    inline void setTreeHeight(int val) CV_OVERRIDE { tree_height_ = val; }
    inline int getPCAIterations() const CV_OVERRIDE { return num_pca_iterations_; }
    inline void setPCAIterations(int val) CV_OVERRIDE { num_pca_iterations_ = val; }
    inline bool getAdjustOutliers() const CV_OVERRIDE { return adjust_outliers_; }
    inline void setAdjustOutliers(bool val) CV_OVERRIDE { adjust_outliers_ = val; }
    inline bool getUseRNG() const CV_OVERRIDE { return useRNG; }
    inline void setUseRNG(bool val) CV_OVERRIDE { useRNG = val; }

protected:

    bool adjust_outliers_;
    double sigma_s_;
    double sigma_r_;
    int tree_height_;
    int num_pca_iterations_;
    bool useRNG;

private:

    Size srcSize;
    Size smallSize;
    int jointCnNum;
    int srcCnNum;

    vector<Mat> jointCn;
    vector<Mat> srcCn;

    vector<Mat> etaFull;

    vector<Mat> sum_w_ki_Psi_blur_;
    Mat sum_w_ki_Psi_blur_0_;

    Mat w_k;
    Mat Psi_splat_0_small;
    vector<Mat> Psi_splat_small;

    Mat1f minDistToManifoldSquared;

    int curTreeHeight;
    float sigma_r_over_sqrt_2;

    RNG rnd;

private: /*inline functions*/

    double getNormalizer(int depth)
    {
        double normalizer = 1.0;

        if (depth == CV_8U)
            normalizer = 1.0 / 0xFF;
        else if (depth == CV_16U)
            normalizer = 1.0 / 0xFFFF;

        return normalizer;
    }

    double getResizeRatio()
    {
        double df = min(sigma_s_ / 4.0, 256.0 * sigma_r_);
        df = floor_to_power_of_two(df);
        df = max(1.0, df);
        return df;
    }

    Size getSmallSize()
    {
        double df = getResizeRatio();
        return Size( cvRound(srcSize.width * (1.0/df)), cvRound(srcSize.height*(1.0/df)) ) ;
    }

    void downsample(const Mat& src, Mat& dst)
    {
        double df = getResizeRatio();
        CV_DbgAssert(src.empty() || src.size() == srcSize);
        resize(src, dst, Size(), 1.0 / df, 1.0 / df, INTER_LINEAR);
        CV_DbgAssert(dst.size() == smallSize);
    }

    void upsample(const Mat& src, Mat& dst)
    {
        CV_DbgAssert(src.empty() || src.size() == smallSize);
        resize(src, dst, srcSize, 0, 0);
    }

    void downsample(const vector<Mat>& srcv, vector<Mat>& dstv)
    {
        mapParallel(&AdaptiveManifoldFilterN::downsample, srcv, dstv);
    }

    void upsample(const vector<Mat>& srcv, vector<Mat>& dstv)
    {
        mapParallel(&AdaptiveManifoldFilterN::upsample, srcv, dstv);
    }

private:

    void initBuffers(InputArray src_, InputArray joint_);

    void initSrcAndJoint(InputArray src_, InputArray joint_);

    void buildManifoldsAndPerformFiltering(vector<Mat>& eta, Mat1b& cluster, int treeLevel);

    void gatherResult(InputArray src_, OutputArray dst_);

    void compute_w_k(vector<Mat>& etak, Mat& dst, float sigma, int curTreeLevel);

    void computeClusters(Mat1b& cluster, Mat1b& cluster_minus, Mat1b& cluster_plus);

    void computeEta(Mat& teta, Mat1b& cluster, vector<Mat>& etaDst);


    static void h_filter(const Mat1f& src, Mat& dst, float sigma);

    static void RFFilterPass(vector<Mat>& joint, vector<Mat>& Psi_splat, Mat& Psi_splat_0, vector<Mat>& Psi_splat_dst, Mat& Psi_splat_0_dst, float ss, float sr);

    static void computeDTHor(vector<Mat>& srcCn, Mat& dst, float ss, float sr);

    static void computeDTVer(vector<Mat>& srcCn, Mat& dst, float ss, float sr);

    static void computeEigenVector(const vector<Mat>& X, const Mat1b& mask, Mat1f& vecDst, int num_pca_iterations, const Mat1f& vecRand);

    static void computeOrientation(const vector<Mat>& X, const Mat1f& vec, Mat1f& dst);

private: /*Parallelization routines*/

    typedef void (AdaptiveManifoldFilterN::*MapFunc)(const Mat& src, Mat& dst);

    void mapParallel(MapFunc func, const vector<Mat>& srcv, vector<Mat>& dstv)
    {
        dstv.resize(srcv.size());
        parallel_for_(Range(0, (int)srcv.size()), MapPrallelLoopBody(this, func, srcv, dstv));
    }

    struct MapPrallelLoopBody : public cv::ParallelLoopBody
    {
        MapPrallelLoopBody(AdaptiveManifoldFilterN *_instancePtr, MapFunc _transform, const vector<Mat>& _srcv, vector<Mat>& _dstv)
            : instancePtr(_instancePtr), transform(_transform), srcv(_srcv), dstv(_dstv)
        {}

        AdaptiveManifoldFilterN *instancePtr;
        MapFunc transform;
        const vector<Mat>& srcv;
        vector<Mat>& dstv;

        void operator () (const Range& range) const CV_OVERRIDE
        {
            for (int i = range.start; i < range.end; i++)
                (instancePtr->*transform)(srcv[i], dstv[i]);
        }
    };

};

AdaptiveManifoldFilterN::AdaptiveManifoldFilterN()
{
    sigma_s_ = 16.0;
    sigma_r_ = 0.2;
    tree_height_ = -1;
    num_pca_iterations_ = 1;
    adjust_outliers_ = false;
    useRNG = true;
}

void AdaptiveManifoldFilterN::initBuffers(InputArray src_, InputArray joint_)
{
    initSrcAndJoint(src_, joint_);

    jointCn.resize(jointCnNum);
    Psi_splat_small.resize(jointCnNum);
    for (int i = 0; i < jointCnNum; i++)
    {
        //jointCn[i].create(srcSize, CV_32FC1);
        Psi_splat_small[i].create(smallSize, CV_32FC1);
    }

    srcCn.resize(srcCnNum);
    sum_w_ki_Psi_blur_.resize(srcCnNum);
    for (int i = 0; i < srcCnNum; i++)
    {
        //srcCn[i].create(srcSize, CV_32FC1);
        sum_w_ki_Psi_blur_[i] = Mat::zeros(srcSize, CV_32FC1);
    }

    sum_w_ki_Psi_blur_0_ = Mat::zeros(srcSize, CV_32FC1);
    w_k.create(srcSize, CV_32FC1);
    Psi_splat_0_small.create(smallSize, CV_32FC1);

    if (adjust_outliers_)
        minDistToManifoldSquared.create(srcSize);
}

void AdaptiveManifoldFilterN::initSrcAndJoint(InputArray src_, InputArray joint_)
{
    srcSize = src_.size();
    smallSize = getSmallSize();
    srcCnNum = src_.channels();

    split(src_, srcCn);
    if (src_.depth() != CV_32F)
    {
        for (int i = 0; i < srcCnNum; i++)
            srcCn[i].convertTo(srcCn[i], CV_32F);
    }

    if (joint_.empty() || joint_.getObj() == src_.getObj())
    {
        jointCnNum = srcCnNum;

        if (src_.depth() == CV_32F)
        {
            jointCn = srcCn;
        }
        else
        {
            jointCn.resize(jointCnNum);
            for (int i = 0; i < jointCnNum; i++)
                srcCn[i].convertTo(jointCn[i], CV_32F, getNormalizer(src_.depth()));
        }
    }
    else
    {
        splitChannels(joint_, jointCn);

        jointCnNum = (int)jointCn.size();
        int jointDepth = jointCn[0].depth();
        Size jointSize = jointCn[0].size();

        CV_Assert( jointSize == srcSize && (jointDepth == CV_8U || jointDepth == CV_16U || jointDepth == CV_32F) );

        if (jointDepth != CV_32F)
        {
            for (int i = 0; i < jointCnNum; i++)
                jointCn[i].convertTo(jointCn[i], CV_32F, getNormalizer(jointDepth));
        }
    }
}

void AdaptiveManifoldFilterN::filter(InputArray src, OutputArray dst, InputArray joint)
{
    CV_Assert(sigma_s_ >= 1 && (sigma_r_ > 0 && sigma_r_ <= 1));
    num_pca_iterations_ = std::max(1, num_pca_iterations_);

    initBuffers(src, joint);

    curTreeHeight = tree_height_ <= 0 ? computeManifoldTreeHeight(sigma_s_, sigma_r_) : tree_height_;

    sigma_r_over_sqrt_2 = (float) (sigma_r_ / sqrt(2.0));

    const double seedCoef = jointCn[0].at<float>(srcSize.height/2, srcSize.width/2);
    const uint64 baseCoef = numeric_limits<uint64>::max() / 0xFFFF;
    rnd.state = static_cast<int64>(baseCoef*seedCoef);

    Mat1b cluster0(srcSize, 0xFF);
    vector<Mat> eta0(jointCnNum);
    for (int i = 0; i < jointCnNum; i++)
        h_filter(jointCn[i], eta0[i], (float)sigma_s_);

    buildManifoldsAndPerformFiltering(eta0, cluster0, 1);

    gatherResult(src, dst);
}

void AdaptiveManifoldFilterN::gatherResult(InputArray src_, OutputArray dst_)
{
    int dDepth = src_.depth();
    vector<Mat> dstCn(srcCnNum);

    if (!adjust_outliers_)
    {
        for (int i = 0; i < srcCnNum; i++)
            divide(sum_w_ki_Psi_blur_[i], sum_w_ki_Psi_blur_0_, dstCn[i], 1.0, dDepth);

        merge(dstCn, dst_);
    }
    else
    {
        Mat1f& alpha = minDistToManifoldSquared;
        double sigmaMember = -0.5 / (sigma_r_*sigma_r_);
        multiply(minDistToManifoldSquared, sigmaMember, minDistToManifoldSquared);
        cv::exp(minDistToManifoldSquared, alpha);

        for (int i = 0; i < srcCnNum; i++)
        {
            Mat& f = srcCn[i];
            Mat& g = dstCn[i];

            divide(sum_w_ki_Psi_blur_[i], sum_w_ki_Psi_blur_0_, g);

            subtract(g, f, g);
            multiply(alpha, g, g);
            add(g, f, g);

            g.convertTo(g, dDepth);
        }

        merge(dstCn, dst_);
    }
}

void AdaptiveManifoldFilterN::buildManifoldsAndPerformFiltering(vector<Mat>& eta, Mat1b& cluster, int treeLevel)
{
    CV_DbgAssert((int)eta.size() == jointCnNum);

    //splatting
    Size etaSize = eta[0].size();
    CV_DbgAssert(etaSize == srcSize || etaSize == smallSize);

    if (etaSize == srcSize)
    {
        compute_w_k(eta, w_k, sigma_r_over_sqrt_2, treeLevel);
        etaFull = eta;
        downsample(eta, eta);
    }
    else
    {
        upsample(eta, etaFull);
        compute_w_k(etaFull, w_k, sigma_r_over_sqrt_2, treeLevel);
    }

    //blurring
    Psi_splat_small.resize(srcCnNum);
    for (int si = 0; si < srcCnNum; si++)
    {
        Mat tmp;
        multiply(srcCn[si], w_k, tmp);
        downsample(tmp, Psi_splat_small[si]);
    }
    downsample(w_k, Psi_splat_0_small);

    vector<Mat>& Psi_splat_small_blur = Psi_splat_small;
    Mat& Psi_splat_0_small_blur = Psi_splat_0_small;

    float rf_ss = (float)(sigma_s_ / getResizeRatio());
    float rf_sr = (float)(sigma_r_over_sqrt_2);
    RFFilterPass(eta, Psi_splat_small, Psi_splat_0_small, Psi_splat_small_blur, Psi_splat_0_small_blur, rf_ss, rf_sr);

    //slicing
    {
        Mat tmp;
        for (int i = 0; i < srcCnNum; i++)
        {
            upsample(Psi_splat_small_blur[i], tmp);
            multiply(tmp, w_k, tmp);
            add(sum_w_ki_Psi_blur_[i], tmp, sum_w_ki_Psi_blur_[i]);
        }
        upsample(Psi_splat_0_small_blur, tmp);
        multiply(tmp, w_k, tmp);
        add(sum_w_ki_Psi_blur_0_, tmp, sum_w_ki_Psi_blur_0_);
    }

    //build new manifolds
    if (treeLevel < curTreeHeight)
    {
        Mat1b cluster_minus, cluster_plus;

        computeClusters(cluster, cluster_minus, cluster_plus);

        vector<Mat> eta_minus(jointCnNum), eta_plus(jointCnNum);
        {
            Mat1f teta = 1.0 - w_k;
            computeEta(teta, cluster_minus, eta_minus);
            computeEta(teta, cluster_plus, eta_plus);
        }

        //free memory to continue deep recursion
        eta.clear();
        cluster.release();

        buildManifoldsAndPerformFiltering(eta_minus, cluster_minus, treeLevel + 1);
        buildManifoldsAndPerformFiltering(eta_plus, cluster_plus, treeLevel + 1);
    }
}

void AdaptiveManifoldFilterN::collectGarbage()
{
    srcCn.clear();
    jointCn.clear();
    etaFull.clear();
    sum_w_ki_Psi_blur_.clear();
    Psi_splat_small.clear();

    sum_w_ki_Psi_blur_0_.release();
    w_k.release();
    Psi_splat_0_small.release();
    minDistToManifoldSquared.release();
}

void AdaptiveManifoldFilterN::h_filter(const Mat1f& src, Mat& dst, float sigma)
{
    CV_DbgAssert(src.depth() == CV_32F);

    const float a = exp(-sqrt(2.0f) / sigma);

    dst.create(src.size(), CV_32FC1);

    for (int y = 0; y < src.rows; ++y)
    {
        const float* src_row = src[y];
        float* dst_row = dst.ptr<float>(y);

        dst_row[0] = src_row[0];
        for (int x = 1; x < src.cols; ++x)
        {
            dst_row[x] = src_row[x] + a * (dst_row[x - 1] - src_row[x]);
        }
        for (int x = src.cols - 2; x >= 0; --x)
        {
            dst_row[x] = dst_row[x] + a * (dst_row[x + 1] - dst_row[x]);
        }
    }

    for (int y = 1; y < src.rows; ++y)
    {
        float* dst_cur_row = dst.ptr<float>(y);
        float* dst_prev_row = dst.ptr<float>(y-1);

        rf_vert_row_pass(dst_cur_row, dst_prev_row, a, src.cols);
    }
    for (int y = src.rows - 2; y >= 0; --y)
    {
        float* dst_cur_row = dst.ptr<float>(y);
        float* dst_prev_row = dst.ptr<float>(y+1);

        rf_vert_row_pass(dst_cur_row, dst_prev_row, a, src.cols);
    }
}

void AdaptiveManifoldFilterN::compute_w_k(vector<Mat>& etak, Mat& dst, float sigma, int curTreeLevel)
{
    CV_DbgAssert((int)etak.size() == jointCnNum);

    dst.create(srcSize, CV_32FC1);
    float argConst = -0.5f / (sigma*sigma);

    for (int i = 0; i < srcSize.height; i++)
    {
        float *dstRow = dst.ptr<float>(i);

        for (int cn = 0; cn < jointCnNum; cn++)
        {
            float *eta_kCnRow = etak[cn].ptr<float>(i);
            float *jointCnRow = jointCn[cn].ptr<float>(i);

            if (cn == 0)
            {
                sqr_dif(dstRow, eta_kCnRow, jointCnRow, srcSize.width);
            }
            else
            {
                add_sqr_dif(dstRow, eta_kCnRow, jointCnRow, srcSize.width);
            }
        }

        if (adjust_outliers_)
        {
            float *minDistRow = minDistToManifoldSquared.ptr<float>(i);

            if (curTreeLevel != 1)
            {
                min_(minDistRow, minDistRow, dstRow, srcSize.width);
            }
            else
            {
                std::memcpy(minDistRow, dstRow, srcSize.width*sizeof(float));
            }
        }

        mul(dstRow, dstRow, argConst, srcSize.width);
        //Exp_32f(dstRow, dstRow, srcSize.width);
    }

    cv::exp(dst, dst);
}

void AdaptiveManifoldFilterN::computeDTHor(vector<Mat>& srcCn, Mat& dst, float sigma_s, float sigma_r)
{
    int cnNum = (int)srcCn.size();
    int h = srcCn[0].rows;
    int w = srcCn[0].cols;

    float sigmaRatioSqr = (float) SQR(sigma_s / sigma_r);
    float lnAlpha       = (float) (-sqrt(2.0) / sigma_s);

    dst.create(h, w-1, CV_32F);

    for (int i = 0; i < h; i++)
    {
        float *dstRow = dst.ptr<float>(i);

        for (int cn = 0; cn < cnNum; cn++)
        {
            float *curCnRow = srcCn[cn].ptr<float>(i);

            if (cn == 0)
                sqr_dif(dstRow, curCnRow, curCnRow + 1, w - 1);
            else
                add_sqr_dif(dstRow, curCnRow, curCnRow + 1, w - 1);
        }

        mad(dstRow, dstRow, sigmaRatioSqr, 1.0f, w - 1);
        sqrt_(dstRow, dstRow, w - 1);
        mul(dstRow, dstRow, lnAlpha, w - 1);
        //Exp_32f(dstRow, dstRow, w - 1);
    }

    cv::exp(dst, dst);
}

void AdaptiveManifoldFilterN::computeDTVer(vector<Mat>& srcCn, Mat& dst, float sigma_s, float sigma_r)
{
    int cnNum = (int)srcCn.size();
    int h = srcCn[0].rows;
    int w = srcCn[0].cols;

    dst.create(h-1, w, CV_32F);

    float sigmaRatioSqr = (float) SQR(sigma_s / sigma_r);
    float lnAlpha       = (float) (-sqrt(2.0) / sigma_s);

    for (int i = 0; i < h-1; i++)
    {
        float *dstRow = dst.ptr<float>(i);

        for (int cn = 0; cn < cnNum; cn++)
        {
            float *srcRow1 = srcCn[cn].ptr<float>(i);
            float *srcRow2 = srcCn[cn].ptr<float>(i+1);

            if (cn == 0)
                sqr_dif(dstRow, srcRow1, srcRow2, w);
            else
                add_sqr_dif(dstRow, srcRow1, srcRow2, w);
        }

        mad(dstRow, dstRow, sigmaRatioSqr, 1.0f, w);
        sqrt_(dstRow, dstRow, w);

        mul(dstRow, dstRow, lnAlpha, w);
        //Exp_32f(dstRow, dstRow, w);
    }

    cv::exp(dst, dst);
}

void AdaptiveManifoldFilterN::RFFilterPass(vector<Mat>& joint, vector<Mat>& Psi_splat, Mat& Psi_splat_0, vector<Mat>& Psi_splat_dst, Mat& Psi_splat_0_dst, float ss, float sr)
{
    int h = joint[0].rows;
    int w = joint[0].cols;
    int cnNum = (int)Psi_splat.size();

    Mat adth, adtv;
    computeDTHor(joint, adth, ss, sr);
    computeDTVer(joint, adtv, ss, sr);

    Psi_splat_0_dst.create(h, w, CV_32FC1);
    Psi_splat_dst.resize(cnNum);
    for (int cn = 0; cn < cnNum; cn++)
        Psi_splat_dst[cn].create(h, w, CV_32FC1);

    Ptr<DTFilter> dtf = createDTFilterRF(adth, adtv, ss, sr, 1);
    for (int cn = 0; cn < cnNum; cn++)
        dtf->filter(Psi_splat[cn], Psi_splat_dst[cn]);
    dtf->filter(Psi_splat_0, Psi_splat_0_dst);
}

void AdaptiveManifoldFilterN::computeClusters(Mat1b& cluster, Mat1b& cluster_minus, Mat1b& cluster_plus)
{

    Mat1f difOreientation;
    if (jointCnNum > 1)
    {
        Mat1f initVec(1, jointCnNum);
        if (useRNG)
        {
            rnd.fill(initVec, RNG::UNIFORM, -0.5, 0.5);
        }
        else
        {
            for (int i = 0; i < (int)initVec.total(); i++)
                initVec(0, i) = (i % 2 == 0) ? 0.5f : -0.5f;
        }

        vector<Mat> difEtaSrc(jointCnNum);
        for (int i = 0; i < jointCnNum; i++)
            subtract(jointCn[i], etaFull[i], difEtaSrc[i]);

        Mat1f eigenVec(1, jointCnNum);
        computeEigenVector(difEtaSrc, cluster, eigenVec, num_pca_iterations_, initVec);

        computeOrientation(difEtaSrc, eigenVec, difOreientation);
        CV_DbgAssert(difOreientation.size() == srcSize);
    }
    else
    {
        subtract(jointCn[0], etaFull[0], difOreientation);
    }

    compare(difOreientation, 0, cluster_minus, CMP_LT);
    bitwise_and(cluster_minus, cluster, cluster_minus);

    compare(difOreientation, 0, cluster_plus, CMP_GE);
    bitwise_and(cluster_plus, cluster, cluster_plus);
}

void AdaptiveManifoldFilterN::computeEta(Mat& teta, Mat1b& cluster, vector<Mat>& etaDst)
{
    CV_DbgAssert(teta.size() == srcSize && cluster.size() == srcSize);

    Mat1f tetaMasked = Mat1f::zeros(srcSize);
    teta.copyTo(tetaMasked, cluster);

    float sigma_s = (float)(sigma_s_ / getResizeRatio());

    Mat1f tetaMaskedBlur;
    downsample(tetaMasked, tetaMaskedBlur);
    h_filter(tetaMaskedBlur, tetaMaskedBlur, sigma_s);

    Mat mul;
    etaDst.resize(jointCnNum);
    for (int i = 0; i < jointCnNum; i++)
    {
        multiply(tetaMasked, jointCn[i], mul);
        downsample(mul, etaDst[i]);
        h_filter(etaDst[i], etaDst[i], sigma_s);
        divide(etaDst[i], tetaMaskedBlur, etaDst[i]);
    }
}

void AdaptiveManifoldFilterN::computeEigenVector(const vector<Mat>& X, const Mat1b& mask, Mat1f& vecDst, int num_pca_iterations, const Mat1f& vecRand)
{
    int cnNum = (int)X.size();
    int height = X[0].rows;
    int width = X[0].cols;

    vecDst.create(1, cnNum);
    CV_Assert(vecRand.size() == Size(cnNum, 1) && vecDst.size() == Size(cnNum, 1));
    CV_Assert(mask.rows == height && mask.cols == width);

    const float *pVecRand = vecRand.ptr<float>();
    Mat1d vecDstd(1, cnNum, 0.0);
    double *pVecDst = vecDstd.ptr<double>();
    Mat1f Xw(height, width);

    for (int iter = 0; iter < num_pca_iterations; iter++)
    {
        for (int  i = 0; i < height; i++)
        {
            const uchar *maskRow = mask.ptr<uchar>(i);
            float *mulRow = Xw.ptr<float>(i);

            //first multiplication
            for (int cn = 0; cn < cnNum; cn++)
            {
                const float *srcRow = X[cn].ptr<float>(i);
                const float cnVal = pVecRand[cn];

                if (cn == 0)
                {
                    for (int j = 0; j < width; j++)
                        mulRow[j] = cnVal*srcRow[j];
                }
                else
                {
                    for (int j = 0; j < width; j++)
                        mulRow[j] += cnVal*srcRow[j];
                }
            }

            for (int j = 0; j < width; j++)
                if (!maskRow[j]) mulRow[j] = 0.0f;

            //second multiplication
            for (int cn = 0; cn < cnNum; cn++)
            {
                float curCnSum = 0.0f;
                const float *srcRow = X[cn].ptr<float>(i);

                for (int j = 0; j < width; j++)
                    curCnSum += mulRow[j]*srcRow[j];

                //TODO: parallel reduce
                pVecDst[cn] += curCnSum;
            }
        }
    }

    divide(vecDstd, norm(vecDstd), vecDst);
}

void AdaptiveManifoldFilterN::computeOrientation(const vector<Mat>& X, const Mat1f& vec, Mat1f& dst)
{
    int height = X[0].rows;
    int width = X[0].cols;
    int cnNum = (int)X.size();
    dst.create(height, width);
    CV_DbgAssert(vec.rows == 1 && vec.cols == cnNum);

    const float *pVec = vec.ptr<float>();

    for (int i = 0; i < height; i++)
    {
        float *dstRow = dst.ptr<float>(i);

        for (int cn = 0; cn < cnNum; cn++)
        {
            const float *srcRow = X[cn].ptr<float>(i);
            const float cnVal = pVec[cn];

            if (cn == 0)
            {
                for (int j = 0; j < width; j++)
                    dstRow[j] = cnVal*srcRow[j];
            }
            else
            {
                for (int j = 0; j < width; j++)
                    dstRow[j] += cnVal*srcRow[j];
            }
        }
    }
}


}


namespace cv
{
namespace ximgproc
{

Ptr<AdaptiveManifoldFilter> AdaptiveManifoldFilter::create()
{
    return Ptr<AdaptiveManifoldFilter>(new AdaptiveManifoldFilterN());
}

Ptr<AdaptiveManifoldFilter> createAMFilter(double sigma_s, double sigma_r, bool adjust_outliers)
{
    Ptr<AdaptiveManifoldFilter> amf(new AdaptiveManifoldFilterN());

    amf->setSigmaS(sigma_s);
    amf->setSigmaR(sigma_r);
    amf->setAdjustOutliers(adjust_outliers);

    return amf;
}

void amFilter(InputArray joint, InputArray src, OutputArray dst, double sigma_s, double sigma_r, bool adjust_outliers)
{
    Ptr<AdaptiveManifoldFilter> amf = createAMFilter(sigma_s, sigma_r, adjust_outliers);
    amf->filter(src, dst, joint);
}

}
}
