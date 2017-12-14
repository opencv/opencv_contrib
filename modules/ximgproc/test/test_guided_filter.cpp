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

#include "test_precomp.hpp"

namespace cvtest
{

using namespace std;
using namespace std::tr1;
using namespace testing;
using namespace cv;
using namespace cv::ximgproc;

#ifndef SQR
#define SQR(x) ((x)*(x))
#endif

static string getOpenCVExtraDir()
{
    return cvtest::TS::ptr()->get_data_path();
}

static Mat convertTypeAndSize(Mat src, int dstType, Size dstSize)
{
    Mat dst;
    int srcCnNum = src.channels();
    int dstCnNum = CV_MAT_CN(dstType);
    CV_Assert(srcCnNum == 3);

    if (srcCnNum == dstCnNum)
    {
        src.copyTo(dst);
    }
    else if (dstCnNum == 1 && srcCnNum == 3)
    {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else if (dstCnNum == 1 && srcCnNum == 4)
    {
        cvtColor(src, dst, COLOR_BGRA2GRAY);
    }
    else
    {
        vector<Mat> srcCn;
        split(src, srcCn);
        srcCn.resize(dstCnNum);

        uint64 seed = 10000 * src.rows + 1000 * src.cols + 100 * dstSize.height + 10 * dstSize.width + dstType;
        RNG rnd(seed);

        for (int i = srcCnNum; i < dstCnNum; i++)
        {
            Mat& donor = srcCn[i % srcCnNum];

            double minVal, maxVal;
            minMaxLoc(donor, &minVal, &maxVal);

            Mat randItem(src.size(), CV_MAKE_TYPE(src.depth(), 1));
            randn(randItem, 0, (maxVal - minVal) / 100);

            add(donor, randItem, srcCn[i]);
        }

        merge(srcCn, dst);
    }

    dst.convertTo(dst, dstType);
    resize(dst, dst, dstSize, 0, 0, INTER_LINEAR_EXACT);

    return dst;
}

class GuidedFilterRefImpl : public GuidedFilter
{
    int height, width, rad, chNum;
    Mat det;
    Mat *channels, *exps, **vars, **A;
    double eps;

    void meanFilter(const Mat &src, Mat & dst);

    void computeCovGuide();

    void computeCovGuideInv();

    void applyTransform(int cNum, Mat *Ichannels, Mat *beta, Mat **alpha, int dDepth);

    void computeCovGuideAndSrc(int cNum, Mat **vars_I, Mat *Ichannels, Mat *exp_I);

    void computeBeta(int cNum, Mat *beta, Mat *exp_I, Mat **alpha);

    void computeAlpha(int cNum, Mat **alpha, Mat **vars_I);

public:

    GuidedFilterRefImpl(InputArray guide_, int rad, double eps);

    void filter(InputArray src, OutputArray dst, int dDepth = -1);

    ~GuidedFilterRefImpl();
};

void GuidedFilterRefImpl::meanFilter(const Mat &src, Mat & dst)
{
    boxFilter(src, dst, CV_32F, Size(2 * rad + 1, 2 * rad + 1), Point(-1, -1), true, BORDER_REFLECT);
}

GuidedFilterRefImpl::GuidedFilterRefImpl(InputArray _guide, int _rad, double _eps) :
  height(_guide.rows()), width(_guide.cols()), rad(_rad), chNum(_guide.channels()), eps(_eps)
{
    Mat guide = _guide.getMat();
    CV_Assert(chNum > 0 && chNum <= 3);

    channels = new Mat[chNum];
    exps     = new Mat[chNum];

    A    = new Mat *[chNum];
    vars = new Mat *[chNum];
    for (int i = 0; i < chNum; ++i)
    {
        A[i]    = new Mat[chNum];
        vars[i] = new Mat[chNum];
    }

    split(guide, channels);
    for (int i = 0; i < chNum; ++i)
    {
        channels[i].convertTo(channels[i], CV_32F);
        meanFilter(channels[i], exps[i]);
    }

    computeCovGuide();

    computeCovGuideInv();
}

void GuidedFilterRefImpl::computeCovGuide()
{
    static const int pY[] = { 0, 0, 1, 0, 1, 2 };
    static const int pX[] = { 0, 1, 1, 2, 2, 2 };

    int numOfIterations = (SQR(chNum) - chNum) / 2 + chNum;
    for (int k = 0; k < numOfIterations; ++k)
    {
        int i = pY[k], j = pX[k];

        vars[i][j] = channels[i].mul(channels[j]);
        meanFilter(vars[i][j], vars[i][j]);
        vars[i][j] -= exps[i].mul(exps[j]);

        if (i == j)
            vars[i][j] += eps * Mat::ones(height, width, CV_32F);
        else
            vars[j][i] = vars[i][j];
    }
}

void GuidedFilterRefImpl::computeCovGuideInv()
{
    static const int pY[] = { 0, 0, 1, 0, 1, 2 };
    static const int pX[] = { 0, 1, 1, 2, 2, 2 };

    int numOfIterations = (SQR(chNum) - chNum) / 2 + chNum;
    if (chNum == 3)
    {
        for (int k = 0; k < numOfIterations; ++k){
            int i = pY[k], i1 = (pY[k] + 1) % 3, i2 = (pY[k] + 2) % 3;
            int j = pX[k], j1 = (pX[k] + 1) % 3, j2 = (pX[k] + 2) % 3;

            A[i][j] = vars[i1][j1].mul(vars[i2][j2])
                - vars[i1][j2].mul(vars[i2][j1]);
        }
    }
    else if (chNum == 2)
    {
        A[0][0] = vars[1][1];
        A[1][1] = vars[0][0];
        A[0][1] = -vars[0][1];
    }
    else if (chNum == 1)
        A[0][0] = Mat::ones(height, width, CV_32F);

    for (int i = 0; i < chNum; ++i)
        for (int j = 0; j < i; ++j)
            A[i][j] = A[j][i];

    det = vars[0][0].mul(A[0][0]);
    for (int k = 0; k < chNum - 1; ++k)
        det += vars[0][k + 1].mul(A[0][k + 1]);
}

GuidedFilterRefImpl::~GuidedFilterRefImpl(){
    delete [] channels;
    delete [] exps;

    for (int i = 0; i < chNum; ++i)
    {
        delete [] A[i];
        delete [] vars[i];
    }

    delete [] A;
    delete [] vars;
}

void GuidedFilterRefImpl::filter(InputArray src_, OutputArray dst_, int dDepth)
{
    if (dDepth == -1) dDepth = src_.depth();
    dst_.create(height, width, src_.type());
    Mat src = src_.getMat();
    Mat dst = dst_.getMat();
    int cNum = src.channels();

    CV_Assert(height == src.rows && width == src.cols);

    Mat *Ichannels, *exp_I, **vars_I, **alpha, *beta;
    Ichannels = new Mat[cNum];
    exp_I     = new Mat[cNum];
    beta      = new Mat[cNum];

    vars_I = new Mat *[chNum];
    alpha  = new Mat *[chNum];
    for (int i = 0; i < chNum; ++i){
        vars_I[i] = new Mat[cNum];
        alpha[i]  = new Mat[cNum];
    }

    split(src, Ichannels);
    for (int i = 0; i < cNum; ++i)
    {
        Ichannels[i].convertTo(Ichannels[i], CV_32F);
        meanFilter(Ichannels[i], exp_I[i]);
    }

    computeCovGuideAndSrc(cNum, vars_I, Ichannels, exp_I);

    computeAlpha(cNum, alpha, vars_I);

    computeBeta(cNum, beta, exp_I, alpha);

    for (int i = 0; i < chNum + 1; ++i)
        for (int j = 0; j < cNum; ++j)
            if (i < chNum)
                meanFilter(alpha[i][j], alpha[i][j]);
            else
                meanFilter(beta[j], beta[j]);

    applyTransform(cNum, Ichannels, beta, alpha, dDepth);
    merge(Ichannels, cNum, dst);

    delete [] Ichannels;
    delete [] exp_I;
    delete [] beta;

    for (int i = 0; i < chNum; ++i)
    {
        delete [] vars_I[i];
        delete [] alpha[i];
    }
    delete [] vars_I;
    delete [] alpha;
}

void GuidedFilterRefImpl::computeAlpha(int cNum, Mat **alpha, Mat **vars_I)
{
    for (int i = 0; i < chNum; ++i)
        for (int j = 0; j < cNum; ++j)
        {
            alpha[i][j] = vars_I[0][j].mul(A[i][0]);
            for (int k = 1; k < chNum; ++k)
                alpha[i][j] += vars_I[k][j].mul(A[i][k]);
            alpha[i][j] /= det;
        }
}

void GuidedFilterRefImpl::computeBeta(int cNum, Mat *beta, Mat *exp_I, Mat **alpha)
{
    for (int i = 0; i < cNum; ++i)
    {
        beta[i] = exp_I[i];
        for (int j = 0; j < chNum; ++j)
            beta[i] -= alpha[j][i].mul(exps[j]);
    }
}

void GuidedFilterRefImpl::computeCovGuideAndSrc(int cNum, Mat **vars_I, Mat *Ichannels, Mat *exp_I)
{
    for (int i = 0; i < chNum; ++i)
        for (int j = 0; j < cNum; ++j)
        {
            vars_I[i][j] = channels[i].mul(Ichannels[j]);
            meanFilter(vars_I[i][j], vars_I[i][j]);
            vars_I[i][j] -= exp_I[j].mul(exps[i]);
        }
}

void GuidedFilterRefImpl::applyTransform(int cNum, Mat *Ichannels, Mat *beta, Mat **alpha, int dDepth)
{
    for (int i = 0; i < cNum; ++i)
    {
        Ichannels[i] = beta[i];
        for (int j = 0; j < chNum; ++j)
            Ichannels[i] += alpha[j][i].mul(channels[j]);
        Ichannels[i].convertTo(Ichannels[i], dDepth);
    }
}

typedef tuple<int, string, string> GFParams;
typedef TestWithParam<GFParams> GuidedFilterTest;

TEST_P(GuidedFilterTest, accuracy)
{
    GFParams params = GetParam();

    int guideCnNum = 3;
    int srcCnNum = get<0>(params);

    string guideFileName = get<1>(params);
    string srcFileName = get<2>(params);

    int seed = 100 * guideCnNum + 50 * srcCnNum + 5*(int)guideFileName.length() + (int)srcFileName.length();
    RNG rng(seed);

    Mat guide = imread(getOpenCVExtraDir() + guideFileName);
    Mat src = imread(getOpenCVExtraDir() + srcFileName);
    ASSERT_TRUE(!guide.empty() && !src.empty());

    Size dstSize(guide.cols + 1 + rng.uniform(0, 3), guide.rows);

    guide = convertTypeAndSize(guide, CV_MAKE_TYPE(guide.depth(), guideCnNum), dstSize);
    src = convertTypeAndSize(src, CV_MAKE_TYPE(src.depth(), srcCnNum), dstSize);

    for (int iter = 0; iter < 2; iter++)
    {
        int radius = rng.uniform(0, 50);
        double eps = rng.uniform(0.0, SQR(255.0));

        cv::setNumThreads(cv::getNumberOfCPUs());
        Mat res;
        Ptr<GuidedFilter> gf = createGuidedFilter(guide, radius, eps);
        gf->filter(src, res);

        cv::setNumThreads(1);
        Mat resRef;
        Ptr<GuidedFilter> gfRef(new GuidedFilterRefImpl(guide, radius, eps));
        gfRef->filter(src, resRef);

        double normInf = cv::norm(res, resRef, NORM_INF);
        double normL2 = cv::norm(res, resRef, NORM_L2) / guide.total();

        EXPECT_LE(normInf, 1.0);
        EXPECT_LE(normL2, 1.0/64.0);
    }
}

TEST_P(GuidedFilterTest, smallParamsIssue)
{
    GFParams params = GetParam();
    string guideFileName = get<1>(params);
    string srcFileName = get<2>(params);
    int guideCnNum = 3;
    int srcCnNum = get<0>(params);

    Mat guide = imread(getOpenCVExtraDir() + guideFileName);
    Mat src = imread(getOpenCVExtraDir() + srcFileName);
    ASSERT_TRUE(!guide.empty() && !src.empty());

    Size dstSize(guide.cols, guide.rows);
    guide = convertTypeAndSize(guide, CV_MAKE_TYPE(guide.depth(), guideCnNum), dstSize);
    src = convertTypeAndSize(src, CV_MAKE_TYPE(src.depth(), srcCnNum), dstSize);
    Mat output;

    ximgproc::guidedFilter(guide, src, output, 3, 1e-6);

    size_t whitePixels = 0;
    for(int i = 0; i < output.cols; i++)
    {
        for(int j = 0; j < output.rows; j++)
        {
            if(output.channels() == 1)
            {
                if(output.ptr<uchar>(i)[j] == 255)
                    whitePixels++;
            }
            else if(output.channels() == 3)
            {
                Vec3b currentPixel = output.ptr<Vec3b>(i)[j];
                if(currentPixel == Vec3b(255, 255, 255))
                    whitePixels++;
            }
        }
    }
    double whiteRate = whitePixels / (double) output.total();
    EXPECT_LE(whiteRate, 0.1);
}

INSTANTIATE_TEST_CASE_P(TypicalSet, GuidedFilterTest,
    Combine(
    Values(1, 3),
    Values("cv/shared/lena.png", "cv/shared/baboon.png"),
    Values("cv/shared/lena.png", "cv/shared/baboon.png")
));

}
