// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ximgproc/sparse_table_morphology.hpp"
#include <vector>

namespace opencv_test {
namespace {

void assertArraysIdentical(InputArray ary1, InputArray ary2)
{
    Mat xormat = ary1.getMat() ^ ary2.getMat();
    CV_Assert(cv::countNonZero(xormat.reshape(1)) == 0);
}
Mat im(int type)
{
    int depth = CV_MAT_DEPTH(type);
    int ch = CV_MAT_CN(type);
    Mat img = imread(cvtest::TS::ptr()->get_data_path() + "cv/shared/lena.png");
    CV_Assert(img.type() == CV_8UC3);

    if (ch == 1) cv::cvtColor(img, img, ColorConversionCodes::COLOR_BGR2GRAY, ch);
    if (depth == CV_8S) img /= 2;
    img.convertTo(img, depth);
    if (depth == CV_16S) img *= (1 << 7);
    if (depth == CV_16U) img *= (1 << 8);
    if (depth == CV_32S) img *= (1 << 23);
    if (depth == CV_32F) img /= (1 << 8);
    if (depth == CV_64F) img /= (1 << 8);

    return img;
}
Mat kn4() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(4, 4)); }
Mat kn5() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(5, 5)); }
Mat knBig() { return getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(201, 201)); }
Mat kn1Zero() { return Mat::zeros(1, 1, CV_8UC1); }
Mat kn1One() { return Mat::ones(1, 1, CV_8UC1); }
Mat knEmpty() { return Mat(); }
Mat knZeros() { return Mat::zeros(5, 5, CV_8UC1); }
Mat knOnes() { return Mat::ones(5, 5, CV_8UC1); }
Mat knAsymm (){ return (Mat_<uchar>(5, 5) << 0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0); }
Mat knRnd(int size, int density)
{
    Mat rndMat(size, size, CV_8UC1);
    theRNG().state = getTickCount();
    randu(rndMat, 2, 102);
    density++;
    rndMat.setTo(0, density < rndMat);
    rndMat.setTo(1, 1 < rndMat);
    return rndMat;
}

/*
* dilate regression tests.
*/
void dilate_rgr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT,
    const Scalar& bdrVal = morphologyDefaultBorderValue())
{
    Mat expected, actual;
    dilate(src, expected, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::dilate(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_dilate, regression_8UC1) { dilate_rgr(im(CV_8UC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_8UC3) { dilate_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16UC1) { dilate_rgr(im(CV_16UC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16UC3) { dilate_rgr(im(CV_16UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16SC1) { dilate_rgr(im(CV_16SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16SC3) { dilate_rgr(im(CV_16SC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_32FC1) { dilate_rgr(im(CV_32FC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_32FC3) { dilate_rgr(im(CV_32FC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_64FC1) { dilate_rgr(im(CV_64FC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_64FC3) { dilate_rgr(im(CV_64FC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_kn5) { dilate_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_kn4) { dilate_rgr(im(CV_8UC3), kn4()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_kn1Zero) { dilate_rgr(im(CV_8UC3), kn1Zero()); }
TEST(ximgproc_StMorph_dilate, regression_kn1One) { dilate_rgr(im(CV_8UC3), kn1One()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_knEmpty) { dilate_rgr(im(CV_8UC3), knEmpty()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_knZeros) { dilate_rgr(im(CV_8UC3), knZeros()); }
TEST(ximgproc_StMorph_dilate, regression_knOnes) { dilate_rgr(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_dilate, regression_knBig) { dilate_rgr(im(CV_8UC3), knBig()); }
TEST(ximgproc_StMorph_dilate, regression_knAsymm) { dilate_rgr(im(CV_8UC3), knAsymm()); }
TEST(ximgproc_StMorph_dilate, regression_ancMid) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1)); }
TEST(ximgproc_StMorph_dilate, regression_ancEdge1) { dilate_rgr(im(CV_8UC3), kn5(), Point(0, 0)); }
TEST(ximgproc_StMorph_dilate, regression_ancEdge2) { dilate_rgr(im(CV_8UC3), kn5(), Point(4, 4)); }
TEST(ximgproc_StMorph_dilate, wtf_regression_it0) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 0); }
TEST(ximgproc_StMorph_dilate, regression_it1) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 1); }
TEST(ximgproc_StMorph_dilate, regression_it2) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 2); }
/*
* dilate feature tests.
*/
void dilate_ftr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT,
    const Scalar& bdrVal = morphologyDefaultBorderValue())
{
    Mat expected, actual;
    stMorph::dilate(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    // todo: generate expected result.
    // assertArraysIdentical(expected, actual);
}
/* CV_8S, CV_16F are not supported by morph.simd::getMorphologyFilter */
TEST(ximgproc_StMorph_dilate, feature_8SC1) { dilate_ftr(im(CV_8SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_8SC3) { dilate_ftr(im(CV_8SC3), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_32SC1) { dilate_ftr(im(CV_32SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_32SC3) { dilate_ftr(im(CV_32SC3), kn5()); }

/*
* erode regression tests.
*/
void erode_rgr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT,
    const Scalar& bdrVal = morphologyDefaultBorderValue())
{
    Mat expected, actual;
    erode(src, expected, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::erode(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_erode, regression_8UC1) { erode_rgr(im(CV_8UC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_8UC3) { erode_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16UC1) { erode_rgr(im(CV_16UC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16UC3) { erode_rgr(im(CV_16UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16SC1) { erode_rgr(im(CV_16SC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16SC3) { erode_rgr(im(CV_16SC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_32FC1) { erode_rgr(im(CV_32FC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_32FC3) { erode_rgr(im(CV_32FC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_64FC1) { erode_rgr(im(CV_64FC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_64FC3) { erode_rgr(im(CV_64FC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_kn5) { erode_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_kn4) { erode_rgr(im(CV_8UC3), kn4()); }
TEST(ximgproc_StMorph_erode, wtf_regression_kn1Zero) { erode_rgr(im(CV_8UC3), kn1Zero()); }
TEST(ximgproc_StMorph_erode, regression_kn1One) { erode_rgr(im(CV_8UC3), kn1One()); }
TEST(ximgproc_StMorph_erode, wtf_regression_knEmpty) { erode_rgr(im(CV_8UC3), knEmpty()); }
TEST(ximgproc_StMorph_erode, wtf_regression_knZeros) { erode_rgr(im(CV_8UC3), knZeros()); }
TEST(ximgproc_StMorph_erode, regression_knOnes) { erode_rgr(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_erode, regression_knBig) { erode_rgr(im(CV_8UC3), knBig()); }
TEST(ximgproc_StMorph_erode, regression_knAsymm) { erode_rgr(im(CV_8UC3), knAsymm()); }
TEST(ximgproc_StMorph_erode, regression_ancMid) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1)); }
TEST(ximgproc_StMorph_erode, regression_ancEdge1) { erode_rgr(im(CV_8UC3), kn5(), Point(0, 0)); }
TEST(ximgproc_StMorph_erode, regression_ancEdge2) { erode_rgr(im(CV_8UC3), kn5(), Point(4, 4)); }
TEST(ximgproc_StMorph_erode, wtf_regression_it0) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 0); }
TEST(ximgproc_StMorph_erode, regression_it1) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 1); }
TEST(ximgproc_StMorph_erode, regression_it2) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 2); }
/*
* erode feature tests.
*/
void erode_ftr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT,
    const Scalar& bdrVal = morphologyDefaultBorderValue())
{
    Mat expected, actual;
    stMorph::erode(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    // todo: generate expected result.
    // assertArraysIdentical(expected, actual);
}
/* CV_8S, CV_16F are not supported by morph.simd::getMorphologyFilter */
TEST(ximgproc_StMorph_erode, feature_8SC1) { erode_ftr(im(CV_8SC1), kn5()); }
TEST(ximgproc_StMorph_erode, feature_8SC3) { erode_ftr(im(CV_8SC3), kn5()); }
TEST(ximgproc_StMorph_erode, feature_32SC1) { erode_ftr(im(CV_32SC1), kn5()); }
TEST(ximgproc_StMorph_erode, feature_32SC3) { erode_ftr(im(CV_32SC3), kn5()); }

/*
* morphologyEx regression tests.
*/
void ex_rgr(InputArray src, MorphTypes op, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT,
    const Scalar& bdrVal = morphologyDefaultBorderValue())
{
    Mat expected, actual;
    morphologyEx(src, expected, op, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::morphologyEx(src, actual, op, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_ex, regression_erode) { ex_rgr(im(CV_8UC3), MORPH_ERODE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_dilage) { ex_rgr(im(CV_8UC3), MORPH_DILATE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_open) { ex_rgr(im(CV_8UC3), MORPH_OPEN, kn5()); }
TEST(ximgproc_StMorph_ex, regression_close) { ex_rgr(im(CV_8UC3), MORPH_CLOSE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_gradient) { ex_rgr(im(CV_8UC3), MORPH_GRADIENT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_tophat) { ex_rgr(im(CV_8UC3), MORPH_TOPHAT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_blackhat) { ex_rgr(im(CV_8UC3), MORPH_BLACKHAT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_hitmiss)
{
    EXPECT_THROW( { ex_rgr(im(CV_8UC1), MORPH_HITMISS, kn5()); }, cv::Exception);
}

stMorph::kernelDecompInfo ftr_decomp(InputArray kernel)
{
    auto kdi = stMorph::decompKernel(kernel);
    Mat expected = kernel.getMat();
    Mat actual = Mat::zeros(kernel.size(), kernel.type());
    for (uint r = 0; r < kdi.stRects.size(); r++)
    {
        for (uint c = 0; c < kdi.stRects[r].size(); c++)
        {
            for (Point p : kdi.stRects[r][c])
            {
                Rect rect(p.x, p.y, 1 << c, 1 << r);
                actual(rect).setTo(1);
            }
        }
    }
    assertArraysIdentical(expected, actual);
    return kdi;
}
Mat VisualizeCovering(Mat& kernel, const stMorph::kernelDecompInfo& kdi)
{
    const int rate = 20;
    const int fluct = 3;
    const int colors = 20;
    resize(kernel * 255, kernel, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    cvtColor(kernel, kernel, cv::COLOR_GRAY2BGR);
    Scalar color[colors]{
        Scalar(255, 127, 127), Scalar(255, 127, 191), Scalar(255, 127, 255), Scalar(191, 127, 255),
        Scalar(127, 127, 255), Scalar(127, 191, 255), Scalar(127, 255, 255), Scalar(127, 255, 191),
        Scalar(127, 255, 127), Scalar(191, 255, 127), Scalar(255, 255, 127), Scalar(255, 191, 127)
    };
    int i = 0;
    for (int r = 0; r < kdi.rows; r++)
        cv::line(kernel, Point(0, r * rate), Point(kdi.cols * rate, r * rate), Scalar(0));
    for (int c = 0; c < kdi.cols; c++)
        cv::line(kernel, Point(c * rate, 0), Point(c * rate, kdi.rows * rate), Scalar(0));
    for (uint row = 0; row < kdi.stRects.size(); row++)
    {
        for (uint col = 0; col < kdi.stRects[row].size(); col++)
        {
            Size s(1 << col, 1 << row);
            for (Point p : kdi.stRects[row][col])
            {
                Rect rect(p, s);
                int l = (rect.x) * rate + i % fluct + 2;
                int t = (rect.y) * rate + i % fluct + 2;
                int r = (rect.x + rect.width) * rate - fluct + i % fluct - 1;
                int b = (rect.y + rect.height) * rate - fluct + i % fluct - 1;
                Point lt(l, t);
                Point lb(l, b);
                Point rb(r, b);
                Point rt(r, t);
                cv::line(kernel, lt, lb, color[i % colors], 1);
                cv::line(kernel, lb, rb, color[i % colors], 1);
                cv::line(kernel, rb, rt, color[i % colors], 1);
                cv::line(kernel, rt, lt, color[i % colors], 1);
                i++;
            }
        }
    }
    return kernel;
}
Mat VisualizePlanning(stMorph::kernelDecompInfo kdi)
{
    int g = 30;
    int rows = kdi.plan.rows;
    int cols = kdi.plan.cols;
    Scalar vCol = Scalar(20, 20, 255);
    Scalar eCol = Scalar(100, 100, 100);
    Mat m = Mat::zeros(rows * g, cols * g, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            Rect nodeRect(col * g + g / 2 - 5, row * g + g / 2 - 5, 11, 11);
            if (kdi.stRects[row][col].size() > 0)
                cv::rectangle(m, nodeRect, vCol, -1);
            else
                cv::rectangle(m, nodeRect, vCol, 1);
        }
    }
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            Vec2b p = kdi.plan.at<Vec2b>(r, c);
            Point sp(c * g + g / 2, r * g + g / 2);
            if (p[0] == 1)
            {
                Point ep = Point(c * g + g / 2, (r + 1) * g + g / 2);
                cv::line(m, sp, ep, eCol, 2);
            }
            if (p[1] == 1)
            {
                Point ep = Point((c + 1) * g + g / 2, r * g + g / 2);
                cv::line(m, sp, ep, eCol, 2);
            }
        }
    }
    return m;
}
TEST(ximgproc_StMorph_decomp, feature_rnd1) { ftr_decomp(knRnd(1000, 1)); }
TEST(ximgproc_StMorph_decomp, feature_rnd10) { ftr_decomp(knRnd(1000, 10)); }
TEST(ximgproc_StMorph_decomp, feature_rnd30) { ftr_decomp(knRnd(1000, 30)); }
TEST(ximgproc_StMorph_decomp, feature_rnd50) { ftr_decomp(knRnd(1000, 50)); }
TEST(ximgproc_StMorph_decomp, feature_rnd80) { ftr_decomp(knRnd(1000, 80)); }
TEST(ximgproc_StMorph_decomp, feature_rnd90) { ftr_decomp(knRnd(1000, 90)); }
TEST(ximgproc_StMorph_decomp, feature_visualize) {
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    auto kdi = ftr_decomp(kernel);
    Mat covering = VisualizeCovering(kernel, kdi);
    Mat plan = VisualizePlanning(kdi);
#if 0
    imshow("Covering", covering);
    imshow("Plan", plan);
    waitKey();
    destroyAllWindows();
#endif
}

TEST(ximgproc_StMorph_eval, pdi)
{
    Mat img = im(CV_8UC3);
    Mat dst;
    int sizes[]{ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 45, 51, 55, 61, 71,
                81, 91, 101, 121, 151, 171, 201, 221, 251, 301, 351, 401, 451, 501 };

    std::ofstream ss("opencvlog_pdi.txt", std::ios_base::out);

    for (int c = 0; c < 3; c++)
    for (int i: sizes)
    {
        ss << i;
        Size sz(i, i);
        cv::TickMeter meter;
        Mat kn;
        stMorph::kernelDecompInfo kdi;

        // cv-rect
        kn = getStructuringElement(MORPH_RECT, sz);
        if (i <= 401)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // cv-cross
        kn = getStructuringElement(MORPH_CROSS, sz);
        if (i <= 401)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // cv-ellipse
        kn = getStructuringElement(MORPH_ELLIPSE, sz);
        if (i <= 23)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // st-rect
        kn = getStructuringElement(MORPH_RECT, sz);
        kdi = stMorph::decompKernel(kn);
        meter.start();
        stMorph::erode(img, dst, kdi);
        meter.stop();
        ss << "\t" << meter.getTimeMilli();
        meter.reset();

        // st-cross
        kn = getStructuringElement(MORPH_CROSS, sz);
        kdi = stMorph::decompKernel(kn);
        meter.start();
        stMorph::erode(img, dst, kdi);
        meter.stop();
        ss << "\t" << meter.getTimeMilli();
        meter.reset();

        // st-ellipse
        kn = getStructuringElement(MORPH_ELLIPSE, sz);
        kdi = stMorph::decompKernel(kn);
        meter.start();
        stMorph::erode(img, dst, kdi);
        meter.stop();
        ss << "\t" << meter.getTimeMilli() << "\n";
        meter.reset();
    }
    ss.close();
}

TEST(ximgproc_StMorph_eval, integrated)
{
    Mat img = im(CV_8UC3);
    Mat dst;
    int sizes[]{ 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 35, 41, 45, 51, 55, 61, 71,
                81, 91, 101, 121, 151, 171, 201, 221, 251, 301, 351, 401, 451, 501 };

    std::ofstream ss("opencvlog_integrated.txt", std::ios_base::out);

    for (int c = 0; c < 3; c++)
    for (int i: sizes)
    {
        ss << i;
        Size sz(i, i);
        cv::TickMeter meter;
        Mat kn;

        // cv-rect
        kn = getStructuringElement(MORPH_RECT, sz);
        if (i <= 401)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // cv-cross
        kn = getStructuringElement(MORPH_CROSS, sz);
        if (i <= 401)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // cv-ellipse
        kn = getStructuringElement(MORPH_ELLIPSE, sz);
        if (i <= 23)
        {
            meter.start();
            cv::erode(img, dst, kn);
            meter.stop();
            ss << "\t" << meter.getTimeMilli();
            meter.reset();
        }
        else
        {
            ss << "\t";
        }

        // st-rect
        kn = getStructuringElement(MORPH_RECT, sz);
        meter.start();
        stMorph::erode(img, dst, kn);
        meter.stop();
        ss << "\t" << meter.getTimeMilli();
        meter.reset();

        // st-cross
        kn = getStructuringElement(MORPH_CROSS, sz);
        meter.start();
        stMorph::erode(img, dst, kn);
        meter.stop();
        ss << "\t" << meter.getTimeMilli();
        meter.reset();

        // st-ellipse
        kn = getStructuringElement(MORPH_ELLIPSE, sz);
        meter.start();
        stMorph::erode(img, dst, kn);
        meter.stop();
        ss << "\t" << meter.getTimeMilli() << "\n";
        meter.reset();
    }
    ss.close();
}

}} // opencv_test:: ::
