// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

Mat testOilPainting(Mat imgSrc, int halfSize, int dynRatio, int colorSpace)
{
    vector<int> histogramme;
    vector<Vec3f> moyenneRGB;
    Mat dst(imgSrc.size(), imgSrc.type());
    Mat lum;
    if (imgSrc.channels() != 1)
    {
        cvtColor(imgSrc, lum, colorSpace);
        if (lum.channels() > 1)
        {
            extractChannel(lum, lum, 0);
        }
    }
    else
        lum = imgSrc.clone();
    lum = lum / dynRatio;
    if (dst.channels() == 3)
        for (int y = 0; y < imgSrc.rows; y++)
        {
            Vec3b *vDst = dst.ptr<Vec3b>(y);
            for (int x = 0; x < imgSrc.cols; x++, vDst++) //for each pixel
            {
                Mat mask(lum.size(), CV_8UC1, Scalar::all(0));
                Rect r(Point(x - halfSize, y - halfSize), Size(2 * halfSize + 1, 2 * halfSize + 1));
                r = r & Rect(Point(0, 0), lum.size());
                mask(r).setTo(255);
                int histSize[] = { 256 };
                float hranges[] = { 0, 256 };
                const float* ranges[] = { hranges };
                Mat hist;
                int channels[] = { 0 };
                calcHist(&lum, 1, channels, mask, hist, 1, histSize, ranges, true, false);
                double maxVal = 0;
                Point pMin, pMax;
                minMaxLoc(hist, 0, &maxVal, &pMin, &pMax);
                mask.setTo(0, lum != static_cast<int>(pMax.y));
                Scalar v = mean(imgSrc, mask);
                *vDst = Vec3b(static_cast<uchar>(v[0]), static_cast<uchar>(v[1]), static_cast<uchar>(v[2]));
            }
        }
    else
        for (int y = 0; y < imgSrc.rows; y++)
        {
            uchar *vDst = dst.ptr<uchar>(y);
            for (int x = 0; x < imgSrc.cols; x++, vDst++) //for each pixel
            {
                Mat mask(lum.size(), CV_8UC1, Scalar::all(0));
                Rect r(Point(x - halfSize, y - halfSize), Size(2 * halfSize + 1, 2 * halfSize + 1));
                r = r & Rect(Point(0, 0), lum.size());
                mask(r).setTo(255);
                int histSize[] = { 256 };
                float hranges[] = { 0, 256 };
                const float* ranges[] = { hranges };
                Mat hist;
                int channels[] = { 0 };
                calcHist(&lum, 1, channels, mask, hist, 1, histSize, ranges, true, false);
                double maxVal = 0;
                Point pMin, pMax;
                minMaxLoc(hist, 0, &maxVal, &pMin, &pMax);
                mask.setTo(0, lum != static_cast<int>(pMax.y));
                Scalar v = mean(imgSrc, mask);
                *vDst = static_cast<uchar>(v[0]);
            }
        }
    return dst;
}

TEST(xphoto_oil_painting, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "cv/inpaint/";
    Mat orig = imread(folder+"exp1.png", IMREAD_COLOR);
    ASSERT_TRUE(!orig.empty());
    resize(orig, orig, Size(100, 100));
    Mat dst1, dst2, dd;
    xphoto::oilPainting(orig, dst1, 3, 5, COLOR_BGR2GRAY);
    dst2 = testOilPainting(orig, 3, 5, COLOR_BGR2GRAY);
    absdiff(dst1, dst2, dd);
    vector<Mat> plane;
    split(dd, plane);
    for (auto p : plane)
    {
        double maxVal;
        Point pIdx;
        minMaxLoc(p, NULL, &maxVal, NULL, &pIdx);
        ASSERT_LE(p.at<uchar>(pIdx), 2);
    }
    Mat orig2 = imread(folder + "exp1.png",IMREAD_GRAYSCALE);
    ASSERT_TRUE(!orig2.empty());
    resize(orig2, orig2, Size(100, 100));
    Mat dst3, dst4, ddd;
    xphoto::oilPainting(orig2, dst3, 3, 5, COLOR_BGR2GRAY);
    dst4 = testOilPainting(orig2, 3, 5, COLOR_BGR2GRAY);
    absdiff(dst3, dst4, ddd);
    double maxVal;
    Point pIdx;
    minMaxLoc(ddd, NULL, &maxVal, NULL, &pIdx);
    ASSERT_LE(ddd.at<uchar>(pIdx), 2);
}

}} // namespace
