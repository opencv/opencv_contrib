// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {


TEST(ximgproc_fourierdescriptors,test_FD_AND_FIT)
{
    Mat fd;
    vector<Point2f> ctr(16);
    float Rx = 100, Ry = 100;
    Point2f g(0, 0);
    float angleOri = 0;
    for (int i = 0; i < static_cast<int>(ctr.size()); i++)
    {
        float theta = static_cast<float>(2 * CV_PI / static_cast<int>(ctr.size()) * i + angleOri);
        ctr[i] = Point2f(Rx * cos(theta) + g.x, Ry * sin(theta) + g.y);

    }
    ximgproc::fourierDescriptor(ctr, fd);
    CV_Assert(cv::norm(fd.at<Vec2f>(0, 0)) < ctr.size() * FLT_EPSILON && cv::norm(fd.at<Vec2f>(0, 1) - Vec2f(Rx, 0)) < ctr.size() * FLT_EPSILON);
    Rx = 100, Ry = 50;
    g = Point2f(50, 20);
    for (int i = 0; i < static_cast<int>(ctr.size()); i++)
    {
        float theta = static_cast<float>(2 * CV_PI / static_cast<int>(ctr.size()) * i + angleOri);
        ctr[i] = Point2f(Rx * cos(theta) + g.x, Ry * sin(theta) + g.y);
    }
    ximgproc::fourierDescriptor(ctr, fd);
    CV_Assert(cv::norm(fd.at<Vec2f>(0, 0) - Vec2f(g)) < 1 &&
        fabs(fd.at<Vec2f>(0, 1)[0] + fd.at<Vec2f>(0, static_cast<int>(ctr.size()) - 1)[0] - Rx) < 1 &&
        fabs(fd.at<Vec2f>(0, 1)[0] - fd.at<Vec2f>(0, static_cast<int>(ctr.size()) - 1)[0] - Ry) < 1);
    Rx = 70, Ry = 100;
    g = Point2f(30, 100);
    angleOri = static_cast<float>(CV_PI / 4);
    for (int i = 0; i < static_cast<int>(ctr.size()); i++)
    {
        float theta = static_cast<float>(2 * CV_PI / static_cast<int>(ctr.size()) * i + CV_PI / 4);
        ctr[i] = Point2f(Rx * cos(theta) + g.x, Ry * sin(theta) + g.y);
    }
    ximgproc::fourierDescriptor(ctr, fd);
    CV_Assert(cv::norm(fd.at<Vec2f>(0, 0) - Vec2f(g)) < 1);
    CV_Assert(cv::norm(Vec2f((Rx + Ry)*cos(angleOri) / 2, (Rx + Ry)*sin(angleOri) / 2) - fd.at<Vec2f>(0, 1)) < 1);
    CV_Assert(cv::norm(Vec2f((Rx - Ry)*cos(angleOri) / 2, -(Rx - Ry)*sin(angleOri) / 2) - fd.at<Vec2f>(0, static_cast<int>(ctr.size()) - 1)) < 1);

    RNG rAlea;
    g.x = 0; g.y = 0;
    ctr.resize(256);
    for (int i = 0; i < static_cast<int>(ctr.size()); i++)
    {
        ctr[i] = Point2f(rAlea.uniform(0.0F, 1.0F), rAlea.uniform(0.0F, 1.0F));
        g += ctr[i];
    }
    g.x = g.x / ctr.size();
    g.y = g.y / ctr.size();
    double rotAngle = 35;
    double s = 0.1515;
    Mat r = getRotationMatrix2D(g, rotAngle, 0.1515);
    vector<Point2f> unknownCtr;
    vector<Point2f> ctrShift;
    int valShift = 170;
    for (int i = 0; i < static_cast<int>(ctr.size()); i++)
        ctrShift.push_back(ctr[(i + valShift) % ctr.size()]);
    cv::transform(ctrShift, unknownCtr, r);
    ximgproc::ContourFitting fit;
    fit.setFDSize(16);
    Mat t;
    double dist;
    fit.estimateTransformation(unknownCtr, ctr, t, &dist, false);
    CV_Assert(fabs(t.at<double>(0, 0)*ctr.size() + valShift) < 10 || fabs((1 - t.at<double>(0, 0))*ctr.size() - valShift) < 10);
    CV_Assert(fabs(t.at<double>(0, 1) - rotAngle / 180.*CV_PI) < 0.1);
    CV_Assert(fabs(t.at<double>(0, 2) - 1 / s) < 0.1);
    ctr.resize(4);
    ctr[0] = Point2f(0, 0);
    ctr[1] = Point2f(16, 0);
    ctr[2] = Point2f(16, 16);
    ctr[3] = Point2f(0, 16);
    double squareArea = contourArea(ctr), lengthSquare = arcLength(ctr, true);
    Mat ctrs;
    ximgproc::contourSampling(ctr, ctrs, 64);
    CV_Assert(fabs(squareArea - contourArea(ctrs)) < FLT_EPSILON);
    CV_Assert(fabs(lengthSquare - arcLength(ctrs, true)) < FLT_EPSILON);
}



}} // namespace
