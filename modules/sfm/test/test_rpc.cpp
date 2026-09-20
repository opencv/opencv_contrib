// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Portions adapted from VXL, commit ecacd8cb08773d1dc100605e6d634706d34c9fc5.
// Original files and cases are identified next to the corresponding fixtures below.
//
// Copyright 2000-2013 VXL Contributors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of their
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#include "test_precomp.hpp"
#include <opencv2/sfm/rpc.hpp>

namespace opencv_test { namespace {

static RPCModel vxlFixture(bool inverse)
{
    // VXL core/vpgl/tests/test_rational_camera.cxx, "test rational camera projection
    // (generic)", and core/vpgl/algo/tests/test_backproject.cxx, at the commit above.
    // Both fixtures use these coefficients, but different latitude/line scales.
    // Adaptation: copy their scalar coefficients into the public OpenCV model.
    RPCModel model;
    model.order = RPC_ORDER_VXL;
    model.coefficients = Matx<double, 4, 20>::zeros();
    model.coefficients(0, 0) = 0.1;
    model.coefficients(0, 10) = 0.071;
    model.coefficients(0, 7) = 0.01;
    model.coefficients(0, 9) = 0.3;
    model.coefficients(0, 15) = 1.0;
    model.coefficients(0, 18) = 1.0;
    model.coefficients(0, 19) = 0.75;
    model.coefficients(1, 0) = 0.1;
    model.coefficients(1, 10) = 0.05;
    model.coefficients(1, 17) = 0.01;
    model.coefficients(1, 9) = 1.0;
    model.coefficients(1, 15) = 1.0;
    model.coefficients(1, 18) = 1.0;
    model.coefficients(1, 19) = 1.0;
    model.coefficients(2, 0) = 0.02;
    model.coefficients(2, 10) = 0.014;
    model.coefficients(2, 7) = 0.1;
    model.coefficients(2, 9) = 0.4;
    model.coefficients(2, 15) = 0.5;
    model.coefficients(2, 18) = 0.01;
    model.coefficients(2, 19) = 0.33;
    model.coefficients(3, 0) = 0.1;
    model.coefficients(3, 10) = 0.05;
    model.coefficients(3, 17) = 0.03;
    model.coefficients(3, 9) = 1.0;
    model.coefficients(3, 15) = 1.0;
    model.coefficients(3, 18) = 0.3;
    model.coefficients(3, 19) = 1.0;
    model.worldOffset = Vec3d(150, 100, 10);
    model.worldScale = Vec3d(50, inverse ? 120 : 125, 5);
    model.imageOffset = Vec2d(500, 200);
    model.imageScale = Vec2d(1000, inverse ? 400 : 500);
    return model;
}

static void expectInverse(const RPCModel& model, const Vec4d& plane,
                          const Point2d& image, const Point3d& initial,
                          const RPCLocalFrame* frame = 0, double tolerance = 1e-4)
{
    std::vector<Point2d> images(1, image);
    std::vector<Point3d> initials(1, initial);
    Mat points, success;
    backProjectPointsRPC(images, model, plane, initials, points, success, frame, tolerance);
    ASSERT_EQ(CV_64FC3, points.type());
    ASSERT_EQ(Size(1, 1), points.size());
    ASSERT_EQ(CV_8U, success.type());
    ASSERT_EQ(Size(1, 1), success.size());
    ASSERT_EQ(1, success.at<uchar>(0));
    const Vec3d point = points.at<Vec3d>(0);
    ASSERT_TRUE(std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2]));
    const double normalLength = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] +
                                          plane[2] * plane[2]);
    const double planeError = std::abs(plane[0] * point[0] + plane[1] * point[1] +
                                       plane[2] * point[2] + plane[3]) / normalLength;
    EXPECT_LE(planeError, 1e-9 * std::max(1.0, cv::norm(point)));
    Mat projected;
    projectPointsRPC(points, model, projected, frame);
    const Vec2d residual = projected.at<Vec2d>(0) - Vec2d(image.x, image.y);
    EXPECT_LE(cv::norm(residual), tolerance + 1e-9);
}

TEST(Sfm_RPC, defaultCameraAndNormalization)
{
    RPCModel model;
    std::vector<Point3d> points;
    points.push_back(Point3d(1, 2, 10));
    points.push_back(Point3d(-3, 4, -5));
    std::vector<Point2d> projected;
    projectPointsRPC(points, model, projected);
    ASSERT_EQ(points.size(), projected.size());
    EXPECT_EQ(Point2d(1, 2), projected[0]);
    EXPECT_EQ(Point2d(-3, 4), projected[1]);

    model.worldOffset = Vec3d(100, -20, 50);
    model.worldScale = Vec3d(10, 4, 2);
    model.imageOffset = Vec2d(12, -7);
    model.imageScale = Vec2d(300, 80);
    model.coefficients(1, 0) = 2;
    model.coefficients(3, 0) = 4;
    points.assign(1, Point3d(102.5, -23, 54));
    projectPointsRPC(points, model, projected);
    EXPECT_DOUBLE_EQ(49.5, projected[0].x);
    EXPECT_DOUBLE_EQ(-22, projected[0].y);
    // Scaling a numerator and its denominator together leaves the ratio unchanged.
    for (int i = 0; i < 20; ++i)
    {
        model.coefficients(0, i) *= -3;
        model.coefficients(1, i) *= -3;
    }
    projectPointsRPC(points, model, projected);
    EXPECT_DOUBLE_EQ(49.5, projected[0].x);
    EXPECT_DOUBLE_EQ(-22, projected[0].y);
}

TEST(Sfm_RPC, everyCoefficientInEveryOrder)
{
    // Independent exponent tables transcribe the documented polynomial definitions;
    // they do not share a permutation or evaluator with the implementation.
    const int exponents[3][20][3] = {
        {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1},
         {2,0,0}, {0,2,0}, {0,0,2}, {1,1,1}, {3,0,0}, {1,2,0}, {1,0,2},
         {2,1,0}, {0,3,0}, {0,1,2}, {2,0,1}, {0,2,1}, {0,0,3}},
        {{0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1},
         {1,1,1}, {2,0,0}, {0,2,0}, {0,0,2}, {3,0,0}, {2,1,0}, {2,0,1},
         {1,2,0}, {0,3,0}, {0,2,1}, {1,0,2}, {0,1,2}, {0,0,3}},
        {{3,0,0}, {2,1,0}, {2,0,1}, {2,0,0}, {1,2,0}, {1,1,1}, {1,1,0},
         {1,0,2}, {1,0,1}, {1,0,0}, {0,3,0}, {0,2,1}, {0,2,0}, {0,1,2},
         {0,1,1}, {0,1,0}, {0,0,3}, {0,0,2}, {0,0,1}, {0,0,0}}
    };
    const RPCCoefficientOrder orders[] = {
        RPC_ORDER_RPC00B, RPC_ORDER_RPC00A, RPC_ORDER_VXL
    };
    std::vector<Point3d> points;
    points.push_back(Point3d(0.2, -0.3, 0.7));
    points.push_back(Point3d(-0.6, 0.8, -0.4));
    for (int order = 0; order < 3; ++order)
    {
        for (int term = 0; term < 20; ++term)
        {
            SCOPED_TRACE(testing::Message() << "order=" << order << ", term=" << term);
            RPCModel model;
            model.order = orders[order];
            model.coefficients = Matx<double, 4, 20>::zeros();
            const int constant = order == 2 ? 19 : 0;
            model.coefficients(0, term) = 1;
            model.coefficients(1, constant) = 1;
            model.coefficients(2, constant) = 1;
            model.coefficients(3, term) = 1;
            std::vector<Point2d> projected;
            projectPointsRPC(points, model, projected);
            ASSERT_EQ(points.size(), projected.size());
            for (size_t i = 0; i < points.size(); ++i)
            {
                const double value = std::pow(points[i].x, exponents[order][term][0]) *
                                     std::pow(points[i].y, exponents[order][term][1]) *
                                     std::pow(points[i].z, exponents[order][term][2]);
                EXPECT_NEAR(value, projected[i].x, 1e-14);
                EXPECT_NEAR(1 / value, projected[i].y, 1e-12);
            }
        }
    }
}

TEST(Sfm_RPC, upstreamVXLProjectionFixture)
{
    // Adapted from VXL core/vpgl/tests/test_rational_camera.cxx,
    // "test rational camera projection (generic)", pinned above. Reuses all eight
    // input points and rounded reference values, preserving its 0.01-pixel tolerance;
    // replaces VXL wrapper-overload duplication with one batched OpenCV invocation.
    const Point3d input[] = {
        Point3d(150,100,10), Point3d(150,100,15), Point3d(150,225,10), Point3d(150,225,15),
        Point3d(200,100,10), Point3d(200,100,15), Point3d(200,225,10), Point3d(200,225,15)
    };
    const Point2d expected[] = {
        Point2d(1250,365), Point2d(1370.65,327.82), Point2d(1388.29,405.854),
        Point2d(1421.9,379.412), Point2d(1047.62,378.572), Point2d(1194.53,376.955),
        Point2d(1205.08,400.635), Point2d(1276.68,397.414)
    };
    std::vector<Point3d> points(input, input + 8);
    std::vector<Point2d> projected;
    projectPointsRPC(points, vxlFixture(false), projected);
    ASSERT_EQ(8u, projected.size());
    for (size_t i = 0; i < projected.size(); ++i)
    {
        EXPECT_NEAR(expected[i].x, projected[i].x, 0.01);
        EXPECT_NEAR(expected[i].y, projected[i].y, 0.01);
    }
}

TEST(Sfm_RPC, stridedInputsAndEmptyBatches)
{
    Mat storage = Mat::zeros(3, 7, CV_64F);
    Mat points = storage.colRange(1, 4);
    ASSERT_FALSE(points.isContinuous());
    for (int i = 0; i < points.rows; ++i)
    {
        points.at<double>(i, 0) = 3 - i;
        points.at<double>(i, 1) = -2 + 2 * i;
        points.at<double>(i, 2) = 10 + i;
    }
    Mat projected;
    projectPointsRPC(points, RPCModel(), projected);
    ASSERT_EQ(Size(1, 3), projected.size());
    ASSERT_EQ(CV_64FC2, projected.type());
    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(Vec2d(3 - i, -2 + 2 * i), projected.at<Vec2d>(i));

    Mat imageStorage = Mat::zeros(3, 5, CV_64F);
    Mat images = imageStorage.colRange(1, 3);
    projected.reshape(1).copyTo(images);
    ASSERT_FALSE(images.isContinuous());
    Mat recovered, success;
    backProjectPointsRPC(images, RPCModel(), Vec4d(0, 0, 1, -5), points, recovered, success);
    ASSERT_EQ(Size(1, 3), recovered.size());
    for (int i = 0; i < 3; ++i)
    {
        ASSERT_EQ(1, success.at<uchar>(i));
        EXPECT_EQ(Vec3d(3 - i, -2 + 2 * i, 5), recovered.at<Vec3d>(i));
    }

    std::vector<Point3d> noPoints;
    std::vector<Point2d> noImages;
    projectPointsRPC(noPoints, RPCModel(), projected);
    EXPECT_TRUE(projected.empty());
    backProjectPointsRPC(noImages, RPCModel(), Vec4d(0,0,1,0), noPoints, recovered, success);
    EXPECT_TRUE(recovered.empty());
    EXPECT_TRUE(success.empty());
    localToGeographicRPC(noPoints, RPCLocalFrame(), recovered);
    EXPECT_TRUE(recovered.empty());
    geographicToLocalRPC(noPoints, RPCLocalFrame(), recovered);
    EXPECT_TRUE(recovered.empty());
}

TEST(Sfm_RPC, rejectsMalformedModelsAndPoints)
{
    std::vector<Point3d> points(1, Point3d(1, 2, 3));
    Mat output;
    const double invalid[] = {0, -1, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()};
    for (size_t value = 0; value < sizeof(invalid) / sizeof(invalid[0]); ++value)
    {
        for (int component = 0; component < 5; ++component)
        {
            RPCModel model;
            if (component < 3)
                model.worldScale[component] = invalid[value];
            else
                model.imageScale[component - 3] = invalid[value];
            EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
        }
    }
    RPCModel model;
    model.order = static_cast<RPCCoefficientOrder>(99);
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
    model = RPCModel();
    model.coefficients(0, 17) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
    model = RPCModel();
    model.worldOffset[2] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
    model = RPCModel();
    model.imageOffset[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
    EXPECT_THROW(projectPointsRPC(Mat::zeros(2, 3, CV_32F), RPCModel(), output), cv::Exception);
    EXPECT_THROW(projectPointsRPC(Mat::zeros(2, 4, CV_64F), RPCModel(), output), cv::Exception);
    points[0].y = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(projectPointsRPC(points, RPCModel(), output), cv::Exception);
}

TEST(Sfm_RPC, projectionSingularities)
{
    RPCModel model;
    model.coefficients(1, 0) = 0;
    std::vector<Point3d> points(1, Point3d(1, 2, 3));
    Mat output;
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);

    model = RPCModel();
    model.imageScale[0] = std::numeric_limits<double>::max();
    points[0].x = 2;
    EXPECT_THROW(projectPointsRPC(points, model, output), cv::Exception);
}

TEST(Sfm_RPC, upstreamVXLBackprojectionFixture)
{
    // Adapted from VXL core/vpgl/algo/tests/test_backproject.cxx, pinned above:
    // "X-Y/Y-Z/X-Z plane backprojection convergence" and "arbitrary plane
    // backprojection convergence". Retains its camera, pixels, planes and seeds.
    // Unlike its root-distance assertions, check the geometric contract because
    // the source explicitly demonstrates more than one valid root.
    const RPCModel model = vxlFixture(true);
    const Point2d image(1250, 332);
    expectInverse(model, Vec4d(0, 0, 1, -10), image, Point3d(200, 150, 10));
    expectInverse(model, Vec4d(1, 0, 0, -150), image, Point3d(150, 150, 15));
    expectInverse(model, Vec4d(0, 1, 0, -100), image, Point3d(125, 100, 8));
    expectInverse(model, Vec4d(0, -1, 25, -150), Point2d(1199.1003963, 346.589238723),
                  Point3d(150, 100, 10));
}

TEST(Sfm_RPC, inversePlaneTiesAndScaleInvariance)
{
    RPCModel model;
    model.coefficients(0, 3) = 2;
    model.coefficients(2, 3) = 3;
    // Independent linear camera (x+2z, y+3z) sees (2,-3,4) at (10,9).
    // In particular, (1,1,0,1) cannot be handled by dividing by the z coefficient.
    const Vec4d planes[] = {
        Vec4d(1,0,0,-2), Vec4d(0,1,0,3), Vec4d(0,0,1,-4),
        Vec4d(1,1,0,1), Vec4d(1,1,1,-3), Vec4d(2,-1,3,-19)
    };
    for (size_t i = 0; i < sizeof(planes) / sizeof(planes[0]); ++i)
    {
        SCOPED_TRACE(testing::Message() << "plane=" << i);
        expectInverse(model, planes[i], Point2d(10, 9), Point3d(-1, 1, 0));
        expectInverse(model, planes[i] * -7, Point2d(10, 9), Point3d(-1, 1, 0));
    }
}

TEST(Sfm_RPC, inverseAlreadyCorrectAndFailures)
{
    RPCModel constant;
    constant.coefficients(0, 1) = 0;
    constant.coefficients(2, 2) = 0;
    std::vector<Point2d> images;
    images.push_back(Point2d(0, 0));
    images.push_back(Point2d(1, 2));
    std::vector<Point3d> initial(2, Point3d(7, -2, 3));
    Mat points, success;
    backProjectPointsRPC(images, constant, Vec4d(0,0,1,-3), initial, points, success);
    ASSERT_EQ(Size(1, 2), success.size());
    EXPECT_EQ(1, success.at<uchar>(0));
    EXPECT_EQ(Vec3d(7, -2, 3), points.at<Vec3d>(0));
    EXPECT_EQ(0, success.at<uchar>(1));
    for (int coordinate = 0; coordinate < 3; ++coordinate)
        EXPECT_TRUE(std::isnan(points.at<Vec3d>(1)[coordinate]));

    // An undefined polynomial ratio is a per-point inverse failure, not success
    // from a NaN comparison or an exception aborting the batch.
    constant.coefficients(1, 0) = 0;
    EXPECT_NO_THROW(backProjectPointsRPC(images, constant, Vec4d(0,0,1,-3),
                                         initial, points, success));
    for (int i = 0; i < 2; ++i)
    {
        EXPECT_EQ(0, success.at<uchar>(i));
        for (int coordinate = 0; coordinate < 3; ++coordinate)
            EXPECT_TRUE(std::isnan(points.at<Vec3d>(i)[coordinate]));
    }
}

TEST(Sfm_RPC, rejectsInvalidInverseArguments)
{
    const RPCModel model;
    const Vec4d plane(0, 0, 1, -3);
    std::vector<Point2d> images(1, Point2d(1, 2));
    std::vector<Point3d> initial(1, Point3d(1, 2, 3));
    Mat output, success;
    EXPECT_THROW(backProjectPointsRPC(images, model, Vec4d(0,0,0,1), initial,
                                      output, success), cv::Exception);
    EXPECT_THROW(backProjectPointsRPC(images, model,
                                      Vec4d(0,0,1,std::numeric_limits<double>::infinity()),
                                      initial, output, success), cv::Exception);
    const double invalid[] = {0, -1, std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::quiet_NaN()};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i)
        EXPECT_THROW(backProjectPointsRPC(images, model, plane, initial, output, success,
                                          0, invalid[i]), cv::Exception);
    EXPECT_THROW(backProjectPointsRPC(images, model, plane, initial, output, success,
                                      0, 0.05, 0), cv::Exception);
    EXPECT_THROW(backProjectPointsRPC(Mat::zeros(1,2,CV_32F), model, plane, initial,
                                      output, success), cv::Exception);
    EXPECT_THROW(backProjectPointsRPC(Mat::zeros(1,3,CV_64F), model, plane, initial,
                                      output, success), cv::Exception);
    std::vector<Point3d> tooMany(2, Point3d());
    EXPECT_THROW(backProjectPointsRPC(images, model, plane, tooMany, output, success),
                 cv::Exception);
    initial[0].x = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(backProjectPointsRPC(images, model, plane, initial, output, success),
                 cv::Exception);
    initial[0].x = 1;
    images[0].y = std::numeric_limits<double>::infinity();
    EXPECT_THROW(backProjectPointsRPC(images, model, plane, initial, output, success),
                 cv::Exception);
}

TEST(Sfm_RPC, localFrameComposition)
{
    const RPCLocalFrame frame(Point3d(-122, 37, 100), 0.4);
    std::vector<Point3d> local;
    local.push_back(Point3d(15, -25, 5));
    local.push_back(Point3d(-60, 40, -3));
    local.push_back(Point3d(0, 0, 0));
    std::vector<Point3d> geographic, roundtrip;
    localToGeographicRPC(local, frame, geographic);
    geographicToLocalRPC(geographic, frame, roundtrip);
    ASSERT_EQ(local.size(), roundtrip.size());
    for (size_t i = 0; i < local.size(); ++i)
        EXPECT_LE(cv::norm(local[i] - roundtrip[i]), 2e-8);

    RPCModel model;
    model.worldOffset = Vec3d(-122, 37, 100);
    model.worldScale = Vec3d(0.01, 0.01, 100);
    model.imageOffset = Vec2d(500, 800);
    model.imageScale = Vec2d(1000, 1200);
    model.coefficients(0, 3) = 0.1;
    model.coefficients(2, 3) = -0.2;
    std::vector<Point2d> direct, composed;
    projectPointsRPC(geographic, model, direct);
    projectPointsRPC(local, model, composed, &frame);
    ASSERT_EQ(direct.size(), composed.size());
    for (size_t i = 0; i < direct.size(); ++i)
        EXPECT_LE(cv::norm(direct[i] - composed[i]), 1e-12);
    expectInverse(model, Vec4d(0,0,1,-5), composed[0], Point3d(0,0,5), &frame);
    expectInverse(model, Vec4d(1,1,1,5), composed[0], Point3d(0,0,-5), &frame);
}

TEST(Sfm_RPC, inverseToleranceWithLargeImageScale)
{
    RPCModel model;
    model.worldOffset = Vec3d(-71, 42, 100);
    model.worldScale = Vec3d(0.05, 0.04, 500);
    model.imageOffset = Vec2d(4000, 3000);
    model.imageScale = Vec2d(4000, 3000);
    model.coefficients(0, 3) = 0.1;
    model.coefficients(2, 3) = -0.05;
    const RPCLocalFrame frame(Point3d(-71, 42, 100), CV_PI / 6);
    const std::vector<Point3d> ground = {Point3d(100, 200, 10), Point3d(-50, 25, 10)};
    std::vector<Point2d> images, projected;
    projectPointsRPC(ground, model, images, &frame);
    std::vector<Point3d> initial(2, Point3d(0, 0, 10)), recovered;
    std::vector<uchar> status;
    backProjectPointsRPC(images, model, Vec4d(0, 0, 1, -10), initial, recovered,
                         status, &frame, 1e-7);
    ASSERT_EQ(2u, status.size());
    ASSERT_EQ(1, status[0]);
    ASSERT_EQ(1, status[1]);
    projectPointsRPC(recovered, model, projected, &frame);
    for (size_t i = 0; i < 2; ++i)
    {
        EXPECT_LE(cv::norm(projected[i] - images[i]), 1e-7);
        EXPECT_EQ(10, recovered[i].z);
    }
}

}} // namespace opencv_test
