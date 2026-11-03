// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <opencv2/sfm/rpc.hpp>
#include <cmath>
#include <limits>

namespace opencv_test { namespace {

TEST(Sfm_RPC_Local, equatorialScalesAndOrientation)
{
    // At the equator N=a and M=b^2/a. These independent one-degree controls
    // distinguish east/north radii, degrees/radians and ellipsoidal height.
    const double majorAxis = 6378137.0;
    const double minorAxis = majorAxis * (1.0 - 1.0 / 298.257223563);
    const double height = 250.0;
    const double eastDegree = (majorAxis + height) * (CV_PI / 180.0);
    const double northDegree = (minorAxis * minorAxis / majorAxis + height) * (CV_PI / 180.0);
    const RPCLocalFrame frame(Point3d(170, 0, height));
    const std::vector<Point3d> local = {Point3d(), Point3d(eastDegree, 0, 8),
                                     Point3d(0, northDegree, -13)};
    std::vector<Point3d> geographic;
    localToGeographicRPC(local, frame, geographic);
    ASSERT_EQ(local.size(), geographic.size());
    EXPECT_EQ(frame.origin, geographic[0]);
    EXPECT_NEAR(171, geographic[1].x, 1e-12);
    EXPECT_NEAR(0, geographic[1].y, 1e-12);
    EXPECT_EQ(height + 8, geographic[1].z);
    EXPECT_NEAR(170, geographic[2].x, 1e-12);
    EXPECT_NEAR(1, geographic[2].y, 1e-12);
    EXPECT_EQ(height - 13, geographic[2].z);

    const RPCLocalFrame rotated(frame.origin, CV_PI / 2);
    const std::vector<Point3d> localRotated = {Point3d(northDegree, 0, 0),
                                            Point3d(0, eastDegree, 0)};
    localToGeographicRPC(localRotated, rotated, geographic);
    EXPECT_NEAR(170, geographic[0].x, 1e-12);
    EXPECT_NEAR(1, geographic[0].y, 1e-12);
    EXPECT_NEAR(169, geographic[1].x, 1e-12);
    EXPECT_NEAR(0, geographic[1].y, 1e-12);
}

TEST(Sfm_RPC_Local, curvatureInBothHemispheres)
{
    // WGS84 curvature radii at +/-45 degrees, independently evaluated from
    // a=6378137 and inverse flattening=298.257223563, rounded to a nanometre.
    const double meridional45 = 6367381.815619548;
    const double primeVertical45 = 6388838.290121148;
    const Point3d origins[] = {Point3d(-71, 45, 1000), Point3d(151, -45, -20)};
    for (size_t i = 0; i < 2; ++i)
    {
        const RPCLocalFrame frame(origins[i]);
        const double eastDegree = (primeVertical45 + frame.origin.z) /
                                  std::sqrt(2.0) * (CV_PI / 180.0);
        const double northDegree = (meridional45 + frame.origin.z) * (CV_PI / 180.0);
        const std::vector<Point3d> local = {Point3d(eastDegree, northDegree, 17)};
        std::vector<Point3d> geographic;
        localToGeographicRPC(local, frame, geographic);
        ASSERT_EQ(1u, geographic.size());
        EXPECT_NEAR(frame.origin.x + 1, geographic[0].x, 1e-12);
        EXPECT_NEAR(frame.origin.y + 1, geographic[0].y, 1e-12);
        EXPECT_EQ(frame.origin.z + 17, geographic[0].z);
    }
}

TEST(Sfm_RPC_Local, inverseRotationsAndOrigins)
{
    const Point3d origins[] = {Point3d(-71.4049, 41.8216, -30),
                              Point3d(151.2, -33.5, 200), Point3d(180, 0, 0)};
    const double angles[] = {0, 1e-7, -1e-7, CV_PI / 6, -CV_PI / 4, CV_PI / 2};
    const std::vector<Point3d> local = {Point3d(), Point3d(202.47, 200, 50),
                                      Point3d(-150, 87.5, -12)};
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            const RPCLocalFrame frame(origins[i], angles[j]);
            std::vector<Point3d> geographic, recovered;
            localToGeographicRPC(local, frame, geographic);
            geographicToLocalRPC(geographic, frame, recovered);
            ASSERT_EQ(local.size(), recovered.size());
            for (size_t k = 0; k < local.size(); ++k)
                EXPECT_LE(cv::norm(recovered[k] - local[k]), 5e-8) << i << ", " << j << ", " << k;
        }
    }
    std::vector<Point3d> geographic;
    localToGeographicRPC(local, RPCLocalFrame(Point3d(180, 0, 0)), geographic);
    EXPECT_GT(geographic[1].x, 180); // Fixed local approximation does not wrap longitude.
}

TEST(Sfm_RPC_Local, stridedInputsAndEmptyOutputs)
{
    Mat storage(3, 7, CV_64F, Scalar(-999));
    Mat points = storage(Rect(2, 0, 3, 3));
    ASSERT_FALSE(points.isContinuous());
    for (int i = 0; i < points.rows; ++i)
    {
        points.at<double>(i, 0) = 11 + 7 * i;
        points.at<double>(i, 1) = -8 + i;
        points.at<double>(i, 2) = 4 - 2 * i;
    }
    const Mat untouched = storage.clone();
    const RPCLocalFrame frame(Point3d(12, -23, 17), -0.37);
    Mat geographic, expected;
    localToGeographicRPC(points, frame, geographic);
    localToGeographicRPC(points.clone(), frame, expected);
    EXPECT_EQ(CV_64FC3, geographic.type());
    EXPECT_EQ(3, geographic.rows);
    EXPECT_EQ(1, geographic.cols);
    EXPECT_EQ(0, cv::norm(geographic, expected, NORM_INF));
    EXPECT_EQ(0, cv::norm(storage, untouched, NORM_INF));

    Mat geographicStorage(3, 3, CV_64FC3, Scalar());
    Mat geographicView = geographicStorage.col(1);
    geographic.copyTo(geographicView);
    ASSERT_FALSE(geographicView.isContinuous());
    Mat recovered;
    geographicToLocalRPC(geographicView, frame, recovered);
    EXPECT_LT(cv::norm(recovered.reshape(1), points, NORM_INF), 5e-8);

    Mat rowPoints = points.clone().reshape(3, 1);
    localToGeographicRPC(rowPoints, frame, expected);
    EXPECT_EQ(0, cv::norm(geographic, expected, NORM_INF));

    Mat empty(0, 3, CV_64F);
    localToGeographicRPC(empty, frame, geographic);
    EXPECT_TRUE(geographic.empty());
    geographicToLocalRPC(std::vector<Point3d>(), frame, recovered);
    EXPECT_TRUE(recovered.empty());
}

TEST(Sfm_RPC_Local, invalidArguments)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    EXPECT_THROW(RPCLocalFrame(Point3d(nan, 0, 0)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(0, 0, infinity)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(181, 0, 0)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(-181, 0, 0)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(0, 90, 0)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(0, -90, 0)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(0, 0, -6378137)), cv::Exception);
    EXPECT_THROW(RPCLocalFrame(Point3d(), nan), cv::Exception);

    RPCLocalFrame frame;
    Mat output, valid = Mat::zeros(1, 3, CV_64F);
    EXPECT_THROW(localToGeographicRPC(Mat::zeros(2, 3, CV_32F), frame, output), cv::Exception);
    EXPECT_THROW(geographicToLocalRPC(Mat::zeros(2, 2, CV_64F), frame, output), cv::Exception);
    Mat invalid = valid.clone();
    invalid.at<double>(0, 0) = nan;
    EXPECT_THROW(localToGeographicRPC(invalid, frame, output), cv::Exception);
    invalid.at<double>(0, 0) = infinity;
    EXPECT_THROW(geographicToLocalRPC(invalid, frame, output), cv::Exception);
    frame.origin.y = 90; // Public fields must be validated at use, not only construction.
    EXPECT_THROW(localToGeographicRPC(valid, frame, output), cv::Exception);
    EXPECT_THROW(geographicToLocalRPC(valid, frame, output), cv::Exception);
}

TEST(Sfm_RPC_Local, analyticInverseAtLargeFiniteHeight)
{
    // The product of angular scales underflows here; inversion must not depend
    // on a generic 3-by-3 determinant. Only horizontal displacements are used
    // because adding a metre to this height cannot be represented in a double.
    const RPCLocalFrame frame(Point3d(0, 0, 1e200), 0.37);
    const std::vector<Point3d> local = {Point3d(1, -2, 0)};
    std::vector<Point3d> geographic, recovered;
    localToGeographicRPC(local, frame, geographic);
    geographicToLocalRPC(geographic, frame, recovered);
    ASSERT_EQ(1u, recovered.size());
    EXPECT_NEAR(local[0].x, recovered[0].x, 1e-14);
    EXPECT_NEAR(local[0].y, recovered[0].y, 1e-14);
    EXPECT_EQ(0, recovered[0].z);
}

TEST(Sfm_RPC_Local, outputUnaffectedByLateInvalidPoint)
{
    const RPCLocalFrame frame;
    Mat points = Mat::zeros(2, 3, CV_64F);
    points.at<double>(1, 0) = std::numeric_limits<double>::quiet_NaN();
    Mat output(2, 1, CV_64FC3, Scalar(17, 23, 42));
    const Mat original = output.clone();
    EXPECT_THROW(localToGeographicRPC(points, frame, output), cv::Exception);
    EXPECT_EQ(0, cv::norm(output, original, NORM_INF));
    EXPECT_THROW(geographicToLocalRPC(points, frame, output), cv::Exception);
    EXPECT_EQ(0, cv::norm(output, original, NORM_INF));
}

}} // namespace opencv_test
