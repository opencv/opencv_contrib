// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv::ppf_match_3d;

static Mat generateAsymmetricEllipsoidPatch()
{
    const float a = 10.0f, b = 17.0f, c = 25.0f;
    std::vector<Vec6f> pts;
    for (int iu = 2; iu <= 14; ++iu)
    {
        float u = 0.15f + 1.15f * (float)iu / 16.0f;
        for (int iv = 1; iv <= 15; ++iv)
        {
            float v = 0.10f + 1.35f * (float)iv / 16.0f;
            float x = a * std::sin(u) * std::cos(v);
            float y = b * std::sin(u) * std::sin(v);
            float z = c * std::cos(u);

            Vec3f n(x / (a * a), y / (b * b), z / (c * c));
            n /= cv::norm(n);
            pts.push_back(Vec6f(x, y, z, n[0], n[1], n[2]));
        }
    }
    Mat pc((int)pts.size(), 6, CV_32FC1);
    for (size_t i = 0; i < pts.size(); ++i)
    {
        float* row = pc.ptr<float>((int)i);
        for (int k = 0; k < 6; ++k)
            row[k] = pts[i][k];
    }
    return pc;
}

static Matx44d makeTestPose(const Vec3d& rvec, const Vec3d& tvec)
{
    double theta = cv::norm(rvec);
    Matx33d R = Matx33d::eye();
    if (theta > 1e-12)
    {
        Vec3d k = rvec / theta;
        Matx33d K(0, -k[2], k[1],
                  k[2], 0, -k[0],
                  -k[1], k[0], 0);
        R = Matx33d::eye() + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
    }
    Matx44d T = Matx44d::eye();
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
            T(r, c) = R(r, c);
        T(r, 3) = tvec[r];
    }
    return T;
}

static Mat toNx1C6(const Mat& pcNx6)
{
    return pcNx6.clone().reshape(6, pcNx6.rows);
}

static Mat to1xNC6(const Mat& pcNx6)
{
    return pcNx6.clone().reshape(6, 1);
}

static Mat to3D_Nx1x6(const Mat& pcNx6)
{
    int sz[3] = { pcNx6.rows, 1, 6 };
    Mat out(3, sz, CV_32FC1);
    std::memcpy(out.data, pcNx6.data, pcNx6.total() * sizeof(float));
    return out;
}

static Mat toNonContiguousNx1C6(const Mat& pcNx6)
{
    Mat canvas(pcNx6.rows, 3, CV_32FC(6));
    canvas.reshape(1) = -123.0f;
    Mat roi = canvas.col(1);
    toNx1C6(pcNx6).copyTo(roi);
    CV_Assert(!roi.isContinuous());
    return roi;
}

static Mat toNonContiguousNx6C1(const Mat& pcNx6)
{
    Mat canvas(pcNx6.rows, 9, CV_32FC1, Scalar::all(-123.0));
    Mat roi = canvas.colRange(1, 7);
    pcNx6.copyTo(roi);
    CV_Assert(!roi.isContinuous());
    return roi;
}

TEST(SurfaceMatching_PointCloudShape, ppf_and_icp_shapes_issue_3981)
{
    Mat modelNx6 = generateAsymmetricEllipsoidPatch();
    Matx44d modelToScene = makeTestPose(Vec3d(0.06, -0.05, 0.08), Vec3d(1.2, -0.8, 1.5));
    Mat sceneNx6 = transformPCPose(modelNx6, modelToScene);

    PPF3DDetector detectorRef(0.08, 0.08);
    detectorRef.trainModel(modelNx6);
    std::vector<Pose3DPtr> posesRef;
    detectorRef.match(sceneNx6, posesRef, 0.2, 0.08);
    ASSERT_FALSE(posesRef.empty());

    ICP icp(50, 0.0001f, 2.5f, 3);
    double residualRef = -1.0;
    Matx44d poseICPRef = Matx44d::eye();
    ASSERT_EQ(0, icp.registerModelToScene(modelNx6, sceneNx6, residualRef, poseICPRef));
    EXPECT_NEAR(0.0, cv::norm(poseICPRef, modelToScene, NORM_INF), 1e-2);

    std::vector<Mat> modelVariants = {
        toNx1C6(modelNx6),
        to1xNC6(modelNx6),
        to3D_Nx1x6(modelNx6),
        toNonContiguousNx1C6(modelNx6),
        toNonContiguousNx6C1(modelNx6)
    };
    std::vector<Mat> sceneVariants = {
        toNx1C6(sceneNx6),
        to1xNC6(sceneNx6),
        to3D_Nx1x6(sceneNx6),
        toNonContiguousNx1C6(sceneNx6),
        toNonContiguousNx6C1(sceneNx6)
    };

    PPF3DDetector detectorReused(0.08, 0.08);
    for (size_t idx = 0; idx < modelVariants.size(); ++idx)
    {
        detectorReused.trainModel(modelVariants[idx]);
        std::vector<Pose3DPtr> poses;
        detectorReused.match(sceneVariants[idx], poses, 0.2, 0.08);
        ASSERT_EQ(posesRef.size(), poses.size()) << "variant=" << idx;
        EXPECT_NEAR(0.0, cv::norm(posesRef[0]->pose, poses[0]->pose, NORM_INF), 1e-5) << "variant=" << idx;

        double residual = -1.0;
        Matx44d poseICP = Matx44d::eye();
        ASSERT_EQ(0, icp.registerModelToScene(modelVariants[idx], sceneVariants[idx], residual, poseICP)) << "variant=" << idx;
        EXPECT_NEAR(residualRef, residual, 1e-5) << "variant=" << idx;
        EXPECT_NEAR(0.0, cv::norm(poseICPRef, poseICP, NORM_INF), 1e-5) << "variant=" << idx;

        std::vector<Pose3DPtr> posesForICP = { poses[0]->clone() };
        ASSERT_EQ(0, icp.registerModelToScene(modelVariants[idx], sceneVariants[idx], posesForICP)) << "variant=" << idx;
        EXPECT_NEAR(0.0, cv::norm(posesForICP[0]->pose, modelToScene, NORM_INF), 5e-2) << "variant=" << idx;
    }
}

TEST(SurfaceMatching_PointCloudShape, transformPCPose_Nx3)
{
    Mat pcNx6 = generateAsymmetricEllipsoidPatch();
    Mat pcNx3 = pcNx6.colRange(0, 3).clone();
    Matx44d pose = makeTestPose(Vec3d(0.2, -0.1, 0.15), Vec3d(3.0, -2.0, 4.0));

    Mat tf6 = transformPCPose(pcNx6, pose);
    Mat tf3 = transformPCPose(pcNx3, pose);
    EXPECT_EQ(Size(3, pcNx3.rows), tf3.size());
    EXPECT_LE(cv::norm(tf3, tf6.colRange(0, 3), NORM_INF), 1e-6);
}

TEST(SurfaceMatching_PointCloudShape, invalid_inputs_issue_3981)
{
    Mat validNx6 = generateAsymmetricEllipsoidPatch();
    Mat pcNx3 = validNx6.colRange(0, 3).clone();
    Mat pcNx1C3 = pcNx3.clone().reshape(3, pcNx3.rows);
    Mat pcNx4(20, 4, CV_32FC1, Scalar::all(1.0));
    Mat pc2x3C6(2, 3, CV_32FC(6));
    pc2x3C6.reshape(1) = 1.0f;
    Mat pc64F;
    validNx6.convertTo(pc64F, CV_64F);

    PPF3DDetector detector(0.08, 0.08);
    EXPECT_THROW(detector.trainModel(Mat()), cv::Exception);
    EXPECT_THROW(detector.trainModel(pc64F), cv::Exception);
    EXPECT_THROW(detector.trainModel(pcNx3), cv::Exception);
    EXPECT_THROW(detector.trainModel(pcNx1C3), cv::Exception);
    EXPECT_THROW(detector.trainModel(pcNx4), cv::Exception);
    EXPECT_THROW(detector.trainModel(pc2x3C6), cv::Exception);

    detector.trainModel(validNx6);
    std::vector<Pose3DPtr> poses;
    EXPECT_THROW(detector.match(Mat(), poses), cv::Exception);
    EXPECT_THROW(detector.match(pcNx3, poses), cv::Exception);
    EXPECT_THROW(detector.match(pcNx1C3, poses), cv::Exception);

    ICP icp;
    double residual = 0.0;
    Matx44d pose;
    EXPECT_THROW(icp.registerModelToScene(Mat(), validNx6, residual, pose), cv::Exception);
    EXPECT_THROW(icp.registerModelToScene(validNx6, Mat(), residual, pose), cv::Exception);
    EXPECT_THROW(icp.registerModelToScene(pcNx3, validNx6, residual, pose), cv::Exception);
    EXPECT_THROW(icp.registerModelToScene(validNx6, pcNx3, residual, pose), cv::Exception);
    EXPECT_THROW(icp.registerModelToScene(pc64F, validNx6, residual, pose), cv::Exception);
}

}} // namespace
