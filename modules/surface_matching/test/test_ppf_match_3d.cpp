// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

static Mat makeSyntheticBoxWithNormals()
{
    std::vector<Vec6f> pts;
    for (int u = -4; u <= 4; ++u)
    {
        for (int v = -3; v <= 3; ++v)
        {
            const float x = u * 0.1f;
            const float y = v * 0.1f;
            pts.push_back(Vec6f(x, y, 0.25f, 0.0f, 0.0f, 1.0f));
            pts.push_back(Vec6f(x, 0.35f, y * 0.7f, 0.0f, 1.0f, 0.0f));
            pts.push_back(Vec6f(0.45f, x * 0.7f, y * 0.7f, 1.0f, 0.0f, 0.0f));
        }
    }
    Mat pc(static_cast<int>(pts.size()), 6, CV_32FC1);
    for (size_t i = 0; i < pts.size(); ++i)
    {
        float* row = pc.ptr<float>(static_cast<int>(i));
        for (int c = 0; c < 6; ++c)
            row[c] = pts[i][c];
    }
    return pc;
}

TEST(SurfaceMatching_PPF3D, RetrainModelAndMatchPoseRecovery)
{
    Mat model = makeSyntheticBoxWithNormals();
    PPF3DDetector detector(0.1, 0.1, 24);

    // Train twice on the same detector instance to exercise clearTrainingModels() retrain path
    detector.trainModel(model);
    detector.trainModel(model);

    const Vec3f shift(0.3f, -0.2f, 0.5f);
    Mat scene = model.clone();
    for (int i = 0; i < scene.rows; ++i)
    {
        float* row = scene.ptr<float>(i);
        row[0] += shift[0];
        row[1] += shift[1];
        row[2] += shift[2];
    }

    std::vector<Pose3DPtr> results;
    detector.match(scene, results, 0.5, 0.1);
    ASSERT_FALSE(results.empty());
    ASSERT_FALSE(results[0].empty());
    EXPECT_GT(results[0]->numVotes, 0u);

    const Vec3d estT = results[0]->t;
    EXPECT_NEAR(shift[0], estT[0], 0.18);
    EXPECT_NEAR(shift[1], estT[1], 0.18);
    EXPECT_NEAR(shift[2], estT[2], 0.18);
}

}} // namespace
