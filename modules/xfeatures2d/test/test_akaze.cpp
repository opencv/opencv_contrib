// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Features2d_AKAZE, detect_and_compute_split)
{
    Mat testImg(100, 100, CV_8U);
    RNG rng(101);
    rng.fill(testImg, RNG::UNIFORM, Scalar(0), Scalar(255), true);

    Ptr<Feature2D> ext = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.001f, 1, 1, KAZE::DIFF_PM_G2);
    vector<KeyPoint> detAndCompKps;
    Mat desc;
    ext->detectAndCompute(testImg, noArray(), detAndCompKps, desc);

    vector<KeyPoint> detKps;
    ext->detect(testImg, detKps);

    ASSERT_EQ(detKps.size(), detAndCompKps.size());

    for(size_t i = 0; i < detKps.size(); i++)
        ASSERT_EQ(detKps[i].hash(), detAndCompKps[i].hash());
}

TEST(Features2d_AKAZE, compute_preserves_given_keypoints_with_max_points)
{
    Mat img(160, 160, CV_8UC1);
    theRNG().fill(img, RNG::UNIFORM, 20, 90);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            if ((r + c) % 2 == 0)
                img(Rect(c * 20 + 3, r * 20 + 3, 14, 14)).setTo(Scalar(220));

    Ptr<AKAZE> unlimited = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 2, 2);
    std::vector<KeyPoint> detected;
    unlimited->detect(img, detected);
    ASSERT_GT(detected.size(), 25u);

    // Order ascending by response (opposite of max_points descending sort) so reordering is caught
    std::sort(detected.begin(), detected.end(),
              [](const KeyPoint& a, const KeyPoint& b) {
                  if (a.response != b.response) return a.response < b.response;
                  return a.pt.x < b.pt.x;
              });

    std::vector<KeyPoint> refKps = detected;
    Mat refDesc;
    unlimited->compute(img, refKps, refDesc);
    ASSERT_EQ(detected.size(), refKps.size());
    ASSERT_EQ(static_cast<int>(detected.size()), refDesc.rows);

    const int kLimit = 10;
    Ptr<AKAZE> limited = AKAZE::create(AKAZE::DESCRIPTOR_MLDB, 0, 3, 0.0005f, 2, 2);
    limited->setMaxPoints(kLimit);

    std::vector<KeyPoint> kpCompute = detected;
    Mat descCompute;
    limited->compute(img, kpCompute, descCompute);
    ASSERT_EQ(detected.size(), kpCompute.size());
    ASSERT_EQ(static_cast<int>(detected.size()), descCompute.rows);
    for (size_t i = 0; i < detected.size(); ++i)
    {
        EXPECT_EQ(detected[i].pt, kpCompute[i].pt) << "i=" << i;
        EXPECT_EQ(0.0, cvtest::norm(descCompute.row(static_cast<int>(i)), refDesc.row(static_cast<int>(i)), NORM_HAMMING)) << "i=" << i;
    }

    std::vector<KeyPoint> kpProvided = detected;
    Mat descProvided;
    limited->detectAndCompute(img, noArray(), kpProvided, descProvided, true);
    ASSERT_EQ(detected.size(), kpProvided.size());
    EXPECT_EQ(0.0, cvtest::norm(descProvided, refDesc, NORM_HAMMING));

    // Detection without provided keypoints must still honor max_points
    std::vector<KeyPoint> kpDetect, kpDac;
    Mat descDac;
    limited->detect(img, kpDetect);
    EXPECT_EQ(static_cast<size_t>(kLimit), kpDetect.size());
    limited->detectAndCompute(img, noArray(), kpDac, descDac, false);
    EXPECT_EQ(static_cast<size_t>(kLimit), kpDac.size());
    EXPECT_EQ(kLimit, descDac.rows);
}

/**
 * This test is here to guard propagation of NaNs that happens on this image. NaNs are guarded
 * by debug asserts in AKAZE, which should fire for you if you are lucky.
 *
 * This test also reveals problems with uninitialized memory that happens only on this image.
 * This is very hard to hit and depends a lot on particular allocator. Run this test in valgrind and check
 * for uninitialized values if you think you are hitting this problem again.
 */
TEST(Features2d_AKAZE, uninitialized_and_nans)
{
    Mat b1 = imread(cvtest::TS::ptr()->get_data_path() + "../stitching/b1.png");
    ASSERT_FALSE(b1.empty());

    vector<KeyPoint> keypoints;
    Mat desc;
    Ptr<Feature2D> akaze = AKAZE::create();
    akaze->detectAndCompute(b1, noArray(), keypoints, desc);
}

}} // namespace
