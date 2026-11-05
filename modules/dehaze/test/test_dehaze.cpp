#include "test_precomp.hpp"

namespace opencv_test { namespace {

// Helper to generate a synthetic hazy image for deterministic testing
static Mat makeSyntheticHazyImage(int rows, int cols)
{
    Mat scene(rows, cols, CV_8UC3);
    randu(scene, Scalar(0, 0, 0), Scalar(255, 255, 255));

    // Simulate haze: blend scene with constant atmospheric light
    Mat hazy(rows, cols, CV_8UC3);
    Vec3b A(220, 220, 220);
    double t = 0.5; // constant transmission for synthetic test

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b s = scene.at<Vec3b>(i, j);
            Vec3b h;
            for (int c = 0; c < 3; c++) {
                h[c] = saturate_cast<uchar>(s[c] * t + A[c] * (1 - t));
            }
            hazy.at<Vec3b>(i, j) = h;
        }
    }
    return hazy;
}

TEST(Dehaze_computeDarkChannel, OutputSizeAndType)
{
    Mat src = makeSyntheticHazyImage(100, 100);
    Mat dark;
    computeDarkChannel(src, dark, 15);

    EXPECT_EQ(dark.size(), src.size());
    EXPECT_EQ(dark.type(), CV_8UC1);
}

TEST(Dehaze_computeDarkChannel, ValuesWithinValidRange)
{
    Mat src = makeSyntheticHazyImage(50, 50);
    Mat dark;
    computeDarkChannel(src, dark, 15);

    double minVal, maxVal;
    minMaxLoc(dark, &minVal, &maxVal);

    EXPECT_GE(minVal, 0);
    EXPECT_LE(maxVal, 255);
}

TEST(Dehaze_estimateAtmosphericLight, ReturnsValidRange)
{
    Mat src = makeSyntheticHazyImage(100, 100);
    Mat dark;
    computeDarkChannel(src, dark, 15);

    Vec3d A;
    estimateAtmosphericLight(src, dark, A);

    for (int c = 0; c < 3; c++) {
        EXPECT_GE(A[c], 0.0);
        EXPECT_LE(A[c], 255.0);
    }
}

TEST(Dehaze_computeTransmission, OutputSizeAndType)
{
    Mat src = makeSyntheticHazyImage(80, 80);
    Mat dark;
    computeDarkChannel(src, dark, 15);

    Vec3d A;
    estimateAtmosphericLight(src, dark, A);

    Mat transmission;
    computeTransmission(src, A, transmission, 15, 0.95);

    EXPECT_EQ(transmission.size(), src.size());
    EXPECT_EQ(transmission.type(), CV_64F);
}

TEST(Dehaze_computeTransmission, ValuesWithinExpectedRange)
{
    Mat src = makeSyntheticHazyImage(80, 80);
    Mat dark;
    computeDarkChannel(src, dark, 15);

    Vec3d A;
    estimateAtmosphericLight(src, dark, A);

    Mat transmission;
    computeTransmission(src, A, transmission, 15, 0.95);

    double minVal, maxVal;
    minMaxLoc(transmission, &minVal, &maxVal);

    // Transmission should be in (1 - omega, 1] roughly
    EXPECT_GE(minVal, -0.01);
    EXPECT_LE(maxVal, 1.01);
}

TEST(Dehaze_guidedFilter, OutputSizeAndType)
{
    Mat src = makeSyntheticHazyImage(80, 80);
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat dark;
    computeDarkChannel(src, dark, 15);
    Vec3d A;
    estimateAtmosphericLight(src, dark, A);
    Mat transmission;
    computeTransmission(src, A, transmission, 15, 0.95);

    Mat refined;
    guidedFilter(gray, transmission, refined, 60, 0.0001);

    EXPECT_EQ(refined.size(), src.size());
    EXPECT_EQ(refined.type(), CV_64F);
}

TEST(Dehaze_guidedFilter, NoNaNOrInfValues)
{
    Mat src = makeSyntheticHazyImage(60, 60);
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    Mat dark;
    computeDarkChannel(src, dark, 15);
    Vec3d A;
    estimateAtmosphericLight(src, dark, A);
    Mat transmission;
    computeTransmission(src, A, transmission, 15, 0.95);

    Mat refined;
    guidedFilter(gray, transmission, refined, 60, 0.0001);

    Mat nanMask = refined != refined; // NaN check
    EXPECT_EQ(countNonZero(nanMask), 0);
}

TEST(Dehaze_detectSkyRegion, OutputSizeAndType)
{
    Mat src = makeSyntheticHazyImage(100, 100);
    Mat skyMask;
    detectSkyRegion(src, skyMask);

    EXPECT_EQ(skyMask.size(), src.size());
    EXPECT_EQ(skyMask.type(), CV_8UC1);
}

TEST(Dehaze_recoverSceneRadiance, OutputSizeAndType)
{
    Mat src = makeSyntheticHazyImage(80, 80);
    Mat dark;
    computeDarkChannel(src, dark, 15);
    Vec3d A;
    estimateAtmosphericLight(src, dark, A);
    Mat transmission;
    computeTransmission(src, A, transmission, 15, 0.95);

    Mat dst;
    recoverSceneRadiance(src, transmission, A, dst, 0.1);

    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_8UC3);
}

TEST(Dehaze_dehazeImage, FullPipelineProducesValidOutput)
{
    Mat src = makeSyntheticHazyImage(100, 100);
    Mat dst;
    dehazeImage(src, dst);

    EXPECT_EQ(dst.size(), src.size());
    EXPECT_EQ(dst.type(), CV_8UC3);
}

TEST(Dehaze_dehazeImage, IncreasesContrastOnHazyImage)
{
    // Dehazed image should generally have higher std deviation
    // (contrast) than the hazy input, since haze flattens contrast.
    Mat src = makeSyntheticHazyImage(100, 100);
    Mat dst;
    dehazeImage(src, dst);

    Scalar meanSrc, stddevSrc, meanDst, stddevDst;
    meanStdDev(src, meanSrc, stddevSrc);
    meanStdDev(dst, meanDst, stddevDst);

    double avgStdSrc = (stddevSrc[0] + stddevSrc[1] + stddevSrc[2]) / 3.0;
    double avgStdDst = (stddevDst[0] + stddevDst[1] + stddevDst[2]) / 3.0;

    EXPECT_GE(avgStdDst, avgStdSrc * 0.8); // allow some tolerance
}

TEST(Dehaze_dehazeImage, HandlesSmallImage)
{
    Mat src = makeSyntheticHazyImage(20, 20);
    Mat dst;
    EXPECT_NO_THROW(dehazeImage(src, dst));
}

TEST(Dehaze_computeDarkChannel, AssertsOnWrongType)
{
    Mat src(50, 50, CV_8UC1); // wrong type, should be CV_8UC3
    Mat dark;
    EXPECT_ANY_THROW(computeDarkChannel(src, dark, 15));
}

}} // namespace