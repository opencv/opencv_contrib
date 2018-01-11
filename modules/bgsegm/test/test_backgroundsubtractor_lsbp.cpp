#include "test_precomp.hpp"
#include <set>

using namespace std;
using namespace cv;
using namespace cvtest;

static string getDataDir() { return TS::ptr()->get_data_path(); }

static string getLenaImagePath() { return getDataDir() + "shared/lena.png"; }

// Simple synthetic illumination invariance test
TEST(BackgroundSubtractor_LSBP, IlluminationInvariance)
{
    RNG rng;
    Mat input(100, 100, CV_32FC3);

    rng.fill(input, RNG::UNIFORM, 0.0f, 0.1f);

    Mat lsv1, lsv2;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv1, input);
    input *= 10;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv2, input);

    ASSERT_LE(cv::norm(lsv1, lsv2), 0.04f);
}

TEST(BackgroundSubtractor_LSBP, Correctness)
{
    Mat input(3, 3, CV_32FC3);

    float n = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            input.at<Point3f>(i, j) = Point3f(n, n, n);
            ++n;
        }

    Mat lsv;
    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0903614f), 0.001f);

    input = 1;
    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0f), 0.001f);
}

TEST(BackgroundSubtractor_LSBP, Discrimination)
{
    Point2i LSBPSamplePoints[32];
    for (int i = 0; i < 32; ++i) {
        const double phi = i * CV_2PI / 32.0;
        LSBPSamplePoints[i] = Point2i(int(4 * std::cos(phi)), int(4 * std::sin(phi)));
    }

    Mat lena = imread(getLenaImagePath());
    Mat lsv;

    lena.convertTo(lena, CV_32FC3);

    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, lena);

    Scalar mean, var;
    meanStdDev(lsv, mean, var);

    EXPECT_GE(mean[0], 0.02);
    EXPECT_LE(mean[0], 0.04);
    EXPECT_GE(var[0], 0.03);

    Mat desc;
    bgsegm::BackgroundSubtractorLSBPDesc::computeFromLocalSVDValues(desc, lsv, LSBPSamplePoints);
    Size sz = desc.size();
    std::set<int> distinctive_elements;

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            distinctive_elements.insert(desc.at<int>(i, j));

    EXPECT_GE(distinctive_elements.size(), 35000U);
}

static double scoreBitwiseReduce(const Mat& mask, const Mat& gtMask, uchar v1, uchar v2) {
    Mat result;
    cv::bitwise_and(mask == v1, gtMask == v2, result);
    return cv::countNonZero(result);
}

template<typename T>
static double evaluateBGSAlgorithm(Ptr<T> bgs) {
    Mat background = imread(getDataDir() + "shared/fruits.png");
    Mat object = imread(getDataDir() + "shared/baboon.png");
    cv::resize(object, object, Size(100, 100), 0, 0, INTER_LINEAR_EXACT);
    Ptr<bgsegm::SyntheticSequenceGenerator> generator = bgsegm::createSyntheticSequenceGenerator(background, object);

    double f1_mean = 0;
    unsigned total = 0;

    for (int frameNum = 1; frameNum <= 400; ++frameNum) {
        Mat frame, gtMask;
        generator->getNextFrame(frame, gtMask);

        Mat mask;
        bgs->apply(frame, mask);

        Size sz = frame.size();
        EXPECT_EQ(sz, gtMask.size());
        EXPECT_EQ(gtMask.size(), mask.size());
        EXPECT_EQ(mask.type(), gtMask.type());
        EXPECT_EQ(mask.type(), CV_8U);

        // We will give the algorithm some time for the proper background model inference.
        // Almost all background subtraction algorithms have a problem with cold start and require some time for background model initialization.
        // So we will not count first part of the frames in the score.
        if (frameNum > 300) {
            const double tp = scoreBitwiseReduce(mask, gtMask, 255, 255);
            const double fp = scoreBitwiseReduce(mask, gtMask, 255, 0);
            const double fn = scoreBitwiseReduce(mask, gtMask, 0, 255);

            if (tp + fn + fp > 0) {
                const double f1_score = 2.0 * tp / (2.0 * tp + fn + fp);
                f1_mean += f1_score;
                ++total;
            }
        }
    }

    f1_mean /= total;
    return f1_mean;
}

TEST(BackgroundSubtractor_LSBP, Accuracy)
{
    EXPECT_GE(evaluateBGSAlgorithm(bgsegm::createBackgroundSubtractorGSOC()), 0.9);
    EXPECT_GE(evaluateBGSAlgorithm(bgsegm::createBackgroundSubtractorLSBP()), 0.25);
}
