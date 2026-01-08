// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/ximgproc/run_length_morphology.hpp"
#include "opencv2/imgproc.hpp"

namespace opencv_test {
namespace {

const Size img_size(640, 480);
const int tile_size(20);
typedef tuple<cv::MorphTypes, int, int> RLMParams;
typedef tuple<cv::MorphTypes, int, int, int> RLMSParams;

class RLTestBase
{
public:
    RLTestBase() { }

protected:
    std::vector<Mat> test_image;
    std::vector<Mat> test_image_rle;

    void generateCheckerBoard(Mat& image);
    void generateRandomImage(Mat& image);

    bool areImagesIdentical(Mat& pixelImage, Mat& rleImage);
    bool arePixelImagesIdentical(Mat& image1, Mat& image2);

    void setUp_impl();
};

void RLTestBase::generateCheckerBoard(Mat& image)
{
    image.create(img_size, CV_8UC1);
    for (int iy = 0; iy < img_size.height; iy += tile_size)
    {
        Range rowRange(iy, std::min(iy + tile_size, img_size.height));
        for (int ix = 0; ix < img_size.width; ix += tile_size)
        {
            Range colRange(ix, std::min(ix + tile_size, img_size.width));
            Mat tile(image, rowRange, colRange);
            bool bBright = ((iy + ix) % (2 * tile_size) == 0);
            tile = (bBright ? Scalar(255) : Scalar(0));
        }
    }
}

void RLTestBase::generateRandomImage(Mat& image)
{
    image.create(img_size, CV_8UC1);
    randu(image, Scalar::all(0), Scalar::all(255));
}


void RLTestBase::setUp_impl()
{
    test_image.resize(2);
    test_image_rle.resize(2);
    generateCheckerBoard(test_image[0]);
    rl::threshold(test_image[0], test_image_rle[0], 100.0, THRESH_BINARY);

    cv::Mat theRandom;
    generateRandomImage(theRandom);
    double dThreshold = 254.0;
    cv::threshold(theRandom, test_image[1], dThreshold, 255.0, THRESH_BINARY);

    rl::threshold(theRandom, test_image_rle[1], dThreshold, THRESH_BINARY);

}

bool RLTestBase::areImagesIdentical(Mat& pixelImage, Mat& rleImage)
{
    cv::Mat rleConverted;
    rleConverted = cv::Mat::zeros(pixelImage.rows, pixelImage.cols, CV_8UC1);
    rl::paint(rleConverted, rleImage, Scalar(255.0));
    return arePixelImagesIdentical(pixelImage, rleConverted);
}

bool RLTestBase::arePixelImagesIdentical(Mat& image1, Mat& image2)
{
    cv::Mat diff;
    cv::absdiff(image1, image2, diff);
    int nDiff = cv::countNonZero(diff);
    return (nDiff == 0);
}



class RL_Identical_Result_Simple : public RLTestBase, public ::testing::TestWithParam<RLMSParams>
{
public:
    RL_Identical_Result_Simple() { }
protected:
    virtual void SetUp() { setUp_impl(); }
};

TEST_P(RL_Identical_Result_Simple, simple)
{
    Mat resPix, resRLE;
    RLMSParams param = GetParam();
    cv::MorphTypes elementType = get<0>(param);
    int nSize = get<1>(param);
    int image = get<2>(param);
    int op = get<3>(param);
    Mat element = getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1),
        Point(nSize, nSize));


    morphologyEx(test_image[image], resPix, op, element);

    Mat elementRLE = rl::getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1));
    rl::morphologyEx(test_image_rle[image], resRLE, op, elementRLE);

    ASSERT_TRUE(areImagesIdentical(resPix, resRLE));
}

INSTANTIATE_TEST_CASE_P(TypicalSET, RL_Identical_Result_Simple, Combine(Values(MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE),
    Values(1, 5, 11), Values(0, 1), Values(MORPH_ERODE, MORPH_DILATE, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)));


class RL_Identical_Result : public RLTestBase, public ::testing::TestWithParam<RLMParams>
{
public:
    RL_Identical_Result() { }
protected:
    virtual void SetUp() { setUp_impl(); }
};


TEST_P(RL_Identical_Result, erosion_no_boundary)
{
    Mat resPix, resRLE;
    RLMParams param = GetParam();
    cv::MorphTypes elementType = get<0>(param);
    int nSize = get<1>(param);
    int image = get<2>(param);
    Mat element = getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1),
        Point(nSize, nSize));

    erode(test_image[image], resPix, element, cv::Point(-1,-1), 1, BORDER_CONSTANT, cv::Scalar(0));

    Mat elementRLE = rl::getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1));
    rl::erode(test_image_rle[image], resRLE, elementRLE, false);

    ASSERT_TRUE(areImagesIdentical(resPix, resRLE));
}

TEST_P(RL_Identical_Result, erosion_with_offset)
{
    Mat resPix, resRLE;
    RLMParams param = GetParam();
    cv::MorphTypes elementType = get<0>(param);
    int nSize = get<1>(param);
    int image = get<2>(param);
    int nOffset = nSize - 1;
    Mat element = getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1),
        Point(nSize, nSize));

    erode(test_image[image], resPix, element, cv::Point(nSize + nOffset, nSize + nOffset));

    Mat elementRLE = rl::getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1));
    rl::erode(test_image_rle[image], resRLE, elementRLE, true, Point(nOffset, nOffset));

    ASSERT_TRUE(areImagesIdentical(resPix, resRLE));
}

TEST_P(RL_Identical_Result, dilation_with_offset)
{
    Mat resPix, resRLE;
    RLMParams param = GetParam();
    cv::MorphTypes elementType = get<0>(param);
    int nSize = get<1>(param);
    int image = get<2>(param);
    int nOffset = nSize - 1;
    Mat element = getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1),
        Point(nSize, nSize));

    dilate(test_image[image], resPix, element, cv::Point(nSize + nOffset, nSize + nOffset));

    Mat elementRLE = rl::getStructuringElement(elementType, Size(nSize * 2 + 1, nSize * 2 + 1));
    rl::dilate(test_image_rle[image], resRLE, elementRLE, Point(nOffset, nOffset));

    ASSERT_TRUE(areImagesIdentical(resPix, resRLE));
}

INSTANTIATE_TEST_CASE_P(TypicalSET, RL_Identical_Result, Combine(Values(MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE), Values(1,5,11), Values(0,1)));

class RL_CreateCustomKernel : public RLTestBase, public testing::Test
{
public:
    RL_CreateCustomKernel() { }
protected:
    virtual void SetUp() { setUp_impl(); }
};


TEST_F(RL_CreateCustomKernel, check_valid)
{
    // create a diamond
    int nSize = 21;
    std::vector<Point3i> runs;
    for (int i = 0; i < nSize; ++i)
    {
        runs.emplace_back(Point3i(-i, i, -nSize + i));
        runs.emplace_back(Point3i(-i, i, nSize - i));
    }
    runs.emplace_back(Point3i(-nSize, nSize, 0));
    Mat kernel, dest;
    rl::createRLEImage(runs, kernel);
    ASSERT_TRUE(rl::isRLMorphologyPossible(kernel));

    rl::erode(test_image_rle[0], dest, kernel);
    //only one row means: no runs, all pixels off
    ASSERT_TRUE(dest.rows == 1);
}

typedef tuple<int> RLPParams;

class RL_Paint : public RLTestBase, public ::testing::TestWithParam<RLPParams>
{
public:
RL_Paint() { }
protected:
    virtual void SetUp() { setUp_impl(); }
};

TEST_P(RL_Paint, same_result)
{
    Mat converted, pixBinary, painted;
    RLPParams param = GetParam();
    int rType = get<0>(param);

    double dThreshold = 100.0;
    double dMaxValue = 105.0;
    test_image[1].convertTo(converted, rType);
    cv::threshold(converted, pixBinary, dThreshold, dMaxValue, THRESH_BINARY);

    painted.create(test_image[1].rows, test_image[1].cols, rType);
    painted = cv::Scalar(0.0);
    rl::paint(painted, test_image_rle[1], Scalar(dMaxValue));
    ASSERT_TRUE(arePixelImagesIdentical(pixBinary, painted));
}

INSTANTIATE_TEST_CASE_P(TypicalSET, RL_Paint, Values(CV_8U, CV_16U, CV_16S, CV_32F, CV_64F));

}
}