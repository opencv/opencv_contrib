#include "test_precomp.hpp"

#include <vector>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

const Size img_size(640, 480);
const int FLD_TEST_SEED = 0x134679;
const int EPOCHS = 20;

class FLDBase : public testing::Test
{
    public:
        FLDBase() { }

    protected:
        Mat test_image;
        vector<Vec4f> lines;
        RNG rng;
        int passedtests;

        void GenerateWhiteNoise(Mat& image);
        void GenerateConstColor(Mat& image);
        void GenerateLines(Mat& image, const unsigned int numLines);
        void GenerateBrokenLines(Mat& image, const unsigned int numLines);
        void GenerateRotatedRect(Mat& image);
        virtual void SetUp();
};

class ximgproc_FLD: public FLDBase
{
    public:
        ximgproc_FLD() { }
    protected:

};

void FLDBase::GenerateWhiteNoise(Mat& image)
{
    image = Mat(img_size, CV_8UC1);
    rng.fill(image, RNG::UNIFORM, 0, 256);
}

void FLDBase::GenerateConstColor(Mat& image)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 256)));
}

void FLDBase::GenerateLines(Mat& image, const unsigned int numLines)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 128)));

    for(unsigned int i = 0; i < numLines; ++i)
    {
        int y = rng.uniform(10, img_size.width - 10);
        Point p1(y, 10);
        Point p2(y, img_size.height - 10);
        line(image, p1, p2, Scalar(255), 2);
    }
}

void FLDBase::GenerateBrokenLines(Mat& image, const unsigned int numLines)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(rng.uniform(0, 128)));

    for(unsigned int i = 0; i < numLines; ++i)
    {
        int y = rng.uniform(10, img_size.width - 10);
        Point p1(y, 10);
        Point p2(y, img_size.height/2);
        line(image, p1, p2, Scalar(255), 2);
        p1 = Point2i(y, img_size.height/2 + 3);
        p2 = Point2i(y, img_size.height - 10);
        line(image, p1, p2, Scalar(255), 2);
    }
}

void FLDBase::GenerateRotatedRect(Mat& image)
{
    image = Mat::zeros(img_size, CV_8UC1);

    Point center(rng.uniform(img_size.width/4, img_size.width*3/4),
            rng.uniform(img_size.height/4, img_size.height*3/4));
    Size rect_size(rng.uniform(img_size.width/8, img_size.width/6),
            rng.uniform(img_size.height/8, img_size.height/6));
    float angle = rng.uniform(0.f, 360.f);

    Point2f vertices[4];

    RotatedRect rRect = RotatedRect(center, rect_size, angle);

    rRect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        line(image, vertices[i], vertices[(i + 1) % 4], Scalar(255), 3);
    }
}

void FLDBase::SetUp()
{
    lines.clear();
    test_image = Mat();
    rng = RNG(FLD_TEST_SEED);
    passedtests = 0;
}


TEST_F(ximgproc_FLD, whiteNoise)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateWhiteNoise(test_image);
        Ptr<FastLineDetector> detector = createFastLineDetector(20);
        detector->detect(test_image, lines);

        if(40u >= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_FLD, constColor)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateConstColor(test_image);
        Ptr<FastLineDetector> detector = createFastLineDetector();
        detector->detect(test_image, lines);

        if(0u == lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_FLD, lines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateLines(test_image, numOfLines);
        Ptr<FastLineDetector> detector = createFastLineDetector();
        detector->detect(test_image, lines);
        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_FLD, mergeLines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateBrokenLines(test_image, numOfLines);
        Ptr<FastLineDetector> detector = createFastLineDetector(10, 1.414213562f, true);
        detector->detect(test_image, lines);
        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_FLD, rotatedRect)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateRotatedRect(test_image);
        Ptr<FastLineDetector> detector = createFastLineDetector();
        detector->detect(test_image, lines);

        if(2u <= lines.size())  ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}
