// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

const Size img_size(320, 240);
const int FLD_TEST_SEED = 0x134679;
const int EPOCHS = 10;

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
        void GenerateEdgeLines(Mat& image, const unsigned int numLines);
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

class ximgproc_ED: public FLDBase
{
    public:
        ximgproc_ED()
        {
            detector = createEdgeDrawing();
        }
    protected:
        Ptr<EdgeDrawing> detector;

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

void FLDBase::GenerateEdgeLines(Mat& image, const unsigned int numLines)
{
    image = Mat(img_size, CV_8UC1, Scalar::all(0));

    for(unsigned int i = 0; i < numLines; ++i)
    {
        int y = rng.uniform(10, img_size.width - 10);
        Point p1(y, 10);
        Point p2(y, img_size.height - 10);
        line(image, p1, p2, Scalar(255), 1);
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

TEST_F(ximgproc_FLD, edgeLines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateEdgeLines(test_image, numOfLines);
        Ptr<FastLineDetector> detector = createFastLineDetector(10, 1.414213562f, 50, 50, 0);
        detector->detect(test_image, lines);
        if(numOfLines == lines.size()) ++passedtests;
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

//************** EDGE DRAWING *******************

TEST_F(ximgproc_ED, whiteNoise)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateWhiteNoise(test_image);
        detector->detectEdges(test_image);
        detector->detectLines(lines);
        if(2u >= lines.size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_ED, constColor)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateConstColor(test_image);
        detector->detectEdges(test_image);
        if(0u == detector->getSegments().size()) ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_ED, lines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateLines(test_image, numOfLines);
        detector->detectEdges(test_image);
        detector->detectLines(lines);
        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_ED, mergeLines)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        const unsigned int numOfLines = 1;
        GenerateBrokenLines(test_image, numOfLines);
        detector->detectEdges(test_image);
        detector->detectLines(lines);
        if(numOfLines * 2 == lines.size()) ++passedtests;  // * 2 because of Gibbs effect
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_ED, rotatedRect)
{
    for (int i = 0; i < EPOCHS; ++i)
    {
        GenerateRotatedRect(test_image);
        detector->detectEdges(test_image);
        detector->detectLines(lines);

        if(6u <= lines.size())  ++passedtests;
    }
    ASSERT_EQ(EPOCHS, passedtests);
}

TEST_F(ximgproc_ED, ManySmallCircles)
{
    string picture_name = "cv/imgproc/beads.jpg";

    string filename = cvtest::TS::ptr()->get_data_path() + picture_name;
    test_image = imread(filename, IMREAD_GRAYSCALE);
    EXPECT_FALSE(test_image.empty()) << "Invalid test image: " << filename;

    vector<Vec6d> ellipses;
    detector->detectEdges(test_image);
    detector->detectLines(lines);
    detector->detectEllipses(ellipses);

    size_t segments_size = 6458;
    size_t lines_size = 6264;
    size_t ellipses_size = 2449;
    EXPECT_EQ(detector->getSegments().size(), segments_size);
    EXPECT_EQ(lines.size(), lines_size);
    EXPECT_EQ(ellipses.size(), ellipses_size);
}
}} // namespace
