// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

const Size img_size(320, 240);
const int FLD_TEST_SEED = 0x134679;
const int EPOCHS = 5;

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
        string filename = cvtest::TS::ptr()->get_data_path() + "cv/imgproc/beads.jpg";
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

TEST_F(ximgproc_ED, detectLinesAndEllipses)
{
    Mat gray_image;
    vector<Vec6d> ellipses;

    test_image = imread(filename);
    EXPECT_FALSE(test_image.empty()) << "Invalid test image: " << filename;

    cvtColor(test_image, test_image, COLOR_BGR2BGRA);
    cvtColor(test_image, gray_image, COLOR_BGR2GRAY);

    detector->detectEdges(gray_image);
    detector->detectEllipses(ellipses);
    detector->detectLines(lines);

    size_t segments_size = 6458;
    size_t lines_size = 6264;
    size_t ellipses_size = 2449;
    EXPECT_EQ(detector->getSegments().size(), segments_size);
    EXPECT_GE(lines.size(), lines_size);
    EXPECT_LE(lines.size(), lines_size + 2);
    EXPECT_EQ(ellipses.size(), ellipses_size);

    detector->params.PFmode = true;

    detector->detectEdges(gray_image);
    detector->detectEllipses(ellipses);
    detector->detectLines(lines);

    segments_size = 2717;
    lines_size = 6197;
    ellipses_size = 2446;
    EXPECT_EQ(detector->getSegments().size(), segments_size);
    EXPECT_GE(lines.size(), lines_size);
    EXPECT_LE(lines.size(), lines_size + 2);
    EXPECT_EQ(ellipses.size(), ellipses_size);

    detector->params.MinLineLength = 10;

    detector->detectEdges(test_image);
    detector->detectEllipses(ellipses);
    detector->detectLines(lines);
    detector->detectEllipses(ellipses);
    segments_size = 6230;
    lines_size = 11133;
    ellipses_size = 2431;
    EXPECT_EQ(detector->getSegments().size(), segments_size);
    EXPECT_GE(lines.size(), lines_size);
    EXPECT_LE(lines.size(), lines_size + 2);
    EXPECT_GE(ellipses.size(), ellipses_size);
    EXPECT_LE(ellipses.size(), ellipses_size + 2);
}

// Regression test for the detectLines() heap buffer overflow. The line-fitting
// scratch buffers are sized (width+height)*8, but a single edge segment is a
// 1-pixel-wide chain that can wind through the whole image, so its length is
// bounded by width*height, not by the perimeter. A segment longer than the
// buffer overflowed it at the "x[k]=segment[k].x" fill loop (heap corruption,
// reported by AddressSanitizer). The image below is one long serpentine stroke
// that yields a single segment far larger than the buffer.
TEST_F(ximgproc_ED, detectLinesLongWindingSegment)
{
    Mat img(400, 400, CV_8UC1, Scalar(255));
    const int pitch = 16, margin = 20;
    Point prev(margin, margin);
    bool down = true;
    for (int x = margin; x < img.cols - margin; x += pitch)
    {
        Point p1(x, down ? margin : img.rows - margin);
        Point p2(x, down ? img.rows - margin : margin);
        line(img, prev, p1, Scalar(0), 2, LINE_8);
        line(img, p1, p2, Scalar(0), 2, LINE_8);
        prev = p2;
        down = !down;
    }
    detector->detectEdges(img);

    // Precondition: at least one segment is longer than the (width+height)*8
    // scratch buffer that detectLines() allocates. Otherwise the test would not
    // exercise the overflow.
    size_t max_seg = 0;
    for (const std::vector<Point>& s : detector->getSegments())
        max_seg = std::max(max_seg, s.size());
    EXPECT_GT(max_seg, size_t((img.cols + img.rows) * 8))
        << "test image did not produce an over-long segment; adjust the image";

    // Before the fix this overflowed the line-fitting buffers (heap-buffer-overflow
    // under AddressSanitizer); after the fix it completes cleanly.
    detector->detectLines(lines);
}
}} // namespace
