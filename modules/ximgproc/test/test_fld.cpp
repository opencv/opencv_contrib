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

TEST_F(ximgproc_ED, reusedDetectorMatchesFreshAcrossSizes)
{
    RNG& testRng = theRNG();

    auto makeLineScene = [&testRng](int w, int h) {
        int bw = std::min(w, 160);
        int bh = std::min(h, 120);
        Mat base(bh, bw, CV_8UC1);
        testRng.fill(base, RNG::UNIFORM, 0, 40);
        for (int y = 0; y < bh; ++y)
            for (int x = 0; x < bw; ++x)
                base.at<uchar>(y, x) = saturate_cast<uchar>(
                    base.at<uchar>(y, x) + (((x / 3 + y / 5) % 2) * 195));
        if (w == bw && h == bh)
            return base;
        Mat out;
        resize(base, out, Size(w, h), 0, 0, INTER_LINEAR);
        GaussianBlur(out, out, Size(5, 5), 1.5);
        return out;
    };

    auto makeArcScene = [&testRng](int w, int h) {
        Mat img(h, w, CV_8UC1);
        testRng.fill(img, RNG::UNIFORM, 120, 136);
        if (w < 120 || h < 120)
        {
            int r = std::max(8, std::min(w, h) / 3);
            circle(img, Point(w / 2, h / 2), r, Scalar(15), 2, LINE_8);
            return img;
        }
        for (int cy = 40; cy + 40 < h; cy += 72)
        {
            for (int cx = 40; cx + 40 < w; cx += 72)
            {
                double startAngle = testRng.uniform(0.0, 180.0);
                ellipse(img, Point(cx, cy), Size(26, 26), 0.0,
                        startAngle, startAngle + 122.0, Scalar(15), 2, LINE_8);
            }
        }
        return img;
    };

    const Size seq[] = {
        Size(64, 48),
        Size(960, 720),
        Size(128, 96),
        Size(64, 160),
        Size(32, 320),   // same area and logNT as 64x160, different (w + h) / 8
        Size(128, 128),
        Size(64, 256),   // same area and logNT as 128x128, different (w + h) / 8
        Size(64, 720),   // only height changes
        Size(960, 720)   // only width changes
    };

    Ptr<EdgeDrawing> reusedLineEd = createEdgeDrawing();
    for (size_t idx = 0; idx < sizeof(seq) / sizeof(seq[0]); ++idx)
    {
        Mat img = makeLineScene(seq[idx].width, seq[idx].height);
        vector<Vec4f> reusedLines, freshLines;
        reusedLineEd->detectEdges(img);
        reusedLineEd->detectLines(reusedLines);

        Ptr<EdgeDrawing> freshEd = createEdgeDrawing();
        freshEd->detectEdges(img);
        freshEd->detectLines(freshLines);

        EXPECT_EQ(freshLines.size(), reusedLines.size()) << "detectLines size at " << seq[idx];
        EXPECT_EQ(freshEd->getSegmentIndicesOfLines(), reusedLineEd->getSegmentIndicesOfLines())
            << "segmentIndicesOfLines at " << seq[idx];
        if (freshLines.size() == reusedLines.size())
        {
            for (size_t i = 0; i < freshLines.size(); ++i)
                EXPECT_EQ(freshLines[i], reusedLines[i]) << "line " << i << " at " << seq[idx];
        }
    }

    Ptr<EdgeDrawing> reusedArcEd = createEdgeDrawing();
    for (size_t idx = 0; idx < sizeof(seq) / sizeof(seq[0]); ++idx)
    {
        Mat img = makeArcScene(seq[idx].width, seq[idx].height);
        vector<Vec6d> reusedEllipses, freshEllipses;
        reusedArcEd->detectEdges(img);
        reusedArcEd->detectEllipses(reusedEllipses);

        Ptr<EdgeDrawing> freshEd = createEdgeDrawing();
        freshEd->detectEdges(img);
        freshEd->detectEllipses(freshEllipses);

        EXPECT_EQ(freshEllipses.size(), reusedEllipses.size()) << "detectEllipses size at " << seq[idx];
        if (freshEllipses.size() == reusedEllipses.size())
        {
            for (size_t i = 0; i < freshEllipses.size(); ++i)
                EXPECT_EQ(freshEllipses[i], reusedEllipses[i]) << "ellipse " << i << " at " << seq[idx];
        }
    }

    // Cross-method reuse: detectEllipses(small) -> detectLines(big) and detectLines(small) -> detectEllipses(big).
    {
        Mat smallArc = makeArcScene(64, 48);
        Mat bigLine = makeLineScene(960, 720);
        vector<Vec6d> tmpEllipses;
        vector<Vec4f> crossLines, refLines;

        Ptr<EdgeDrawing> crossEd = createEdgeDrawing();
        crossEd->detectEdges(smallArc);
        crossEd->detectEllipses(tmpEllipses);
        crossEd->detectEdges(bigLine);
        crossEd->detectLines(crossLines);

        Ptr<EdgeDrawing> refEd = createEdgeDrawing();
        refEd->detectEdges(bigLine);
        refEd->detectEllipses(tmpEllipses);
        refEd->detectLines(refLines);
        EXPECT_EQ(refLines.size(), crossLines.size()) << "detectEllipses(small) -> detectLines(big)";
    }
    {
        Mat smallLine = makeLineScene(64, 48);
        Mat bigArc = makeArcScene(960, 720);
        vector<Vec4f> tmpLines;
        vector<Vec6d> crossEllipses, refEllipses;

        Ptr<EdgeDrawing> crossEd = createEdgeDrawing();
        crossEd->detectEdges(smallLine);
        crossEd->detectLines(tmpLines);
        crossEd->detectEdges(bigArc);
        crossEd->detectEllipses(crossEllipses);

        Ptr<EdgeDrawing> refEd = createEdgeDrawing();
        refEd->detectEdges(bigArc);
        refEd->detectLines(tmpLines);
        refEd->detectEllipses(refEllipses);
        EXPECT_EQ(refEllipses.size(), crossEllipses.size()) << "detectLines(small) -> detectEllipses(big)";
    }

    // NFA validation rejects pure uniform noise on a large image (guards against leaving dummy logNT active).
    {
        Mat noise(720, 960, CV_8UC1);
        testRng.fill(noise, RNG::UNIFORM, 0, 256);
        Ptr<EdgeDrawing> noiseEd = createEdgeDrawing();
        vector<Vec4f> noiseLines;
        noiseEd->detectEdges(noise);
        noiseEd->detectLines(noiseLines);
        EXPECT_LE(noiseLines.size(), 5u);
    }

    // Segment-less tiny image ((width + height) / 8 == 0) and a (width + height) / 8 == 1 image
    // (where LUTSize == 1 matched the uninitialized sentinel), followed by reuse on 320x240.
    Mat tiny(3, 4, CV_8UC1);
    testRng.fill(tiny, RNG::UNIFORM, 0, 256);
    Mat smallLut1(5, 9, CV_8UC1, Scalar(255));
    rectangle(smallLut1, Point(1, 1), Point(7, 3), Scalar(0), 1);
    Mat afterTiny = makeLineScene(320, 240);
    Ptr<EdgeDrawing> tinyEd = createEdgeDrawing();
    Ptr<EdgeDrawing> freshAfterTiny = createEdgeDrawing();
    vector<Vec4f> tinyLines, afterTinyLines, freshAfterTinyLines;
    vector<Vec6d> tinyEllipses;
    tinyEd->detectEdges(tiny);
    tinyEd->detectLines(tinyLines);
    tinyEd->detectEllipses(tinyEllipses);
    tinyEd->detectEdges(smallLut1);
    tinyEd->detectLines(tinyLines);
    tinyEd->detectEllipses(tinyEllipses);
    tinyEd->detectEdges(afterTiny);
    tinyEd->detectLines(afterTinyLines);
    freshAfterTiny->detectEdges(afterTiny);
    freshAfterTiny->detectLines(freshAfterTinyLines);
    EXPECT_EQ(freshAfterTinyLines.size(), afterTinyLines.size());
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

// With Sigma < 1.0, detectEdges() used to let smoothImage share the caller's buffer
// (edge_drawing.cpp:411), so every later GaussianBlur() into smoothImage blurred the
// caller's image in place: in detectEllipses() (:2718), in the PFmode branch of
// detectEdges() (:430), and in the next detectEdges() with Sigma >= 1.0 (:413/:415),
// which wrote into the previous input. Each phase below uses its own detector.
TEST_F(ximgproc_ED, inputNotModifiedWithSigmaBelowOne)
{
    Mat circ(200, 200, CV_8UC1, Scalar(255));
    circle(circ, Point(100, 100), 60, Scalar(0), 2, LINE_8);
    Mat rect(200, 200, CV_8UC1, Scalar(255));
    rectangle(rect, Point(50, 50), Point(150, 150), Scalar(0), 2, LINE_8);
    vector<Vec6d> ellipses;
    Mat img, other;
    Ptr<EdgeDrawing> ed;

    // 1. detectEllipses()
    img = circ.clone();
    ed = createEdgeDrawing();
    ed->params.Sigma = 0.0;
    ed->detectEdges(img);
    ed->detectEllipses(ellipses);
    EXPECT_EQ(0, countNonZero(img != circ))
        << "detectEllipses() modified the caller's input image in place";

    // 2. detectEdges() with PFmode, without detectEllipses()
    img = circ.clone();
    ed = createEdgeDrawing();
    ed->params.Sigma = 0.0;
    ed->params.PFmode = true;
    ed->detectEdges(img);
    EXPECT_EQ(0, countNonZero(img != circ))
        << "detectEdges() with PFmode modified the caller's input image in place";

    // 3. the next detectEdges() with Sigma >= 1.0 on an image of the same size
    img = circ.clone();
    other = rect.clone();
    ed = createEdgeDrawing();
    ed->params.Sigma = 0.0;
    ed->detectEdges(img);
    ed->params.Sigma = 1.0;   // the default value
    ed->detectEdges(other);
    EXPECT_EQ(0, countNonZero(img != circ))
        << "a later detectEdges() call overwrote an earlier input image";
    EXPECT_EQ(0, countNonZero(other != rect))
        << "detectEdges() modified the caller's input image in place";

    // 4. as 1, but Sigma is raised before detectEllipses(): whether the input is
    //    written must not depend on the parameter value at that moment
    img = circ.clone();
    ed = createEdgeDrawing();
    ed->params.Sigma = 0.0;
    ed->detectEdges(img);
    ed->params.Sigma = 1.0;
    ed->detectEllipses(ellipses);
    EXPECT_EQ(0, countNonZero(img != circ))
        << "detectEllipses() modified the caller's input image after Sigma was raised";
}
}} // namespace
