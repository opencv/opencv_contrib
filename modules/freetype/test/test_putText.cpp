// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

struct DrawingParams
{
    string title;
    int mattype;
    string fontname;
};

::std::ostream& operator<<(::std::ostream& os, const DrawingParams& prm) {
      return os << prm.title;
}

const DrawingParams drawing_list[] =
{
    { "CV_8UC1-Mplus1-Regular", CV_8UC1, "freetype/mplus/Mplus1-Regular.ttf"},
    { "CV_8UC3-Mplus1-Regular", CV_8UC3, "freetype/mplus/Mplus1-Regular.ttf"},
    { "CV_8UC4-Mplus1-Regular", CV_8UC4, "freetype/mplus/Mplus1-Regular.ttf"},
};

/********************
 * putText()::boundry
 *******************/
typedef testing::TestWithParam<DrawingParams> BoundaryTest;

TEST_P(BoundaryTest, default)
{
    const DrawingParams params = GetParam();
    const string title    = params.title;
    const int mattype     = params.mattype;
    const string fontname = params.fontname;

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + fontname;

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Mat dst(600,600, mattype, Scalar::all(255) );

    Scalar col(128,64,255,192);
    EXPECT_NO_THROW( ft2->putText(dst, title, Point( 100, 200), 20, col, -1, LINE_AA, true ) );

    const int textHeight = 30;
    for ( int iy = -50 ; iy <= +50 ; iy++ )
    {
        Point textOrg( 50, iy );
        const string text = "top boundary";
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_4,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_8,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_AA,  true ) );
    }

    for ( int iy = -50 ; iy <= +50 ; iy++ )
    {
        Point textOrg( 400, dst.cols + iy );
        const string text = "bottom boundary";
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_4,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_8,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_AA,  true ) );
    }

    for ( int ix = -50 ; ix <= +50 ; ix++ )
    {
        Point textOrg( ix, 100 );
        const string text = "left boundary";
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_4,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_8,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_AA,  true ) );
    }

    for ( int ix = -50 ; ix <= +50 ; ix++ )
    {
        Point textOrg( dst.rows + ix, 500 );
        const string text = "bottom boundary";
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_4,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_8,  true ) );
        EXPECT_NO_THROW( ft2->putText(dst, text, textOrg, textHeight, col, -1, LINE_AA,  true ) );
    }

    if (cvtest::debugLevel > 0 )
    {
        imwrite( cv::format("%s-boundary.png", title.c_str()), dst );
    }
}

INSTANTIATE_TEST_CASE_P(Freetype_putText, BoundaryTest,
                        testing::ValuesIn(drawing_list)) ;

/*********************
 * putText()::Ligature
 *********************/

// See https://github.com/opencv/opencv_contrib/issues/2627

static Mat clipRoiAs8UC1( Mat &dst, Rect roi_rect )
{
    Mat roi = Mat(dst, roi_rect).clone();
    switch( roi.type() ){
    case CV_8UC4: cvtColor(roi,roi,COLOR_BGRA2GRAY); break;
    case CV_8UC3: cvtColor(roi,roi,COLOR_BGR2GRAY); break;
    case CV_8UC1: default: break; // Do nothing
    }
    return roi;
}

typedef testing::TestWithParam<DrawingParams> LigatureTest;
TEST_P(LigatureTest, regression2627)
{
    const DrawingParams params = GetParam();
    const string title    = params.title;
    const int mattype     = params.mattype;
    const string fontname = params.fontname;

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + fontname;

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Mat dst(600,600, mattype, Scalar(0,0,0,255) );
    Scalar col(255,255,255,255);
    EXPECT_NO_THROW( ft2->putText(dst, title, Point(  0, 50), 30, col, -1, LINE_AA, true ) );

    vector<string> texts = {
        "ffi", // ff will be combined to single glyph.
        "fs",
        "fi",
        "ff",
        "ae",
        "tz",
        "oe",
        "\xE3\x81\xAF",             // HA ( HIRAGANA )
        "\xE3\x81\xAF\xE3\x82\x99", // BA ( HA + VOICED SOUND MARK )
        "\xE3\x81\xAF\xE3\x82\x9A", // PA ( HA + SEMI-VOICED SOUND MARK )
        "\xE3\x83\x8F",             // HA ( KATAKANA )
        "\xE3\x83\x8F\xE3\x82\x99", // BA ( HA + VOICED SOUND MARK )
        "\xE3\x83\x8F\xE3\x82\x9A", // PA ( HA + SEMI-VOICED SOUND MARK )
    };

    const int fontHeight = 20;
    const int margin = fontHeight / 2; // for current glyph right edgeto next glyph left edge

    const int start_x    =  40;
    const int start_y    = 100;
    const int skip_x     = 100;
    const int skip_y     =  25;

    int tx = start_x;
    int ty = start_y;

    for (auto it = texts.begin(); it != texts.end(); it++ )
    {
        if ( ty + fontHeight * 3 > dst.rows ) {
            ty = start_y;
            tx = tx + skip_x;
        }

        EXPECT_NO_THROW( ft2->putText(dst, *it,  Point(tx,ty), fontHeight, col, -1, LINE_4,  true ) );

        { // Check for next glyph area.
            const Rect roi_rect = Rect( tx + fontHeight + margin, ty - fontHeight, fontHeight, fontHeight );
            const Mat roi = clipRoiAs8UC1(dst, roi_rect);
            EXPECT_EQ(0, countNonZero(roi) );
        }
        ty += skip_y;

        EXPECT_NO_THROW( ft2->putText(dst, *it,  Point(tx,ty), fontHeight, col, -1, LINE_8,  true ) );
        { // Check for next glyph area.
            const Rect roi_rect = Rect( tx + fontHeight + margin, ty - fontHeight, fontHeight, fontHeight );
            const Mat roi = clipRoiAs8UC1(dst, roi_rect);
            EXPECT_EQ(0, countNonZero(roi) );
        }
        ty += skip_y;

        EXPECT_NO_THROW( ft2->putText(dst, *it,  Point(tx,ty), fontHeight, col,  1, LINE_AA, true ) );
        { // Check for next glyph area.
            const Rect roi_rect = Rect( tx + fontHeight + margin, ty - fontHeight, fontHeight, fontHeight );
            const Mat roi = clipRoiAs8UC1(dst, roi_rect);
            EXPECT_EQ(0, countNonZero(roi) );
        }
        ty += skip_y;
    }

    if (cvtest::debugLevel > 0 )
    {
        imwrite( cv::format("%s-contrib2627.png", title.c_str()), dst );
    }
}

INSTANTIATE_TEST_CASE_P(Freetype_putText, LigatureTest,
                        testing::ValuesIn(drawing_list));

}} // namespace
