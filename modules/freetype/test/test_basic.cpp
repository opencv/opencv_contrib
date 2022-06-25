// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

struct MattypeParams
{
    string title;
    int mattype;
    bool expect_success;
};

::std::ostream& operator<<(::std::ostream& os, const MattypeParams& prm) {
      return os << prm.title;
}

const MattypeParams mattype_list[] =
{
    { "CV_8UC1",  CV_8UC1,  true},  { "CV_8UC2",  CV_8UC2,  false},
    { "CV_8UC3",  CV_8UC3,  true},  { "CV_8UC4",  CV_8UC4,  true},

    { "CV_8SC1",  CV_8SC1,  false}, { "CV_8SC2",  CV_8SC2,  false},
    { "CV_8SC3",  CV_8SC3,  false}, { "CV_8SC4",  CV_8SC4,  false},
    { "CV_16UC1", CV_16UC1, false}, { "CV_16UC2", CV_16UC2, false},
    { "CV_16UC3", CV_16UC3, false}, { "CV_16UC4", CV_16UC4, false},
    { "CV_16SC1", CV_16SC1, false}, { "CV_16SC2", CV_16SC2, false},
    { "CV_16SC3", CV_16SC3, false}, { "CV_16SC4", CV_16SC4, false},
    { "CV_32SC1", CV_32SC1, false}, { "CV_32SC2", CV_32SC2, false},
    { "CV_32SC3", CV_32SC3, false}, { "CV_32SC4", CV_32SC4, false},
    { "CV_32FC1", CV_32FC1, false}, { "CV_32FC2", CV_32FC2, false},
    { "CV_32FC3", CV_32FC3, false}, { "CV_32FC4", CV_32FC4, false},
    { "CV_64FC1", CV_64FC1, false}, { "CV_64FC2", CV_64FC2, false},
    { "CV_64FC3", CV_64FC3, false}, { "CV_64FC4", CV_64FC4, false},
    { "CV_16FC1", CV_16FC1, false}, { "CV_16FC2", CV_16FC2, false},
    { "CV_16FC3", CV_16FC3, false}, { "CV_16FC4", CV_16FC4, false},
};

/******************
 * Basically usage
 *****************/

TEST(Freetype_Basic, success )
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Mat dst(600,600, CV_8UC3, Scalar::all(255) );
    Scalar col(128,64,255,192);
    EXPECT_NO_THROW( ft2->putText(dst, "Basic,success", Point( 0,  50), 50, col, -1, LINE_AA, true ) );
}

/******************
 * loadFontData()
 *****************/

TEST(Freetype_loadFontData, nonexist_file)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "UNEXITSTFONT"; /* NON EXISTS FONT DATA */

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_THROW( ft2->loadFontData( fontdata, 0 ), cv::Exception );
    Mat dst(600,600, CV_8UC3, Scalar::all(255) );
    Scalar col(128,64,255,192);
    EXPECT_THROW( ft2->putText(dst, "nonexist_file", Point( 0,  50), 50, col, -1, LINE_AA, true ), cv::Exception );
}

TEST(Freetype_loadFontData, forget_calling)
{
    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";
//    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Mat dst(600,600, CV_8UC3, Scalar::all(255) );

    Scalar col(128,64,255,192);
    EXPECT_THROW( ft2->putText(dst, "forget_calling", Point( 0,  50), 50, col, -1, LINE_AA, true ), cv::Exception );
}

TEST(Freetype_loadFontData, call_multiple)
{
    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";

    for( int i = 0 ; i < 100 ; i ++ )
    {
        EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );
    }

    Mat dst(600,600, CV_8UC3, Scalar::all(255) );
    Scalar col(128,64,255,192);
    EXPECT_NO_THROW( ft2->putText(dst, "call_mutilple", Point( 0,  50), 50, col, -1, LINE_AA, true ) );
}

typedef testing::TestWithParam<int> idx_range;

TEST_P(idx_range, failed )
{
    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";
    EXPECT_THROW( ft2->loadFontData( fontdata, GetParam() ), cv::Exception );
}

const int idx_failed_list[] =
{
    INT_MIN,
    INT_MIN + 1,
    -1,
    1,
    2,
    INT_MAX - 1,
    INT_MAX
};

INSTANTIATE_TEST_CASE_P(Freetype_loadFontData, idx_range,
                        testing::ValuesIn(idx_failed_list));

/******************
 * setSplitNumber()
 *****************/

typedef testing::TestWithParam<int> ctol_range;

TEST_P(ctol_range, success)
{
    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );

    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );
    EXPECT_NO_THROW( ft2->setSplitNumber(GetParam()) );

    Mat dst(600,600, CV_8UC3, Scalar::all(255) );
    Scalar col(128,64,255,192);
    EXPECT_NO_THROW( ft2->putText(dst, "CtoL", Point( 0,  50), 50, col, 1, LINE_4, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_4: oOpPqQ", Point( 40, 100), 50, col, 1, LINE_4, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_8: oOpPqQ", Point( 40, 150), 50, col, 1, LINE_8, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_AA:oOpPqQ", Point( 40, 150), 50, col, 1, LINE_AA, true ) );
}

const int ctol_list[] =
{
    1,
    8,
    16,
    32,
    64,
    128,
    // INT_MAX -1,  // Hang-up
    // INT_MAX      // Hang-up
};

INSTANTIATE_TEST_CASE_P(Freetype_setSplitNumber, ctol_range,
                        testing::ValuesIn(ctol_list));


/********************
 * putText()::common
 *******************/

TEST(Freetype_putText, invalid_img )
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Scalar col(128,64,255,192);
    { /* empty mat */
        Mat dst;
        EXPECT_THROW( ft2->putText(dst, "Invalid_img(empty Mat)", Point( 0,  50), 50, col, -1, LINE_AA, true ), cv::Exception );
    }
    { /* not mat(scalar) */
        Scalar dst;
        EXPECT_THROW( ft2->putText(dst, "Invalid_img(Scalar)", Point( 0,  50), 50, col, -1, LINE_AA, true ), cv::Exception );
    }
}

typedef testing::TestWithParam<MattypeParams> MatType_Test;

TEST_P(MatType_Test, default)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string fontdata = root + "freetype/mplus/Mplus1-Regular.ttf";

    const MattypeParams params = static_cast<MattypeParams>(GetParam());
    const string title        = params.title;
    const int mattype         = params.mattype;
    const bool expect_success = params.expect_success;

    cv::Ptr<cv::freetype::FreeType2> ft2;
    EXPECT_NO_THROW( ft2 = cv::freetype::createFreeType2() );
    EXPECT_NO_THROW( ft2->loadFontData( fontdata, 0 ) );

    Mat dst(600,600, mattype, Scalar::all(255) );

    Scalar col(128,64,255,192);

    if ( expect_success == false )
    {
        EXPECT_THROW( ft2->putText(dst, title, Point( 0,  50), 50, col, -1, LINE_AA, true ), cv::Exception );
        return;
    }

    EXPECT_NO_THROW( ft2->putText(dst, title,                Point( 0,  50), 50, col, -1, LINE_AA, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_4  FILL(mono)", Point(40, 100), 50, col, -1, LINE_4,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_8  FILL(mono)", Point(40, 150), 50, col, -1, LINE_8,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_AA FILL(blend)",Point(40, 200), 50, col, -1, LINE_AA, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_4  OUTLINE(1)", Point(40, 250), 50, col,  1, LINE_4,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_8  OUTLINE(1)", Point(40, 300), 50, col,  1, LINE_8,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_AA OUTLINE(1)", Point(40, 350), 50, col,  1, LINE_AA, true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_4  OUTLINE(5)", Point(40, 400), 50, col,  5, LINE_4,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_8  OUTLINE(5)", Point(40, 450), 50, col,  5, LINE_8,  true ) );
    EXPECT_NO_THROW( ft2->putText(dst, "LINE_AA OUTLINE(5)", Point(40, 500), 50, col,  5, LINE_AA, true ) );
    putText(dst, "LINE_4 putText(th=1)" , Point( 40,550), FONT_HERSHEY_SIMPLEX, 0.5, col, 1, LINE_4);
    putText(dst, "LINE_8 putText(th=1)" , Point( 40,565), FONT_HERSHEY_SIMPLEX, 0.5, col, 1, LINE_8);
    putText(dst, "LINE_AA putText(th=1)", Point( 40,580), FONT_HERSHEY_SIMPLEX, 0.5, col, 1, LINE_AA);
    putText(dst, "LINE_4 putText(th=2)" , Point( 240,550),FONT_HERSHEY_SIMPLEX, 0.5, col, 2, LINE_4);
    putText(dst, "LINE_8 putText(th=2)" , Point( 240,565),FONT_HERSHEY_SIMPLEX, 0.5, col, 2, LINE_8);
    putText(dst, "LINE_AA putText(th=2)", Point( 240,580),FONT_HERSHEY_SIMPLEX, 0.5, col, 2, LINE_AA);

    if (cvtest::debugLevel > 0 )
    {
        imwrite( cv::format("%s-MatType.png", title.c_str()), dst );
    }
}

INSTANTIATE_TEST_CASE_P(Freetype_putText, MatType_Test,
                        testing::ValuesIn(mattype_list));

}} // namespace
