// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

using namespace cv;

namespace opencv_test { namespace {

TEST(intensity_transform_logTransform, accuracy)
{
    uchar image_data[] = {
         51, 211, 212,  38,  48,  25, 189,  16,  64, 197,
        104, 137,  60,  10,  78, 234, 186, 149,  37, 236,
        128,  80,   6,  53,   7,  65, 233,  15, 216,  42,
        108, 132, 136, 194, 117, 128, 214,  46, 220, 119,
        101, 126, 148,  22,  86, 206,  91, 125, 234,  24,
        162, 136,  46, 247, 245,  81, 157, 126,  73, 173,
        120, 230, 117, 111, 145, 168, 169, 187,  23, 109,
          0, 184,  23,  43, 108, 201,  13, 170, 249, 228,
        107,  59,  73, 254, 116, 156, 209, 155, 149,  95,
         24, 245, 136, 107, 192, 114,  69,  80, 199,   8
    };

    Mat_<uchar> image(10, 10, image_data);

    Mat res;
    cv::intensity_transform::logTransform(image, res);

    uchar expectedRes_data[] = {
        182, 247, 247, 169, 179, 150, 241, 130, 192, 243,
        214, 227, 189, 110, 201, 251, 241, 231, 167, 252,
        224, 202,  90, 184,  96, 193, 251, 128, 248, 173,
        216, 225, 226, 243, 220, 224, 247, 177, 248, 220,
        213, 223, 230, 144, 206, 245, 208, 223, 251, 148,
        234, 226, 177, 254, 253, 203, 233, 223, 198, 237,
        221, 250, 220, 217, 229, 236, 236, 241, 146, 216,
          0, 240, 146, 174, 216, 244, 121, 237, 254, 250,
        215, 188, 198, 255, 219, 233, 246, 232, 231, 210,
        148, 253, 226, 215, 242, 218, 196, 202, 244, 101
    };

    Mat_<uchar> expectedRes(10, 10, expectedRes_data);

    EXPECT_LE(cvtest::norm(res, expectedRes, NORM_INF), 1);
}

TEST(intensity_transform_gammaCorrection, accuracy1)
{
    uchar image_data[] = {
         51, 211, 212,  38,  48,  25, 189,  16,  64, 197,
        104, 137,  60,  10,  78, 234, 186, 149,  37, 236,
        128,  80,   6,  53,   7,  65, 233,  15, 216,  42,
        108, 132, 136, 194, 117, 128, 214,  46, 220, 119,
        101, 126, 148,  22,  86, 206,  91, 125, 234,  24,
        162, 136,  46, 247, 245,  81, 157, 126,  73, 173,
        120, 230, 117, 111, 145, 168, 169, 187,  23, 109,
          0, 184,  23,  43, 108, 201,  13, 170, 249, 228,
        107,  59,  73, 254, 116, 156, 209, 155, 149,  95,
         24, 245, 136, 107, 192, 114,  69,  80, 199,   8
    };

    Mat_<uchar> image(10, 10, image_data);

    Mat res;
    cv::intensity_transform::gammaCorrection(image, res, 1.0);

    uchar expectedRes_data[] = {
         51, 211, 212,  38,  48,  25, 189,  16,  64, 197,
        104, 137,  60,  10,  78, 234, 186, 149,  37, 236,
        128,  80,   6,  53,   7,  65, 233,  15, 216,  42,
        108, 132, 136, 194, 117, 128, 214,  46, 220, 119,
        101, 126, 148,  22,  86, 206,  91, 125, 234,  24,
        162, 136,  46, 247, 245,  81, 157, 126,  73, 173,
        120, 230, 117, 111, 145, 168, 169, 187,  23, 109,
          0, 184,  23,  43, 108, 201,  13, 170, 249, 228,
        107,  59,  73, 254, 116, 156, 209, 155, 149,  95,
         24, 245, 136, 107, 192, 114,  69,  80, 199,   8
    };

    Mat_<uchar> expectedRes(10, 10, expectedRes_data);

    EXPECT_LE(cvtest::norm(res, expectedRes, NORM_INF), 1);
}

TEST(intensity_transform_gammaCorrection, accuracy2)
{
    uchar image_data[] = {
         51, 211, 212,  38,  48,  25, 189,  16,  64, 197,
        104, 137,  60,  10,  78, 234, 186, 149,  37, 236,
        128,  80,   6,  53,   7,  65, 233,  15, 216,  42,
        108, 132, 136, 194, 117, 128, 214,  46, 220, 119,
        101, 126, 148,  22,  86, 206,  91, 125, 234,  24,
        162, 136,  46, 247, 245,  81, 157, 126,  73, 173,
        120, 230, 117, 111, 145, 168, 169, 187,  23, 109,
          0, 184,  23,  43, 108, 201,  13, 170, 249, 228,
        107,  59,  73, 254, 116, 156, 209, 155, 149,  95,
         24, 245, 136, 107, 192, 114,  69,  80, 199,   8
    };

    Mat_<uchar> image(10, 10, image_data);

    Mat res;
    cv::intensity_transform::gammaCorrection(image, res, (float)(0.4));

    uchar expectedRes_data[] = {
        133, 236, 236, 119, 130, 100, 226,  84, 146, 229,
        178, 198, 142,  69, 158, 246, 224, 205, 117, 247,
        193, 160,  56, 136,  60, 147, 245,  82, 238, 123,
        180, 195, 198, 228, 186, 193, 237, 128, 240, 187,
        176, 192, 205,  95, 165, 234, 168, 191, 246,  99,
        212, 198, 128, 251, 250, 161, 210, 192, 154, 218,
        188, 244, 186, 182, 203, 215, 216, 225,  97, 181,
          0, 223,  97, 125, 180, 231,  77, 216, 252, 243,
        180, 141, 154, 254, 186, 209, 235, 208, 205, 171,
         99, 250, 198, 180, 227, 184, 151, 160, 230, 63
    };

    Mat_<uchar> expectedRes(10, 10, expectedRes_data);

    EXPECT_LE(cvtest::norm(res, expectedRes, NORM_INF), 1);
}

TEST(intensity_transform_autoscaling, accuracy)
{
    uchar image_data[] = {
         32,  59, 164, 127, 151, 107, 167,  62, 195, 143,
         54, 166, 104,  27, 152,  20,  35, 135,  12, 198,
        107,  63,  90, 169,  67, 135, 136,  14,  94, 115,
         34, 150, 169, 171, 130,  39, 190, 108, 103,  32,
         57,  83, 146,  37,  81, 143, 144,  47,  87,  49,
         32, 108,  17, 165, 127, 137, 108,  35, 179, 175,
         40, 148, 174,  79, 146, 119, 103, 168, 167, 160,
         66, 107, 164,  19,  85, 126,  58,  95,  15, 131,
         88,  58, 162,  90, 147, 125,  61, 157,  60, 104,
        128, 193,  69, 104,  94, 196,  11,  66,  18, 179
    };

    Mat_<uchar> image(10, 10, image_data);

    Mat res;
    cv::intensity_transform::autoscaling(image, res);

    uchar expectedRes_data[] = {
         29,  65, 209, 158, 191, 131, 213,  70, 251, 180,
         59, 211, 127,  22, 192,  12,  33, 169,   1, 255,
        131,  71, 108, 215,  76, 169, 170,   4, 113, 142,
         31, 190, 215, 218, 162,  38, 244, 132, 125,  29,
         63,  98, 184,  35,  95, 180, 181,  49, 104,  52,
         29, 132,   8, 210, 158, 172, 132,  33, 229, 224,
         40, 187, 222,  93, 184, 147, 125, 214, 213, 203,
         75, 131, 209,  11, 101, 157,  64, 115,   5, 164,
        105,  64, 206, 108, 185, 155,  68, 199,  67, 127,
        160, 248,  79, 127, 113, 252,   0,  75,  10, 229
    };

    Mat_<uchar> expectedRes(10, 10, expectedRes_data);

    EXPECT_LE(cvtest::norm(res, expectedRes, NORM_INF), 1);
}

TEST(intensity_transform_contrastStretching, accuracy)
{
    uchar image_data[] = {
         32,  59, 164, 127, 151, 107, 167,  62, 195, 143,
         54, 166, 104,  27, 152,  20,  35, 135,  12, 198,
        107,  63,  90, 169,  67, 135, 136,  14,  94, 115,
         34, 150, 169, 171, 130,  39, 190, 108, 103,  32,
         57,  83, 146,  37,  81, 143, 144,  47,  87,  49,
         32, 108,  17, 165, 127, 137, 108,  35, 179, 175,
         40, 148, 174,  79, 146, 119, 103, 168, 167, 160,
         66, 107, 164,  19,  85, 126,  58,  95,  15, 131,
         88,  58, 162,  90, 147, 125,  61, 157,  60, 104,
        128, 193,  69, 104,  94, 196,  11,  66,  18, 179
    };

    Mat_<uchar> image(10, 10, image_data);

    Mat res;
    cv::intensity_transform::contrastStretching(image, res, 70, 15, 120, 240);

    uchar expectedRes_data[] = {
          6,  12, 244, 240, 243, 181, 245,  13, 248, 242,
         11, 245, 168,   5, 243,   4,   7, 241,   2, 248,
        181,  13, 105, 245,  14, 241, 241,   3, 123, 217,
          7, 243, 245, 245, 241,   8, 247, 186, 163,   6,
         12,  73, 242,   7,  64, 242, 242,  10,  91,  10,
          6, 186,   3, 245, 240, 241, 186,   7, 246, 246,
          8, 243, 246,  55, 242, 235, 163, 245, 245, 244,
         14, 181, 244,   4,  82, 240,  12, 127,   3, 241,
         96,  12, 244, 105, 243, 240,  13, 244,  12, 168,
        240, 248,  14, 168, 123, 248,   2,  14,   3, 246
    };

    Mat_<uchar> expectedRes(10, 10, expectedRes_data);

    EXPECT_LE(cvtest::norm(res, expectedRes, NORM_INF), 1);
}

typedef testing::TestWithParam<std::string> intensity_transform_BIMEF;

TEST_P(intensity_transform_BIMEF, accuracy)
{
#ifdef HAVE_EIGEN
    const std::string directory = "intensity_transform/BIMEF/";
    std::string filename = GetParam();

    const std::string inputFilename = cvtest::findDataFile(directory + filename + ".png");
    Mat img = imread(inputFilename);
    EXPECT_TRUE(!img.empty());
    Mat imgBIMEF;
    BIMEF(img, imgBIMEF);

    const std::string referenceFilename = cvtest::findDataFile(directory + filename + "_ref.png");
    Mat imgRef = imread(referenceFilename);
    EXPECT_TRUE(!imgRef.empty());

    EXPECT_EQ(imgBIMEF.rows, imgRef.rows);
    EXPECT_EQ(imgBIMEF.cols, imgRef.cols);
    EXPECT_EQ(imgBIMEF.type(), imgRef.type());
    double rmse = sqrt(cv::norm(imgBIMEF, imgRef, NORM_L2SQR) / (imgRef.total()*imgRef.channels()));
    std::cout << "BIMEF, RMSE for " << filename << ": " << rmse << std::endl;
    const float max_rmse = 9;
    EXPECT_LE(rmse, max_rmse);
#endif
}

const string BIMEF_accuracy_cases[] = {
    "P1000205_resize",
    "P1010676_resize",
    "P1010815_resize"
};

INSTANTIATE_TEST_CASE_P(/*nothing*/, intensity_transform_BIMEF,
    testing::ValuesIn(BIMEF_accuracy_cases)
);

}} // namespace
