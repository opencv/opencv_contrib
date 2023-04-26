// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{
namespace
{

std::string qrcode_images_name[] = {
    "version_1_top.jpg",
    "version_2_left.jpg", "version_2_up.jpg", "version_2_top.jpg",
    "version_3_down.jpg", "version_3_top.jpg",
    "version_4_top.jpg",
    "version_5_down.jpg", "version_5_left.jpg", "version_5_up.jpg", "version_5_top.jpg",
    "russian.jpg", "kanji.jpg", "link_wiki_cv.jpg"};

std::string qrcode_images_multiple[] = {"2_qrcodes.png", "3_qrcodes.png", "3_close_qrcodes.png",
                                        "4_qrcodes.png", "5_qrcodes.png", "7_qrcodes.png"};

bool Find_Models_Files(std::vector<std::string>& models) {
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_version = "_2021-01";
    path_detect_prototxt = findDataFile("dnn/wechat"+model_version+"/detect.prototxt", false);
    path_detect_caffemodel = findDataFile("dnn/wechat"+model_version+"/detect.caffemodel", false);
    path_sr_prototxt = findDataFile("dnn/wechat"+model_version+"/sr.prototxt", false);
    path_sr_caffemodel = findDataFile("dnn/wechat"+model_version+"/sr.caffemodel", false);
    models = {path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel};
    return true;
}

typedef ::perf::TestBaseWithParam< std::string > Perf_Objdetect_QRCode;

PERF_TEST_P_(Perf_Objdetect_QRCode, detect_and_decode_without_nn)
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Mat > corners;
    auto detector = wechat_qrcode::WeChatQRCode();

    TEST_CYCLE()
    {   
        auto decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_FALSE(decoded_info[0].empty());
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Objdetect_QRCode, detect_and_decode)
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Mat > corners;
    std::vector<std::string> models;
    ASSERT_TRUE(Find_Models_Files(models));
    auto detector = wechat_qrcode::WeChatQRCode(models[0], models[1], models[2], models[3]);
    TEST_CYCLE()
    {
        auto decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_FALSE(decoded_info[0].empty());
    }
    SANITY_CHECK_NOTHING();
}

typedef ::perf::TestBaseWithParam< std::string > Perf_Objdetect_QRCode_Multi;

PERF_TEST_P_(Perf_Objdetect_QRCode_Multi, detect_and_decode_without_nn) 
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Mat > corners;
    auto detector = wechat_qrcode::WeChatQRCode();

    TEST_CYCLE()
    {   
        auto decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_TRUE(decoded_info.size());
        for(size_t i = 0; i < decoded_info.size(); i++)
        {
            ASSERT_FALSE(decoded_info[i].empty());
        }
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Objdetect_QRCode_Multi, detect_and_decode)
{
    const std::string name_current_image = GetParam();
    const std::string root = "cv/qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector<std::string> models;
    ASSERT_TRUE(Find_Models_Files(models));
    auto detector = wechat_qrcode::WeChatQRCode(models[0], models[1], models[2], models[3]);
    std::vector< Mat > corners;                                                
    TEST_CYCLE()
    {   
        auto decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_TRUE(decoded_info.size());
        for(size_t i = 0; i < decoded_info.size(); i++)
        {
            ASSERT_FALSE(decoded_info[i].empty());
        }
    }
    SANITY_CHECK_NOTHING();
}

typedef ::perf::TestBaseWithParam< tuple< std::string, Size > > Perf_Objdetect_Not_QRCode;

PERF_TEST_P_(Perf_Objdetect_Not_QRCode, detect_and_decode_without_nn)
{
    std::string type_gen = get<0>(GetParam());
    Size resolution = get<1>(GetParam());
    Mat not_qr_code(resolution, CV_8UC1, Scalar(0));
    if (type_gen == "random")
    {
        RNG rng;
        rng.fill(not_qr_code, RNG::UNIFORM, Scalar(0), Scalar(1));
    }
    if (type_gen == "chessboard")
    {
        uint8_t next_pixel = 0;
        for (int r = 0; r < not_qr_code.rows * not_qr_code.cols; r++)
        {
            int i = r / not_qr_code.cols;
            int j = r % not_qr_code.cols;
            not_qr_code.ptr<uchar>(i)[j] = next_pixel;
            next_pixel = 255 - next_pixel;
        }
    }
    std::vector< Mat > corners;
    auto detector = wechat_qrcode::WeChatQRCode();

    TEST_CYCLE() 
    {
        auto decoded_info = detector.detectAndDecode(not_qr_code, corners);
        ASSERT_FALSE(decoded_info.size());
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Objdetect_Not_QRCode, detect_and_decode)
{
    std::string type_gen = get<0>(GetParam());
    Size resolution = get<1>(GetParam());
    Mat not_qr_code(resolution, CV_8UC1, Scalar(0));
    if (type_gen == "random")
    {
        RNG rng;
        rng.fill(not_qr_code, RNG::UNIFORM, Scalar(0), Scalar(1));
    }
    if (type_gen == "chessboard")
    {
        uint8_t next_pixel = 0;
        for (int r = 0; r < not_qr_code.rows * not_qr_code.cols; r++)
        {
            int i = r / not_qr_code.cols;
            int j = r % not_qr_code.cols;
            not_qr_code.ptr<uchar>(i)[j] = next_pixel;
            next_pixel = 255 - next_pixel;
        }
    }
    std::vector< Mat > corners;
    std::vector<std::string> models;
    ASSERT_TRUE(Find_Models_Files(models));
    auto detector = wechat_qrcode::WeChatQRCode(models[0], models[1], models[2], models[3]);

    TEST_CYCLE() 
    {
        auto decoded_info = detector.detectAndDecode(not_qr_code, corners);
        ASSERT_FALSE(decoded_info.size());
    }
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode_Multi, testing::ValuesIn(qrcode_images_multiple));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_Not_QRCode,
      ::testing::Combine(
            ::testing::Values("zero", "random", "chessboard"),
            ::testing::Values(Size(640, 480),   Size(1280, 720))
      ));

} // namespace
} // namespace
