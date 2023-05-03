// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"
namespace opencv_test
{
namespace
{
std::string qrcode_model_path[] = {"", "dnn/wechat_2021-01"};

std::string qrcode_images_name[] = {
    "version_1_top.jpg",
    "version_2_left.jpg", "version_2_up.jpg", "version_2_top.jpg",
    "version_3_down.jpg", "version_3_top.jpg",
    "version_4_top.jpg",
    "version_5_down.jpg", "version_5_left.jpg", "version_5_up.jpg", "version_5_top.jpg",
    "russian.jpg", "kanji.jpg", "link_wiki_cv.jpg"};

std::string qrcode_images_multiple[] = {"2_qrcodes.png", "3_qrcodes.png", "3_close_qrcodes.png",
                                        "4_qrcodes.png", "5_qrcodes.png", "7_qrcodes.png"};

WeChatQRCode createQRDetectorWithDNN(std::string& model_path)
{
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    if (!model_path.empty())
    {
        path_detect_prototxt = findDataFile(model_path + "/detect.prototxt", false);
        path_detect_caffemodel = findDataFile(model_path + "/detect.caffemodel", false);
        path_sr_prototxt = findDataFile(model_path + "/sr.prototxt", false);
        path_sr_caffemodel = findDataFile(model_path + "/sr.caffemodel", false);
    }
    return WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel);
}

typedef ::perf::TestBaseWithParam< tuple< std::string,std::string > > Perf_Objdetect_QRCode;

PERF_TEST_P_(Perf_Objdetect_QRCode, detect_and_decode)
{
    std::string model_path = get<0>(GetParam());
    std::string name_current_image = get<1>(GetParam());
    const std::string root = "cv/qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Mat > corners;
    std::vector< String > decoded_info;
    auto detector = createQRDetectorWithDNN(model_path);
    // warmup
    if (!model_path.empty())
    {
        decoded_info = detector.detectAndDecode(src, corners);
    }
    TEST_CYCLE()
    {
        decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_FALSE(decoded_info[0].empty());
    }
    SANITY_CHECK_NOTHING();
}

typedef ::perf::TestBaseWithParam< tuple< std::string,std::string > > Perf_Objdetect_QRCode_Multi;

PERF_TEST_P_(Perf_Objdetect_QRCode_Multi, detect_and_decode)
{
    std::string model_path = get<0>(GetParam());
    std::string name_current_image = get<1>(GetParam());
    const std::string root = "cv/qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE), straight_barcode;
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    std::vector< Mat > corners;
    std::vector< String > decoded_info;
    auto detector = createQRDetectorWithDNN(model_path);
    // warmup
    if (!model_path.empty())
    {
        decoded_info = detector.detectAndDecode(src, corners);
    }
    TEST_CYCLE()
    {
        decoded_info = detector.detectAndDecode(src, corners);
        ASSERT_TRUE(decoded_info.size());
    }
    for(size_t i = 0; i < decoded_info.size(); i++)
    {
        ASSERT_FALSE(decoded_info[i].empty());
    }
    SANITY_CHECK_NOTHING();
}

typedef ::perf::TestBaseWithParam< tuple<std::string, std::string, Size> >Perf_Objdetect_Not_QRCode;

PERF_TEST_P_(Perf_Objdetect_Not_QRCode, detect_and_decode)
{
    std::string model_path = get<0>(GetParam());
    std::string type_gen = get<1>(GetParam());
    Size resolution = get<2>(GetParam());
    Mat not_qr_code(resolution, CV_8UC1, Scalar(0));
    if (type_gen == "random")
    {
        RNG rng;
        rng.fill(not_qr_code, RNG::UNIFORM, Scalar(0), Scalar(1));
    }
    else if (type_gen == "chessboard")
    {
        uint8_t next_pixel = 255;
        for (int j = 0; j < not_qr_code.cols; j++)
        {
            not_qr_code.ptr<uchar>(0)[j] = next_pixel;
            next_pixel = 255 - next_pixel;
        }
        for (int r = not_qr_code.cols; r < not_qr_code.rows * not_qr_code.cols; r++)
        {
            int i = r / not_qr_code.cols;
            int j = r % not_qr_code.cols;
            not_qr_code.ptr<uchar>(i)[j] = 255 - not_qr_code.ptr<uchar>(i-1)[j];
        }
    }
    std::vector< Mat > corners;
    std::vector< String > decoded_info;
    auto detector = createQRDetectorWithDNN(model_path);
    // warmup
    if (!model_path.empty())
    {
        decoded_info = detector.detectAndDecode(not_qr_code, corners);
    }
    TEST_CYCLE()
    {
        decoded_info = detector.detectAndDecode(not_qr_code, corners);
        ASSERT_FALSE(decoded_info.size());
    }
    SANITY_CHECK_NOTHING();
}


INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode,
      ::testing::Combine(
            ::testing::ValuesIn(qrcode_model_path),
            ::testing::ValuesIn(qrcode_images_name)
      ));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_QRCode_Multi,
      ::testing::Combine(
            ::testing::ValuesIn(qrcode_model_path),
            ::testing::ValuesIn(qrcode_images_multiple)
      ));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Objdetect_Not_QRCode,
      ::testing::Combine(
            ::testing::ValuesIn(qrcode_model_path),
            ::testing::Values("zero", "random", "chessboard"),
            ::testing::Values(Size(640, 480),   Size(1280, 720))
      ));

} // namespace
} // namespace
