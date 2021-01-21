// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "test_precomp.hpp"

namespace opencv_test {
namespace {
std::string qrcode_images_name[] = {
    "version_1_down.jpg", /*"version_1_left.jpg",  "version_1_right.jpg", "version_1_up.jpg",*/
    "version_1_top.jpg",
    /*"version_2_down.jpg",*/ "version_2_left.jpg", /*"version_2_right.jpg",*/
    "version_2_up.jpg",
    "version_2_top.jpg",
    "version_3_down.jpg",
    "version_3_left.jpg",
    /*"version_3_right.jpg",*/ "version_3_up.jpg",
    "version_3_top.jpg",
    "version_4_down.jpg",
    "version_4_left.jpg",
    /*"version_4_right.jpg",*/ "version_4_up.jpg",
    "version_4_top.jpg",
    "version_5_down.jpg",
    "version_5_left.jpg",
    /*"version_5_right.jpg",*/ "version_5_up.jpg",
    "version_5_top.jpg",
    "russian.jpg",
    "kanji.jpg", /*"link_github_ocv.jpg",*/
    "link_ocv.jpg",
    "link_wiki_cv.jpg"};

std::string qrcode_images_close[] = {/*"close_1.png",*/ "close_2.png", "close_3.png", "close_4.png",
                                     "close_5.png"};
std::string qrcode_images_monitor[] = {"monitor_1.png", "monitor_2.png", "monitor_3.png",
                                       "monitor_4.png", "monitor_5.png"};
std::string qrcode_images_curved[] = {"curved_1.jpg", /*"curved_2.jpg", "curved_3.jpg",
                                      "curved_4.jpg",*/
                                      "curved_5.jpg", "curved_6.jpg",
                                      /*"curved_7.jpg", "curved_8.jpg"*/};
// std::string qrcode_images_multiple[] = {"2_qrcodes.png", "3_close_qrcodes.png", "3_qrcodes.png",
//                                         "4_qrcodes.png", "5_qrcodes.png",       "6_qrcodes.png",
//                                         "7_qrcodes.png", "8_close_qrcodes.png"};

typedef testing::TestWithParam<std::string> Objdetect_QRCode;
TEST_P(Objdetect_QRCode, regression) {
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    // can not find the model file
    // so we temporarily comment it out
    // auto detector = wechat_qrcode::WeChatQRCode(
    //     findDataFile("detect.prototxt", false), findDataFile("detect.caffemodel", false),
    //     findDataFile("sr.prototxt", false), findDataFile("sr.caffemodel", false));
    auto detector = wechat_qrcode::WeChatQRCode();
    auto decoded_info = detector.detectAndDecode(src, points);

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["test_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u)
            << "Can't find validation data entries in 'test_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++) {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image) {
                std::string original_info = config["info"];
                string decoded_str;
                if (decoded_info.size()) {
                    decoded_str = decoded_info[0];
                }
                EXPECT_EQ(decoded_str, original_info);
                return;  // done
            }
        }
        std::cerr << "Not found results for '" << name_current_image
                  << "' image in config file:" << dataset_config << std::endl
                  << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
                  << std::endl;
    }
}

typedef testing::TestWithParam<std::string> Objdetect_QRCode_Close;
TEST_P(Objdetect_QRCode_Close, regression) {
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/close/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    // can not find the model file
    // so we temporarily comment it out
    // auto detector = wechat_qrcode::WeChatQRCode(
    //     findDataFile("detect.prototxt", false), findDataFile("detect.caffemodel", false),
    //     findDataFile("sr.prototxt", false), findDataFile("sr.caffemodel", false));
    auto detector = wechat_qrcode::WeChatQRCode();
    auto decoded_info = detector.detectAndDecode(src, points);

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["close_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u)
            << "Can't find validation data entries in 'close_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++) {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image) {
                std::string original_info = config["info"];
                string decoded_str;
                if (decoded_info.size()) {
                    decoded_str = decoded_info[0];
                }
                EXPECT_EQ(decoded_str, original_info);
                return;  // done
            }
        }
        std::cerr << "Not found results for '" << name_current_image
                  << "' image in config file:" << dataset_config << std::endl
                  << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
                  << std::endl;
    }
}

typedef testing::TestWithParam<std::string> Objdetect_QRCode_Monitor;
TEST_P(Objdetect_QRCode_Monitor, regression) {
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/monitor/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    // can not find the model file
    // so we temporarily comment it out
    // auto detector = wechat_qrcode::WeChatQRCode(
    //     findDataFile("detect.prototxt", false), findDataFile("detect.caffemodel", false),
    //     findDataFile("sr.prototxt", false), findDataFile("sr.caffemodel", false));
    auto detector = wechat_qrcode::WeChatQRCode();
    auto decoded_info = detector.detectAndDecode(src, points);

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["monitor_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u)
            << "Can't find validation data entries in 'monitor_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++) {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image) {
                std::string original_info = config["info"];
                string decoded_str;
                if (decoded_info.size()) {
                    decoded_str = decoded_info[0];
                }
                EXPECT_EQ(decoded_str, original_info);
                return;  // done
            }
        }
        std::cerr << "Not found results for '" << name_current_image
                  << "' image in config file:" << dataset_config << std::endl
                  << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
                  << std::endl;
    }
}

typedef testing::TestWithParam<std::string> Objdetect_QRCode_Curved;
TEST_P(Objdetect_QRCode_Curved, regression) {
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/curved/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    // can not find the model file
    // so we temporarily comment it out
    // auto detector = wechat_qrcode::WeChatQRCode(
    //     findDataFile("detect.prototxt", false), findDataFile("detect.caffemodel", false),
    //     findDataFile("sr.prototxt", false), findDataFile("sr.caffemodel", false));
    auto detector = wechat_qrcode::WeChatQRCode();
    auto decoded_info = detector.detectAndDecode(src, points);

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["test_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u)
            << "Can't find validation data entries in 'test_images': " << dataset_config;

        for (size_t index = 0; index < images_count; index++) {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image) {
                std::string original_info = config["info"];
                string decoded_str;
                if (decoded_info.size()) {
                    decoded_str = decoded_info[0];
                }
                EXPECT_EQ(decoded_str, original_info);
                return;  // done
            }
        }
        std::cerr << "Not found results for '" << name_current_image
                  << "' image in config file:" << dataset_config << std::endl
                  << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
                  << std::endl;
    }
}

typedef testing::TestWithParam<std::string> Objdetect_QRCode_Multi;
TEST_P(Objdetect_QRCode_Multi, regression) {
    const std::string name_current_image = GetParam();
    const std::string root = "qrcode/multiple/";

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    // can not find the model file
    // so we temporarily comment it out
    // auto detector = wechat_qrcode::WeChatQRCode(
    //     findDataFile("detect.prototxt", false), findDataFile("detect.caffemodel", false),
    //     findDataFile("sr.prototxt", false), findDataFile("sr.caffemodel", false));
    auto detector = wechat_qrcode::WeChatQRCode();
    vector<string> decoded_info = detector.detectAndDecode(src, points);

    const std::string dataset_config = findDataFile(root + "dataset_config.json");
    FileStorage file_config(dataset_config, FileStorage::READ);
    ASSERT_TRUE(file_config.isOpened()) << "Can't read validation data: " << dataset_config;
    {
        FileNode images_list = file_config["multiple_images"];
        size_t images_count = static_cast<size_t>(images_list.size());
        ASSERT_GT(images_count, 0u)
            << "Can't find validation data entries in 'test_images': " << dataset_config;
        for (size_t index = 0; index < images_count; index++) {
            FileNode config = images_list[(int)index];
            std::string name_test_image = config["image_name"];
            if (name_test_image == name_current_image) {
                size_t count_eq_info = 0;
                for (int i = 0; i < int(decoded_info.size()); i++) {
                    for (int j = 0; j < int(config["info"].size()); j++) {
                        std::string original_info = config["info"][j];
                        if (original_info == decoded_info[i]) {
                            count_eq_info++;
                            break;
                        }
                    }
                }
                EXPECT_EQ(config["info"].size(), count_eq_info);
                return;  // done
            }
        }
        std::cerr << "Not found results for '" << name_current_image
                  << "' image in config file:" << dataset_config << std::endl
                  << "Re-run tests with enabled UPDATE_QRCODE_TEST_DATA macro to update test data."
                  << std::endl;
    }
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Close, testing::ValuesIn(qrcode_images_close));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Monitor, testing::ValuesIn(qrcode_images_monitor));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Curved, testing::ValuesIn(qrcode_images_curved));
// INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Multi, testing::ValuesIn(qrcode_images_multiple));

}  // namespace
}  // namespace opencv_test
