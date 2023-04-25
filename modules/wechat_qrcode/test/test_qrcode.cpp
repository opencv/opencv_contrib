// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

#include "test_precomp.hpp"
#include "opencv2/objdetect.hpp"

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
std::string qrcode_images_multiple[] = {/*"2_qrcodes.png",*/ "3_close_qrcodes.png", /*"3_qrcodes.png",
                                          "4_qrcodes.png", "5_qrcodes.png",  "6_qrcodes.png",*/
                                          "7_qrcodes.png"/*, "8_close_qrcodes.png"*/};

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
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_version = "_2021-01";
    path_detect_prototxt = findDataFile("dnn/wechat"+model_version+"/detect.prototxt", false);
    path_detect_caffemodel = findDataFile("dnn/wechat"+model_version+"/detect.caffemodel", false);
    path_sr_prototxt = findDataFile("dnn/wechat"+model_version+"/sr.prototxt", false);
    path_sr_caffemodel = findDataFile("dnn/wechat"+model_version+"/sr.caffemodel", false);

    std::string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    vector<Mat> points;
    auto detector = wechat_qrcode::WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt,
                                                path_sr_caffemodel);
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

TEST(Objdetect_QRCode_points_position, rotate45) {
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_version = "_2021-01";
    path_detect_prototxt = findDataFile("dnn/wechat"+model_version+"/detect.prototxt", false);
    path_detect_caffemodel = findDataFile("dnn/wechat"+model_version+"/detect.caffemodel", false);
    path_sr_prototxt = findDataFile("dnn/wechat"+model_version+"/sr.prototxt", false);
    path_sr_caffemodel = findDataFile("dnn/wechat"+model_version+"/sr.caffemodel", false);

    auto detector = wechat_qrcode::WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt,
                                                path_sr_caffemodel);

    const cv::String expect_msg = "OpenCV";
    QRCodeEncoder::Params params;
    params.version = 5; // 37x37
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    Mat qrImage;
    qrcode_enc->encode(expect_msg, qrImage);
    Mat image(800, 800, CV_8UC1);
    const int pixInBlob = 4;
    Size qrSize = Size((21+(params.version-1)*4)*pixInBlob,(21+(params.version-1)*4)*pixInBlob);
    Rect2f rec(static_cast<float>((image.cols - qrSize.width)/2),
               static_cast<float>((image.rows - qrSize.height)/2),
               static_cast<float>(qrSize.width),
               static_cast<float>(qrSize.height));
    vector<float> goldCorners = {rec.x, rec.y,
                                 rec.x+rec.width, rec.y,
                                 rec.x+rec.width, rec.y+rec.height,
                                 rec.x, rec.y+rec.height};
    Mat roiImage = image(rec);
    cv::resize(qrImage, roiImage, qrSize, 1., 1., INTER_NEAREST);

    vector<Mat> points1;
    auto decoded_info1 = detector.detectAndDecode(image, points1);
    ASSERT_EQ(1ull, decoded_info1.size());
    ASSERT_EQ(expect_msg, decoded_info1[0]);
    EXPECT_NEAR(0, cvtest::norm(Mat(goldCorners), points1[0].reshape(1, 8), NORM_INF), 8.);

    const double angle = 45;
    Point2f pc(image.cols/2.f, image.rows/2.f);
    Mat rot = getRotationMatrix2D(pc, angle, 1.);
    warpAffine(image, image, rot, image.size());
    vector<float> rotateGoldCorners;
    for (int i = 0; i < static_cast<int>(goldCorners.size()); i+= 2) {
        rotateGoldCorners.push_back(static_cast<float>(rot.at<double>(0, 0) * goldCorners[i] +
                                    rot.at<double>(0, 1) * goldCorners[i+1] + rot.at<double>(0, 2)));
        rotateGoldCorners.push_back(static_cast<float>(rot.at<double>(1, 0) * goldCorners[i] +
                                    rot.at<double>(1, 1) * goldCorners[i+1] + rot.at<double>(1, 2)));
    }
    vector<Mat> points2;
    auto decoded_info2 = detector.detectAndDecode(image, points2);
    ASSERT_EQ(1ull, decoded_info2.size());
    ASSERT_EQ(expect_msg, decoded_info2[0]);
    EXPECT_NEAR(0, cvtest::norm(Mat(rotateGoldCorners), points2[0].reshape(1, 8), NORM_INF), 11.);
}

INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode, testing::ValuesIn(qrcode_images_name));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Close, testing::ValuesIn(qrcode_images_close));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Monitor, testing::ValuesIn(qrcode_images_monitor));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Curved, testing::ValuesIn(qrcode_images_curved));
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Multi, testing::ValuesIn(qrcode_images_multiple));

TEST(Objdetect_QRCode_Big, regression) {
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_version = "_2021-01";
    path_detect_prototxt = findDataFile("dnn/wechat"+model_version+"/detect.prototxt", false);
    path_detect_caffemodel = findDataFile("dnn/wechat"+model_version+"/detect.caffemodel", false);
    path_sr_prototxt = findDataFile("dnn/wechat"+model_version+"/sr.prototxt", false);
    path_sr_caffemodel = findDataFile("dnn/wechat"+model_version+"/sr.caffemodel", false);

    auto detector = wechat_qrcode::WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt,
                                                path_sr_caffemodel);

    const cv::String expect_msg = "OpenCV";
    QRCodeEncoder::Params params;
    params.version = 4; // 33x33
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    Mat qrImage;
    qrcode_enc->encode(expect_msg, qrImage);
    Mat largeImage(4032, 3024, CV_8UC1);
    const int pixInBlob = 4;
    Size qrSize = Size((21+(params.version-1)*4)*pixInBlob,(21+(params.version-1)*4)*pixInBlob);
    Mat roiImage = largeImage(Rect((largeImage.cols - qrSize.width)/2, (largeImage.rows - qrSize.height)/2,
                                   qrSize.width, qrSize.height));
    cv::resize(qrImage, roiImage, qrSize, 1., 1., INTER_NEAREST);

    vector<Mat> points;
    detector.setScaleFactor(0.25f);
    auto decoded_info = detector.detectAndDecode(largeImage, points);
    ASSERT_EQ(1ull, decoded_info.size());
    ASSERT_EQ(expect_msg, decoded_info[0]);
}

TEST(Objdetect_QRCode_Tiny, regression) {
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_version = "_2021-01";
    path_detect_prototxt = findDataFile("dnn/wechat"+model_version+"/detect.prototxt", false);
    path_detect_caffemodel = findDataFile("dnn/wechat"+model_version+"/detect.caffemodel", false);
    path_sr_prototxt = findDataFile("dnn/wechat"+model_version+"/sr.prototxt", false);
    path_sr_caffemodel = findDataFile("dnn/wechat"+model_version+"/sr.caffemodel", false);

    auto detector = wechat_qrcode::WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt,
                                                path_sr_caffemodel);

    const cv::String expect_msg = "OpenCV";
    QRCodeEncoder::Params params;
    params.version = 4; // 33x33
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    Mat qrImage;
    qrcode_enc->encode(expect_msg, qrImage);
    Mat tinyImage(80, 80, CV_8UC1);
    const int pixInBlob = 2;
    Size qrSize = Size((21+(params.version-1)*4)*pixInBlob,(21+(params.version-1)*4)*pixInBlob);
    Mat roiImage = tinyImage(Rect((tinyImage.cols - qrSize.width)/2, (tinyImage.rows - qrSize.height)/2,
                                   qrSize.width, qrSize.height));
    cv::resize(qrImage, roiImage, qrSize, 1., 1., INTER_NEAREST);

    vector<Mat> points;
    auto decoded_info = detector.detectAndDecode(tinyImage, points);
    ASSERT_EQ(1ull, decoded_info.size());
    ASSERT_EQ(expect_msg, decoded_info[0]);
}


typedef testing::TestWithParam<std::string> Objdetect_QRCode_Easy_Multi;
TEST_P(Objdetect_QRCode_Easy_Multi, regression) {
    string path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt, path_sr_caffemodel;
    string model_path = GetParam();

    if (!model_path.empty()) {
        path_detect_prototxt = findDataFile(model_path + "/detect.prototxt", false);
        path_detect_caffemodel = findDataFile(model_path + "/detect.caffemodel", false);
        path_sr_prototxt = findDataFile(model_path + "/sr.prototxt", false);
        path_sr_caffemodel = findDataFile(model_path + "/sr.caffemodel", false);
    }

    auto detector = wechat_qrcode::WeChatQRCode(path_detect_prototxt, path_detect_caffemodel, path_sr_prototxt,
                                                path_sr_caffemodel);

    const cv::String expect_msg1 = "OpenCV1", expect_msg2 = "OpenCV2";
    QRCodeEncoder::Params params;
    params.version = 4; // 33x33
    Ptr<QRCodeEncoder> qrcode_enc = cv::QRCodeEncoder::create(params);
    Mat qrImage1, qrImage2;
    qrcode_enc->encode(expect_msg1, qrImage1);
    qrcode_enc->encode(expect_msg2, qrImage2);
    const int pixInBlob = 2;
    const int offset = 14;
    const int qr_size = (params.version - 1) * 4 + 21;
    Mat tinyImage = Mat::zeros(qr_size*pixInBlob+offset, (qr_size*pixInBlob+offset)*2, CV_8UC1);
    Size qrSize = Size(qrImage1.cols, qrImage1.rows);

    Mat roiImage = tinyImage(Rect((tinyImage.cols/2 - qrSize.width)/2, (tinyImage.rows - qrSize.height)/2,
                                   qrSize.width, qrSize.height));
    cv::resize(qrImage1, roiImage, qrSize, 1., 1., INTER_NEAREST);

    roiImage = tinyImage(Rect((tinyImage.cols/2 - qrSize.width)/2+tinyImage.cols/2, (tinyImage.rows - qrSize.height)/2,
                                   qrSize.width, qrSize.height));
    cv::resize(qrImage2, roiImage, qrSize, 1., 1., INTER_NEAREST);

    vector<Mat> points;
    auto decoded_info = detector.detectAndDecode(tinyImage, points);
    ASSERT_EQ(2ull, decoded_info.size());
    ASSERT_TRUE((expect_msg1 == decoded_info[0] && expect_msg2 == decoded_info[1]) ||
                (expect_msg1 == decoded_info[1] && expect_msg2 == decoded_info[0]));
}

std::string qrcode_model_path[] = {"", "dnn/wechat_2021-01"};
INSTANTIATE_TEST_CASE_P(/**/, Objdetect_QRCode_Easy_Multi, testing::ValuesIn(qrcode_model_path));

TEST(Objdetect_QRCode_bug, issue_3478) {
    auto detector = wechat_qrcode::WeChatQRCode();
    std::string image_path = findDataFile("qrcode/issue_3478.png");
    Mat src = imread(image_path, IMREAD_GRAYSCALE);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    std::vector<std::string> outs = detector.detectAndDecode(src);
    ASSERT_EQ(1, (int) outs.size());
    ASSERT_EQ(16, (int) outs[0].size());
    ASSERT_EQ("KFCVW50         ", outs[0]);
}

}  // namespace
}  // namespace opencv_test
