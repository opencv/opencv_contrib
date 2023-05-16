// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test{namespace{

typedef ::perf::TestBaseWithParam< string > Perf_Barcode_multi;
typedef ::perf::TestBaseWithParam< string > Perf_Barcode_single;

PERF_TEST_P_(Perf_Barcode_multi, detect)
{
    const string name_current_image = GetParam();
    const string root = "cv/barcode/multiple/";

    auto bardet = barcode::BarcodeDetector();
    vector< Point > corners;
    string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    cout << src.size << endl;
    TEST_CYCLE() ASSERT_TRUE(bardet.detect(src, corners));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Barcode_multi, decode)
{
    const string name_current_image = GetParam();
    const string root = "cv/barcode/multiple/";

    auto bardet = barcode::BarcodeDetector();
    vector<cv::String> decoded_info;
    vector<barcode::BarcodeType> decoded_type;
    vector< Point > corners;
    string image_path = findDataFile(root + name_current_image);

    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    cout << src.size << endl;
    bardet.detect(src, corners);

    TEST_CYCLE() ASSERT_TRUE(bardet.decode(src, corners, decoded_info, decoded_type));
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Barcode_single, detect)
{
    const string name_current_image = GetParam();
    const string root = "cv/barcode/single/";
    auto bardet = barcode::BarcodeDetector();
    vector< Point > corners;
    string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;

    cout << src.size << endl;
    TEST_CYCLE() ASSERT_TRUE(bardet.detect(src, corners));
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(Perf_Barcode_single, decode)
{
    const string name_current_image = GetParam();
    const string root = "cv/barcode/single/";

    auto bardet = barcode::BarcodeDetector();
    vector<cv::String> decoded_info;
    vector<barcode::BarcodeType> decoded_type;

    string image_path = findDataFile(root + name_current_image);
    Mat src = imread(image_path);
    ASSERT_FALSE(src.empty()) << "Can't read image: " << image_path;
    vector< Point > corners;

    cout << src.size << endl;
    bardet.detect(src, corners);

    TEST_CYCLE() ASSERT_TRUE(bardet.decode(src, corners, decoded_info, decoded_type));
    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Barcode_multi, ::testing::Values("4_barcodes.jpg"));
INSTANTIATE_TEST_CASE_P(/*nothing*/, Perf_Barcode_single, ::testing::Values("book.jpg", "bottle_1.jpg", "bottle_2.jpg"));

}} //namespace