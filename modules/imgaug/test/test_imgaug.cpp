// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test{ namespace{


TEST(Aug_RandomCrop, no_padding){
    cout << "run test: no_padding" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);

    int th = 200;
    int tw = 200;

    string ref_path = findDataFile("imgaug/random_crop_test_0.jpg");
    Mat ref = imread(ref_path);

    int seed = 0;

    cv::imgaug::setSeed(seed);
    cv::imgaug::RandomCrop aug(Size(tw, th));
    Mat out;
    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomCrop, padding){
    cout << "run test: padding" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);

    int seed = 0;

    int th = 200;
    int tw = 200;
    Vec4d padding {10, 20, 30, 40};

    string ref_path = findDataFile("imgaug/random_crop_test_1.jpg");
    Mat ref = imread(ref_path);

    imgaug::setSeed(seed);
    cv::imgaug::RandomCrop aug(Size(tw, th), padding);
    Mat out;
    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomFlip, diagonal){
    cout << "run test: random flip (diagonal)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/random_flip_test_2.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomFlip aug(0, 1);
    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_Resize, basic){
    cout << "run test: resize (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/resize_test_3.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::Resize aug(cv::Size(256, 128));
    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_CenterCrop, basic){
    cout << "run test: center crop (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/center_crop_test_4.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::CenterCrop aug(cv::Size(400, 300));
    aug.call(input, out);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Pad, basic){
    cout << "run test: pad (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/pad_test_5.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::Pad aug(Vec4i(10, 20, 30, 40), Scalar(0));
    aug.call(input, out);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_RandomResizedCrop, basic){
    cout << "run test: random resized crop (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    cv::Size size(1024, 512);
    uint64 seed = 10;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("imgaug/random_resized_crop_test_6.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomResizedCrop aug(size);

    aug.call(input, out);
    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_RandomRotation, not_expand){
    cout << "run test: random rotation (not_expand)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    cv::Vec2d degrees(-10, 10);
    uint64 seed = 5;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("imgaug/random_rotation_test_7.jpg");
    Mat ref = imread(ref_path);

    cv::imgaug::RandomRotation aug(degrees);

    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_GrayScale, basic){
    cout << "run test: gray scale (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/gray_scale_test_8.jpg");
    Mat ref = imread(ref_path, IMREAD_GRAYSCALE);

    cv::imgaug::GrayScale aug;

    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_GaussianBlur, basic){
    cout << "run test: gaussian blur (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/gaussian_blur_test_9.jpg");
    Mat ref = imread(ref_path);
    cv::imgaug::setSeed(15);
    cv::imgaug::GaussianBlur aug(Size(5, 5));

    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Normalize, basic){
    cout << "run test: gaussian blur (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/normalize_test_10.jpg");
    Mat ref = imread(ref_path);
    cv::imgaug::setSeed(15);
    // Mean and std for ImageNet is [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] in order of RGB.
    // For order of BGR, they should be (0.406, 0.456, 0.485), (0.225, 0.224, 0.229)
    cv::imgaug::Normalize aug(Scalar(0.406, 0.456, 0.485), Scalar(0.225, 0.224, 0.229));
    aug.call(input, out);
    out.convertTo(out, CV_8UC3, 255);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_ColorJitter, basic){
    cout << "run test: color jitter (basic)" << endl;
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat input = imread(img_path);
    Mat out;

    string ref_path = findDataFile("imgaug/color_jitter_test_11.jpg");
    Mat ref = imread(ref_path);
    cv::imgaug::setSeed(15);
    // Mean and std for ImageNet is [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] in order of RGB.
    // For order of BGR, they should be (0.406, 0.456, 0.485), (0.225, 0.224, 0.229)
    cv::imgaug::ColorJitter aug(cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(0, 2), cv::Vec2d(-0.5, 0.5));
    aug.call(input, out);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols ) {
        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


}}
