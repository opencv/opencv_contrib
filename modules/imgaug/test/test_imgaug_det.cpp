// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test{ namespace{

void read_annotation(const String& path, std::vector<Rect>& bboxes, std::vector<int>& labels){
    FILE* fp;
    fp = fopen(path.c_str(), "rt");

    int n;
    int sig;
    sig = fscanf(fp, "%d", &n);
    CV_Assert(sig != EOF);

    for(int i=0; i < n; i++){
        int x, y, w, h, l;
        sig = fscanf(fp, "%d %d %d %d %d\n", &x, &y, &w, &h, &l);
        CV_Assert(sig != EOF);
        bboxes.push_back(Rect(x, y, w, h));
        labels.push_back(l);
    }

    fclose(fp);
}


TEST(Aug_Det_RandomFlip, vertical){
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat src = imread(img_path);
    Mat out;

    int seed = 0;
    cv::imgaug::setSeed(seed);


    string ref_path = findDataFile("imgaug/det_random_flip_test_0.jpg");
    Mat ref = imread(ref_path);

    std::vector<Rect> ref_bboxes;
    std::vector<int> ref_labels;

    String ref_data = findDataFile("imgaug/det_random_flip_test_0.dat");
    read_annotation(ref_data, ref_bboxes, ref_labels);


    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels{1, 2};

    int flipCode = 0;
    cv::imgaug::det::RandomFlip aug(flipCode);
    aug.call(src, out, bboxes, labels);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols && ref_bboxes.size() == bboxes.size() && ref_labels.size() == labels.size()) {
        EXPECT_EQ(bboxes, ref_bboxes);
        EXPECT_EQ(labels, ref_labels);

        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}

TEST(Aug_Det_Resize, small){
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat src = imread(img_path);
    Mat out;

    int seed = 0;
    cv::imgaug::setSeed(seed);


    string ref_path = findDataFile("imgaug/det_resize_test_0.jpg");
    Mat ref = imread(ref_path);

    std::vector<Rect> ref_bboxes;
    std::vector<int> ref_labels;

    String ref_data = findDataFile("imgaug/det_resize_test_0.dat");
    read_annotation(ref_data, ref_bboxes, ref_labels);


    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels{1, 2};

    Size size(224, 224);
    cv::imgaug::det::Resize aug(size);
    aug.call(src, out, bboxes, labels);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols && ref_bboxes.size() == bboxes.size() && ref_labels.size() == labels.size()) {
        EXPECT_EQ(bboxes, ref_bboxes);
        EXPECT_EQ(labels, ref_labels);

        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Det_Convert, BGR2GRAY){
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat src = imread(img_path);
    Mat out;

    int seed = 0;
    cv::imgaug::setSeed(seed);


    string ref_path = findDataFile("imgaug/det_convert_test_0.jpg");
    Mat ref = imread(ref_path, IMREAD_GRAYSCALE);

    std::vector<Rect> ref_bboxes;
    std::vector<int> ref_labels;

    String ref_data = findDataFile("imgaug/det_convert_test_0.dat");
    read_annotation(ref_data, ref_bboxes, ref_labels);


    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels{1, 2};

    int code = COLOR_BGR2GRAY;
    cv::imgaug::det::Convert aug(code);
    aug.call(src, out, bboxes, labels);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols && ref_bboxes.size() == bboxes.size() && ref_labels.size() == labels.size()) {
        EXPECT_EQ(bboxes, ref_bboxes);
        EXPECT_EQ(labels, ref_labels);

        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Det_RandomTranslation, no_drop){
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat src = imread(img_path);
    Mat out;

    int seed = 0;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("imgaug/det_random_translation_test_0.jpg");
    Mat ref = imread(ref_path, IMREAD_COLOR);

    std::vector<Rect> ref_bboxes;
    std::vector<int> ref_labels;

    String ref_data = findDataFile("imgaug/det_random_translation_test_0.dat");
    read_annotation(ref_data, ref_bboxes, ref_labels);

    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels{1, 2};

    Vec2d trans(20, 20);
    cv::imgaug::det::RandomTranslation aug(trans);
    aug.call(src, out, bboxes, labels);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols && ref_bboxes.size() == bboxes.size() && ref_labels.size() == labels.size()) {
        EXPECT_EQ(bboxes, ref_bboxes);
        EXPECT_EQ(labels, ref_labels);

        // Calculate the L2 relative error between images.
        double errorL2 = cv::norm( out, ref, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double error = errorL2 / (double)( out.rows * out.cols );
        EXPECT_LE(error, 0.1);
    }else{
        ts->set_failed_test_info(TS::FAIL_MISMATCH);
    }
}


TEST(Aug_Det_RandomRotation, no_drop){
    cvtest::TS* ts = cvtest::TS::ptr();
    string img_path = findDataFile("imgaug/lena.jpg");
    Mat src = imread(img_path);
    Mat out;

    int seed = 0;
    cv::imgaug::setSeed(seed);

    string ref_path = findDataFile("imgaug/det_random_rotation_test_0.jpg");
    Mat ref = imread(ref_path, IMREAD_COLOR);

    std::vector<Rect> ref_bboxes;
    std::vector<int> ref_labels;

    String ref_data = findDataFile("imgaug/det_random_rotation_test_0.dat");
    read_annotation(ref_data, ref_bboxes, ref_labels);

    std::vector<Rect> bboxes{
            Rect{112, 40, 249, 343},
            Rect{61, 273, 113, 228}
    };

    std::vector<int> labels{1, 2};

    Vec2d degrees(-30, 30);
    cv::imgaug::det::RandomRotation aug(degrees);
    aug.call(src, out, bboxes, labels);

    if ( out.rows > 0 && out.rows == ref.rows && out.cols > 0 && out.cols == ref.cols && ref_bboxes.size() == bboxes.size() && ref_labels.size() == labels.size()) {
        EXPECT_EQ(bboxes, ref_bboxes);
        EXPECT_EQ(labels, ref_labels);

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
