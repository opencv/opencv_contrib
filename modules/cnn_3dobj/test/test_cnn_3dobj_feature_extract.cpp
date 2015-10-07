/*
 *  Created on: Aug 14, 2015
 *      Author: Yida Wang
 */

#include "test_precomp.hpp"

using namespace cv;
using namespace cv::cnn_3dobj;

class CV_CNN_Feature_Test : public cvtest::BaseTest
{
public:
    CV_CNN_Feature_Test();
protected:
    void run(int);
};

CV_CNN_Feature_Test::CV_CNN_Feature_Test()
{
}

/**
 * This test checks the following:
 * Feature extraction by the triplet trained CNN model
 */
void CV_CNN_Feature_Test::run(int)
{
    String caffemodel = String(ts->get_data_path()) + "3d_triplet_iter_30000.caffemodel";
    String network_forIMG = cvtest::TS::ptr()->get_data_path() + "3d_triplet_testIMG.prototxt";
    String mean_file = "no";
    std::vector<String> ref_img;
    String target_img = String(ts->get_data_path()) + "1_8.png";
    String feature_blob = "feat";
    String device = "CPU";
    int dev_id = 0;

    cv::Mat img_base = cv::imread(target_img, -1);
    if (img_base.empty())
    {
        ts->printf(cvtest::TS::LOG, "could not read reference image %s\n", target_img.c_str(), "make sure the path of images are set properly.");
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        return;
    }
    cv::cnn_3dobj::descriptorExtractor descriptor(device, dev_id);
    if (strcmp(mean_file.c_str(), "no") == 0)
        descriptor.loadNet(network_forIMG, caffemodel);
    else
        descriptor.loadNet(network_forIMG, caffemodel, mean_file);

    cv::Mat feature_test;
    descriptor.extract(img_base, feature_test, feature_blob);
    Mat feature_reference = (Mat_<float>(1,16) << -134.03548, -203.48265, -105.96752, 55.343075, -211.36378, 487.85968, -182.15063, 62.229042, 297.19876, 206.07578, 291.74951, -19.906454, -464.09152, 135.79895, 420.43616, 2.2887282);
    printf("Reference feature is computed by Caffe extract_features tool by \n To generate values for different images, use extract_features \n with the resetted image list in prototxt.");
    float dist = norm(feature_test - feature_reference);
    if (dist > 5) {
      ts->printf(cvtest::TS::LOG, "Extracted featrue is not the same from the one extracted from Caffe.");
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }
}

TEST(CNN_FEATURE, accuracy) { CV_CNN_Feature_Test test; test.safe_run(); }
