/*
 *  Created on: Aug 14, 2015
 *      Author: yidawang
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
    string caffemodel = ts->get_data_path() + "cnn_3dobj/samples/data/3d_triplet_iter_20000.caffemodel";
    string network_forIMG   = ts->get_data_path() + "cnn_3dobj/samples/data/3d_triplet_testIMG.prototxt";
    string mean_file    = "no";
    string target_img   = ts->get_data_path() + "cnn_3dobj/samples/data/images_all/2_24.png";
    string feature_blob = "feat";
    string device = "CPU";
    int dev_id = 0;

    cv::cnn_3dobj::descriptorExtractor descriptor(device, dev_id);
    if (strcmp(mean_file.c_str(), "no") == 0)
        descriptor.loadNet(network_forIMG, caffemodel);
    else
        descriptor.loadNet(network_forIMG, caffemodel, mean_file);
    cv::Mat img = cv::imread(target_img, -1);
    if (img.empty()) {
      ts->printf(cvtest::TS::LOG, "could not read image %s\n", target_img.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }
    cv::Mat feature_test;
    descriptor.extract(img, feature_test, feature_blob);
    if (feature_test.empty()) {
      ts->printf(cvtest::TS::LOG, "could not extract feature from image %s\n", target_img.c_str());
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }

}

TEST(VIDEO_BGSUBGMG, accuracy) { CV_CNN_Feature_Test test; test.safe_run(); }
