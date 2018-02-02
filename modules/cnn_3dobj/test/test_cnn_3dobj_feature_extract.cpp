// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: Yida Wang
#include "test_precomp.hpp"

namespace opencv_test { namespace {
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
    String caffemodel = cvtest::findDataFile("contrib/cnn_3dobj/3d_triplet_iter_30000.caffemodel");
    String network_forIMG = cvtest::findDataFile("contrib/cnn_3dobj/3d_triplet_testIMG.prototxt");
    String mean_file = "no";
    std::vector<String> ref_img;
    String target_img = cvtest::findDataFile("contrib/cnn_3dobj/4_78.png");
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
    if (mean_file == "no")
        descriptor.loadNet(network_forIMG, caffemodel);
    else
        descriptor.loadNet(network_forIMG, caffemodel, mean_file);

    cv::Mat feature_test;
    descriptor.extract(img_base, feature_test, feature_blob);
    // Reference feature is computed by Caffe extract_features tool.
    // To generate values for different images, use extract_features with the resetted image list in prototxt.
    Mat feature_reference = (Mat_<float>(1,3) << -312.4805, 8.4768486, -224.98953);
    float dist = norm(feature_test - feature_reference);
    if (dist > 5) {
      ts->printf(cvtest::TS::LOG, "Extracted featrue is not the same from the one extracted from Caffe.");
      ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
      return;
    }
}

TEST(CNN_FEATURE, accuracy) { CV_CNN_Feature_Test test; test.safe_run(); }

}} // namespace
