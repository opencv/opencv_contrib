// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.


#include "test_precomp.hpp"

namespace opencv_test { namespace {


static float disparity_MAE(const Mat &reference, const Mat &estimation)
{
    int elems=0;
    float error=0;
    float ref_invalid=0;
    for (int row=0; row< reference.rows; row++){
        for (int col=0; col<reference.cols; col++){
            float ref_val = reference.at<float>(row, col);
            float estimated_val = estimation.at<float>(row, col);
            if (ref_val == 0){
                ref_invalid++;
            }
            // filter out pixels with unknown reference value and pixels whose disparity did not get estimated.
            if (estimated_val == 0 || ref_val == 0 || std::isnan(estimated_val)){
                continue;
            }
            else{
                error+=abs(ref_val - estimated_val);
                elems+=1;
            }
        }
    }
    return error/elems;
}


// void CV_QdsMatchingTest::run(int)
TEST(qds_getDisparity, accuracy)
{
    //load data
    Mat image1, image2, gt;
    image1 = imread(cvtest::TS::ptr()->get_data_path() + "stereomatching/datasets/cones/im2.png", IMREAD_GRAYSCALE);
    image2 = imread(cvtest::TS::ptr()->get_data_path() + "stereomatching/datasets/cones/im6.png", IMREAD_GRAYSCALE);
    gt = imread(cvtest::TS::ptr()->get_data_path() + "stereomatching/datasets/cones/disp2.png", IMREAD_GRAYSCALE);

    // reference scale factor is based on this https://github.com/opencv/opencv_extra/blob/master/testdata/cv/stereomatching/datasets/datasets.xml
    gt.convertTo(gt, CV_32F);
    gt =gt/4;

    //test inputs
    ASSERT_FALSE(image1.empty() || image2.empty() || gt.empty()) << "Issue with input data";

    //configure disparity algorithm
    cv::Size frameSize = image1.size();
    Ptr<stereo::QuasiDenseStereo> qds_matcher = stereo::QuasiDenseStereo::create(frameSize);


    //compute disparity
    qds_matcher->process(image1, image2);
    Mat outDisp = qds_matcher->getDisparity();

    // test input output size consistency
    ASSERT_EQ(gt.size(), outDisp.size()) << "Mismatch input/output dimensions";
    ASSERT_LT(disparity_MAE(gt, outDisp),2) << "EPE should be 1.1053 for this sample/hyperparamters (Tested on version 4.5.1)";
}



}} // namespace