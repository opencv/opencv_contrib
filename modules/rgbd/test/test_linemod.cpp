// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(RGBD_Linemod, handlesUnalignedSpreadDestination)
{
    Mat source(32, 218, CV_8UC3);
    randu(source, 0, 255);
    Mat mask(source.size(), CV_8U, Scalar::all(255));

    Ptr<cv::linemod::Detector> detector = cv::linemod::getDefaultLINE();
    EXPECT_GE(detector->addTemplate(std::vector<Mat>(1, source), "object", mask), 0);
}

}} // namespace opencv_test
