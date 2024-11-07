/*
 * Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
*/

#include "opencv2/ts.hpp"
#include "opencv2/fastcv/moments.hpp"

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<bool,Size,int>> fcv_momentsTest;

TEST_P(fcv_momentsTest, accuracy)
{
    const bool binaryImage = get<0>(GetParam());
    const Size srcSize = get<1>(GetParam());
    const MatDepth srcType = get<2>(GetParam());
    Mat src(srcSize, srcType);

	for(int j = 0; j < srcSize.width; ++j)
        for(int i = 0; i < srcSize.height; ++i)
		{
			if(srcType == CV_8UC1)
				src.at<uchar>(i, j) = cv::randu<uchar>();
			else if(srcType == CV_32SC1)
				src.at<int>(i, j) = cv::randu<int>();
			else if(srcType == CV_32FC1)
				src.at<float>(i, j) = cv::randu<float>();
	    }

	cv::Moments m = cv::fastcv::moments(src, binaryImage);

    int len_m = sizeof(m)/sizeof(m.m00);
    EXPECT_FALSE(len_m != 24);
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, fcv_momentsTest, Combine(
                   Values(false, true),
                   Values(TYPICAL_MAT_SIZES),
                   Values(CV_8UC1, CV_32SC1, CV_32FC1)			   
));

}
}
