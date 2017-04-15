/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "test_precomp.hpp"

TEST(CV_Face_BIF, can_create_default) {
    cv::Ptr<cv::face::BIF> bif;
    EXPECT_NO_THROW(bif = cv::face::createBIF());
    EXPECT_FALSE(bif.empty());
}

TEST(CV_Face_BIF, fails_when_zero_bands) {
    EXPECT_ANY_THROW(cv::face::createBIF(0));
}

TEST(CV_Face_BIF, fails_when_too_many_bands) {
    EXPECT_ANY_THROW(cv::face::createBIF(9));
}

TEST(CV_Face_BIF, fails_when_zero_rotations) {
    EXPECT_ANY_THROW(cv::face::createBIF(8, 0));
}

TEST(CV_Face_BIF, can_compute) {
    cv::Mat image(60, 60, CV_32F);
    cv::theRNG().fill(image, cv::RNG::UNIFORM, -1, 1);

    cv::Ptr<cv::face::BIF> bif = cv::face::createBIF();
    cv::Mat fea;
    EXPECT_NO_THROW(bif->compute(image, fea));
    EXPECT_EQ(cv::Size(1, 13188), fea.size());
}
