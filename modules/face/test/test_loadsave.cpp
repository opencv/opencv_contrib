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

// regression for #1267
// let's make sure, that both Algorithm::save(String) and
// FaceRecognizer::write(String) lead to the same result

void make_test_data(std::vector<cv::Mat> &images, std::vector<int> &labels) {
    for (int i=0; i<5; i++) {
        cv::Mat m(100,100,CV_8U);
        cv::randu(m,0,255);
        images.push_back(m);
        labels.push_back(i);
    }
}

TEST(CV_Face_SAVELOAD, use_save) {
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    make_test_data(images, labels);
    cv::Ptr<cv::face::FaceRecognizer> model1 = cv::face::LBPHFaceRecognizer::create();
    model1->train(images,labels);
    model1->save("fr.xml");
    int p1 = model1->predict(images[2]);
    cv::Ptr<cv::face::FaceRecognizer> model2 = cv::face::LBPHFaceRecognizer::create();
    model2->read("fr.xml");
    EXPECT_EQ(model2->empty(), false);
    EXPECT_EQ(p1, model2->predict(images[2]));
}

TEST(CV_Face_SAVELOAD, use_write) {
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    make_test_data(images, labels);
    cv::Ptr<cv::face::FaceRecognizer> model1 = cv::face::LBPHFaceRecognizer::create();
    model1->train(images,labels);
    model1->write("fr.xml");
    int p1 = model1->predict(images[2]);
    cv::Ptr<cv::face::FaceRecognizer> model2 = cv::face::LBPHFaceRecognizer::create();
    model2->read("fr.xml");
    EXPECT_EQ(model2->empty(), false);
    EXPECT_EQ(p1, model2->predict(images[2]));
}
