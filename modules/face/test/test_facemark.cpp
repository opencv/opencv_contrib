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
#include "opencv2/imgcodecs.hpp"
#include "opencv2/face.hpp"
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::face;

TEST(CV_Face_Facemark, can_load_annotation) {
    string annotation_file = cvtest::findDataFile("face/therock.pts", true);


}

TEST(CV_Face_Facemark, test_utilities) {
    string image_file = cvtest::findDataFile("face/therock.jpg", true);
    string annotation_file = cvtest::findDataFile("face/therock.pts", true);
    string cascade_filename =
        cvtest::findDataFile("face/lbpcascade_frontalface.xml", true);

    std::vector<Point2f> facial_points;
    EXPECT_NO_THROW(loadFacePoints(annotation_file,facial_points));

    Mat img = imread(image_file);
    EXPECT_NO_THROW(drawFacemarks(img, facial_points, Scalar(0,0,255)));

    std::vector<Rect> faces;
    EXPECT_TRUE(getFacesHaar(img, faces, cascade_filename));
}
