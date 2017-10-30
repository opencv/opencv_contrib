// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
This file contains results of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/face.hpp"
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::face;

TEST(CV_Face_Facemark, test_utilities) {
    string image_file = cvtest::findDataFile("face/david1.jpg", true);
    string annotation_file = cvtest::findDataFile("face/david1.pts", true);
    string cascade_filename =
        cvtest::findDataFile("cascadeandhog/cascades/lbpcascade_frontalface.xml", true);

    std::vector<Point2f> facial_points;
    EXPECT_NO_THROW(loadFacePoints(annotation_file,facial_points));

    Mat img = imread(image_file);
    EXPECT_NO_THROW(drawFacemarks(img, facial_points, Scalar(0,0,255)));

    CParams params(cascade_filename);
    std::vector<Rect> faces;
    EXPECT_TRUE(getFaces(img, faces, &params));
}
