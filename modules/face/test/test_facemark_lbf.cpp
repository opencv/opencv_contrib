// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
This file contains results of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

/*Usage:
 download the opencv_extra from https://github.com/opencv/opencv_extra
 and then execute the following commands:
 export OPENCV_TEST_DATA_PATH=/home/opencv/opencv_extra/testdata
 <build_folder>/bin/opencv_test_face
*/

#include "test_precomp.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/face.hpp"
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier cascade_detector;
static bool myCustomDetector( InputArray image, OutputArray ROIs, void * config = 0 ){
    Mat gray;
    std::vector<Rect> & faces = *(std::vector<Rect>*) ROIs.getObj();
    faces.clear();

    if(config!=0){
        //do nothing
    }

    if(image.channels()>1){
        cvtColor(image.getMat(),gray,COLOR_BGR2GRAY);
    }else{
        gray = image.getMat().clone();
    }
    equalizeHist( gray, gray );

    cascade_detector.detectMultiScale( gray, faces, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30) );
    return true;
}

TEST(CV_Face_FacemarkLBF, can_create_default) {
    FacemarkLBF::Params params;
    params.n_landmarks = 68;

    Ptr<Facemark> facemark;
    EXPECT_NO_THROW(facemark = FacemarkLBF::create(params));
    EXPECT_FALSE(facemark.empty());
}

TEST(CV_Face_FacemarkLBF, can_set_custom_detector) {
    string cascade_filename =
        cvtest::findDataFile("cascadeandhog/cascades/lbpcascade_frontalface.xml", true);

    EXPECT_TRUE(cascade_detector.load(cascade_filename));

    Ptr<Facemark> facemark = FacemarkLBF::create();
    EXPECT_TRUE(facemark->setFaceDetector(myCustomDetector));
}

TEST(CV_Face_FacemarkLBF, test_workflow) {

    string i1 = cvtest::findDataFile("face/david1.jpg", true);
    string p1 = cvtest::findDataFile("face/david1.pts", true);
    string i2 = cvtest::findDataFile("face/david2.jpg", true);
    string p2 = cvtest::findDataFile("face/david2.pts", true);

    std::vector<string> images_train;
    images_train.push_back(i1);
    images_train.push_back(i2);

    std::vector<String> points_train;
    points_train.push_back(p1);
    points_train.push_back(p2);

    string cascade_filename =
        cvtest::findDataFile("cascadeandhog/cascades/lbpcascade_frontalface.xml", true);
    FacemarkLBF::Params params;
    params.cascade_face = cascade_filename;
    params.verbose = false;
    params.save_model = false;

    Ptr<Facemark> facemark = FacemarkLBF::create(params);

    Mat image;
    std::vector<Point2f> landmarks;
    for(size_t i=0;i<images_train.size();i++){
        image = imread(images_train[i].c_str());
        EXPECT_TRUE(loadFacePoints(points_train[i].c_str(),landmarks));
        EXPECT_TRUE(landmarks.size()>0);
        EXPECT_TRUE(facemark->addTrainingSample(image, landmarks));
    }

    EXPECT_NO_THROW(facemark->training());

    /*------------ Fitting Part ---------------*/
    cascade_detector.load(cascade_filename);
    facemark->setFaceDetector(myCustomDetector);

    string image_filename = cvtest::findDataFile("face/david1.jpg", true);
    image = imread(image_filename.c_str());
    EXPECT_TRUE(!image.empty());

    std::vector<Rect> rects;
    std::vector<std::vector<Point2f> > facial_points;

    EXPECT_TRUE(facemark->getFaces(image, rects));
    EXPECT_TRUE(rects.size()>0);
    EXPECT_TRUE(facemark->fit(image, rects, facial_points));
    EXPECT_TRUE(facial_points[0].size()>0);
}
