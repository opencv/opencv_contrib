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
#include "opencv2/core.hpp"
#include "opencv2/objdetect.hpp"
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::face;
CascadeClassifier face_cascade;
bool myDetector( InputArray image, OutputArray ROIs );
bool myDetector( InputArray image, OutputArray ROIs ){
    Mat gray;
    std::vector<Rect> faces;
    if(image.channels()>1){
        cvtColor(image.getMat(),gray,COLOR_BGR2GRAY);
    }
    else{
        gray = image.getMat().clone();
    }
    equalizeHist( gray, gray );
    face_cascade.detectMultiScale( gray, faces, 1.1, 3,0, Size(30, 30) );
    Mat(faces).copyTo(ROIs);
    return true;
}

TEST(CV_Face_FacemarkKazemi, can_create_default) {
    string cascade_name = cvtest::findDataFile("face/lbpcascade_frontalface_improved.xml", true);
    string configfile_name = cvtest::findDataFile("face/config.xml", true);
    EXPECT_TRUE(face_cascade.load(cascade_name));
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<Facemark> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector(myDetector));
    EXPECT_FALSE(facemark.empty());
}

TEST(CV_Face_FacemarkKazemi, can_loadTrainingData) {
    string filename = cvtest::findDataFile("face/lbpcascade_frontalface_improved.xml", true);
    string configfile_name = cvtest::findDataFile("face/config.xml", true);
    EXPECT_TRUE(face_cascade.load(filename));
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<Facemark> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector(myDetector));
    vector<String> filenames;
    filename = cvtest::findDataFile("face/1.txt", true);
    filenames.push_back(filename);
    filename = cvtest::findDataFile("face/2.txt", true);
    filenames.push_back(filename);
    vector<String> imagenames;
    vector< vector<Point2f> > trainlandmarks,Trainlandmarks;
    vector<Rect> rectangles;
    //Test getData function
    EXPECT_NO_THROW(loadTrainingData(filenames,trainlandmarks,imagenames));
    vector<Mat> trainimages;
    for(unsigned long i=0;i<imagenames.size();i++){
        string img = cvtest::findDataFile(imagenames[i], true);
        Mat src = imread(img);
        EXPECT_TRUE(!src.empty());
        trainimages.push_back(src);
        Trainlandmarks.push_back(trainlandmarks[i]);
    }
    string modelfilename = "face_landmark_model.dat";
    Size scale = Size(460,460);
    EXPECT_TRUE(facemark->training(trainimages,Trainlandmarks,configfile_name,scale,modelfilename));
}
TEST(CV_Face_FacemarkKazemi, can_detect_landmarks) {
    string cascade_name = cvtest::findDataFile("face/lbpcascade_frontalface_improved.xml", true);
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    Ptr<Facemark> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector(myDetector));
    string imgname = cvtest::findDataFile("face/detect.jpg");
    string modelfilename = cvtest::findDataFile("face/face_landmark_model.dat",true);
    Mat img = imread(imgname);
    EXPECT_TRUE(!img.empty());
    EXPECT_FALSE(facemark.empty());
    EXPECT_NO_THROW(facemark->loadModel(modelfilename));
    vector<Rect> faces;
    //Detect faces in the current image
    EXPECT_TRUE(facemark->getFaces(img,faces));
    //vector to store the landmarks of all the faces in the image
    vector< vector<Point2f> > shapes;
    EXPECT_NO_THROW(facemark->fit(img,faces,shapes));
    shapes.clear();
}