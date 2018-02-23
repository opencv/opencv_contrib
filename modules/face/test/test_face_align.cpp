// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {
using namespace cv::face;

static bool myDetector( InputArray image, OutputArray ROIs, CascadeClassifier* face_cascade)
{
    Mat gray;
    std::vector<Rect> faces;
    if(image.channels()>1){
        cvtColor(image.getMat(),gray,COLOR_BGR2GRAY);
    }
    else{
        gray = image.getMat().clone();
    }
    equalizeHist( gray, gray );
    face_cascade->detectMultiScale( gray, faces, 1.1, 3, 0, Size(30, 30) );
    Mat(faces).copyTo(ROIs);
    return true;
}

TEST(CV_Face_FacemarkKazemi, can_create_default) {
    string cascade_name = cvtest::findDataFile("face/lbpcascade_frontalface_improved.xml", true);
    string configfile_name = cvtest::findDataFile("face/config.xml", true);
    CascadeClassifier face_cascade;
    EXPECT_TRUE(face_cascade.load(cascade_name));
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<FacemarkKazemi> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector((cv::face::FN_FaceDetector)myDetector, &face_cascade));
    EXPECT_FALSE(facemark.empty());
}

TEST(CV_Face_FacemarkKazemi, can_loadTrainingData) {
    string filename = cvtest::findDataFile("face/lbpcascade_frontalface_improved.xml", true);
    string configfile_name = cvtest::findDataFile("face/config.xml", true);
    CascadeClassifier face_cascade;
    EXPECT_TRUE(face_cascade.load(filename));
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<FacemarkKazemi> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector((cv::face::FN_FaceDetector)myDetector, &face_cascade));
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
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    Ptr<FacemarkKazemi> facemark;
    EXPECT_NO_THROW(facemark = FacemarkKazemi::create(params));
    EXPECT_TRUE(facemark->setFaceDetector((cv::face::FN_FaceDetector)myDetector, &face_cascade));
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

}} // namespace
