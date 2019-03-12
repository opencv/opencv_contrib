/*
This file was part of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

/*----------------------------------------------
 * Usage:
 * facemark_demo_lbf <face_cascade_model> <saved_model_filename> <training_images> <annotation_files> [test_files]
 *
 * Example:
 * facemark_demo_lbf ../face_cascade.xml ../LBF.model ../images_train.txt ../points_train.txt ../test.txt
 *
 * Notes:
 * the user should provides the list of training images_train
 * accompanied by their corresponding landmarks location in separated files.
 * example of contents for images_train.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 * example of contents for points_train.txt:
 * ../trainset/image_0001.pts
 * ../trainset/image_0002.pts
 * where the image_xxxx.pts contains the position of each face landmark.
 * example of the contents:
 *  version: 1
 *  n_points:  68
 *  {
 *  115.167660 220.807529
 *  116.164839 245.721357
 *  120.208690 270.389841
 *  ...
 *  }
 * example of the dataset is available at https://ibug.doc.ic.ac.uk/download/annotations/ibug.zip
 *--------------------------------------------------*/

#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

using namespace std;
using namespace cv;
using namespace cv::face;

static bool myDetector( InputArray image, OutputArray roi, CascadeClassifier *face_detector);
static bool parseArguments(int argc, char** argv, String & cascade,
   String & model, String & images, String & annotations, String & testImages
);

int main(int argc, char** argv)
{
    String cascade_path,model_path,images_path, annotations_path, test_images_path;
    if(!parseArguments(argc, argv, cascade_path,model_path,images_path, annotations_path, test_images_path))
       return -1;

    /*create the facemark instance*/
    FacemarkLBF::Params params;
    params.model_filename = model_path;
    params.cascade_face = cascade_path;
    Ptr<FacemarkLBF> facemark = FacemarkLBF::create(params);

    CascadeClassifier face_cascade;
    face_cascade.load(params.cascade_face.c_str());
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);

    /*Loads the dataset*/
    std::vector<String> images_train;
    std::vector<String> landmarks_train;
    loadDatasetList(images_path,annotations_path,images_train,landmarks_train);

    Mat image;
    std::vector<Point2f> facial_points;
    for(size_t i=0;i<images_train.size();i++){
        printf("%i/%i :: %s\n", (int)(i+1), (int)images_train.size(),images_train[i].c_str());
        image = imread(images_train[i].c_str());
        loadFacePoints(landmarks_train[i],facial_points);
        facemark->addTrainingSample(image, facial_points);
    }

    /*train the Algorithm*/
    facemark->training();

    /*test using some images*/
    String testFiles(images_path), testPts(annotations_path);
    if(!test_images_path.empty()){
        testFiles = test_images_path;
        testPts = test_images_path; //unused
    }
    std::vector<String> images;
    std::vector<String> facePoints;
    loadDatasetList(testFiles, testPts, images, facePoints);

    std::vector<Rect> rects;
    CascadeClassifier cc(params.cascade_face.c_str());
    for(size_t i=0;i<images.size();i++){
        std::vector<std::vector<Point2f> > landmarks;
        cout<<images[i];
        Mat img = imread(images[i]);
        facemark->getFaces(img, rects);
        facemark->fit(img, rects, landmarks);

        for(size_t j=0;j<rects.size();j++){
            drawFacemarks(img, landmarks[j], Scalar(0,0,255));
            rectangle(img, rects[j], Scalar(255,0,255));
        }

        if(rects.size()>0){
            cout<<endl;
            imshow("result", img);
            waitKey(0);
        }else{
            cout<<"face not found"<<endl;
        }
    }
}

bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

bool parseArguments(int argc, char** argv,
    String & cascade,
    String & model,
    String & images,
    String & annotations,
    String & test_images
){
    const String keys =
        "{ @c cascade         |      | (required) path to the face cascade xml file fo the face detector }"
        "{ @i images          |      | (required) path of a text file contains the list of paths to all training images}"
        "{ @a annotations     |      | (required) Path of a text file contains the list of paths to all annotations files}"
        "{ @m model           |      | (required) path to save the trained model }"
        "{ t test-images      |      | Path of a text file contains the list of paths to the test images}"
        "{ help h usage ?     |      | facemark_demo_lbf -cascade -images -annotations -model [-t] \n"
         " example: facemark_demo_lbf ../face_cascade.xml ../images_train.txt ../points_train.txt ../lbf.model}"
    ;
    CommandLineParser parser(argc, argv,keys);
    parser.about("hello");

    if (parser.has("help")){
        parser.printMessage();
        return false;
    }

    cascade = String(parser.get<String>("cascade"));
    model = String(parser.get<string>("model"));
    images = String(parser.get<string>("images"));
    annotations = String(parser.get<string>("annotations"));
    test_images = String(parser.get<string>("t"));

    cout<<"cascade : "<<cascade.c_str()<<endl;
    cout<<"model : "<<model.c_str()<<endl;
    cout<<"images : "<<images.c_str()<<endl;
    cout<<"annotations : "<<annotations.c_str()<<endl;

    if(cascade.empty() || model.empty() || images.empty() || annotations.empty()){
        std::cerr << "one or more required arguments are not found" << '\n';

        parser.printMessage();
        return false;
    }

    return true;
}
