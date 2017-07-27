/*----------------------------------------------
 * Usage:
 * facemark_demo_lbf
 *
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
 #include <opencv2/opencv.hpp>
 #include <opencv2/face.hpp>

 using namespace std;
 using namespace cv;

 int main(int argc, char** argv )
 {
     /*create the facemark instance*/
     FacemarkLBF::Params params;
     params.saved_file_name = "ibug68.model";
     params.cascade_face = "../data/haarcascade_frontalface_alt.xml";
     Ptr<Facemark> facemark = FacemarkLBF::create(params);

     /*train the Algorithm*/
     String imageFiles = "../data/images_train.txt";
     String ptsFiles = "../data/points_train.txt";
     facemark->training(imageFiles, ptsFiles);

     String testFiles = "../data/images_test.txt";
     String testPts = "../data/points_test.txt";
     std::vector<String> images;
     std::vector<std::vector<Point2f> > facePoints;
     facemark->loadTrainingData(testFiles, testPts, images, facePoints, 0.0);

     std::vector<Rect> rects;
     CascadeClassifier cc(params.cascade_face.c_str());
     for(int i=0;i<images.size();i++){
         Mat img = imread(images[i]);

         std::vector<Point2f> landmarks;
         facemark->fit(img, landmarks);
         facemark->drawPoints(img, landmarks, Scalar(0,0,255));
         imshow("result", img);
         waitKey(0);
     }

 }
