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

  CascadeClassifier face_cascade;
  bool myDetector( InputArray image, OutputArray roi, void * config=0 );
  bool parseArguments(int argc, char** argv, CommandLineParser & , String & cascade,
      String & model, String & images, String & annotations, String & testImages
  );

  int main(int argc, char** argv)
  {
      CommandLineParser parser(argc, argv,"");
      String cascade_path,model_path,images_path, annotations_path, test_images_path;
      if(!parseArguments(argc, argv, parser,cascade_path,model_path,images_path, annotations_path, test_images_path))
         return -1;

      /*create the facemark instance*/
      FacemarkLBF::Params params;
      params.model_filename = model_path;
      params.cascade_face = cascade_path;
      Ptr<Facemark> facemark = FacemarkLBF::create(params);

      face_cascade.load(params.cascade_face.c_str());
      facemark->setFaceDetector(myDetector);

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

  bool myDetector( InputArray image, OutputArray roi, void * config ){
      Mat gray;
      std::vector<Rect> & faces = *(std::vector<Rect>*) roi.getObj();
      faces.clear();

      if(config!=0){
          //do nothing
      }

      if(image.channels()>1){
          cvtColor(image,gray,CV_BGR2GRAY);
      }else{
          gray = image.getMat().clone();
      }
      equalizeHist( gray, gray );

      face_cascade.detectMultiScale( gray, faces, 1.4, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30) );

      return true;
  }

  bool parseArguments(int argc, char** argv, CommandLineParser & parser,
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
     parser = CommandLineParser(argc, argv,keys);
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
