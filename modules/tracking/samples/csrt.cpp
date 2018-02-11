/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <fstream>
#include "samples_utility.hpp"

using namespace std;
using namespace cv;

#include <chrono>

int main( int argc, char** argv ){
  // show help
  if(argc<2){
    cout<<
      " Usage: example_tracking_csrt <video_name>\n"
      " examples:\n"
      " example_tracking_csrt Bolt/img/%04.jpg\n"
      " example_tracking_csrt Bolt/img/%04.jpg Bolt/grouondtruth.txt\n"
      " example_tracking_csrt faceocc2.webm\n"
      << endl;
    return 0;
  }

  // declare variables to measure time
  std::chrono::time_point<std::chrono::system_clock> t1,t2,t4,t5,time_;

  // create the tracker
  Ptr<TrackerCSRT> tracker = TrackerCSRT::create();

  // const char* param_file_path = "/home/amuhic/Workspace/3_dip/params.yml";
  // FileStorage fs(params_file_path, FileStorage::WRITE);
  // tracker->write(fs);
  // FileStorage fs(param_file_path, FileStorage::READ);
  // tracker->read( fs.root());

  // set input video
  std::string video = argv[1];
  VideoCapture cap(video);
  // and read first frame
  Mat frame;
  cap >> frame;

  // target bounding box
  Rect2d roi;
  if(argc > 2) {
    // read first line of ground-truth file
    std::string groundtruthPath = argv[2];
    std::ifstream gtIfstream(groundtruthPath);
    std::string gtLine;
    getline(gtIfstream, gtLine);
    gtIfstream.close();

    // parse the line by elements
    std::stringstream gtStream(gtLine);
    std::string element;
    std::vector<float> elements;
    while(std::getline(gtStream, element, ','))
    {
      elements.push_back(round(std::stof(element)));
    }

    if(elements.size() == 4) {
      // ground-truth is rectangle
      roi = cv::Rect(elements[0], elements[1], elements[2], elements[3]);
    } else if(elements.size() == 8) {
      // ground-truth is polygon
      float xMin = std::round(std::min(elements[0], std::min(elements[2], std::min(elements[4], elements[6]))));
      float yMin = std::round(std::min(elements[1], std::min(elements[3], std::min(elements[5], elements[7]))));
      float xMax = std::round(std::max(elements[0], std::max(elements[2], std::max(elements[4], elements[6]))));
      float yMax = std::round(std::max(elements[1], std::max(elements[3], std::max(elements[5], elements[7]))));
      roi = cv::Rect(xMin, yMin, xMax-xMin, yMax-yMin);

      // create mask from polygon and set it to the tracker
      cv::Rect aaRect = cv::Rect(xMin, yMin, xMax - xMin, yMax - yMin);
      cout << aaRect.size()<<endl;
      Mat mask = Mat::zeros(aaRect.size(), CV_8UC1);
      const int n = 4;
      std::vector<cv::Point> poly_points(n);
      //Translate x and y to rects start position
      int sx = aaRect.x;
      int sy = aaRect.y;
      for (int i = 0; i < n; ++i) {
        poly_points[i] = Point( elements[2*i] - sx, elements[2*i+1] - sy );
      }
      cv::fillConvexPoly(mask, poly_points, Scalar(1.0), CV_AA);
      mask.convertTo(mask, CV_32FC1);
      tracker->setInitialMask(mask);
    } else {
        std::cout << "Number of ground-truth elements is not 4 or 8." << std::endl;
    }

  } else {
    // second argument is not given - user selects target
    roi = selectROI("tracker", frame, true, false);
  }

  //quit if ROI was not selected
  if(roi.width==0 || roi.height==0)
    return 0;

  // initialize the tracker
  t1 = std::chrono::system_clock::now();
  tracker->init(frame,roi);
  t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> time_acc = t2-t1;

  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");
  int frame_idx = 1;
  for ( ;; ) {
    // get frame from the video
    cap >> frame;

    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    // update the tracking result
    t4 = std::chrono::system_clock::now();
    bool isfound = tracker->update(frame,roi);
    t5 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = t5-t4;
    time_acc += elapsed_seconds;
    frame_idx++;

    if(!isfound) {
        cout << "The target has been lost...\n";
        waitKey(0);
        return 0;
    }

    // draw the tracked object and show the image
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
    imshow("tracker",frame);

    //quit on ESC button
    if(waitKey(1)==27)break;
  }

  cout<< "Elapsed sec: " << time_acc.count() << endl;
  cout<< "fps: " << ((double)(frame_idx)) / time_acc.count() << endl;
}
