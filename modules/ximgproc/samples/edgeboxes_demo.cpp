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

#include "opencv2/ximgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

static void help()
{
    std::cout << std::endl <<
    "This sample demonstrates structured edge detection and edgeboxes." << std::endl <<
    "Usage:" << std::endl <<
    "./edgeboxes_demo [<model>] [<input_image>]" << std::endl;
}

int main(int argc, char **argv)
{

  if (argc < 3)
  {
    help();
    return -1;
  }

  Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(argv[1]);

  Mat im;
  im = imread(argv[2]);

  Mat rgb_im;
  cvtColor(im, rgb_im, COLOR_BGR2RGB);
  rgb_im.convertTo(rgb_im, CV_32F, 1.0 / 255.0f);

  Mat edge_im;
  pDollar->detectEdges(rgb_im, edge_im);

  // computes orientation from edge map
  Mat O;
  pDollar->computeOrientation(edge_im, O);

  // apply edge nms
  Mat edge_nms;
  pDollar->edgesNms(edge_im, O, edge_nms, 2, 0, 1, true);

  std::vector<Rect> boxes;
  Ptr<EdgeBoxes> edgeboxes = createEdgeBoxes();
  edgeboxes->setMaxBoxes(30);
  edgeboxes->getBoundingBoxes(edge_nms, O, boxes);

  for(int i = 0; i < (int)boxes.size(); i++)
  {
      Point p1(boxes[i].x, boxes[i].y), p2(boxes[i].x + boxes[i].width, boxes[i].y + boxes[i].height);
      Scalar color(0, 255, 0);
      rectangle(im, p1, p2, color, 1);
  }

  imshow("im", im);
  waitKey(0);

  return 0;
 }
