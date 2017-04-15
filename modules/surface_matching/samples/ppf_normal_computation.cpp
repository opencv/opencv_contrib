//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include <iostream>
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"

using namespace std;

static void help(const string& errorMessage)
{
  cout << "Program init error : " << errorMessage << endl;
  cout << "\nUsage : ppf_normal_computation [input model file] [output model file]" << endl;
  cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    help("Not enough input arguments");
    exit(1);
  }

  string modelFileName = (string)argv[1];
  string outputFileName = (string)argv[2];
  cv::Mat points, pointsAndNormals;

  cout << "Loading points\n";
  cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 1).copyTo(points);

  cout << "Computing normals\n";
  cv::Vec3d viewpoint(0, 0, 0);
  cv::ppf_match_3d::computeNormalsPC3d(points, pointsAndNormals, 6, false, viewpoint);

  std::cout << "Writing points\n";
  cv::ppf_match_3d::writePLY(pointsAndNormals, outputFileName.c_str());
  //the following function can also be used for debugging purposes
  //cv::ppf_match_3d::writePLYVisibleNormals(pointsAndNormals, outputFileName.c_str());

  std::cout << "Done\n";
  return 0;
}
