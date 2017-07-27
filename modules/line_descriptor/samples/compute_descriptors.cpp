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
 // Copyright (C) 2014, Biagio Montesano, all rights reserved.
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

#include <iostream>
#include <opencv2/opencv_modules.hpp>

#ifdef HAVE_OPENCV_FEATURES2D

#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::line_descriptor;


static const char* keys =
{ "{@image_path | | Image path }" };

static void help()
{
  std::cout << "\nThis example shows the functionalities of lines extraction " << "and descriptors computation furnished by BinaryDescriptor class\n"
            << "Please, run this sample using a command in the form\n" << "./example_line_descriptor_compute_descriptors <path_to_input_image>"
            << std::endl;
}

int main( int argc, char** argv )
{
  /* get parameters from command line */
  CommandLineParser parser( argc, argv, keys );
  String image_path = parser.get<String>( 0 );

  if( image_path.empty() )
  {
    help();
    return -1;
  }

  /* load image */
  cv::Mat imageMat = imread( image_path, 1 );
  if( imageMat.data == NULL )
  {
    std::cout << "Error, image could not be loaded. Please, check its path" << std::endl;
  }

  /* create a binary mask */
  cv::Mat mask = Mat::ones( imageMat.size(), CV_8UC1 );

  /* create a pointer to a BinaryDescriptor object with default parameters */
  Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

  /* compute lines */
  std::vector<KeyLine> keylines;
  bd->detect( imageMat, keylines, mask );

  /* compute descriptors */
  cv::Mat descriptors;

  bd->compute( imageMat, keylines, descriptors);

}

#else

int main()
{
    std::cerr << "OpenCV was built without features2d module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_FEATURES2D
