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

#include <vector>

using namespace cv;
using namespace cv::line_descriptor;

static const std::string images[] =
{ "cameraman.jpg", "church.jpg", "church2.png", "einstein.jpg", "stuff.jpg" };

static const char* keys =
{ "{@image_path | | Image path }" };

static void help()
{
  std::cout << "\nThis example shows the functionalities of radius matching " << "Please, run this sample using a command in the form\n"
      << "./example_line_descriptor_radius_matching <path_to_input_images>/" << std::endl;
}

int main( int argc, char** argv )
{
  /* get parameters from comand line */
  CommandLineParser parser( argc, argv, keys );
  String pathToImages = parser.get < String > ( 0 );

  /* create structures for hosting KeyLines and descriptors */
  int num_elements = sizeof ( images ) / sizeof ( images[0] );
  std::vector < Mat > descriptorsMat;
  std::vector < std::vector<KeyLine> > linesMat;

  /*create a pointer to a BinaryDescriptor object */
  Ptr < BinaryDescriptor > bd = BinaryDescriptor::createBinaryDescriptor();

  /* compute lines and descriptors */
  for ( int i = 0; i < num_elements; i++ )
  {
    /* get path to image */
    std::stringstream image_path;
    image_path << pathToImages << images[i];
    std::cout << image_path.str().c_str() << std::endl;

    /* load image */
    Mat loadedImage = imread( image_path.str().c_str(), 1 );
    if( loadedImage.data == NULL )
    {
      std::cout << "Could not load images." << std::endl;
      help();
      exit( -1 );
    }

    /* compute lines and descriptors */
    std::vector < KeyLine > lines;
    Mat computedDescr;
    bd->detect( loadedImage, lines );
    bd->compute( loadedImage, lines, computedDescr );

    descriptorsMat.push_back( computedDescr );
    linesMat.push_back( lines );

  }

  /* compose a queries matrix */
  Mat queries;
  for ( size_t j = 0; j < descriptorsMat.size(); j++ )
  {
    if( descriptorsMat[j].rows >= 5 )
      queries.push_back( descriptorsMat[j].rowRange( 0, 5 ) );

    else if( descriptorsMat[j].rows > 0 && descriptorsMat[j].rows < 5 )
      queries.push_back( descriptorsMat[j] );
  }

  std::cout << "It has been generated a matrix of " << queries.rows << " descriptors" << std::endl;

  /* create a BinaryDescriptorMatcher object */
  Ptr < BinaryDescriptorMatcher > bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  /* populate matcher */
  bdm->add( descriptorsMat );

  /* compute matches */
  std::vector < std::vector<DMatch> > matches;
  bdm->radiusMatch( queries, matches, 30 );
  std::cout << "size matches sample " << matches.size() << std::endl;

  for ( int i = 0; i < (int) matches.size(); i++ )
  {
    for ( int j = 0; j < (int) matches[i].size(); j++ )
    {
      std::cout << "match: " << matches[i][j].queryIdx << " " << matches[i][j].trainIdx << " " << matches[i][j].distance << std::endl;
    }

  }

}

#else

int main()
{
    std::cerr << "OpenCV was built without features2d module" << std::endl;
    return 0;
}

#endif // HAVE_OPENCV_FEATURES2D
