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
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace saliency;

static const char* keys =
{ "{@saliency_algorithm | | Saliency algorithm <saliencyAlgorithmType.[saliencyAlgorithmTypeSubType]> }"
    "{@video_name      | | video name            }"
    "{@start_frame     |1| Start frame           }"
    "{@training_path   |1| Path of the folder containing the trained files}" };

static void help()
{
  cout << "\nThis example shows the functionality of \"Saliency \""
       "Call:\n"
       "./example_saliency_computeSaliency <saliencyAlgorithmSubType> <video_name> <start_frame> \n"
       << endl;
}

int main( int argc, char** argv )
{
  CommandLineParser parser( argc, argv, keys );

  String saliency_algorithm = parser.get<String>( 0 );
  String video_name = parser.get<String>( 1 );
  int start_frame = parser.get<int>( 2 );
  String training_path = parser.get<String>( 3 );

  if( saliency_algorithm.empty() || video_name.empty() )
  {
    help();
    return -1;
  }

  //open the capture
  VideoCapture cap;
  cap.open( video_name );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() )
  {
    help();
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    parser.printMessage();
    return -1;
  }

  Mat frame;

  //instantiates the specific Saliency
  Ptr<Saliency> saliencyAlgorithm = Saliency::create( saliency_algorithm );

  if( saliencyAlgorithm == NULL )
  {
    cout << "***Error in the instantiation of the saliency algorithm...***\n";
    return -1;
  }

  Mat binaryMap;
  Mat image;

  cap >> frame;
  if( frame.empty() )
  {
    return 0;
  }

  frame.copyTo( image );

  if( saliency_algorithm.find( "SPECTRAL_RESIDUAL" ) == 0 )
  {
    Mat saliencyMap;
    if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
    {
      StaticSaliencySpectralResidual spec;
      spec.computeBinaryMap( saliencyMap, binaryMap );

      imshow( "Saliency Map", saliencyMap );
      imshow( "Original Image", image );
      imshow( "Binary Map", binaryMap );
      waitKey( 0 );
    }

  }
  else if( saliency_algorithm.find( "BING" ) == 0 )
  {
    if( training_path.empty() )
    {

      cout << "Path of trained files missing! " << endl;
      return -1;
    }

    else
    {
      vector<Vec4i> saliencyMap;
      saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath( training_path );
      saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir( training_path + "/Results" );

      if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
      {
        std::cout << "Objectness done" << std::endl;
      }
    }

  }
  else if( saliency_algorithm.find( "BinWangApr2014" ) == 0 )
  {

    //Ptr<Size> size = Ptr<Size>( new Size( image.cols, image.rows ) );
    saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->setImagesize( image.cols, image.rows );
    saliencyAlgorithm.dynamicCast<MotionSaliencyBinWangApr2014>()->init();

    bool paused = false;
    for ( ;; )
    {
      if( !paused )
      {

        cap >> frame;
        cvtColor( frame, frame, COLOR_BGR2GRAY );

        Mat saliencyMap;
        if( saliencyAlgorithm->computeSaliency( frame, saliencyMap ) )
        {
          std::cout << "current frame motion saliency done" << std::endl;
        }

        imshow( "image", frame );
        imshow( "saliencyMap", saliencyMap * 255 );
      }

      char c = (char) waitKey( 2 );
      if( c == 'q' )
        break;
      if( c == 'p' )
        paused = !paused;

    }
  }

  return 0;
}
