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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
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

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/structured_light.hpp>
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

static const char* keys =
{ "{@path | | Path of the folder where the captured pattern images will be save }"
     "{@proj_width      | | Projector width            }"
     "{@proj_height     | | Projector height           }" };

static void help()
{
    cout << "\nThis example shows how to use the \"Structured Light module\" to acquire a graycode pattern"
         "\nCall (with the two cams connected):\n"
         "./example_structured_light_cap_pattern <path> <proj_width> <proj_height> \n"
         << endl;
}

int main( int argc, char** argv )
{

  structured_light::GrayCodePattern::Params params;

  CommandLineParser parser( argc, argv, keys );
  String path = parser.get<String>( 0 );
  params.width = parser.get<int>( 1 );
  params.height = parser.get<int>( 2 );

  if( path.empty() || params.width < 1 || params.height < 1 )
  {
    help();
    return -1;
  }

  // Set up GraycodePattern with params
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );

  // Storage for pattern
  vector<Mat> pattern;
  graycode->generate( pattern );

  cout << pattern.size() << " pattern images + 2 images for shadows mask computation to acquire with both cameras"
         << endl;

  // Generate the all-white and all-black images needed for shadows mask computation
  Mat white;
  Mat black;
  graycode->getImagesForShadowMasks( black, white );

  pattern.push_back( white );
  pattern.push_back( black );

  // Setting pattern window on second monitor (the projector's one)
  namedWindow( "Pattern Window", WINDOW_NORMAL );
  resizeWindow( "Pattern Window", params.width, params.height );
  moveWindow( "Pattern Window", params.width + 316, -20 );
  setWindowProperty( "Pattern Window", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN );

  // Open camera number 1, using libgphoto2
  VideoCapture cap1( CAP_GPHOTO2 );

  if( !cap1.isOpened() )
  {
    // check if cam1 opened
    cout << "cam1 not opened!" << endl;
    help();
    return -1;
  }

  // Open camera number 2
  VideoCapture cap2( 1 );

  if( !cap2.isOpened() )
  {
     // check if cam2 opened
     cout << "cam2 not opened!" << endl;
     help();
     return -1;
  }

  // Turning off autofocus
  cap1.set( CAP_PROP_SETTINGS, 1 );
  cap2.set( CAP_PROP_SETTINGS, 1 );

  int i = 0;
  while( i < (int) pattern.size() )
  {
    cout << "Waiting to save image number " << i + 1 << endl << "Press any key to acquire the photo" << endl;
    imshow( "Pattern Window", pattern[i] );

    Mat frame1;
    Mat frame2;

    cap1 >> frame1;  // get a new frame from camera 1
    cap2 >> frame2;  // get a new frame from camera 2

    if( ( frame1.data ) && ( frame2.data ) )
    {

      Mat tmp;
      cout << "cam 1 size: " << Size( ( int ) cap1.get( CAP_PROP_FRAME_WIDTH ), ( int ) cap1.get( CAP_PROP_FRAME_HEIGHT ) )
           << endl;

      cout << "cam 2 size: " << Size( ( int ) cap2.get( CAP_PROP_FRAME_WIDTH ), ( int ) cap2.get( CAP_PROP_FRAME_HEIGHT ) )
           << endl;

      cout << "zoom cam 1: " << cap1.get( CAP_PROP_ZOOM ) << endl << "zoom cam 2: " << cap2.get( CAP_PROP_ZOOM )
           << endl;

      cout << "focus cam 1: " << cap1.get( CAP_PROP_FOCUS ) << endl << "focus cam 2: " << cap2.get( CAP_PROP_FOCUS )
           << endl;

      cout << "Press enter to save the photo or an other key to re-acquire the photo" << endl;

      namedWindow( "cam1", WINDOW_NORMAL );
      resizeWindow( "cam1", 640, 480 );

      namedWindow( "cam2", WINDOW_NORMAL );
      resizeWindow( "cam2", 640, 480 );

      // Moving window of cam2 to see the image at the same time with cam1
      moveWindow( "cam2", 640 + 75, 0 );

      // Resizing images to avoid issues for high resolution images, visualizing them as grayscale
      resize( frame1, tmp, Size( 640, 480 ) );
      cvtColor( tmp, tmp, COLOR_RGB2GRAY );
      imshow( "cam1", tmp );
      resize( frame2, tmp, Size( 640, 480 ) );
      cvtColor( tmp, tmp, COLOR_RGB2GRAY );
      imshow( "cam2", tmp );

      bool save1 = false;
      bool save2 = false;

      int key = waitKey( 0 );

      // Pressing enter, it saves the output
      if( key == 13 )
      {
        ostringstream name;
        name << i + 1;

        save1 = imwrite( path + "pattern_cam1_im" + name.str() + ".png", frame1 );
        save2 = imwrite( path + "pattern_cam2_im" + name.str() + ".png", frame2 );

        if( ( save1 ) && ( save2 ) )
        {
          cout << "pattern cam1 and cam2 images number " << i + 1 << " saved" << endl << endl;
          i++;
        }
        else
        {
          cout << "pattern cam1 and cam2 images number " << i + 1 << " NOT saved" << endl << endl << "Retry, check the path"<< endl << endl;
        }
      }
      // Pressing escape, the program closes
      if( key == 27 )
      {
        cout << "Closing program" << endl;
      }
    }
    else
    {
      cout << "No frame data, waiting for new frame" << endl;
    }
  }

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}