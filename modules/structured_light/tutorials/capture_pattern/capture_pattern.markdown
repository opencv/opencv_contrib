Capture Gray code pattern tutorial {#tutorial_capture_graycode_pattern}
=============

Goal
----

In this tutorial you will learn how to use the *GrayCodePattern* class to:

-   Generate a Gray code pattern.
-   Project the Gray code pattern.
-   Capture the projected Gray code pattern.

It is important to underline that *GrayCodePattern* class actually implements the 3DUNDERWORLD algorithm described in @cite UNDERWORLD , which is based on a stereo approach: we need to capture the projected pattern at the same time from two different views if we want to reconstruct the 3D model of the scanned object. Thus, an acquisition set consists of the images captured by each camera for each image in the pattern sequence.

Code
----

@include structured_light/samples/cap_pattern.cpp

Explanation
-----------
First of all the pattern images to project must be generated. Since the number of images is a function of the projector's resolution, *GrayCodePattern* class parameters must be set with our projector's width and height. In this way the *generate* method can be called: it fills a vector of Mat with the computed pattern images:

@code{.cpp}
  structured_light::GrayCodePattern::Params params;
    ....
  params.width = parser.get<int>( 1 );
  params.height = parser.get<int>( 2 );
    ....
  // Set up GraycodePattern with params
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );
  // Storage for pattern
  vector<Mat> pattern;
  graycode->generate( pattern );
@endcode

For example, using the default projector resolution (1024 x 768), 40 images have to be projected: 20 for regular color pattern (10 images for the columns sequence and 10 for the rows one) and 20 for the color-inverted pattern, where the inverted pattern images are images with the same structure as the original but with inverted colors. This provides an effective method for easily determining the intensity value of each pixel when it is lit (highest value) and when it is not lit (lowest value) during the decoding step.

Subsequently, to identify shadow regions, the regions of two images where the pixels are not lit by projector's light and thus where there is not code information, the 3DUNDERWORLD algorithm computes a shadow mask for the two cameras views, starting from a white and a black images captured by each camera. So two additional images need to be projected and captured with both cameras:

@code{.cpp}
  // Generate the all-white and all-black images needed for shadows mask computation
  Mat white;
  Mat black;
  graycode->getImagesForShadowMasks( black, white );
  pattern.push_back( white );
  pattern.push_back( black );
@endcode

Thus, the final projection sequence is projected as follows: first the column and its inverted sequence, then the row and its inverted sequence and finally the white and black images.

Once the pattern images have been generated, they must be projected using the full screen option: the images must fill all the projection area, otherwise the projector full resolution is not exploited, a condition on which is based 3DUNDERWORLD implementation.

@code{.cpp}
  // Setting pattern window on second monitor (the projector's one)
  namedWindow( "Pattern Window", WINDOW_NORMAL );
  resizeWindow( "Pattern Window", params.width, params.height );
  moveWindow( "Pattern Window", params.width + 316, -20 );
  setWindowProperty( "Pattern Window", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN );
@endcode

At this point the images can be captured with our digital cameras, using libgphoto2 library, recently included in OpenCV: remember to turn on gPhoto2 option in Cmake.list when building OpenCV.
@code{.cpp}
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
@endcode

The two cameras must work at the same resolution and must have autofocus option disabled, maintaining the same focus during all acquisition. The projector can be positioned in the middle of the cameras.

However, before to proceed with pattern acquisition, the cameras must be calibrated. Once the calibration is performed, there should be no movement of the cameras, otherwise a new calibration will be needed.

After having connected the cameras and the projector to the computer, cap_pattern demo can be launched giving as parameters the path where to save the images, and the projector's width and height, taking care to use the same focus and cameras settings of calibration.

At this point, to acquire the images with both cameras, the user can press any key.

@code{.cpp}
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
     ...
  }
@endcode

If the captured images are good (the user must take care that the projected pattern is viewed from the two cameras), the user can save them pressing the enter key, otherwise pressing any other key he can take another shot.
@code{.cpp}
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
@endcode

The acquistion ends when all the pattern images have saved for both cameras. Then the user can reconstruct the 3D model of the captured scene using the *decode* method of *GrayCodePattern* class (see next tutorial).
