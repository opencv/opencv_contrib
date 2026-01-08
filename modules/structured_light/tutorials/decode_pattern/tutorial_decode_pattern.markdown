Decode Gray code pattern tutorial {#tutorial_decode_graycode_pattern}
=============

Goal
----

In this tutorial you will learn how to use the *GrayCodePattern* class to:

-   Decode a previously acquired Gray code pattern.
-   Generate a disparity map.
-   Generate a pointcloud.

Code
----

@include structured_light/samples/pointcloud.cpp

Explanation
-----------

First of all the needed parameters must be passed to the program.
The first is the name list of previously acquired pattern images, stored in a .yaml file organized as below:

@code{.cpp}
%YAML:1.0
cam1:
   - "/data/pattern_cam1_im1.png"
   - "/data/pattern_cam1_im2.png"
           ..............
   - "/data/pattern_cam1_im42.png"
   - "/data/pattern_cam1_im43.png"
   - "/data/pattern_cam1_im44.png"
cam2:
   - "/data/pattern_cam2_im1.png"
   - "/data/pattern_cam2_im2.png"
           ..............
   - "/data/pattern_cam2_im42.png"
   - "/data/pattern_cam2_im43.png"
   - "/data/pattern_cam2_im44.png"
@endcode

For example, the dataset used for this tutorial has been acquired using a projector with a resolution of 1280x800, so 42 pattern images (from number 1 to 42) + 1 white (number 43) and 1 black (number 44) were captured with both the two cameras.

Then the cameras calibration parameters, stored in another .yml file, together with the width and the height of the projector used to project the pattern, and, optionally, the values of white and black tresholds, must be passed to the tutorial program.

In this way, *GrayCodePattern* class parameters can be set up with the width and the height of the projector used during the pattern acquisition and a pointer to a GrayCodePattern object can be created:

@code{.cpp}
  structured_light::GrayCodePattern::Params params;
     ....
  params.width = parser.get<int>( 2 );
  params.height = parser.get<int>( 3 );
     ....
  // Set up GraycodePattern with params
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create( params );
@endcode

If the white and black thresholds are passed as parameters (these thresholds influence the number of decoded pixels), their values can be set, otherwise the algorithm will use the default values.

@code{.cpp}
  size_t white_thresh = 0;
  size_t black_thresh = 0;
  if( argc == 7 )
  {
    // If passed, setting the white and black threshold, otherwise using default values
    white_thresh = parser.get<size_t>( 4 );
    black_thresh = parser.get<size_t>( 5 );
    graycode->setWhiteThreshold( white_thresh );
    graycode->setBlackThreshold( black_thresh );
  }
@endcode

At this point, to use the *decode* method of *GrayCodePattern* class, the acquired pattern images must be stored in a vector of vector of Mat.
The external vector has a size of two because two are the cameras: the first vector stores the pattern images captured from the left camera, the second those acquired from the right one. The number of pattern images is obviously the same for both cameras and can be retrieved using the getNumberOfPatternImages() method:

@code{.cpp}
  size_t numberOfPatternImages = graycode->getNumberOfPatternImages();
  vector<vector<Mat> > captured_pattern;
  captured_pattern.resize( 2 );
  captured_pattern[0].resize( numberOfPatternImages );
  captured_pattern[1].resize( numberOfPatternImages );

  .....

  for( size_t i = 0; i < numberOfPatternImages; i++ )
  {
    captured_pattern[0][i] = imread( imagelist[i], IMREAD_GRAYSCALE );
    captured_pattern[1][i] = imread( imagelist[i + numberOfPatternImages + 2], IMREAD_GRAYSCALE );
     ......
  }
@endcode

As regards the black and white images, they must be stored in two different vectors of Mat:

@code{.cpp}
  vector<Mat> blackImages;
  vector<Mat> whiteImages;
  blackImages.resize( 2 );
  whiteImages.resize( 2 );
  // Loading images (all white + all black) needed for shadows computation
  cvtColor( color, whiteImages[0], COLOR_RGB2GRAY );
  whiteImages[1] = imread( imagelist[2 * numberOfPatternImages + 2], IMREAD_GRAYSCALE );
  blackImages[0] = imread( imagelist[numberOfPatternImages + 1], IMREAD_GRAYSCALE );
  blackImages[1] = imread( imagelist[2 * numberOfPatternImages + 2 + 1], IMREAD_GRAYSCALE );
@endcode

It is important to underline that all the images, the pattern ones, black and white, must be loaded as grayscale images and rectified before being passed to decode method:

@code{.cpp}
  // Stereo rectify
  cout << "Rectifying images..." << endl;
  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];
  stereoRectify( cam1intrinsics, cam1distCoeffs, cam2intrinsics, cam2distCoeffs, imagesSize, R, T, R1, R2, P1, P2, Q, 0,
                -1, imagesSize, &validRoi[0], &validRoi[1] );
  Mat map1x, map1y, map2x, map2y;
  initUndistortRectifyMap( cam1intrinsics, cam1distCoeffs, R1, P1, imagesSize, CV_32FC1, map1x, map1y );
  initUndistortRectifyMap( cam2intrinsics, cam2distCoeffs, R2, P2, imagesSize, CV_32FC1, map2x, map2y );
         ........
  for( size_t i = 0; i < numberOfPatternImages; i++ )
  {
         ........
    remap( captured_pattern[1][i], captured_pattern[1][i], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
    remap( captured_pattern[0][i], captured_pattern[0][i], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  }
         ........
  remap( color, color, map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[0], whiteImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( whiteImages[1], whiteImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[0], blackImages[0], map2x, map2y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
  remap( blackImages[1], blackImages[1], map1x, map1y, INTER_NEAREST, BORDER_CONSTANT, Scalar() );
@endcode


In this way the *decode* method can be called to decode the pattern and to generate the corresponding disparity map, computed on the first camera (left):
@code{.cpp}
  Mat disparityMap;
  bool decoded = graycode->decode(captured_pattern, disparityMap, blackImages, whiteImages,
                                  structured_light::DECODE_3D_UNDERWORLD);
@endcode

To better visualize the result, a colormap is applied to the computed disparity:
@code{.cpp}
    double min;
    double max;
    minMaxIdx(disparityMap, &min, &max);
    Mat cm_disp, scaledDisparityMap;
    cout << "disp min " << min << endl << "disp max " << max << endl;
    convertScaleAbs( disparityMap, scaledDisparityMap, 255 / ( max - min ) );
    applyColorMap( scaledDisparityMap, cm_disp, COLORMAP_JET );
    // Show the result
    resize( cm_disp, cm_disp, Size( 640, 480 ) );
    imshow( "cm disparity m", cm_disp )
@endcode

![](pics/cm_disparity.png)

At this point the point cloud can be generated using the reprojectImageTo3D method, taking care to convert the computed disparity in a CV_32FC1 Mat (decode method computes a CV_64FC1 disparity map):
@code{.cpp}
  Mat pointcloud;
  disparityMap.convertTo( disparityMap, CV_32FC1 );
  reprojectImageTo3D( disparityMap, pointcloud, Q, true, -1 );
@endcode

Then a mask to remove the unwanted background is computed:
@code{.cpp}
  Mat dst, thresholded_disp;
  threshold( scaledDisparityMap, thresholded_disp, 0, 255, THRESH_OTSU + THRESH_BINARY );
  resize( thresholded_disp, dst, Size( 640, 480 ) );
  imshow( "threshold disp otsu", dst );
@endcode
![](pics/threshold_disp.png)

The white image of cam1 was previously loaded also as a color image, in order to map the color of the object on its reconstructed pointcloud:
@code{.cpp}
  Mat color = imread( imagelist[numberOfPatternImages], IMREAD_COLOR );
@endcode

The background renoval mask is thus applied to the point cloud and to the color image:
@code{.cpp}
  Mat pointcloud_tresh, color_tresh;
  pointcloud.copyTo(pointcloud_tresh, thresholded_disp);
  color.copyTo(color_tresh, thresholded_disp);
@endcode

Finally the computed point cloud of the scanned object can be visualized on viz:
@code{.cpp}
  viz::Viz3d myWindow( "Point cloud with color");
  myWindow.setBackgroundMeshLab();
  myWindow.showWidget( "coosys", viz::WCoordinateSystem());
  myWindow.showWidget( "pointcloud", viz::WCloud( pointcloud_tresh, color_tresh ) );
  myWindow.showWidget( "text2d", viz::WText( "Point cloud", Point(20, 20), 20, viz::Color::green() ) );
  myWindow.spin();
@endcode

![](pics/plane_viz.png)
