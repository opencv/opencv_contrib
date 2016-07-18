/*M///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//
//M*/


#include "test_precomp.hpp"


namespace cv
{
namespace rgbd
{

class CV_RgbdDepthRegistrationTest: public cvtest::BaseTest
{
public:
  CV_RgbdDepthRegistrationTest()
  {
  }
  ~CV_RgbdDepthRegistrationTest()
  {
  }
protected:
  void
  run(int)
  {

      // Test all three input types for no-op registrations (where a depth image is registered to itself)

      int code = noOpRandomRegistrationTest<unsigned short>(100, 2500);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }

      code = noOpRandomRegistrationTest<float>(0.1f, 2.5f);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }

      code = noOpRandomRegistrationTest<double>(0.1, 2.5);
      if( code != cvtest::TS::OK )
      {
          ts->set_failed_test_info(code);
          return;
      }


      // Test sentinel value handling, occlusion, and dilation
      {

          // K from a VGA Kinect
          Mat K = (Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

          int width = 640, height = 480;

          // All elements are zero except for first two along the diagonal
          Mat_<unsigned short> vgaDepth(height, width, (unsigned short)0);
          vgaDepth(0,0) = 1001;
          vgaDepth(1,1) = 1000;

          Mat_<unsigned short> registeredDepth;
          registerDepth(K, K, Mat(), Matx44f::eye(), vgaDepth, Size(width, height), registeredDepth, true);

          // We expect the closer depth of 1000 to occlude the more distant depth and occupy the
          // upper four left pixels in the depth image because of dilation
          Mat_<unsigned short> expectedResult(height, width, (unsigned short)0);
          expectedResult(0,0) = 1000;
          expectedResult(0,1) = 1000;
          expectedResult(1,0) = 1000;
          expectedResult(1,1) = 1000;

          int cmpResult =  cvtest::cmpEps2( ts, registeredDepth, expectedResult, 0, true, "Dilation and occlusion");

          if( cmpResult != cvtest::TS::OK )
          {
              ts->set_failed_test_info(cmpResult);
              return;
          }

      }

      ts->set_failed_test_info(cvtest::TS::OK);

  }
private:

    template <class DepthDepth>
    int noOpRandomRegistrationTest(DepthDepth minDepth, DepthDepth maxDepth)
    {

        // K from a VGA Kinect
        Mat K = (Mat_<float>(3, 3) << 525., 0., 319.5, 0., 525., 239.5, 0., 0., 1.);

        // Create a random depth image
        RNG rng;
        Mat_<DepthDepth> randomVGADepth(480, 640);
        rng.fill(randomVGADepth, RNG::UNIFORM, minDepth, maxDepth);

        Mat registeredDepth;
        registerDepth(K, K, Mat(), Matx44f::eye(), randomVGADepth, Size(640, 480), registeredDepth);

        // See if registeredDepth == depth
        return cvtest::cmpEps2( ts, registeredDepth, randomVGADepth, 1e-5, true, "No-op registration");

    }

};


}
}

TEST(Rgbd_DepthRegistration, compute)
{
  cv::rgbd::CV_RgbdDepthRegistrationTest test;
  test.safe_run();
}
