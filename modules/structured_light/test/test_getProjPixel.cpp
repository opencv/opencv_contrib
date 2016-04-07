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

#include "test_precomp.hpp"

using namespace std;
using namespace cv;

/****************************************************************************************\
*                              GetProjPixel test                                         *
\****************************************************************************************/
class CV_GetProjPixelTest : public cvtest::BaseTest
{
 public:
  CV_GetProjPixelTest();
  ~CV_GetProjPixelTest();
 protected:
  void run(int);
};

CV_GetProjPixelTest::CV_GetProjPixelTest(){}

CV_GetProjPixelTest::~CV_GetProjPixelTest(){}

void CV_GetProjPixelTest::run( int )
{
  // Using default projector resolution (1024 x 768)
  Ptr<structured_light::GrayCodePattern> graycode = structured_light::GrayCodePattern::create();

  // Storage for pattern
  vector<Mat> pattern;

  // Generate the pattern
  graycode->generate( pattern );

  Point projPixel;

  int image_width = pattern[0].cols;
  int image_height = pattern[0].rows;

  for( int i = 0; i < image_width; i++ )
  {
    for( int j = 0; j < image_height; j++ )
    {
      //for a (x,y) pixel of the camera returns the corresponding projector pixel
      bool error = graycode->getProjPixel( pattern, i, j, projPixel );
      EXPECT_FALSE( error );
      EXPECT_EQ( projPixel.y, j );
      EXPECT_EQ( projPixel.x, i );
    }
  }
}

/****************************************************************************************\
*                                Test registration                                     *
\****************************************************************************************/

TEST( GrayCodePattern, getProjPixel )
{
  CV_GetProjPixelTest test;
  test.safe_run();
}
