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

#include "perf_precomp.hpp"

namespace opencv_test { namespace {

typedef perf::TestBaseWithParam<std::string> file_str;

#define IMAGES \
  "cv/line_descriptor/cameraman.jpg", "cv/shared/lena.png"

void createMatFromVec( const std::vector<KeyLine>& linesVec, Mat& output );

void createMatFromVec( const std::vector<KeyLine>& linesVec, Mat& output )
{
  output = Mat( (int) linesVec.size(), 17, CV_32FC1 );

  for ( int i = 0; i < (int) linesVec.size(); i++ )
  {
    std::vector<float> klData;
    KeyLine kl = linesVec[i];
    klData.push_back( kl.angle );
    klData.push_back( (float) kl.class_id );
    klData.push_back( kl.ePointInOctaveX );
    klData.push_back( kl.ePointInOctaveY );
    klData.push_back( kl.endPointX );
    klData.push_back( kl.endPointY );
    klData.push_back( kl.lineLength );
    klData.push_back( (float) kl.numOfPixels );
    klData.push_back( (float) kl.octave );
    klData.push_back( kl.pt.x );
    klData.push_back( kl.pt.y );
    klData.push_back( kl.response );
    klData.push_back( kl.sPointInOctaveX );
    klData.push_back( kl.sPointInOctaveY );
    klData.push_back( kl.size );
    klData.push_back( kl.startPointX );
    klData.push_back( kl.startPointY );

    float* pointerToRow = output.ptr<float>( i );
    for ( int j = 0; j < 17; j++ )
    {
      *pointerToRow = klData[j];
      pointerToRow++;
    }
  }
}

PERF_TEST_P(file_str, detect, testing::Values(IMAGES))
{
  std::string filename = getDataPath( GetParam() );

  Mat frame = imread( filename, 1 );

  if( frame.empty() )
    FAIL()<< "Unable to load source image " << filename;

  Mat lines;
  std::vector<KeyLine> keylines;
  Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

  TEST_CYCLE()
  {
    bd->detect( frame, keylines );
    createMatFromVec( keylines, lines );
  }

  SANITY_CHECK_NOTHING();

}

PERF_TEST_P(file_str, detect_lsd, testing::Values(IMAGES))
{
  std::string filename = getDataPath( GetParam() );
  std::cout << filename.c_str() << std::endl;

  Mat frame = imread( filename, 1 );

  if( frame.empty() )
    FAIL()<< "Unable to load source image " << filename;

  Mat lines;
  std::vector<KeyLine> keylines;
  Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();

  TEST_CYCLE()
  {
    lsd->detect( frame, keylines, 2, 1 );
    createMatFromVec( keylines, lines );
  }

  SANITY_CHECK_NOTHING();

}

}} // namespace
