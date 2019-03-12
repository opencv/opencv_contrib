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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/****************************************************************************************\
*            Regression tests for line detector comparing keylines.                 *
 \****************************************************************************************/

const std::string LINE_DESCRIPTOR_DIR = "line_descriptor";
const std::string IMAGE_FILENAME = "cameraman.jpg";

class CV_BinaryDescriptorDetectorTest : public cvtest::BaseTest
{

 public:
  CV_BinaryDescriptorDetectorTest( std::string fs )
  {
    bd = BinaryDescriptor::createBinaryDescriptor();
    fs_name = fs;
  }

 protected:
  bool isSimilarKeylines( const KeyLine& k1, const KeyLine& k2 );
  void compareKeylineSets( const std::vector<KeyLine>& validKeylines, const std::vector<KeyLine>& calcKeylines );
  void createMatFromVec( const std::vector<KeyLine>& linesVec, Mat& output );
  void createVecFromMat( Mat& inputMat, std::vector<KeyLine>& output );

  void emptyDataTest();
  void regressionTest();
  virtual void run( int );

  Ptr<BinaryDescriptor> bd;
  std::string fs_name;

};

void CV_BinaryDescriptorDetectorTest::emptyDataTest()
{
  /* one image */
  Mat image;
  std::vector<KeyLine> keylines;

  try
  {
    bd->detect( image, keylines );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "detect() on empty image must return empty keylines vector (1).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

  if( !keylines.empty() )
  {
    ts->printf( cvtest::TS::LOG, "detect() on empty image must return empty keylines vector (1).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    return;
  }

  /* more than one image */
  std::vector<Mat> images;
  std::vector<std::vector<KeyLine> > keylineCollection;

  try
  {
    bd->detect( images, keylineCollection );
  }

  catch ( ... )
  {
    ts->printf( cvtest::TS::LOG, "detect() on empty image vector must not generate exception (2).\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
  }

}

void CV_BinaryDescriptorDetectorTest::createMatFromVec( const std::vector<KeyLine>& linesVec, Mat& output )
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

void CV_BinaryDescriptorDetectorTest::createVecFromMat( Mat& inputMat, std::vector<KeyLine>& output )
{
  for ( int i = 0; i < inputMat.rows; i++ )
  {
    std::vector<float> tempFloat;
    KeyLine kl;
    float* pointerToRow = inputMat.ptr<float>( i );

    for ( int j = 0; j < 17; j++ )
    {
      tempFloat.push_back( *pointerToRow );
      pointerToRow++;
    }

    kl.angle = tempFloat[0];
    kl.class_id = (int) tempFloat[1];
    kl.ePointInOctaveX = tempFloat[2];
    kl.ePointInOctaveY = tempFloat[3];
    kl.endPointX = tempFloat[4];
    kl.endPointY = tempFloat[5];
    kl.lineLength = tempFloat[6];
    kl.numOfPixels = (int) tempFloat[7];
    kl.octave = (int) tempFloat[8];
    kl.pt.x = tempFloat[9];
    kl.pt.y = tempFloat[10];
    kl.response = tempFloat[11];
    kl.sPointInOctaveX = tempFloat[12];
    kl.sPointInOctaveY = tempFloat[13];
    kl.size = tempFloat[14];
    kl.startPointX = tempFloat[15];
    kl.startPointY = tempFloat[16];

    output.push_back( kl );
  }
}

bool CV_BinaryDescriptorDetectorTest::isSimilarKeylines( const KeyLine& k1, const KeyLine& k2 )
{
  const float maxPtDif = 1.f;
  const float maxSizeDif = 1.f;
  const float maxAngleDif = 2.f;
  const float maxResponseDif = 0.1f;

  float dist = (float)cv::norm(k1.pt - k2.pt);
  return ( dist < maxPtDif && fabs( k1.size - k2.size ) < maxSizeDif && abs( k1.angle - k2.angle ) < maxAngleDif
      && abs( k1.response - k2.response ) < maxResponseDif && k1.octave == k2.octave && k1.class_id == k2.class_id );
}

void CV_BinaryDescriptorDetectorTest::compareKeylineSets( const std::vector<KeyLine>& validKeylines, const std::vector<KeyLine>& calcKeylines )
{
  const float maxCountRatioDif = 0.01f;

  // Compare counts of validation and calculated keylines.
  float countRatio = (float) validKeylines.size() / (float) calcKeylines.size();
  if( countRatio < 1 - maxCountRatioDif || countRatio > 1.f + maxCountRatioDif )
  {
    ts->printf( cvtest::TS::LOG, "Bad keylines count ratio (validCount = %d, calcCount = %d).\n", validKeylines.size(), calcKeylines.size() );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
    return;
  }

  int progress = 0;
  int progressCount = (int) ( validKeylines.size() * calcKeylines.size() );
  int badLineCount = 0;
  int commonLineCount = max( (int) validKeylines.size(), (int) calcKeylines.size() );
  for ( size_t v = 0; v < validKeylines.size(); v++ )
  {
    int nearestIdx = -1;
    float minDist = std::numeric_limits<float>::max();

    for ( size_t c = 0; c < calcKeylines.size(); c++ )
    {
      progress = update_progress( progress, (int) ( v * calcKeylines.size() + c ), progressCount, 0 );
      float curDist = (float)cv::norm(calcKeylines[c].pt - validKeylines[v].pt);
      if( curDist < minDist )
      {
        minDist = curDist;
        nearestIdx = (int) c;
      }
    }

    assert( minDist >= 0 );
    if( !isSimilarKeylines( validKeylines[v], calcKeylines[nearestIdx] ) )
      badLineCount++;
  }

  ts->printf( cvtest::TS::LOG, "badLineCount = %d; validLineCount = %d; calcLineCount = %d\n", badLineCount, validKeylines.size(),
              calcKeylines.size() );

  if( badLineCount > 0.9 * commonLineCount )
  {
    ts->printf( cvtest::TS::LOG, " - Bad accuracy!\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    return;
  }

  ts->printf( cvtest::TS::LOG, " - OK\n" );
}

void CV_BinaryDescriptorDetectorTest::regressionTest()
{
  assert( bd );
  std::string imgFilename = std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/" + IMAGE_FILENAME;
  std::string resFilename = std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/" + fs_name + ".yaml";

  // Read the test image.
  Mat image = imread( imgFilename );
  if( image.empty() )
  {
    ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    return;
  }

  // open a storage for reading
  FileStorage fs( resFilename, FileStorage::READ );

  // Compute keylines.
  std::vector<KeyLine> calcKeylines;
  bd->detect( image, calcKeylines );

  if( fs.isOpened() )  // Compare computed and valid keylines.
  {
    // Read validation keylines set.
    std::vector<KeyLine> validKeylines;
    Mat storedKeylines;
    fs["keylines"] >> storedKeylines;
    createVecFromMat( storedKeylines, validKeylines );

    if( validKeylines.empty() )
    {
      ts->printf( cvtest::TS::LOG, "keylines can not be read.\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    compareKeylineSets( validKeylines, calcKeylines );
  }

  else  // Write detector parameters and computed keylines as validation data.
  {
    fs.open( resFilename, FileStorage::WRITE );
    if( !fs.isOpened() )
    {
      ts->printf( cvtest::TS::LOG, "File %s can not be opened to write.\n", resFilename.c_str() );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    else
    {
      fs << "detector_params" << "{";
      bd->write( fs );
      fs << "}";
      Mat lines;
      createMatFromVec( calcKeylines, lines );
      fs << "keylines" << lines;
    }
  }
}

void CV_BinaryDescriptorDetectorTest::run( int )
{
  if( !bd )
  {
    ts->printf( cvtest::TS::LOG, "Feature detector is empty.\n" );
    ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    return;
  }

  emptyDataTest();
  regressionTest();

  ts->set_failed_test_info( cvtest::TS::OK );
}

/****************************************************************************************\
*                                Tests registrations                                     *
 \****************************************************************************************/

TEST( BinaryDescriptor_Detector, DISABLED_regression )
{
  CV_BinaryDescriptorDetectorTest test( std::string( "edl_detector_keylines_cameraman" ) );
  test.safe_run();
}

}} // namespace
