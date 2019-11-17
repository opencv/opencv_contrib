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

template<class Distance>
class CV_BD_DescriptorsTest : public cvtest::BaseTest
{

 public:
  typedef typename Distance::ValueType ValueType;
  typedef typename Distance::ResultType DistanceType;

  CV_BD_DescriptorsTest( std::string fs, DistanceType _maxDist ): maxDist(_maxDist)
  {
    bd = BinaryDescriptor::createBinaryDescriptor();
    fs_name = fs;
  }

 protected:
//  void compareDescriptors( const Mat& validDescriptors, const Mat& calcDescriptors );
//  void createVecFromMat( Mat& inputMat, std::vector<KeyLine>& output );
//  virtual bool writeDescriptors( Mat& descs );
//  virtual Mat readDescriptors();
//  void emptyDataTest();
//  void regressionTest();
//  virtual void run( int );

  Ptr<BinaryDescriptor> bd;
  std::string fs_name;
  const DistanceType maxDist;
  Distance distance;

//};

  void compareDescriptors( const Mat& validDescriptors, const Mat& calcDescriptors )
  {
    if( validDescriptors.size != calcDescriptors.size || validDescriptors.type() != calcDescriptors.type() )
    {
      ts->printf( cvtest::TS::LOG, "Valid and computed descriptors matrices must have the same size and type.\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    CV_Assert( validDescriptors.type() == CV_8U );

    int dimension = validDescriptors.cols;
    DistanceType curMaxDist = std::numeric_limits<DistanceType>::min();
    for ( int y = 0; y < validDescriptors.rows; y++ )
    {
      DistanceType dist = distance( validDescriptors.ptr<ValueType>( y ), calcDescriptors.ptr<ValueType>( y ), dimension );
      if( dist > curMaxDist )
        curMaxDist = dist;
    }

    EXPECT_LT(curMaxDist, maxDist) << "Max distance between valid and computed descriptors";
  }

  Mat readDescriptors()
  {
    Mat descriptors;
    FileStorage fs( std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/" + fs_name, FileStorage::READ );
    fs["descriptors"] >> descriptors;

    return descriptors;
  }

  bool writeDescriptors( Mat& descs )
  {
    FileStorage fs( std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/" + fs_name, FileStorage::WRITE );
    fs << "descriptors" << descs;

    return true;
  }

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

  void createVecFromMat( Mat& inputMat, std::vector<KeyLine>& output )
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

  void emptyDataTest()
  {
    assert( bd );

    // One image.
    Mat image;
    std::vector<KeyLine> keypoints;
    Mat descriptors;

    try
    {
      bd->compute( image, keypoints, descriptors );
    }

    catch ( ... )
    {
      ts->printf( cvtest::TS::LOG, "compute() on empty image and empty keypoints must not generate exception (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    }

    image.create( 50, 50, CV_8UC3 );
    try
    {
      bd->compute( image, keypoints, descriptors );
    }

    catch ( ... )
    {
      ts->printf( cvtest::TS::LOG, "compute() on nonempty image and empty keylines must not generate exception (1).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    }

    // Several images.
    std::vector<Mat> images;
    std::vector<std::vector<KeyLine> > keylinesCollection;
    std::vector<Mat> descriptorsCollection;
    try
    {
      bd->compute( images, keylinesCollection, descriptorsCollection );
    }

    catch ( ... )
    {
      ts->printf( cvtest::TS::LOG, "compute() on empty images and empty keylines collection must not generate exception (2).\n" );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
    }

  }

  void regressionTest()
  {
    assert( bd );

    // Read the test image.
    std::string imgFilename = std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/" + IMAGE_FILENAME;

    Mat img = imread( imgFilename );
    if( img.empty() )
    {
      ts->printf( cvtest::TS::LOG, "Image %s can not be read.\n", imgFilename.c_str() );
      ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
      return;
    }

    std::vector<KeyLine> keylines;
    FileStorage fs( std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/edl_detector_keylines_cameraman.yaml", FileStorage::READ );
    if( fs.isOpened() )
    {
      //read( fs.getFirstTopLevelNode(), keypoints );

      /* load keylines */
      Mat loadedKeylines;
      fs["keylines"] >> loadedKeylines;
      createVecFromMat( loadedKeylines, keylines );

      /* compute descriptors */
      Mat calcDescriptors;
      double t = (double) getTickCount();
      bd->compute( img, keylines, calcDescriptors );
      t = getTickCount() - t;
      ts->printf( cvtest::TS::LOG, "\nAverage time of computing one descriptor = %g ms.\n",
                  t / ( (double) getTickFrequency() * 1000. ) / calcDescriptors.rows );

      ASSERT_EQ((int)keylines.size(), calcDescriptors.rows)
          << "Count of computed descriptors and keylines count must be equal";

      ASSERT_EQ(bd->descriptorSize() / 8, calcDescriptors.cols);
      ASSERT_EQ(bd->descriptorType(), calcDescriptors.type());

      // TODO read and write descriptor extractor parameters and check them
      Mat validDescriptors = readDescriptors();
      if( !validDescriptors.empty() )
        compareDescriptors( validDescriptors, calcDescriptors );
      else
      {
        if( !writeDescriptors( calcDescriptors ) )
        {
          ts->printf( cvtest::TS::LOG, "Descriptors can not be written.\n" );
          ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
          return;
        }
      }

    }
    else
    {
      ts->printf( cvtest::TS::LOG, "Compute and write keylines.\n" );
      fs.open( std::string( ts->get_data_path() ) + LINE_DESCRIPTOR_DIR + "/edl_detector_keylines_cameraman.yaml", FileStorage::WRITE );
      if( fs.isOpened() )
      {
        bd->detect( img, keylines );
        Mat keyLinesToYaml;
        createMatFromVec( keylines, keyLinesToYaml );
        fs << "keylines" << keyLinesToYaml;
      }
      else
      {
        ts->printf( cvtest::TS::LOG, "File for writting keylines can not be opened.\n" );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
      }
    }
  }

  void run( int )
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

 private:
  CV_BD_DescriptorsTest& operator=( const CV_BD_DescriptorsTest& )
  {
    return *this;
  }
};
/****************************************************************************************\
*                                Tests registrations                                     *
 \****************************************************************************************/

TEST( BinaryDescriptor_Descriptors, regression )
{
  CV_BD_DescriptorsTest<Hamming> test( std::string( "lbd_descriptors_cameraman" ), 1 );
  test.safe_run();
}

/****************************************************************************************\
*                                Other tests                                             *
 \****************************************************************************************/

TEST( BinaryDescriptor, DISABLED_no_lines_found )
{
  Mat Image = Mat::zeros(100, 100, CV_8U);
  Ptr<line_descriptor::BinaryDescriptor> binDescriptor =
    line_descriptor::BinaryDescriptor::createBinaryDescriptor();

  std::vector<cv::line_descriptor::KeyLine> keyLines;
  binDescriptor->detect(Image, keyLines);
  ASSERT_EQ(keyLines.size(), 0u);
}

}} // namespace
