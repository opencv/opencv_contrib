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

#include <opencv2/line_descriptor.hpp>

#include "opencv2/core/utility.hpp"
#include "opencv2/core/private.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

static const char* keys =
{ "{@image_path1 | | Image path 1 }"
    "{@image_path2 | | Image path 2 }" };

static void help()
{
  std::cout << "\nThis example shows the functionalities of lines extraction " << "and descriptors computation furnished by BinaryDescriptor class\n"
            << "Please, run this sample using a command in the form\n" << "./example_line_descriptor_compute_descriptors <path_to_input_image 1>"
            << "<path_to_input_image 2>" << std::endl;

}

inline void writeMat( cv::Mat m, std::string name, int n )
{
  std::stringstream ss;
  std::string s;
  ss << n;
  ss >> s;
  std::string fileNameConf = name + s;
  cv::FileStorage fsConf( fileNameConf, cv::FileStorage::WRITE );
  fsConf << "m" << m;

  fsConf.release();
}

inline void loadMat( cv::Mat& m, std::string name )
{

  cv::FileStorage fsConf( name, cv::FileStorage::READ );
  fsConf["m"] >> m;

  fsConf.release();
}

int binaryDist( const uchar * p_descriptor, const uchar * p_trained )
{
  int count = 0;
  for ( int i = 0; i < 32; i++ )
  {
    uchar a = p_descriptor[i];
    uchar a1 = a & 1;
    uchar a2 = a & 2;
    uchar a4 = a & 4;
    uchar a8 = a & 8;
    uchar a16 = a & 16;
    uchar a32 = a & 32;
    uchar a64 = a & 64;
    uchar a128 = a & 128;

    uchar b = p_trained[i];
    uchar b1 = b & 1;
    uchar b2 = b & 2;
    uchar b4 = b & 4;
    uchar b8 = b & 8;
    uchar b16 = b & 16;
    uchar b32 = b & 32;
    uchar b64 = b & 64;
    uchar b128 = b & 128;

    if( a1 == b1 )
      count++;
    if( a2 == b2 )
      count++;
    if( a4 == b4 )
      count++;
    if( a8 == b8 )
      count++;
    if( a16 == b16 )
      count++;
    if( a32 == b32 )
      count++;
    if( a64 == b64 )
      count++;
    if( a128 == b128 )
      count++;
  }
  return count;
}

std::vector<DMatch> computeBruteForceSingleImages( Mat descriptor_query, Mat descriptor_db )
{
  //BRUTE FORCE//

  std::vector<DMatch> matches;

  for ( int i = 0; i < descriptor_query.rows; i++ )
  {

    const uchar * p_descriptor = ( descriptor_query.ptr() ) + i * 32;

    const uchar * p_trained = descriptor_db.ptr();
    int min_dist = 0;
    int min_index = -1;
    for ( int k = 0; k < descriptor_db.rows; k++ )
    {
      int dist = binaryDist( p_descriptor, p_trained + ( k * 32 ) );
      if( dist > min_dist )
      {
        min_dist = dist;
        min_index = k;
      }
    }
    DMatch m( i, min_index, (float) min_dist );
    matches.push_back( m );

  }

  return matches;
}

void computeDescr( Mat sm_image, Mat img )
{
  Mat query = sm_image.clone();
  Mat db = img.clone();

  Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

  /* compute lines */
  std::vector<KeyLine> keylines1, keylines2;
  bd->detect( query, keylines1 );
  bd->detect( db, keylines2 );

  /* compute descriptors */
  cv::Mat descr1, descr2;
  bd->compute( query, keylines1, descr1 );
  bd->compute( db, keylines2, descr2 );

  std::vector<cv::KeyPoint> keypoints_1;
  std::vector<cv::KeyPoint> keypoints_2;
  std::vector<std::pair<cv::KeyPoint, int> > v_pair_k1;
  std::vector<std::pair<cv::KeyPoint, int> > v_pair_k2;
  for ( int i = 0; i < keylines1.size(); i++ )
  {
    KeyLine l = keylines1[i];
    keypoints_1.push_back( cv::KeyPoint( l.startPointX, l.startPointY, 8, l.angle ) );
    v_pair_k1.push_back( std::make_pair( cv::KeyPoint( l.startPointX, l.startPointY, 8, l.angle ), i ) );

  }
  for ( int i = 0; i < keylines2.size(); i++ )
  {
    KeyLine l = keylines2[i];
    keypoints_2.push_back( cv::KeyPoint( l.startPointX, l.startPointY, 8, l.angle ) );
    v_pair_k2.push_back( std::make_pair( cv::KeyPoint( l.startPointX, l.startPointY, 8, l.angle ), i ) );
  }

//                 vector<DMatch> matches = ImageFinderFLANN::computeBruteForceSingleImages(purged_descriptor_query, purged_descriptor_db );
  std::vector<DMatch> matches = computeBruteForceSingleImages( descr1, descr2 );

  Mat img_draw_matches, img_draw_matches_debug;

  std::vector<DMatch> good_matches;
  int thresh_good = 200;
  for ( int i = 0; i < matches.size(); i++ )
  {
    if( matches[i].distance > thresh_good )
    {
      good_matches.push_back( matches[i] );

    }
  }

  srand( (unsigned) time( 0 ) );
  int lowest = 100, highest = 255;
  int range = ( highest - lowest ) + 1;
  unsigned int r, g, b;

  //DISEGNO MATCHES
  std::vector<cv::KeyPoint> fake_k1;
  std::vector<cv::KeyPoint> fake_k2;
  std::vector<cv::DMatch> fake_match;
  drawMatches( sm_image, fake_k1, img, fake_k2, fake_match, img_draw_matches, Scalar::all( -1 ), Scalar::all( -1 ), Mat(),
               DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  for ( int i = 0; i < keylines1.size(); i++ )
  {
    KeyLine line = keylines1[i];
    cv::Point startP( line.sPointInOctaveX, line.sPointInOctaveY );
    cv::Point endP( line.ePointInOctaveX, line.ePointInOctaveY );

    cv::Point midP( ( startP.x + endP.x ) / 2, ( startP.y + endP.y ) / 2 );
    //cv::putText(img_draw_matches, std::to_string(i), midP, 1, 1, Scalar(255,0,0), 1 );

    cv::line( img_draw_matches, startP, endP, Scalar( 0, 0, 255 ) );

  }
  for ( int i = 0; i < keylines2.size(); i++ )
  {
    KeyLine line = keylines2[i];
    cv::Point startP( line.sPointInOctaveX + sm_image.cols, line.sPointInOctaveY );
    cv::Point endP( line.ePointInOctaveX + sm_image.cols, line.ePointInOctaveY );

    cv::Point midP( ( startP.x + endP.x ) / 2, ( startP.y + endP.y ) / 2 );
    //cv::putText(img_draw_matches, std::to_string(i), midP, 1, 1, Scalar(255,0,0), 1 );

    cv::line( img_draw_matches, startP, endP, Scalar( 0, 0, 255 ) );

  }

  for ( int i = 0; i < good_matches.size(); i++ )
  {
    r = lowest + int( rand() % range );
    g = lowest + int( rand() % range );
    b = lowest + int( rand() % range );

    std::pair<cv::KeyPoint, int> tmp_pair_1 = v_pair_k1[good_matches[i].queryIdx];
    std::pair<cv::KeyPoint, int> tmp_pair_2 = v_pair_k2[good_matches[i].trainIdx];
    cv::KeyPoint tmp_key_1 = tmp_pair_1.first;
    cv::KeyPoint tmp_key_2 = tmp_pair_2.first;

    KeyLine line1 = keylines1[tmp_pair_1.second];
    cv::Point startP1( line1.sPointInOctaveX, line1.sPointInOctaveY );
    cv::Point endP1( line1.ePointInOctaveX, line1.ePointInOctaveY );
    cv::line( img_draw_matches, startP1, endP1, Scalar( r, g, b ), 2 );

    KeyLine line2 = keylines2[tmp_pair_2.second];
    cv::Point startP2( line2.sPointInOctaveX + sm_image.cols, line2.sPointInOctaveY );
    cv::Point endP2( line2.ePointInOctaveX + sm_image.cols, line2.ePointInOctaveY );
    cv::line( img_draw_matches, startP2, endP2, Scalar( r, g, b ), 2 );

    cv::Point startP_connect( tmp_key_1.pt.x, tmp_key_1.pt.y );
    cv::Point endP_connect( tmp_key_2.pt.x + sm_image.cols, tmp_key_2.pt.y );
    cv::line( img_draw_matches, startP_connect, endP_connect, Scalar( r, g, b ), 2 );

  }

  imshow( "Imshow", img_draw_matches );
  waitKey();
}

int main( int argc, char** argv )
{
  /* get parameters from comand line */
  CommandLineParser parser( argc, argv, keys );
  String image_path1 = parser.get<String>( 0 );
  String image_path2 = parser.get<String>( 1 );

  if( image_path1.empty() || image_path2.empty() )
  {
    help();
    return -1;
  }

  /* load image */
  cv::Mat imageMat1 = imread( image_path1, 1 );
  cv::Mat imageMat2 = imread( image_path2, 1 );

  if( imageMat1.data == NULL || imageMat2.data == NULL )
  {
    std::cout << "Error, images could not be loaded. Please, check their path" << std::endl;
  }

  /* create binary masks */
  cv::Mat mask1 = Mat::ones( imageMat1.size(), CV_8UC1 );
  cv::Mat mask2 = Mat::ones( imageMat2.size(), CV_8UC1 );

  /* create a pointer to a BinaryDescriptor object with default parameters */
  Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();

  /* compute lines */
  std::vector<KeyLine> keylines1, keylines2;

   bd->detect( imageMat2, keylines2, mask2 );
   bd->detect( imageMat1, keylines1, mask1 );

   //compute descriptors
 /*  cv::Mat descr1, descr2;*/
   cv::Mat descr1, descr2;
   bd->compute( imageMat1, keylines1, descr1 );
   bd->compute( imageMat2, keylines2, descr2 );

  //cv::Mat descr1, descr2;
  //( *bd )( imageMat1, mask1, keylines1, descr1, true, false );

  //( *bd )( imageMat2, mask2, keylines2, descr2, true, false );


  /* create a BinaryDescriptorMatcher object */
  Ptr<BinaryDescriptorMatcher> bdm = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();

  /* require match */
  std::vector<DMatch> matches;
  bdm->match( descr1, descr2, matches );
  /* Mat newd1, newd2;
   loadMat(newd1, "bd_descriptors0");
   loadMat(newd2, "bd_descriptors1");*/
  //matches = computeBruteForceSingleImages(newd1, newd2);
  //matches = computeBruteForceSingleImages( descr1, descr2 );

  std::vector<DMatch> good_matches;
  int thresh_good = 25;
  for(int i = 0; i<matches.size(); i++)
  {
    if(matches[i].distance < thresh_good)
    {
      good_matches.push_back(matches[i]);

    }
  }

  /* plot matches */
  cv::Mat outImg;
  std::vector<char> mask( matches.size(), 1 );
  drawLineMatches( imageMat1, keylines1, imageMat2, keylines2, good_matches , outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,
                   DrawLinesMatchesFlags::DEFAULT );

  imshow( "Matches", outImg );
  waitKey();

  Ptr<LSDDetector> lsd = LSDDetector::createLSDDetector();
  std::vector<KeyLine> klsd1, klsd2;
  Mat lsd_descr1, lsd_descr2;
  lsd->detect(imageMat1, klsd1, 2, 2, mask1);
  lsd->detect(imageMat2, klsd2, 2, 2, mask2);

  bd->compute( imageMat1, klsd1, lsd_descr1 );
  bd->compute( imageMat2, klsd2, lsd_descr2 );

  std::vector<DMatch> lsd_matches;
  bdm->match( lsd_descr1, lsd_descr2, lsd_matches);
  good_matches.clear();
  for(int i = 0; i<lsd_matches.size(); i++)
    {
      if(lsd_matches[i].distance < thresh_good)
      {
        good_matches.push_back(lsd_matches[i]);

      }
    }

  cv::Mat lsd_outImg;
    std::vector<char> lsd_mask( matches.size(), 1 );
    drawLineMatches( imageMat1, klsd1, imageMat2, klsd2, good_matches , lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
                     DrawLinesMatchesFlags::DEFAULT );

  imshow("LSD matches", lsd_outImg);
  waitKey();
}

