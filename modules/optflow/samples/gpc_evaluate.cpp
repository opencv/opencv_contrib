#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/optflow.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>

/* This tool finds correspondences between two images using Global Patch Collider
 * and calculates error using provided ground truth flow.
 *
 * It will look for the file named "forest.yml.gz" with a learned forest.
 * You can obtain the "forest.yml.gz" either by manually training it using another tool with *_train suffix
 * or by downloading one of the files trained on some publicly available dataset from here:
 *
 * https://drive.google.com/open?id=0B7Hb8cfuzrIIZDFscXVYd0NBNFU
 */

using namespace cv;

const String keys = "{help h ?     |             | print this message}"
                    "{@image1      |<none>       | image1}"
                    "{@image2      |<none>       | image2}"
                    "{@groundtruth |<none>       | path to the .flo file}"
                    "{@output      |             | output to a file instead of displaying, output image path}"
                    "{g gpu        |             | use OpenCL}"
                    "{f forest     |forest.yml.gz| path to the forest.yml.gz}";

const int nTrees = 5;

static double normL2( const Point2f &v ) { return sqrt( v.x * v.x + v.y * v.y ); }

static Vec3d getFlowColor( const Point2f &f, const bool logScale = true, const double scaleDown = 5 )
{
  if ( f.x == 0 && f.y == 0 )
    return Vec3d( 0, 0, 1 );

  double radius = normL2( f );
  if ( logScale )
    radius = log( radius + 1 );
  radius /= scaleDown;
  radius = std::min( 1.0, radius );

  double angle = ( atan2( -f.y, -f.x ) + CV_PI ) * 180 / CV_PI;
  return Vec3d( angle, radius, 1 );
}

static void displayFlow( InputArray _flow, OutputArray _img )
{
  const Size sz = _flow.size();
  Mat flow = _flow.getMat();
  _img.create( sz, CV_32FC3 );
  Mat img = _img.getMat();

  for ( int i = 0; i < sz.height; ++i )
    for ( int j = 0; j < sz.width; ++j )
      img.at< Vec3f >( i, j ) = getFlowColor( flow.at< Point2f >( i, j ) );

  cvtColor( img, img, COLOR_HSV2BGR );
}

static bool fileProbe( const char *name ) { return std::ifstream( name ).good(); }

int main( int argc, const char **argv )
{
  CommandLineParser parser( argc, argv, keys );
  parser.about( "Global Patch Collider evaluation tool" );

  if ( parser.has( "help" ) )
  {
    parser.printMessage();
    return 0;
  }

  String fromPath = parser.get< String >( 0 );
  String toPath = parser.get< String >( 1 );
  String gtPath = parser.get< String >( 2 );
  String outPath = parser.get< String >( 3 );
  const bool useOpenCL = parser.has( "gpu" );
  String forestDumpPath = parser.get< String >( "forest" );

  if ( !parser.check() )
  {
    parser.printErrors();
    return 1;
  }

  if ( !fileProbe( forestDumpPath.c_str() ) )
  {
    std::cerr << "Can't open the file with a trained model: `" << forestDumpPath
              << "`.\nYou can obtain this file either by manually training the model using another tool with *_train suffix or by "
                 "downloading one of the files trained on some publicly available dataset from "
                 "here:\nhttps://drive.google.com/open?id=0B7Hb8cfuzrIIZDFscXVYd0NBNFU"
              << std::endl;
    return 1;
  }

  ocl::setUseOpenCL( useOpenCL );

  Ptr< optflow::GPCForest< nTrees > > forest = Algorithm::load< optflow::GPCForest< nTrees > >( forestDumpPath );

  Mat from = imread( fromPath );
  Mat to = imread( toPath );
  Mat gt = optflow::readOpticalFlow( gtPath );
  std::vector< std::pair< Point2i, Point2i > > corr;

  TickMeter meter;
  meter.start();

  forest->findCorrespondences( from, to, corr, optflow::GPCMatchingParams( useOpenCL ) );

  meter.stop();

  std::cout << "Found " << corr.size() << " matches." << std::endl;
  std::cout << "Time:  " << meter.getTimeSec() << " sec." << std::endl;
  double error = 0;
  Mat dispErr = Mat::zeros( from.size(), CV_32FC3 );
  dispErr = Scalar( 0, 0, 1 );
  Mat disp = Mat::zeros( from.size(), CV_32FC3 );
  disp = Scalar( 0, 0, 1 );

  for ( size_t i = 0; i < corr.size(); ++i )
  {
    const Point2f a = corr[i].first;
    const Point2f b = corr[i].second;
    const Point2f c = a + gt.at< Point2f >( corr[i].first.y, corr[i].first.x );
    error += normL2( b - c );
    circle( disp, a, 3, getFlowColor( b - a ), -1 );
    circle( dispErr, a, 3, getFlowColor( b - c, false, 32 ), -1 );
  }

  error /= corr.size();

  std::cout << "Average endpoint error: " << error << " px." << std::endl;

  cvtColor( disp, disp, COLOR_HSV2BGR );
  cvtColor( dispErr, dispErr, COLOR_HSV2BGR );

  Mat dispGroundTruth;
  displayFlow( gt, dispGroundTruth );

  if ( outPath.length() )
  {
    putText( disp, "Sparse matching: Global Patch Collider", Point2i( 24, 40 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
    char buf[256];
    sprintf( buf, "Average EPE: %.2f", error );
    putText( disp, buf, Point2i( 24, 80 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
    sprintf( buf, "Number of matches: %u", (unsigned)corr.size() );
    putText( disp, buf, Point2i( 24, 120 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
    disp *= 255;
    imwrite( outPath, disp );
    return 0;
  }

  namedWindow( "Correspondences", WINDOW_AUTOSIZE );
  imshow( "Correspondences", disp );
  namedWindow( "Error", WINDOW_AUTOSIZE );
  imshow( "Error", dispErr );
  namedWindow( "Ground truth", WINDOW_AUTOSIZE );
  imshow( "Ground truth", dispGroundTruth );
  waitKey( 0 );

  return 0;
}
