#include "opencv2/core/ocl.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/optflow.hpp"
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace cv;
using optflow::OpticalFlowPCAFlow;
using optflow::PCAPrior;

const String keys = "{help h ?     |      | print this message}"
                    "{@image1      |<none>| image1}"
                    "{@image2      |<none>| image2}"
                    "{@groundtruth |<none>| path to the .flo file}"
                    "{@prior       |<none>| path to a prior file for PCAFlow}"
                    "{@output      |<none>| output image path}"
                    "{g gpu        |      | use OpenCL}";

static double normL2( const Point2f &v ) { return sqrt( v.x * v.x + v.y * v.y ); }

static bool fileProbe( const char *name ) { return std::ifstream( name ).good(); }

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

static bool isFlowCorrect( const Point2f &u )
{
  return !cvIsNaN( u.x ) && !cvIsNaN( u.y ) && ( fabs( u.x ) < 1e9 ) && ( fabs( u.y ) < 1e9 );
}

static double calcEPE( const Mat &f1, const Mat &f2 )
{
  double sum = 0;
  Size sz = f1.size();
  size_t cnt = 0;
  for ( int i = 0; i < sz.height; ++i )
    for ( int j = 0; j < sz.width; ++j )
      if ( isFlowCorrect( f1.at< Point2f >( i, j ) ) && isFlowCorrect( f2.at< Point2f >( i, j ) ) )
      {
        sum += normL2( f1.at< Point2f >( i, j ) - f2.at< Point2f >( i, j ) );
        ++cnt;
      }
  return sum / cnt;
}

static void displayResult( Mat &i1, Mat &i2, Mat &gt, Ptr< DenseOpticalFlow > &algo, OutputArray _img, const char *descr,
                           const bool useGpu = false )
{
  Mat flow( i1.size[0], i1.size[1], CV_32FC2 );
  TickMeter meter;
  meter.start();

  if ( useGpu )
    algo->calc( i1, i2, flow.getUMat( ACCESS_RW ) );
  else
    algo->calc( i1, i2, flow );

  meter.stop();
  displayFlow( flow, _img );
  Mat img = _img.getMat();
  putText( img, descr, Point2i( 24, 40 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
  char buf[256];
  sprintf( buf, "Average EPE: %.2f", calcEPE( flow, gt ) );
  putText( img, buf, Point2i( 24, 80 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
  sprintf( buf, "Time: %.2fs", meter.getTimeSec() );
  putText( img, buf, Point2i( 24, 120 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
}

static void displayGT( InputArray _flow, OutputArray _img, const char *descr )
{
  displayFlow( _flow, _img );
  Mat img = _img.getMat();
  putText( img, descr, Point2i( 24, 40 ), FONT_HERSHEY_DUPLEX, 1, Vec3b( 1, 0, 0 ), 2, LINE_AA );
}

int main( int argc, const char **argv )
{
  CommandLineParser parser( argc, argv, keys );
  parser.about( "PCAFlow demonstration" );

  if ( parser.has( "help" ) )
  {
    parser.printMessage();
    return 0;
  }

  String img1 = parser.get< String >( 0 );
  String img2 = parser.get< String >( 1 );
  String groundtruth = parser.get< String >( 2 );
  String prior = parser.get< String >( 3 );
  String outimg = parser.get< String >( 4 );
  const bool useGpu = parser.has( "gpu" );

  if ( !parser.check() )
  {
    parser.printErrors();
    return 1;
  }

  if ( !fileProbe( prior.c_str() ) )
  {
    std::cerr << "Can't open the file with prior! Check the provided path: " << prior << std::endl;
    return 1;
  }

  cv::ocl::setUseOpenCL( useGpu );

  Mat i1 = imread( img1 );
  Mat i2 = imread( img2 );
  Mat gt = readOpticalFlow( groundtruth );

  Mat i1g, i2g;
  cvtColor( i1, i1g, COLOR_BGR2GRAY );
  cvtColor( i2, i2g, COLOR_BGR2GRAY );

  Mat pcaflowDisp, pcaflowpriDisp, farnebackDisp, gtDisp;

  {
    Ptr< DenseOpticalFlow > pcaflow = makePtr< OpticalFlowPCAFlow >( makePtr< PCAPrior >( prior.c_str() ) );
    displayResult( i1, i2, gt, pcaflow, pcaflowpriDisp, "PCAFlow with prior", useGpu );
  }

  {
    Ptr< DenseOpticalFlow > pcaflow = makePtr< OpticalFlowPCAFlow >();
    displayResult( i1, i2, gt, pcaflow, pcaflowDisp, "PCAFlow without prior", useGpu );
  }

  {
    Ptr< DenseOpticalFlow > farneback = optflow::createOptFlow_Farneback();
    displayResult( i1g, i2g, gt, farneback, farnebackDisp, "Farneback", useGpu );
  }

  displayGT( gt, gtDisp, "Ground truth" );

  Mat disp1, disp2;
  vconcat( pcaflowpriDisp, farnebackDisp, disp1 );
  vconcat( pcaflowDisp, gtDisp, disp2 );
  hconcat( disp1, disp2, disp1 );
  disp1 *= 255;

  imwrite( outimg, disp1 );

  return 0;
}
