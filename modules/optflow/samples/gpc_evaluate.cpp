#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/optflow.hpp"
#include <iostream>

using namespace cv;

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
      img.at< Vec3f >( i, j ) = getFlowColor( flow.at<Point2f>( i, j ) );

  cvtColor( img, img, COLOR_HSV2BGR );
}

int main( int argc, const char **argv )
{
  if ( argc != 4 )
  {
    std::cerr << "Usage: " << argv[0] << " ImageFrom ImageTo GroundTruth" << std::endl;
    return 1;
  }

  Ptr< optflow::GPCForest< nTrees > > forest = Algorithm::load< optflow::GPCForest< nTrees > >( "forest.dump" );

  Mat from = imread( argv[1] );
  Mat to = imread( argv[2] );
  Mat gt = optflow::readOpticalFlow( argv[3] );
  std::vector< std::pair< Point2i, Point2i > > corr;

  forest->findCorrespondences( from, to, corr );

  std::cout << "Found " << corr.size() << " matches." << std::endl;
  double error = 0;
  Mat dispErr = Mat::zeros( from.size(), CV_32FC3 );
  dispErr = Scalar( 0, 0, 1 );
  Mat disp = Mat::zeros( from.size(), CV_32FC3 );
  disp = Scalar( 0, 0, 1 );

  for ( size_t i = 0; i < corr.size(); ++i )
  {
    const Point2f a = corr[i].first;
    const Point2f b = corr[i].second;
    const Point2f c = a + gt.at< Point2f >( a.y, a.x );
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

  namedWindow( "Correspondences", WINDOW_AUTOSIZE );
  imshow( "Correspondences", disp );
  namedWindow( "Error", WINDOW_AUTOSIZE );
  imshow( "Error", dispErr );
  namedWindow( "Ground truth", WINDOW_AUTOSIZE );
  imshow( "Ground truth", dispGroundTruth );
  waitKey( 0 );

  return 0;
}
