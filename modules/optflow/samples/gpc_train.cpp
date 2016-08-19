#include "opencv2/optflow.hpp"
#include <iostream>

const int nTrees = 5;

int main( int argc, const char **argv )
{
  int nSequences = argc - 1;

  if ( nSequences <= 0 || nSequences % 3 != 0 )
  {
    std::cerr << "Usage: " << argv[0] << " ImageFrom1 ImageTo1 GroundTruth1 ... ImageFromN ImageToN GroundTruthN" << std::endl;
    return 1;
  }

  nSequences /= 3;
  std::vector< cv::String > img1, img2, gt;

  for ( int i = 0; i < nSequences; ++i )
  {
    img1.push_back( argv[1 + i * 3] );
    img2.push_back( argv[1 + i * 3 + 1] );
    gt.push_back( argv[1 + i * 3 + 2] );
  }

  cv::Ptr< cv::optflow::GPCForest< nTrees > > forest = cv::optflow::GPCForest< nTrees >::create();
  forest->train( img1, img2, gt );
  forest->save( "forest.dump" );

  return 0;
}
