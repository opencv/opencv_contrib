#include "opencv2/optflow.hpp"
#include <iostream>

/* This tool trains the forest for the Global Patch Collider and stores output to the "forest.yml.gz".
 */

using namespace cv;

const String keys = "{help h ?       |             | print this message}"
                    "{max-tree-depth |             | Maximum tree depth to stop partitioning}"
                    "{min-samples    |             | Minimum number of samples in the node to stop partitioning}"
                    "{descriptor-type|0            | Descriptor type. Set to 0 for quality, 1 for speed.}"
                    "{print-progress |             | Set to 0 to enable quiet mode, set to 1 to print progress}"
                    "{f forest       |forest.yml.gz| Path where to store resulting forest. It is recommended to use .yml.gz extension.}";

const int nTrees = 5;

static void fillInputImagesFromCommandLine( std::vector< String > &img1, std::vector< String > &img2, std::vector< String > &gt, int argc,
                                            const char **argv )
{
  for ( int i = 1, j = 0; i < argc; ++i )
  {
    if ( argv[i][0] == '-' )
      continue;
    if ( j % 3 == 0 )
      img1.push_back( argv[i] );
    if ( j % 3 == 1 )
      img2.push_back( argv[i] );
    if ( j % 3 == 2 )
      gt.push_back( argv[i] );
    ++j;
  }
}

int main( int argc, const char **argv )
{
  CommandLineParser parser( argc, argv, keys );
  parser.about( "Global Patch Collider training tool" );

  std::vector< String > img1, img2, gt;
  optflow::GPCTrainingParams params;

  if ( parser.has( "max-tree-depth" ) )
    params.maxTreeDepth = parser.get< unsigned >( "max-tree-depth" );
  if ( parser.has( "min-samples" ) )
    params.minNumberOfSamples = parser.get< unsigned >( "min-samples" );
  if ( parser.has( "descriptor-type" ) )
    params.descriptorType = parser.get< int >( "descriptor-type" );
  if ( parser.has( "print-progress" ) )
    params.printProgress = parser.get< unsigned >( "print-progress" ) != 0;

  fillInputImagesFromCommandLine( img1, img2, gt, argc, argv );

  if ( parser.has( "help" ) || img1.size() != img2.size() || img1.size() != gt.size() || img1.size() == 0 )
  {
    std::cerr << "\nUsage: " << argv[0] << " [params] ImageFrom1 ImageTo1 GroundTruth1 ... ImageFromN ImageToN GroundTruthN\n" << std::endl;
    parser.printMessage();
    return 1;
  }

  Ptr< optflow::GPCForest< nTrees > > forest = optflow::GPCForest< nTrees >::create();
  forest->train( img1, img2, gt, params );
  forest->save( parser.get< String >( "forest" ) );

  return 0;
}
