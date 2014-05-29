#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static const char* keys =
{ "{@saliency_algorithm | | Saliency algorithm <saliencyAlgorithmType.[saliencyAlgorithmTypeSubType]> }"
    "{@video_name      | | video name            }"
    "{@start_frame     |1| Start frame           }" };

static void help()
{
  cout << "\nThis example shows the functionality of \"Saliency \""
       "Call:\n"
       "./example_saliency_computeSaliency <saliencyAlgorithmType.[saliencyAlgorithmTypeSubType]> <video_name> <start_frame> \n"
       << endl;
}

int main( int argc, char** argv )
{
  CommandLineParser parser( argc, argv, keys );

  String saliency_algorithm = parser.get<String>( 0 );
  String video_name = parser.get<String>( 1 );
  int start_frame = parser.get<int>( 2 );

  if( saliency_algorithm.empty() || video_name.empty() )
  {
    help();
    return -1;
  }

  //open the capture
  VideoCapture cap;
  cap.open( video_name );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() )
  {
    help();
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    parser.printMessage();
    return -1;
  }

  Mat frame;

  //instantiates the specific Saliency
  Ptr<Saliency> saliencyAlgorithm = Saliency::create( saliency_algorithm );

  if( saliencyAlgorithm == NULL )
  {
    cout << "***Error in the instantiation of the saliency algorithm...***\n";
    return -1;
  }

  Mat image, saliencyMap, binaryMap;

  cap >> frame;
  if( frame.empty() )
  {
    return 0;
  }

  frame.copyTo( image );

  if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
  {
    StaticSaliencySpectralResidual spec;
    spec.computeBinaryMap( saliencyMap, binaryMap );
    //saliencyAlgorithm->computeBinaryMap( saliencyMap, binaryMap );
    imshow( "Saliency Map", saliencyMap );
    imshow( "Original Image", image );
    imshow( "Binary Map", binaryMap );
    waitKey( 0 );

  }

  return 0;
}
