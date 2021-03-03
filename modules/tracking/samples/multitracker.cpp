/*----------------------------------------------
 * Usage:
 * example_tracking_multitracker <video_name> [algorithm]
 *
 * example:
 * example_tracking_multitracker Bolt/img/%04d.jpg
 * example_tracking_multitracker faceocc2.webm KCF
 *
 * Note: after the OpenCV library is installed,
 * please re-compile this code with "HAVE_OPENCV" parameter activated
 * to enable the high precission of fps computation
 *--------------------------------------------------*/

/* after the OpenCV library is installed
 * please uncomment the the line below and re-compile this code
 * to enable high precission of fps computation
 */
//#define HAVE_OPENCV

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>
#include "samples_utility.hpp"

#ifdef HAVE_OPENCV
#include <opencv2/flann.hpp>
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

using namespace std;
using namespace cv;

int main( int argc, char** argv ){
  // show help
  if(argc<2){
    cout<<
      " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
      " examples:\n"
      " example_tracking_multitracker Bolt/img/%04d.jpg\n"
      " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
      " \n"
      " Note: after the OpenCV library is installed,\n"
      " please re-compile with the HAVE_OPENCV parameter activated\n"
      " to enable the high precission of fps computation.\n"
      << endl;
    return 0;
  }

  // timer
#ifdef HAVE_OPENCV
  cvflann::StartStopTimer timer;
#else
  clock_t timer;
#endif

  // for showing the speed
  double fps;
  String text;
  char buffer [50];

  // set the default tracking algorithm
  String trackingAlg = "KCF";

  // set the tracking algorithm from parameter
  if(argc>2)
    trackingAlg = argv[2];

  // create the tracker
  legacy::MultiTracker trackers;

  // container of the tracked objects
  vector<Rect> ROIs;
  vector<Rect2d> objects;

  // set input video
  String video = argv[1];
  VideoCapture cap(video);

  Mat frame;

  // get bounding box
  cap >> frame;
  selectROIs("tracker",frame,ROIs);

  //quit when the tracked object(s) is not provided
  if(ROIs.size()<1)
    return 0;

  std::vector<Ptr<legacy::Tracker> > algorithms;
  for (size_t i = 0; i < ROIs.size(); i++)
  {
      algorithms.push_back(createTrackerByName_legacy(trackingAlg));
      objects.push_back(ROIs[i]);
  }

  // initialize the tracker
  trackers.add(algorithms,frame,objects);

  // do the tracking
  printf(GREEN "Start the tracking process, press ESC to quit.\n" RESET);
  for ( ;; ){
    // get frame from the video
    cap >> frame;

    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    // start the timer
#ifdef HAVE_OPENCV
    timer.start();
#else
    timer=clock();
#endif

    //update the tracking result
    trackers.update(frame);

    // calculate the processing speed
#ifdef HAVE_OPENCV
    timer.stop();
    fps=1.0/timer.value;
    timer.reset();
#else
    timer=clock();
    trackers.update(frame);
    timer=clock()-timer;
    fps=(double)CLOCKS_PER_SEC/(double)timer;
#endif

    // draw the tracked object
    for(unsigned i=0;i<trackers.getObjects().size();i++)
      rectangle( frame, trackers.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );

    // draw the processing speed
    sprintf (buffer, "speed: %.0f fps", fps);
    text = buffer;
    putText(frame, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));

    // show image with the tracked object
    imshow("tracker",frame);

    //quit on ESC button
    if(waitKey(1)==27)break;
  }

}
