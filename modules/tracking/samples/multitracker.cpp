/*----------------------------------------------
 * Usage:
 * example_tracking_multitracker <video_name> [algorithm]
 *
 * example:
 * example_tracking_multitracker Bolt/img/%04d.jpg
 * example_tracking_multitracker faceocc2.webm KCF
 *
 * Note: after the OpenCV libary is installed,
 * please re-compile this code with "HAVE_OPENCV" parameter activated
 * to enable the high precission of fps computation
 *--------------------------------------------------*/

#define HAVE_OPENCV

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

#ifdef HAVE_OPENCV
#include <opencv2/flann.hpp>
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

using namespace std;
using namespace cv;

class BoxExtractor {
public:
  Rect2d selectROI(Mat img, bool fromCenter = true);
  Rect2d selectROI(const std::string& windowName, Mat img, bool showCrossair = true, bool fromCenter = true);
  void selectROI(const std::string& windowName, Mat img, std::vector<Rect2d> & boundingBox, bool fromCenter = true);

  struct handlerT{
    // basic parameters
    bool isDrawing;
    Rect2d box;
    Mat image;

    // parameters for drawing from the center
    bool drawFromCenter;
    Point2f center;

    // initializer list
    handlerT(): isDrawing(false), drawFromCenter(true) {};
  }params;

  // to store the tracked objects
  vector<handlerT> objects;

private:
  static void mouseHandler(int event, int x, int y, int flags, void *param);
  void opencv_mouse_callback( int event, int x, int y, int , void *param );

  // save the keypressed characted
  int key;
};

int main( int argc, char** argv ){
  // show help
  if(argc<2){
    cout<<
      " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
      " examples:\n"
      " example_tracking_multitracker Bolt/img/%04d.jpg\n"
      " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
      " \n"
      " Note: after the OpenCV libary is installed,\n"
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
  std::string text;
  char buffer [50];

  // ROI selector
  BoxExtractor box;

  // set the default tracking algorithm
  std::string trackingAlg = "KCF";

  // set the tracking algorithm from parameter
  if(argc>2)
    trackingAlg = argv[2];

  // create the tracker
  MultiTracker trackers(trackingAlg);

  // container of the tracked objects
  vector<Rect2d> objects;

  // set input video
  std::string video = argv[1];
  VideoCapture cap(video);

  Mat frame;

  // get bounding box
  cap >> frame;
  box.selectROI("tracker",frame,objects);

  //quit when the tracked object(s) is not provided
  if(objects.size()<1)
    return 0;

  // initialize the tracker
  trackers.add(frame,objects);

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
    for(unsigned i=0;i<trackers.objects.size();i++)
    rectangle( frame, trackers.objects[i], Scalar( 255, 0, 0 ), 2, 1 );

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

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void *param){
    BoxExtractor *self =static_cast<BoxExtractor*>(param);
    self->opencv_mouse_callback(event,x,y,flags,param);
}

void BoxExtractor::opencv_mouse_callback( int event, int x, int y, int , void *param ){
    handlerT * data = (handlerT*)param;
    switch( event ){
      // update the selected bounding box
      case EVENT_MOUSEMOVE:
        if( data->isDrawing ){
          if(data->drawFromCenter){
            data->box.width = 2*(x-data->center.x)/*data->box.x*/;
            data->box.height = 2*(y-data->center.y)/*data->box.y*/;
            data->box.x = data->center.x-data->box.width/2.0;
            data->box.y = data->center.y-data->box.height/2.0;
          }else{
            data->box.width = x-data->box.x;
            data->box.height = y-data->box.y;
          }
        }
      break;

      // start to select the bounding box
      case EVENT_LBUTTONDOWN:
        data->isDrawing = true;
        data->box = cvRect( x, y, 0, 0 );
        data->center = Point2f((float)x,(float)y);
      break;

      // cleaning up the selected bounding box
      case EVENT_LBUTTONUP:
        data->isDrawing = false;
        if( data->box.width < 0 ){
          data->box.x += data->box.width;
          data->box.width *= -1;
        }
        if( data->box.height < 0 ){
          data->box.y += data->box.height;
          data->box.height *= -1;
        }
      break;
    }
}

Rect2d BoxExtractor::selectROI(Mat img, bool fromCenter){
  return selectROI("Bounding Box Extractor", img, fromCenter);
}

Rect2d BoxExtractor::selectROI(const std::string& windowName, Mat img, bool showCrossair, bool fromCenter){

  key=0;

  // set the drawing mode
  params.drawFromCenter = fromCenter;

  // show the image and give feedback to user
  imshow(windowName,img);

  // copy the data, rectangle should be drawn in the fresh image
  params.image=img.clone();

  // select the object
  setMouseCallback( windowName, mouseHandler, (void *)&params );

  // end selection process on SPACE (32) BACKSPACE (27) or ENTER (13)
  while(!(key==32 || key==27 || key==13)){
    // draw the selected object
    rectangle(
      params.image,
      params.box,
      Scalar(255,0,0),2,1
    );

    // draw cross air in the middle of bounding box
    if(showCrossair){
      // horizontal line
      line(
        params.image,
        Point((int)params.box.x,(int)(params.box.y+params.box.height/2)),
        Point((int)(params.box.x+params.box.width),(int)(params.box.y+params.box.height/2)),
        Scalar(255,0,0),2,1
      );

      // vertical line
      line(
        params.image,
        Point((int)(params.box.x+params.box.width/2),(int)params.box.y),
        Point((int)(params.box.x+params.box.width/2),(int)(params.box.y+params.box.height)),
        Scalar(255,0,0),2,1
      );
    }

    // show the image bouding box
    imshow(windowName,params.image);

    // reset the image
    params.image=img.clone();

    //get keyboard event
    key=waitKey(1);
  }


  return params.box;
}

void BoxExtractor::selectROI(const std::string& windowName, Mat img, std::vector<Rect2d> & boundingBox, bool fromCenter){
  vector<Rect2d> box;
  Rect2d temp;
  key=0;

  // show notice to user
  printf(RED "Select an object to track and then press SPACE or ENTER button!\n" RESET);
  printf(RED "Finish the selection process by pressing BACKSPACE button!\n" RESET);

  // while key is not Backspace
  while(key!=27){
    temp=selectROI(windowName, img, true, fromCenter);
    if(temp.width>0 && temp.height>0)
      box.push_back(temp);
  }
  boundingBox=box;
}