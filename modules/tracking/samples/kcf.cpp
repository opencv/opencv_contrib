/*----------------------------------------------
 * Usage:
 * example_tracking_kcf <video path>
 *
 * example:
 * example_tracking_kcf Bolt/img/%04.jpg
 * example_tracking_kcf faceocc2.webm
 *--------------------------------------------------*/

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;
using namespace cv;

class BoxExtractor {
public:
  Rect2d extract(Mat img);
  Rect2d extract(const std::string& windowName, Mat img);

  struct handlerT{
    bool isDrawing;
    Rect2d box;
    Mat image;

    // initializer list
    handlerT(): isDrawing(false) {};
  }params;

private:
  static void mouseHandler(int event, int x, int y, int flags, void *param);
  void opencv_mouse_callback( int event, int x, int y, int , void *param );
};

int main( int , char** argv ){
  BoxExtractor box;

  // create the tracker
  Ptr<Tracker> tracker = Tracker::create( "KCF" );

  Rect2d roi;
  int start_frame = 0;
  std::string video = argv[1];

  VideoCapture cap;
  cap.open( video );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  Mat frame;

  // get bounding box
  cap >> frame;
  roi=box.extract("tracker",frame);

  // initialize the tracker
  tracker->init(frame,roi);

  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");
  for ( ;; ){
    // get frame from the video
    cap >> frame;

    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    // update the tracking result
    tracker->update(frame,roi);

    // draw the tracked object
    rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );

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
          data->box.width = x-data->box.x;
          data->box.height = y-data->box.y;
        }
      break;

      // start to select the bounding box
      case EVENT_LBUTTONDOWN:
        data->isDrawing = true;
        data->box = cvRect( x, y, 0, 0 );
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

Rect2d BoxExtractor::extract(Mat img){
  return extract("Bounding Box Extractor", img);
}

Rect2d BoxExtractor::extract(const std::string& windowName, Mat img){

  int key=0;

  // show the image and give feedback to user
  imshow(windowName,img);
  printf("Select an object to track and then press SPACE/BACKSPACE/ENTER button!\n");

  // copy the data, rectangle should be drawn in the fresh image
  params.image=img.clone();

  // select the object
  setMouseCallback( windowName, mouseHandler, (void *)&params );

  while(!(key==32 || key==27 || key==13)){
    // draw the selected object
    rectangle(
      params.image,
      Point(params.box.x, params.box.y),
      Point(params.box.x+params.box.width,params.box.y+params.box.height),
      Scalar(255,0,0),2,1
    );

    // show the image bouding box
    imshow(windowName,params.image);

    // reset the image
    params.image=img.clone();

    //get keyboard event
    key=waitKey(1);
  }


  return params.box;
}