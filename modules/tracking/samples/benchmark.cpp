#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

using namespace std;
using namespace cv;

/*
 * TODO:
 * how to allow to select initial frame?: pass it on cmdline(TODO: format) -->do normalization ala Kalal's assessment protocol for TLD
 * 1. Draw only gt frame
 *
 * FIXME:
 */

static Mat image;
static Rect boundingBox;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;
FILE* gt=NULL;

static const char* keys =
{   "{@video_name   | | video name}"
    "{@ground_truth | | ground truth }"
    "{@tracker_algorithm | | Tracker algorithm }" 
    "{@bdd_frame | | Bounding frame }" 
};

static void listTrackers(){
  vector<String> algorithms;
  Algorithm::getList(algorithms);
  cout << "\nAvailable tracker algorithms:\n";
  for (size_t i=0; i < algorithms.size(); i++){
      const char* algoname=algorithms[i].c_str();
      char *pos=NULL;
      if((pos=strstr((char*)algoname,"TRACKER."))!=NULL){
          printf("%s\n",pos+8);
      }
  }
}

static void myexit(int code){
    if(gt!=NULL){
        fclose(gt);
    }
    exit(code);
}
static Rect2d lineToRect(char* line){
    Rect2d res;
}

static void help(){
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker <video_name> <ground_truth> <algorithm1> <init_box1> <algorithm2> <init_box2> ...\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
  listTrackers();
}

int main( int argc, char** argv )
{
  CommandLineParser parser( argc, argv, keys );

  String tracker_algorithm = parser.get<String>( 2 );
  String video_name = parser.get<String>( 0 );
  String ground_truth = parser.get<String>( 1 );
  String init_box = parser.get<String>( 3 );
  int start_frame = 0;
  char buf[200];

  if( tracker_algorithm.empty() || video_name.empty() || ground_truth.empty() ){
    help();
    myexit(EXIT_FAILURE);
  }

  FILE* gt=fopen(ground_truth.c_str(),"r");
  if(gt==NULL || (fgets(buf,sizeof(buf),gt)==NULL)){//FIXME: diff msg
      printf("cannot open the ground truth file %s\n",ground_truth.c_str());
      myexit(EXIT_FAILURE);
  }

  int coords[4]={0,0,0,0};
  String initBoundingBox=parser.get<String>(3);
  if(initBoundingBox.empty()){
      printf("please, submit the initial bounding box for %s\n",tracker_algorithm.c_str());
      int i=0;
      if(fgets(buf,sizeof(buf),gt)!=NULL){
        printf("FYI, ground truth is %s\n",buf);
      }
      myexit(EXIT_FAILURE);
  }
  for(size_t npos=0,pos=0,ctr=0;ctr<4;ctr++){
    npos=initBoundingBox.find_first_of(',',pos);
    if(npos==string::npos && ctr<3){
       printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer"
               " cordinates of opposed corners of bdd box\n");
       printf("got: %s\n",initBoundingBox.substr(pos,string::npos).c_str());
       printf("manual selection of bounding box will be employed\n");
       break;
    }
    int num=atoi(initBoundingBox.substr(pos,(ctr==3)?(string::npos):(npos-pos)).c_str());
    if(num<=0){
       printf("bounding box should be given in format \"x1,y1,x2,y2\",where x's and y's are integer"
               " cordinates of opposed corners of bdd box\n");
       printf("got: %s\n",initBoundingBox.substr(pos,npos-pos).c_str());
       printf("program will be terminated\n");
       myexit(EXIT_FAILURE);
    }
    coords[ctr]=num;
    pos=npos+1;
  }
  boundingBox.x = coords[0];
  boundingBox.y = coords[1];
  boundingBox.width = std::abs( coords[2] - coords[0] );
  boundingBox.height = std::abs( coords[3]-coords[1]);

  //open the capture
  VideoCapture cap;
  cap.open( video_name );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() ){
    help();
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    parser.printMessage();
    return -1;
  }

  Mat frame;
  paused = true;
  namedWindow( "Tracking API", 1 );

  //instantiates the specific Tracker
  Ptr<Tracker> tracker = Tracker::create( tracker_algorithm );
  if( tracker == NULL ){
    cout << "***Error in the instantiation of the tracker...***\n";
    return -1;
  }

  //get the first frame
  cap >> frame;
  frame.copyTo( image );
  vector<Scalar> palette;
  palette.push_back(Scalar(255,0,0));
  palette.push_back(Scalar(0,0,255));

  //TODO: draw ground truth
  Rect2d gt_box=lineToRect(buf);
  rectangle( image, boundingBox,palette[1], 2, 1 );
  imshow( "Tracking API", image );
  //TODO: initialize tracker

  paused=false;
  bool initialized = false;
  int frameCounter = 0;

  for ( ;; )
  {
    if( !paused ){

      /*if( !initialized )
      {
        //initializes the tracker
        if( !tracker->init( frame, boundingBox ) )
        {
          cout << "***Could not initialize tracker...***\n";
          return -1;
        }
        initialized = true;
      }*/
        //updates the tracker
      if( tracker->update( frame, boundingBox ) ){
        rectangle( image, boundingBox, palette[1], 2, 1 );
      }
      imshow( "Tracking API", image );
      frameCounter++;
    }

    char c = (char) waitKey( 2 );
    if( c == 'q' )
      break;
    if( c == 'p' )
      paused = !paused;

  }

  fclose(gt);
  return 0;
}
