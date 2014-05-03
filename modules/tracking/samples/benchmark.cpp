#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#define CMDLINEMAX 10

using namespace std;
using namespace cv;

/*
 * TODO:
 * TODO: several videos (<-- cmdline args format)
 * TODO: test 2 trackers: MIL and MADIANFLOW
 do normalization ala Kalal's assessment protocol for TLD
 *
 * FIXME: normal exit, multiple rectangles, TODOs and FIXMEs
 */

static Mat image;
static Rect boundingBox;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;
FILE* gt=NULL;

static const char* keys =
{ "{@tracker_algorithm | | Tracker algorithm }"
    "{@video_name      | | video name        }"
    "{@start_frame     |0| Start frame       }" 
    "{@bounding_frame  |0,0,0,0| Initial bounding frame}"};

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
static inline void myexit(int code){
    printf("terminating...\n");
    if(gt!=NULL){
        fclose(gt);
    }
    exit(code);
}
static Rect2d lineToRect(char* line){
  Rect2d res;
  char * ptr;
  ptr = strtok (line,", ");
  double nums[4]={0};
  for(int i=0; i<4 && ptr != NULL;i++){
    nums[i]=atof(ptr);
    ptr = strtok (NULL,", ");
    if(nums[i]<=0){
        break;
    }
  }
  if(nums[sizeof(nums)/sizeof(nums[0])-1]<=0){
      printf("we had problems with decoding line %s\n",line);
      myexit(EXIT_FAILURE);
  }
  res.x=cv::min(nums[0],nums[2]);
  res.y=cv::min(nums[1],nums[3]);
  res.width=cv::abs(nums[0]-nums[2]);
  res.height=cv::abs(nums[1]-nums[3]);
  return res;
}
static inline double overlap(Rect2d r1,Rect2d r2){
    double a1=r1.area(), a2=r2.area(), a0=(r1&r2).area();
    return a0/(a1+a2-a0);
}
static void help(){
  cout << "\nThis example shows the functionality of \"Long-term optical tracking API\""
       "-- pause video [p] and draw a bounding box around the target to start the tracker\n"
       "Example of <video_name> is in opencv_extra/testdata/cv/tracking/\n"
       "Call:\n"
       "./tracker [<keys and args>] <video_name> <ground_truth> <algorithm1> <init_box1> <algorithm2> <init_box2> ...\n"
       << endl;

  cout << "\n\nHot keys: \n"
       "\tq - quit the program\n"
       "\tp - pause video\n";
  listTrackers();
  exit(EXIT_SUCCESS);
}
static void parseCommandLineArgs(int argc, char** argv,char* videos[],char* gts[],
        int* vc,char* algorithms[],char* initBoxes[][CMDLINEMAX],int* ac){

    vector<String> trackers;
    Algorithm::getList(trackers);
    for(int i=0;i<trackers.size();i++){
        if(strstr(trackers[i].c_str(),"TRACKER.")!=trackers[i].c_str()){
            trackers.erase(trackers.begin()+i);
            i--;
        }
    }

    *ac=*vc=0;
    for(int i=1;i<argc;i++){
        if(argv[i][0]=='-'){
            char *key=(argv[i]+1),*argument=NULL;
            if(std::strcmp("h",key)==0||std::strcmp("help",key)==0){
                help();
            }
            if((argument=strchr(argv[i],'='))==NULL){
                i++;
                argument=argv[i];
            }else{
                argument++;
            }
            continue;
        }
        bool isVideo=true;
        for(int j=0;j<trackers.size();j++){
            if(strcmp(argv[i],trackers[j].c_str()+8)==0){
                isVideo=false;
                break;
            }
        }
        if(isVideo){//TODO: multiple recs
            videos[*vc]=argv[i];
            i++;
            gts[*vc]=(i<argc)?argv[i]:NULL;
            (*vc)++;
        }else{
            algorithms[*ac]=argv[i];
            i++;
            for(int j=0;j<*vc;j++,i++){
                initBoxes[*ac][j]=(i<argc)?argv[i]:NULL;
            }
            i--;(*ac)++;
        }
    }
}
static void assessment(char* video,char* gt, char* algorithms[],char* initBoxes[],int algnum){
  printf("gonna do assessment with %s %s\n");
  printf("algos to assess\n");
  for(int i=0;i<algnum;i++){
      printf("%s %s\n",algorithms[i],initBoxes[algnum*i]);
  }
  /*VideoCapture cap;
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
  paused = true;
  namedWindow( "Tracking API", 1 );

  //instantiates the specific Tracker
  Ptr<Tracker> tracker = Tracker::create( tracker_algorithm );
  if( tracker == NULL )
  {
    cout << "***Error in the instantiation of the tracker...***\n";
    return -1;
  }

  //get the first frame
  cap >> frame;
  frame.copyTo( image );
  if(initBoxWasGivenInCommandLine){
      selectObject=true;
      paused=false;
      boundingBox.x = coords[0];
      boundingBox.y = coords[1];
      boundingBox.width = std::abs( coords[2] - coords[0] );
      boundingBox.height = std::abs( coords[3]-coords[1]);
      printf("bounding box with vertices (%d,%d) and (%d,%d) was given in command line\n",coords[0],coords[1],coords[2],coords[3]);
      rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
  }
  imshow( "Tracking API", image );

  bool initialized = false;
  int frameCounter = 0;

  for ( ;; )
  {
    if( !paused )
    {
      if(initialized){
          cap >> frame;
          if(frame.empty()){
            break;
          }
          frame.copyTo( image );
      }

      if( !initialized && selectObject )
      {
        //initializes the tracker
        if( !tracker->init( frame, boundingBox ) )
        {
          cout << "***Could not initialize tracker...***\n";
          return -1;
        }
        initialized = true;
      }
      else if( initialized )
      {
        //updates the tracker
        if( tracker->update( frame, boundingBox ) )
        {
          rectangle( image, boundingBox, Scalar( 255, 0, 0 ), 2, 1 );
        }
      }
      imshow( "Tracking API", image );
      frameCounter++;
    }

    char c = (char) waitKey( 2 );
    if( c == 'q' )
      break;
    if( c == 'p' )
      paused = !paused;

  }*/
}

int main( int argc, char** argv ){

  int vcount=0,acount=0;
  char* videos[CMDLINEMAX],*gts[CMDLINEMAX],*algorithms[CMDLINEMAX],*initBoxes[CMDLINEMAX][CMDLINEMAX];
  parseCommandLineArgs(argc,argv,videos,gts,&vcount,algorithms,initBoxes,&acount);
  CV_Assert(acount<CMDLINEMAX && vcount<CMDLINEMAX);
  printf("videos and gts\n");
  for(int i=0;i<vcount;i++){
      printf("%s %s\n",videos[i],gts[i]);
  }
  printf("algorithms and boxes\n");
  for(int i=0;i<acount;i++){
      printf("%s ",algorithms[i]);
      for(int j=0;j<vcount;j++){
        printf("%s \n",initBoxes[i][j]);
      }
  }

  String tracker_algorithm;
  String video_name;

  Ptr<Tracker> tracker = Tracker::create( tracker_algorithm );

  return 0;
}
