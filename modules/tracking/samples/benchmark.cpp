#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#define CMDLINEMAX 10

using namespace std;
using namespace cv;

/*
 * TODO: test 2 trackers: MIL and MADIANFLOW
 do normalization ala Kalal's assessment protocol for TLD
 *
 * FIXME: normal exit
 */

static Mat image;
static Rect boundingBox;
static bool paused;
static bool selectObject = false;
static bool startSelection = false;

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
static int lineToRect(char* line,Rect2d& res){
  char * ptr;
  if(line==NULL || line[0]=='\0'){
      return -1;
  }
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
      printf("lineToRect had problems with decoding line %s\n",line);
      return -1;
  }
  res.x=cv::min(nums[0],nums[2]);
  res.y=cv::min(nums[1],nums[3]);
  res.width=cv::abs(nums[0]-nums[2]);
  res.height=cv::abs(nums[1]-nums[3]);
  return 0;
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
        if(isVideo){
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
typedef struct{
    char* videoName;
    vector<int> correctFrames;
    int len;
}assessmentRes;
static assessmentRes assessment(char* video,char* gt_str, char* algorithms[],char* initBoxes_str[],int algnum){

  char buf[200];
  int start_frame=0;
  int linecount=0;
  vector<Scalar> palette;
  palette.push_back(Scalar(255,0,0));//BGR
  palette.push_back(Scalar(0,0,255));
  palette.push_back(Scalar(0,255,255));
  Rect2d boundingBox;

  FILE* gt=fopen(gt_str,"r");
  if(gt==NULL){
      printf("cannot open the ground truth file %s\n",gt_str);
      exit(EXIT_FAILURE);
  }
  for(linecount=0;fgets(buf,sizeof(buf),gt)!=NULL;linecount++);
  if(linecount==0){
      printf("ground truth file %s has no lines\n",gt_str);
      exit(EXIT_FAILURE);
  }
  fseek(gt,0,SEEK_SET);
  fgets(buf,sizeof(buf),gt);

  std::vector<Rect2i> initBoxes(algnum);
  for(int i=0;i<algnum;i++){
      printf("%s %s\n",algorithms[i],initBoxes_str[CMDLINEMAX*i]);
      if(lineToRect(initBoxes_str[CMDLINEMAX*i],boundingBox)<0){
          printf("please, specify bounding box for video %s, algorithm %s\n",video,algorithms[i]);
          printf("FYI, initial bounding box in ground truth is %s\n",buf);
          if(gt!=NULL){
              fclose(gt);
          }
          exit(EXIT_FAILURE);
      }else{
          initBoxes[i].x=boundingBox.x;
          initBoxes[i].y=boundingBox.y;
          initBoxes[i].width=boundingBox.width;
          initBoxes[i].height=boundingBox.height;
      }
  }

  VideoCapture cap;
  cap.open( String(video) );
  cap.set( CAP_PROP_POS_FRAMES, start_frame );

  if( !cap.isOpened() ){
    printf("cannot open video %s\n",video);
    help();
  }

  Mat frame;
  namedWindow( "Tracking API", 1 );

  //instantiates the specific Tracker
  std::vector<Ptr<Tracker> >trackers(algnum);
  for(int i=0;i<algnum;i++){
      trackers[i] = Tracker::create( algorithms[i] );
      if( trackers[i] == NULL ){
        printf("error in the instantiation of the tracker %s\n",algorithms[i]);
        if(gt!=NULL){
            fclose(gt);
        }
        exit(EXIT_FAILURE);
      }
  }

  //get the first frame
  cap >> frame;
  frame.copyTo( image );
  if(lineToRect(buf,boundingBox)<0){
      if(gt!=NULL){
          fclose(gt);
      }
      exit(EXIT_FAILURE);
  }
  rectangle( image, boundingBox,palette[0], 2, 1 );
  for(int i=0;i<trackers.size();i++){
      rectangle(image,initBoxes[i],palette[i+1], 2, 1 );
      if( !trackers[i]->init( frame, initBoxes[i] ) ){
        printf("could not initialize tracker %s with box %s at video %s\n",algorithms[i],initBoxes_str[i],video);
        if(gt!=NULL){
            fclose(gt);
        }
        exit(EXIT_FAILURE);
      }
  }
  imshow( "Tracking API", image );

  int frameCounter = 0;
  std::vector<int> correctFrames(trackers.size(),0);

  for ( ;; ){
    if( !paused ){
      cap >> frame;
      if(frame.empty()){
        break;
      }
      frame.copyTo( image );

      if(fgets(buf,sizeof(buf),gt)==NULL){
          printf("ground truth is over\n");
          break;
      }
      if(lineToRect(buf,boundingBox)<0){
          if(gt!=NULL){
              fclose(gt);
          }
          exit(EXIT_FAILURE);
      }
      rectangle( image, boundingBox,palette[0], 2, 1 );
      
      frameCounter++;
      bool allTrackersOut=true;
      for(int i=0;i<trackers.size();i++){
          if( correctFrames[i]!=0){
              continue;
          }
          if(trackers[i]->update( frame, initBoxes[i] ) && (overlap(initBoxes[i],boundingBox)>=0.5) ){
            allTrackersOut=false;
            rectangle( image, initBoxes[i], palette[i+1], 2, 1 );
          }else{
              correctFrames[i]=frameCounter;
          }
      }
      if(allTrackersOut){
          break;
      }
      imshow( "Tracking API", image );

      char c = (char) waitKey( 2 );
      if( c == 'q' )
        break;
      if( c == 'p' )
        paused = !paused;
      }
  }
  if(gt!=NULL){
      fclose(gt);
  }
  destroyWindow( "Tracking API");
  assessmentRes res;
  res.videoName=video;
  res.correctFrames=correctFrames;
  res.len=linecount;
  return res;
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
  printf("algorithms and boxes (%d)\n",acount);
  for(int i=0;i<acount;i++){
      printf("%s ",algorithms[i]);
      for(int j=0;j<vcount;j++){
        printf("%s ",initBoxes[i][j]);
      }
      printf("\n");
  }

  std::vector<assessmentRes> results;
  for(int i=0;i<vcount;i++){
      results.push_back(assessment(videos[i],gts[i],algorithms,((char**)initBoxes)+i,acount));
  }
  printf("\n\n");
  for(int i=0;i<vcount;i++){
      printf("%20s",results[i].videoName);
      printf("%5d",results[i].len);
      for(int j=0;j<results[i].correctFrames.size();j++){
          printf("%5d",results[i].correctFrames[j]);
      }
      printf("\n");
  }
  return 0;
}
