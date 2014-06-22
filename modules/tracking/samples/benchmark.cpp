#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <time.h>
#include <cstring>
#include <climits>

#define CMDLINEMAX 10
#define ASSESS_TILL INT_MAX

using namespace std;
using namespace cv;

/* FIXME:
 *          strange symbols --> PRF gives 1 for MedianFlow --> FIXME 0.0<>0.5 -->  logs and videos (1,5,6) --> report --> payo --> fek
 * TODO:  
 *          have FUN! with 5 frames and vs MEDIANFLOW
 *          PRF
 *          have FUN!
 *
            do normalization ala Kalal's assessment protocol for TLD
 */

static Mat image;
static bool paused;
vector<Scalar> palette;

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
  char * ptr=line,*pos=ptr;
  if(line==NULL || line[0]=='\0'){
      return -1;
  }
  if(strcmp(line,"NaN,NaN,NaN,NaN\n")==0){
      res.height=res.width=-1.0;
      return 0;
  }

  double nums[4]={0};
  for(int i=0; i<4 && (ptr=strpbrk(ptr,"0123456789-"))!= NULL;i++,ptr=pos){
    nums[i]=strtod(ptr,&pos);
    if(pos==ptr){
      printf("lineToRect had problems with decoding line %s\n",line);
      return -1;
    }
  }
  res.x=cv::min(nums[0],nums[2]);
  res.y=cv::min(nums[1],nums[3]);
  res.width=cv::abs(nums[0]-nums[2]);
  res.height=cv::abs(nums[1]-nums[3]);
  return 0;
}
static inline double overlap(Rect2d r1,Rect2d r2){
    if(r1.width<0 || r2.width<0 || r1.height<0 || r1.width<0)return -1.0;
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
void print_table(char* videos[],int videoNum,char* algorithms[],int algNum,const vector<vector<char*> >& results){
    vector<int> grid(1+algNum,0);
    char spaces[100];memset(spaces,' ',100);
    for(int i=0;i<videoNum;i++){
        grid[0]=std::max(grid[0],(int)strlen(videos[i]));
    }
    for(int i=0;i<algNum;i++){
        grid[i+1]=strlen(algorithms[i]);
        for(int j=0;j<videoNum;j++)
            grid[i+1]=std::max(grid[i+1],(int)strlen(results[j][i]));
    }
    printf("%.*s ",grid[0],spaces);
    for(int i=0;i<algNum;i++)
        printf("%s%.*s",algorithms[i],grid[i+1]+1-strlen(algorithms[i]),spaces);
    printf("\n");
    for(int i=0;i<videoNum;i++){
        printf("%s%.*s",videos[i],grid[0]+1-strlen(videos[i]),spaces);
        for(int j=0;j<algNum;j++)
            printf("%s%.*s",results[i][j],grid[j+1]+1-strlen(results[i][j]),spaces);
        printf("\n");
    }
    printf("*************************************************************\n");
}

struct AssessmentRes{
    class Assessment{
    public:
        virtual int printf(char* buf)=0;
        virtual void printName()=0;
        virtual void assess(const Rect2d& ethalon,const Rect2d& res)=0;
        virtual ~Assessment(){}
    };
    AssessmentRes(int algnum);
    char* videoName;
    vector<vector<Ptr<Assessment> > >results;
    int len;
};
class CorrectFrames : public AssessmentRes::Assessment{
public:
    CorrectFrames(double tol):tol_(tol),correctFrames_(1){}
    int printf(char* buf){return sprintf(buf,"%d",correctFrames_);}
    void printName(){printf("Num of correct frames\n");}
    void assess(const Rect2d& ethalon,const Rect2d& res){if(overlap(ethalon,res)>tol_)correctFrames_++;}
private:
    double tol_;
    int correctFrames_;
};
class AvgTime : public AssessmentRes::Assessment{
public:
    AvgTime(double res):res_(res){}
    int printf(char* buf){return sprintf(buf,"%gms",res_);}
    void printName(){printf("Average frame tracking time\n");}
    void assess(const Rect2d& ethalon,const Rect2d& res){};
private:
    double res_;
};
class PRF : public AssessmentRes::Assessment{
public:
    PRF():occurences_(0),responses_(0),true_responses_(0){};
    void printName(){printf("PRF\n");}
    int printf(char* buf){return sprintf(buf,"%g/%g/%g",(1.0*true_responses_)/responses_,(1.0*true_responses_)/occurences_,
            (2.0*true_responses_)/(responses_+occurences_));}
    void assess(const Rect2d& ethalon,const Rect2d& res){
        if(res.height>=0)responses_++;
        if(ethalon.height>=0)occurences_++;
        if(ethalon.height>=0 && res.height>=0)true_responses_++;
    }
private:
    int occurences_,responses_,true_responses_;
};
AssessmentRes::AssessmentRes(int algnum):len(0),results(algnum){
    for(int i=0;i<results.size();i++){
        if(!false){
            results[i].push_back(Ptr<Assessment>(new CorrectFrames(0.0)));
        }else{
            results[i].push_back(Ptr<Assessment>(new CorrectFrames(0.5)));
        }
        results[i].push_back(Ptr<Assessment>(new PRF()));
    }
}

static AssessmentRes assessment(char* video,char* gt_str, char* algorithms[],char* initBoxes_str[],int algnum){
  char buf[200];
  int start_frame=0;
  int linecount=0;
  Rect2d boundingBox;
  vector<double> averageMillisPerFrame(algnum,0.0);

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

  std::vector<Rect2d> initBoxes(algnum);
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
  AssessmentRes res(trackers.size());

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
      for(int i=0;i<trackers.size();i++){
          bool trackerRes=true;
          clock_t start;start=clock();
          trackerRes=trackers[i]->update( frame, initBoxes[i] );
          start=clock()-start;
          averageMillisPerFrame[i]+=1000.0*start/CLOCKS_PER_SEC;
          if(trackerRes==false){
              initBoxes[i].height=initBoxes[i].width=-1.0;
          }else{
              rectangle( image, initBoxes[i], palette[i+1], 2, 1 );
          }

          if(!true && i==1){
              printf("TLD\n");
              printf("boundingBox=[%f,%f,%f,%f]\n",boundingBox.x,boundingBox.y,boundingBox.width,boundingBox.height);
              printf("initBoxes[i]=[%f,%f,%f,%f]\n",initBoxes[i].x,initBoxes[i].y,initBoxes[i].width,initBoxes[i].height);
              printf("overlap=%f\n",overlap(initBoxes[i],boundingBox));
              exit(0);
          }

          for(int j=0;j<res.results[i].size();j++)
              res.results[i][j]->assess(boundingBox,initBoxes[i]);
      }
      imshow( "Tracking API", image );

      if(frameCounter>=ASSESS_TILL){
          break;
      }

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

  res.len=linecount;
  res.videoName=video;
  for(int i=0;i<res.results.size();i++)
      res.results[i].push_back(Ptr<AssessmentRes::Assessment>(new AvgTime(averageMillisPerFrame[i]/res.len)));
  return res;
}

int main( int argc, char** argv ){
  palette.push_back(Scalar(255,0,0));//BGR
  palette.push_back(Scalar(0,0,255));
  palette.push_back(Scalar(0,255,255));
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

  std::vector<AssessmentRes> results;
  for(int i=0;i<vcount;i++){
      results.push_back(assessment(videos[i],gts[i],algorithms,((char**)initBoxes)+i,acount));
  }
  printf("\n\n");

  char buf[CMDLINEMAX*CMDLINEMAX*40];
  vector<vector<char*> > resultStrings(vcount);
  for(int i=0;i<vcount;i++){
      for(int j=0;j<acount;j++){
          resultStrings[i].push_back(buf+i*CMDLINEMAX*40 + j*40);
      }
  }
  for(int tableCount=0;tableCount<results[0].results[0].size();tableCount++){
      for(int videoCount=0;videoCount<results.size();videoCount++)
          for(int algoCount=0;algoCount<results[0].results.size();algoCount++){
              (results[videoCount].results[algoCount][tableCount])->printf(resultStrings[videoCount][algoCount]);
          }
      print_table(videos,vcount,algorithms,acount,resultStrings);
  }
  return 0;
}
