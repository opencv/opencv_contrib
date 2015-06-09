#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>


using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
  
  if(argc<2) {
    std::cerr << "Use: marker_detector video" << std::endl;
    return 0;
  }

  cv::VideoCapture input;
  input.open(argv[1]);
    
  while( input.grab() ) {
    cv::Mat image,imageCopy;
    input.retrieve(image);

    std::vector< int > ids;
    std::vector< std::vector<cv::Point2f> > imgPoints;
    std::vector< std::vector<cv::Point2f> > rejectedImgPoints;
    
    // detect markers
    cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, imgPoints, ids, rejectedImgPoints);   
   
    // draw results
    if(ids.size()>0) cv::aruco::drawDetectedMarkers(image, imageCopy, imgPoints, ids);
    else image.copyTo(imageCopy);
    if(rejectedImgPoints.size()>0) cv::aruco::drawDetectedMarkers(imageCopy, imageCopy, rejectedImgPoints,cv::noArray(), cv::Scalar(100,0,255));  
    
    cv::imshow("out", imageCopy);
    char key = cv::waitKey(0);
    if(key == 27) break;
  }
  
  return 0;
  
}




