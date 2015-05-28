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
    cv::Mat image;
    input.retrieve(image);

    std::vector< int > ids;
    std::vector< std::vector<cv::Point2f> > imgPoints;
    
    // detect markers
    cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, imgPoints, ids);   
    
    // draw results
    cv::aruco::drawDetectedMarkers(image, imgPoints, ids);  
    
    cv::imshow("out", image);
    char key = cv::waitKey(0);
    if(key == 27) break;
  }
  
  return 0;
  
}




