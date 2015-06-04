#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <vector>
#include <iostream>


using namespace std;
using namespace cv;


int main(int argc, char* argv[])
{
  
  if(argc<2) {
    std::cerr << "Use: board_detector video" << std::endl;
    return 0;
  }

  cv::VideoCapture input;
  input.open(argv[1]);
  
  // Parameters for the Aruco test videos
  cv::Mat camMatrix(3,3,CV_64FC1,cv::Scalar::all(0));
  camMatrix.at<double>(0,0) = 628.158;
  camMatrix.at<double>(0,2) = 324.099;
  camMatrix.at<double>(1,1) = 628.156;
  camMatrix.at<double>(1,2) = 260.908;
  camMatrix.at<double>(2,2) = 1.;
  
  cv::Mat distCoeffs(5,1,CV_64FC1, cv::Scalar::all(0));
  distCoeffs.ptr<double>(0)[0] = 0.0995485;
  distCoeffs.ptr<double>(0)[1] = -0.206384;
  distCoeffs.ptr<double>(0)[2] = 0.00754589;
  distCoeffs.ptr<double>(0)[3] = 0.00336531;
  
  cv::aruco::Board b =  cv::aruco::Board::createPlanarBoard(4, 6, 0.04, 0.008);
  b.ids.clear();
  b.ids.push_back(985);
  b.ids.push_back(838);
  b.ids.push_back(908);
  b.ids.push_back(299);
  b.ids.push_back(428);
  b.ids.push_back(177);
  
  b.ids.push_back(64);
  b.ids.push_back(341);
  b.ids.push_back(760);
  b.ids.push_back(882);
  b.ids.push_back(982);
  b.ids.push_back(977);
  
  b.ids.push_back(477);
  b.ids.push_back(125);
  b.ids.push_back(717);
  b.ids.push_back(791);
  b.ids.push_back(618);
  b.ids.push_back(76);
  
  b.ids.push_back(181);
  b.ids.push_back(1005);
  b.ids.push_back(175);
  b.ids.push_back(684);
  b.ids.push_back(233);
  b.ids.push_back(461);  

  
  while( input.grab() ) {
    cv::Mat image,imageCopy;
    input.retrieve(image);

    std::vector< int > ids;
    std::vector< std::vector<cv::Point2f> > imgPoints;
    cv::Mat rvec, tvec;
    
    // detect markers and estimate pose
    cv::aruco::detectMarkers(image, cv::aruco::DICT_ARUCO, imgPoints, ids);
    if(ids.size()>0) cv::aruco::estimatePoseBoard(imgPoints, ids, b, camMatrix, distCoeffs, rvec, tvec);
    
    // draw results
    if(ids.size()>0) {
        cv::aruco::drawDetectedMarkers(image, imageCopy, imgPoints, ids);
        cv::aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, 0.1);    
    }
    
    cv::imshow("out", imageCopy);
    char key = cv::waitKey(0);
    if(key == 27) break;
  }
  
  return 0;
  
}




