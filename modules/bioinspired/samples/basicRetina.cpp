// include bioinspired module and OpenCV core utilities
#include "opencv2/bioinspired.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

// main function
int main(int argc, char* argv[]) {
  // declare the retina input buffer.
  cv::Mat inputFrame;
  // setup webcam reader and grab a first frame to get its size
  cv::VideoCapture videoCapture(0); 
  videoCapture>>inputFrame;

  // allocate a retina instance with input size equal to the one of the loaded image
  cv::Ptr<cv::bioinspired::Retina> myRetina = cv::bioinspired::createRetina(inputFrame.size());

  /* retina parameters management methods use sample
     -> save current (here default) retina parameters to a xml file (you may use it only one time to get the file and modify it)
  */
  myRetina->write("RetinaDefaultParameters.xml");

  // -> load parameters if file exists
  myRetina->setup("RetinaSpecificParameters.xml");

  // reset all retina buffers (open your eyes)  
  myRetina->clearBuffers();

  // declare retina output buffers
  cv::Mat retinaOutput_parvo;
  cv::Mat retinaOutput_magno;

  //main processing loop
  bool stillProcess=true;
  while(stillProcess){
	
	// if using video stream, then, grabbing a new frame, else, input remains the same
	if (videoCapture.isOpened())
		videoCapture>>inputFrame;
        else
		stillProcess=false;
	// run retina filter
	myRetina->run(inputFrame);
	// Retrieve and display retina output
	myRetina->getParvo(retinaOutput_parvo);
	myRetina->getMagno(retinaOutput_magno);
	cv::imshow("retina input", inputFrame);
	cv::imshow("Retina Parvo", retinaOutput_parvo);
	cv::imshow("Retina Magno", retinaOutput_magno);

	cv::waitKey(5);
  }

}
