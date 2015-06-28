//============================================================================
// Name        : retinademo.cpp
// Author      : Alexandre Benoit, benoit.alexandre.vision@gmail.com
// Version     : 0.1
// Copyright   : LISTIC/GIPSA French Labs, May 2015
// Description : Gipsa/LISTIC Labs quick retina demo in C++, Ansi-style
//============================================================================

// include bioinspired module and OpenCV core utilities
#include "opencv2/bioinspired.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <cstring>

// main function
int main(int argc, char* argv[]) {

  // basic input arguments checking
  if (argc>1)
  {
        std::cout<<"****************************************************"<<std::endl;
	std::cout<<"* Retina demonstration : demonstrates the use of is a wrapper class of the Gipsa/Listic Labs retina model."<<std::endl;
	std::cout<<"* This retina model allows spatio-temporal image processing (applied on a webcam sequences)."<<std::endl;
	std::cout<<"* As a summary, these are the retina model properties:"<<std::endl;
	std::cout<<"* => It applies a spectral whithening (mid-frequency details enhancement)"<<std::endl;
	std::cout<<"* => high frequency spatio-temporal noise reduction"<<std::endl;
	std::cout<<"* => low frequency luminance to be reduced (luminance range compression)"<<std::endl;
	std::cout<<"* => local logarithmic luminance compression allows details to be enhanced in low light conditions\n"<<std::endl;
	std::cout<<"* for more information, reer to the following papers :"<<std::endl;
	std::cout<<"* Benoit A., Caplier A., Durette B., Herault, J., \"USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING\", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011"<<std::endl;
	std::cout<<"* Vision: Images, Signals and Neural Networks: Models of Neural Processing in Visual Perception (Progress in Neural Processing),By: Jeanny Herault, ISBN: 9814273686. WAPI (Tower ID): 113266891."<<std::endl;
	std::cout<<"* => reports comments/remarks at benoit.alexandre.vision@gmail.com"<<std::endl;
	std::cout<<"* => more informations and papers at : http://sites.google.com/site/benoitalexandrevision/"<<std::endl;
	std::cout<<"****************************************************"<<std::endl;
	std::cout<<" NOTE : this program generates the default retina parameters file 'RetinaDefaultParameters.xml'"<<std::endl;
	std::cout<<" => you can use this to fine tune parameters and load them if you save to file 'RetinaSpecificParameters.xml'"<<std::endl;
       
        if (strcmp(argv[1],  "help")==0){
	    std::cout<<"No help provided for now, please test the retina Demo for a more complete program"<<std::endl;
        }
  }

  std::string inputMediaType=argv[1];
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
