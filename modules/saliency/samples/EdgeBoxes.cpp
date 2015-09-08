/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace saliency;

static const char* keys =
{ "{@saliency_algorithm | | Saliency algorithm <saliencyAlgorithmType.[saliencyAlgorithmTypeSubType]> }"
    "{@video_name      | | video name            }"
    "{@start_frame     |1| Start frame           }"
    "{@training_path   |1| Path of the folder containing the trained files}" };

static void help()
{
  cout << "\nThis example shows the functionality of \"Saliency \""
       "Call:\n"
       "./example_saliency_computeSaliency <saliencyAlgorithmSubType> <video_name> <start_frame> \n"
       << endl;
}

int main( int argc, char** argv )
{
	std::cout << "Test EdgeBoxes OpenCV algorithm" << std::endl;
	

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("Edge", WINDOW_AUTOSIZE);// Create a window for display.
	namedWindow("Orientation", WINDOW_AUTOSIZE);// Create a window for display.


	std::string saliency_algorithm = "EdgeBoxes";
  
	vector<Vec4i> saliencyMap;

	Ptr<Saliency> saliencyAlgorithm = Saliency::create(saliency_algorithm);

	Mat edgeImage;
	edgeImage = imread("C:/Users/hisham/Dropbox/trud/edges-master/edge_circle.png", 0);
	edgeImage.convertTo(edgeImage, CV_64F,1/255.0f);
	Mat orientationImage;
	orientationImage = imread("C:/Users/hisham/Dropbox/trud/edges-master/orientation_circle.png", 0);
	orientationImage.convertTo(orientationImage, CV_64F, 1 / 255.0f);
	
	

	
	Mat result= Mat(edgeImage.rows, edgeImage.cols, CV_64F, double(0));
	ObjectnessEdgeBoxes oeb;


	//oeb.computeSaliencyDiagnostic(edgeImage, orientationImage, result);
	oeb.computeSaliencyMap(edgeImage, orientationImage, result);


	imshow("Edge", edgeImage);
	//waitKey(0);

	imshow("Orientation", orientationImage);
	waitKey(10);

	double result_min, result_max;
	minMaxLoc(result, &result_min, &result_max);
	result.convertTo(result, CV_64F, 1 / result_max);

	imshow("Display window", result);                   // Show our image inside it.
	waitKey(0);
	



	//saliencyAlgorithm.dynamicCast<ObjectnessEdgeBoxes>()->computeSaliencyMap(edgeImage, orientationImage, result);

	//saliencyAlgorithm.dynamicCast<ObjectnessBING>()->computeSaliency(image, saliencyMap);

	//saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setTrainingPath(training_path);
	//saliencyAlgorithm.dynamicCast<ObjectnessBING>()->setBBResDir(training_path + "/Results");

	/*
	if (saliencyAlgorithm->computeSaliency(image, saliencyMap))
	{
		std::cout << "Objectness done" << std::endl;
	}
	*/
	getchar();


  return 0;
}
