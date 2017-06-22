/*
 * main.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/saliency.hpp>
#include <vector>
#include <string>
#include <dirent.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;
/* Find best class for the blob (i. e. class with maximal probability) */

int main(int argc, char **argv)
{
	DeepGaze1 g = DeepGaze1();
	vector<Mat> images;
	vector<Mat> fixs;

//Download mit1003 saliency dataset in the working directory
//ALLSTIMULI folder store images
//ALLFIXATIONMAPS foler store training eye fixation
	DIR *dp;
	struct dirent *dirp;
	if((dp = opendir("ALLSTIMULI")) == NULL)
	{
		cout << "Can not open dataset directory" << endl;
	}
	unsigned n = 0;
	while((dirp = readdir(dp)) != NULL)
	{
		if(dirp->d_type != 8) continue;
		if((n++) >= 100) break;
		string f_name(dirp->d_name);
		images.push_back(imread("ALLSTIMULI/" + f_name));
		fixs.push_back(imread("ALLFIXATIONMAPS/" + f_name.substr(0, f_name.find_first_of('.')) + "_fixPts.jpg", 0));
	}

	g.training(images, fixs);
	Mat res(g.saliencyMapGenerator(imread("ALLSTIMULI/i05june05_static_street_boston_p1010764.jpeg")));
    return 0;
} //main
