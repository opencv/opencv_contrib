// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

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
		if((n++) >= 400) break;
		string f_name(dirp->d_name);
		images.push_back(imread("ALLSTIMULI/" + f_name));
		fixs.push_back(imread("ALLFIXATIONMAPS/" + f_name.substr(0, f_name.find_first_of('.')) + "_fixPts.jpg", 0));
	}

	g.training(images, fixs);
	ofstream file;
	Mat res2;
	g.computeSaliency(imread("ALLSTIMULI/i05june05_static_street_boston_p1010764.jpeg"), res2);
	log(res2, res2);
	resize(res2, res2, Size(1024, 768));
	cout << "AUC = " << g.computeAUC(res2, imread("ALLSTIMULI/i05june05_static_street_boston_p1010764_fixPts.jpeg", 0)) << endl;;
	g.saliencyMapVisualize(res2);
	file.open("saliency.csv");
	for (int i = 0;i < res2.rows;i++)
	{
		for (int j=0;j < res2.cols;j++)
		{
			file << res2.at<double>(i, j) << " ";
		}
		file << endl;
	}
	file.close();
    return 0;
} //main
