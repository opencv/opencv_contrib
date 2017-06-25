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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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
