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

void thresholding(Mat img)
{
	Mat res(227, 227, CV_8U, Scalar::all(0));
	vector<double> fixation;
	vector<pair<double, unsigned> > match;
	for(int i = 0;i < img.rows;i++)
	{
		for(int j = 0;j < img.cols;j++)
		{
			fixation.push_back((double)img.at<double>(i, j));
		}
	}
	for(unsigned i = 0; i < fixation.size(); i++)
	{
		match.push_back({fixation[i], i});
	}
	sort(match.begin(), match.end(), [](pair<double, unsigned> a, pair<double, unsigned> b) {
		return b.first < a.first;
	});
	for(unsigned i = 0 ; i <= match.size()*0.01; i++)
	{
		if(i <= match.size() * 0.001)
			res.at<uchar>(match[i].second / 227, match[i].second % 227) = 255;
		else if (i <= match.size() * 0.01)
			res.at<uchar>(match[i].second / 227, match[i].second % 227) = 150;
		else if (i <= match.size() * 0.1)
			res.at<uchar>(match[i].second / 227, match[i].second % 227) = 75;
		else if (i <= match.size() * 0.4)
			res.at<uchar>(match[i].second / 227, match[i].second % 227) = 25;
	}
	resize(res, res, Size(1024, 768), 0, 0, INTER_AREA);
	log(img, img);
	ofstream file("sample.csv");
	for (int i = 0;i < img.rows;i++)
	{
		for (int j=0;j < img.cols;j++)
		{
			file << img.at<double>(i, j) << " ";
		}
		file << endl;
	}
	file.close();
	imshow("img2", res);
	waitKey(0);
}

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
	thresholding(res);
    return 0;
} //main
