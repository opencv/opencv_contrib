/*
 * DeepGaze1.hpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */

#ifndef _OPENCV_DEEPGAZE1_HPP_
#define _OPENCV_DEEPGAZE1_HPP_

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class DeepGaze1
{
private:
	Net net;
	vector<string> layers_names;
	vector<Mat> training_images;
	vector<double> weights;
public:
	DeepGaze1();
	DeepGaze1(string, string, vector<string>, vector<Mat>);
	Mat saliencyMapGenerator(string);
	void training(vector<Mat>);
};




#endif /* _OPENCV_DEEPGAZE1_HPP_ */
