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
#include "saliencyBaseClasses.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>


using namespace cv;
using namespace cv::dnn;
using namespace std;


class DeepGaze1
{
private:
	dnn::Net net;
	std::vector<std::string> layers_names;
	std::vector<double> weights;

public:
	DeepGaze1();
	DeepGaze1(std::string, std::string, std::vector<std::string>);
	bool computeSaliency(InputArray image, OutputArray saliencyMap)
	{
		if(image.empty())
			return false;
		return computeSaliencyImpl(image, saliencyMap);
	}
	void training(std::vector<Mat>&, std::vector<Mat>&);
protected:
    bool computeSaliencyImpl(InputArray image, OutputArray saliencyMap);
    std::vector<Mat> featureMapGenerator(Mat);
    static Mat comb(std::vector<Mat>&, std::vector<double>);
    static Mat softmax(Mat);
    Mat saliencyMapGenerator(Mat);
    static std::vector<double> evalGrad(std::vector<Mat>&, std::vector<unsigned>&, std::vector<double>);
    std::vector<unsigned> batchIndex(unsigned, unsigned);
    static double loss(std::vector<double>, std::vector<double>);
    static std::vector<double> mapSampler(Mat, std::vector<unsigned>);
    std::vector<unsigned> fixationLoc(Mat);
};




#endif /* _OPENCV_DEEPGAZE1_HPP_ */
