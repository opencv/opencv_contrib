/*
 * main.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */

#include "DeepGaze1.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::dnn;
/* Find best class for the blob (i. e. class with maximal probability) */

int main(int argc, char **argv)
{
	DeepGaze1 g = DeepGaze1();
	Mat res(g.saliencyMapGenerator(string("timg.jpeg")));
	Mat t = getGaussianKernel(2, 1);
	exp(t, t);
	t /= sum(t).val[0];
	imshow("image", res);
	waitKey(0);
    return 0;
} //main
