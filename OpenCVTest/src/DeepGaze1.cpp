/*
 * DeepGaze1.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */
#include "DeepGaze1.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;

DeepGaze1::DeepGaze1()
{
	net = dnn::readNetFromCaffe("deploy.prototxt", "bvlc_alexnet.caffemodel");
	layers_names = vector<string>{"conv1", "relu1", "pool1", "norm1", "conv2", "relu2", "pool2", "norm2", "conv3", "relu3", "conv4", "relu4", "conv5", "relu5", "pool5"};
	training_images = vector<Mat>();
	weights = vector<double>(3712, 1.0/3712);
}

DeepGaze1::DeepGaze1(string net_proto, string net_caffemodel, vector<string> selected_layers, vector<Mat> training_set)
{
	net = dnn::readNetFromCaffe(net_proto, net_caffemodel);
	layers_names = selected_layers;
	training_images = training_set;
	weights = vector<double>(layers_names.size(), 1.0);
}

Mat DeepGaze1::saliencyMapGenerator(string input_image)
{
//raw saliency map generate
	Mat img = imread(input_image);
	Mat res(227, 227, CV_64F, Scalar::all(0.0));//hard coded currently, change later

	resize(img, img, Size(227, 227), 0, 0, INTER_AREA);//hard coded
	Mat inputBlob = blobFromImage(img);   //Convert Mat to batch of images
	net.setBlob(".data", inputBlob);        //set the network input
	net.forward();                          //compute output
	for(unsigned i = 0; i < layers_names.size(); i++)
	{
		Mat blob_set = net.getBlob(layers_names[i]);

		for(int j = 0; j < blob_set.size[1]; j++)
		{
			Mat blob = Mat(blob_set.size[2], blob_set.size[3], CV_64F, blob_set.ptr(0, j)).clone();
			resize(blob, blob, Size(227, 227), 0, 0, INTER_AREA);
			blob *= weights[i * layers_names.size() + j];
			filter2D(blob, blob, -1 , getGaussianKernel(2, 1), Point( -1, -1 ), 0, BORDER_DEFAULT );
			res += blob;
		}
	}

	cout << res << endl;
	waitKey(0);
	exp(res, res);
	res = res / sum(res).val[0];
	return res;
}



