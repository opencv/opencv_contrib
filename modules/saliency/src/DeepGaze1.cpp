/*
 * DeepGaze1.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */
#include "precomp.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <numeric>

using namespace std;
using namespace cv;
using namespace cv::dnn;

DeepGaze1::DeepGaze1()
{
	net = dnn::readNetFromCaffe("deploy.prototxt", "bvlc_alexnet.caffemodel");
	layers_names = vector<string>{"conv1", "relu1", "pool1", "norm1", "conv2", "relu2", "pool2", "norm2", "conv3", "relu3", "conv4", "relu4", "conv5", "relu5", "pool5"};
	training_images = vector<Mat>();
	weights = vector<double>(3713, 1.0);
	for(double& i : weights)
	{
		i = (rand() % 10 / 10.0);
	}
}

DeepGaze1::DeepGaze1(string net_proto, string net_caffemodel, vector<string> selected_layers, vector<Mat> training_set)
{
	net = dnn::readNetFromCaffe(net_proto, net_caffemodel);
	layers_names = selected_layers;
	training_images = training_set;
	weights = vector<double>(layers_names.size(), 0);
	for(double& i : weights)
	{
		i = (double)(rand() % 10) / 10.0;
	}
}

vector<Mat> DeepGaze1::featureMapGenerator(Mat img)
{
	//Mat img = imread(input_image);
	vector<Mat> featureMaps;
	Mat me(227, 227, CV_64F, Scalar::all(0.0));//hard coded currently, change later
	Mat std(227, 227, CV_64F, Scalar::all(0.0));
	Mat temp(227, 227, CV_64F, Scalar::all(0.0));


	resize(img, img, Size(227, 227), 0, 0, INTER_AREA);//hard coded
	Mat inputBlob = blobFromImage(img);   //Convert Mat to batch of images
	net.setBlob(".data", inputBlob);        //set the network input
	net.forward();                          //compute output
	for(unsigned i = 0; i < layers_names.size(); i++)
	{
		Mat blob_set = net.getBlob(layers_names[i]);
		for(int j = 0; j < blob_set.size[1]; j++)
		{
			Mat m, s, blob = Mat(blob_set.size[2], blob_set.size[3], CV_64F, blob_set.ptr(0, j)).clone();
			resize(blob, blob, Size(227, 227), 0, 0, INTER_AREA);
			featureMaps.push_back(blob);
		}
	}
	for (unsigned i = 0; i < featureMaps.size(); i++)
	{
		me += (featureMaps[i] / (double)featureMaps.size());
	}
	for (unsigned i = 0; i < featureMaps.size(); i++)
	{
		pow(featureMaps[i] - me, 2, temp);
		std += (temp / (double)featureMaps.size());
	}
	pow(std, 0.5, std);
	for (unsigned i = 0; i < featureMaps.size(); i++)
	{
		featureMaps[i] -= me;
		divide(featureMaps[i], std, featureMaps[i]);
	}
	/*
	for(unsigned i = 0;i < featureMaps.size();i++)
	{
		Scalar me, std;
		meanStdDev(featureMaps[i], me, std);
		featureMaps[i] -= me.val[0];
		if(std.val[0] != 0) featureMaps[i] /= std.val[0];
	}*/
	Mat centerBias(227, 227, CV_64F, Scalar::all(0.0));
	for(int i = 0;i < centerBias.rows;i++)
	{
		for(int j = 0;j < centerBias.cols;j++)
		{
			double temp = max(abs(i - 227/2), abs(i - 227/2)) * 1.414;
			centerBias.at<double>(i, j) = 0.01 * exp(-0.0001 * temp * temp);
			//double ratio = abs(i - centerBias.rows / 2.0) * 2 * abs(j - centerBias.cols / 2.0) * 2 / centerBias.rows / centerBias.cols;
			//centerBias.at<double>(i, j) = 0.01 * exp(-1.2 * ratio);
		}
	}
	featureMaps.push_back(centerBias);
	return featureMaps;
}

bool computeSaliencyImpl(InputArray image, OutputArray saleicnyMap)
{
	vector<Mat> featureMaps = featureMapGenerator(image);
	saliencyMap = softmax(comb(featureMaps, weights));
	return true;
}

Mat DeepGaze1::saliencyMapGenerator(Mat input_image)
{
//raw saliency map generate
	vector<Mat> featureMaps = featureMapGenerator(input_image);
	return softmax(comb(featureMaps, weights));
}

Mat DeepGaze1::comb(vector<Mat>& featureMaps, vector<double> wei)
{
	Mat res(227, 227, CV_64F, Scalar::all(0.0));

	for(unsigned i = 0; i < featureMaps.size(); i++)
	{
		Mat temp = featureMaps[i].clone();
		temp *= wei[i];
		res += temp;
	}
	//filter2D(res, res, -1, getGaussianKernel(30, 1), Point(-1, -1), 0, BORDER_DEFAULT);
	GaussianBlur(res, res, Size(51, 51), 0, 0);
	return res;
}

Mat DeepGaze1::softmax(Mat res)
{
	exp(res, res);
	res = res / sum(res).val[0];

	return res;
}

vector<unsigned> DeepGaze1::batchIndex(unsigned total, unsigned batchSize)
{
	srand(0);
	vector<unsigned> allIndex(total, 0);

	for(unsigned i = 0; i < total; i++)
	{
		allIndex[i] = i;
	}
	for(int i = batchSize - 1; i >= 0; i--)
	{
		swap(allIndex[i], allIndex[rand() % total]);
	}
	return vector<unsigned>(allIndex.begin(), allIndex.begin() + batchSize);
}



vector<unsigned> DeepGaze1::fixationLoc(Mat img)
{
	vector<unsigned> randIndex;
	//Mat img = imread("ALLFIXATIONMAPS/i05june05_static_street_boston_p1010764_fixPts.jpg", 0);
	resize(img, img, Size(227, 227), 0, 0, INTER_AREA);
	vector<unsigned> fixation;
	vector<pair<unsigned, unsigned> > match;
	fixation.assign(img.datastart, img.dataend);
	for(unsigned i = 0; i < fixation.size(); i++)
	{
		match.push_back({fixation[i], i});
	}
	sort(match.begin(), match.end(), [](pair<unsigned, unsigned> a, pair<unsigned, unsigned> b) {
		return b.first < a.first;
	});
	for(unsigned i = 0 ; i <= match.size() * 0.2 && match[i].first > 0 ; i++)
	{
		randIndex.push_back(match[i].second);
	}
	img.release();
	return randIndex;
}

double DeepGaze1::loss(vector<double> saliency_sample, vector<double> wei)
{
	double res = 0, l1 = 0, l2 = 0;

	for(double i : wei)
	{
		l1 += abs(i);
	}
	for(double i : wei)
	{
		l2 += i * i;
	}
	for(unsigned i = 0; i < saliency_sample.size(); i++)
	{
		res -= log(saliency_sample[i]) / saliency_sample.size();
	}
	return res + 0.001 * l1 / l2;
}

vector<double> DeepGaze1::mapSampler(Mat saliency, vector<unsigned> randIndex)
{
	vector<double> saliency_sample;
	for(unsigned i = 0 ; i < randIndex.size() ; i++)
	{
		saliency_sample.push_back(saliency.at<double>(randIndex[i] / saliency.size[1], randIndex[i] % saliency.size[1]));
	}
	return saliency_sample;
}

vector<double> DeepGaze1::evalGrad(vector<Mat>& featureMaps, vector<unsigned> randIndex)
{
	vector<double> grad(featureMaps.size(), 0);

	Mat c = comb(featureMaps, weights);
	Mat saliency_old = softmax(c.clone());
	vector<double> tt_old = mapSampler(saliency_old, randIndex);
	double loss_old = 0;
	loss_old = loss(tt_old, weights);
	for(unsigned i = 0; i < weights.size(); i++)
	{
		Mat saliency_new = c.clone();
		Mat temp(227, 227, CV_64F, Scalar::all(0.0));
		temp += 0.0001 * featureMaps[i];
		saliency_new += temp;
		//filter2D(saliency_new, saliency_new, -1, getGaussianKernel(30, 1), Point(-1, -1), 0, BORDER_DEFAULT);
		GaussianBlur(saliency_new, saliency_new, Size(51, 51), 0, 0);
		saliency_new = softmax(saliency_new);
		vector<double> weights_new(weights);
		weights_new[i] += 0.0001;
		double loss_new = 0;
		vector<double> tt = mapSampler(saliency_new, randIndex);
		loss_new = loss(tt, weights_new);
		grad[i] = (loss_new - loss_old) / 0.0001;
	}
	return grad;
}

void DeepGaze1::training(vector<Mat>& images, vector<Mat>& fixMaps)
{
  vector<unsigned> randIndex = batchIndex(images.size(), images.size());
  vector<vector<unsigned> > fixLoc_list;
  vector<vector<Mat> > featureMaps_list;
  vector<double> grad;
  vector<double> vel(weights.size(), 0);


  for(unsigned i : randIndex)
  {
    vector<Mat> featureMaps = featureMapGenerator(images[i]);
    vector<unsigned> fixLoc = fixationLoc(fixMaps[i]);
    grad = evalGrad(featureMaps, fixLoc);
    for(unsigned j = 0; j < grad.size(); j++)
    {
    	vel[j] = 0.9 * vel[j] + grad[j];
    	weights[j] -= 0.001 * vel[j] * exp(-0.001 * i);
    }
    double avgGrad = accumulate(grad.begin(), grad.end(), 0.0) / weights.size();
    double avgWeight = accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
    if (abs(avgGrad) <= 0.01 || isnan(avgGrad)) break;
    cout << i << " " << avgGrad << " " << avgWeight << endl;
  }

}
