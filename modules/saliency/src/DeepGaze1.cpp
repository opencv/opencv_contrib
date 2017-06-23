/*
 * DeepGaze1.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/saliency/DeepGaze1.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <numeric>
#include <fstream>


using namespace std;
using namespace cv;
using namespace cv::dnn;



DeepGaze1::DeepGaze1()
{
	net = dnn::readNetFromCaffe("deploy.prototxt", "bvlc_alexnet.caffemodel");
	layers_names = vector<string>{"conv5"};
	weights = {0.497487,0.502411,0.456505,0.425814,0.497493,0.512262,0.516318,0.44849,0.424749,0.472097,0.494168,0.488127,0.45547,0.476046,0.44338,0.50457,0.55326,0.534006,0.423589,0.498399,0.517197,0.533391,0.518,0.491569,0.491973,0.468363,0.490844,0.387676,0.56501,0.486951,0.498872,0.476382,0.432565,0.492387,0.572339,0.547571,0.481624,0.471914,0.412187,0.457617,0.547966,0.501938,0.517393,0.523375,0.522948,0.486832,0.507451,0.516924,0.424877,0.510275,0.495443,0.472026,0.125719,0.503043,0.506072,0.440315,0.495942,0.477689,0.466947,0.458948,0.346367,0.527349,0.587935,0.58706,0.494051,0.474483,0.539967,0.532128,0.443884,0.462236,0.494746,0.487751,0.488264,0.470903,0.477316,0.498729,0.482376,0.421621,0.439487,0.505218,0.483866,0.471529,0.480059,0.476024,0.496628,0.510684,0.485402,0.521804,0.480137,0.407191,0.231358,0.0521284,0.614009,0.440855,0.47279,0.250036,0.53501,0.413822,0.444445,0.462979,0.518892,0.585769,0.474131,0.508421,0.509128,0.540619,0.412131,0.439915,0.492447,0.522151,0.474511,0.495,0.430954,0.535171,0.518931,0.544447,0.524002,0.421077,0.517055,0.447245,0.251134,0.511714,0.153483,0.490056,0.453425,0.450157,0.442129,0.436315,0.555194,0.46594,0.422647,0.456453,0.472298,0.532641,0.523724,0.471044,0.501304,0.306234,0.485647,0.516084,0.529696,0.318047,0.482182,0.459202,0.524382,0.461817,0.460216,0.518925,0.565913,0.46236,0.397505,0.536967,0.477518,0.526041,0.484764,0.418136,0.546664,0.332771,0.432856,0.474645,0.534784,0.503048,0.460163,0.433259,0.427519,0.291319,0.36069,0.407249,0.48776,0.434639,0.492274,0.471517,0.436101,0.551626,0.41872,0.424817,0.533052,0.457066,0.508638,0.488303,0.501407,0.503762,0.521199,0.500515,0.510584,0.50581,0.466941,0.51722,0.502145,0.447454,0.430834,0.451845,0.541774,0.505495,0.501585,0.434551,0.536647,0.408516,0.47475,0.576825,0.515408,0.278159,0.527432,0.507564,0.461757,0.468113,0.463559,0.511806,0.379718,0.346922,0.395977,0.489966,0.513447,0.450056,0.511192,0.486209,0.481624,0.425786,0.48788,0.514394,0.457541,0.430562,0.438185,0.417461,0.501014,0.49015,0.541213,0.47548,0.471689,0.36902,0.566105,0.573095,0.474797,0.546882,0.492789,0.505544,0.497233,0.418915,0.481781,0.103597,0.474253,0.478368,0.511184,0.503792,0.497224,0.493456,0.450304,0.44323,0.477985,0.508655,0.425664,0.210711,0.541917,0.505914,0.476779,0.417452,2.1771};
}

DeepGaze1::DeepGaze1(string net_proto, string net_caffemodel, vector<string> selected_layers)
{
	net = dnn::readNetFromCaffe(net_proto, net_caffemodel);
	layers_names = selected_layers;
	weights = vector<double>(layers_names.size(), 0);
	for(double& i : weights)
	{
		i = (double)(rand() % 10) / 10.0;
	}
}

vector<Mat> DeepGaze1::featureMapGenerator(Mat img)
{
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
			resize(blob, blob, Size(227, 227));
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
	Mat centerBias(227, 227, CV_64F, Scalar::all(0.0));
	for(int i = 0;i < centerBias.rows;i++)
	{
		for(int j = 0;j < centerBias.cols;j++)
		{
			double temp = max(abs(i - 227/2), abs(i - 227/2)) * 1.414;
			centerBias.at<double>(i, j) = exp(-0.0002 * temp * temp);
		}
	}
	featureMaps.push_back(centerBias);
	return featureMaps;
}

bool DeepGaze1::computeSaliencyImpl(InputArray image, OutputArray saliencyMap)
{
	vector<Mat> featureMaps = featureMapGenerator(image.getMat());
	saliencyMap.assign(softmax(comb(featureMaps, weights)));
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
	GaussianBlur(res, res, Size(35, 35), 0, 0);
	return res;
}

Mat DeepGaze1::softmax(Mat res)
{
	exp(res, res);
	double temp = sum(res).val[0];
	res = res / temp;

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
	for(unsigned i = 0 ; ((i < match.size()) && (match[i].first > 0)) ; i++)
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

vector<double> DeepGaze1::evalGrad(vector<Mat>& featureMaps, vector<unsigned>& randIndex, vector<double> wei)
{
	vector<double> grad(featureMaps.size(), 0);

	Mat c = comb(featureMaps, wei);
	Mat saliency_old = c.clone();
	saliency_old = softmax(saliency_old);
	vector<double> tt_old = mapSampler(saliency_old, randIndex);
	double loss_old = 0;
	loss_old = loss(tt_old, wei);
	for(unsigned i = 0; i < wei.size(); i++)
	{
		Mat saliency_new = c.clone();
		Mat temp(227, 227, CV_64F, Scalar::all(0.0));
		temp += 0.0000000001 * featureMaps[i];
		GaussianBlur(temp, temp, Size(35, 35), 0, 0);
		saliency_new += temp;
		saliency_new = softmax(saliency_new);
		vector<double> weights_new(wei);
		weights_new[i] += 0.0000000001;
		double loss_new = 0;
		vector<double> tt = mapSampler(saliency_new, randIndex);
		loss_new = loss(tt, weights_new);
		grad[i] = (loss_new - loss_old) / 0.0000000001;
	}
	return grad;
}

void DeepGaze1::training(vector<Mat>& images, vector<Mat>& fixMaps)
{
  vector<unsigned> randIndex = batchIndex(images.size(), images.size());
  vector<vector<unsigned> > fixLoc_list;
  vector<vector<Mat> > featureMaps_list;
  vector<unsigned> fixLoc;
  vector<Mat> featureMaps;
  vector<double> grad;
  vector<double> vel(weights.size(), 0);

  unsigned n = 0;
  for(unsigned i : randIndex)
  {
    vector<Mat> featureMaps = featureMapGenerator(images[i]);
    vector<unsigned> fixLoc = fixationLoc(fixMaps[i]);
    grad = evalGrad(featureMaps, fixLoc, weights);
    for(unsigned j = 0; j < grad.size(); j++)
    {
    	vel[j] = 0.9 * vel[j] + grad[j];
    	weights[j] -= 0.01 * vel[j] * exp(-0.01 * n);
    }
    n++;
    double avgGrad = accumulate(grad.begin(), grad.end(), 0.0) / weights.size();
    double avgWeight = accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
    cout << n << " " << avgGrad << " " << avgWeight << endl;
  }
}
