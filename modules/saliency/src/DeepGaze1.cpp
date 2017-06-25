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
 * DeepGaze1.cpp
 *
 *  Created on: Jun 1, 2017
 *      Author: qsx
 */

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "precomp.hpp"
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
using namespace cv::saliency;

namespace cv
{
namespace saliency
{

DeepGaze1::DeepGaze1()
{
	net = dnn::readNetFromCaffe("deploy.prototxt", "bvlc_alexnet.caffemodel");
	layers_names = vector<string>{"conv5"};
	weights = {0.469576,0.475909,0.445926,0.476597,0.505775,0.457083,0.486233,0.460788,0.46106,0.471132,0.461589,0.383889,0.470089,0.461857,0.606081,0.469439,0.473654,0.438096,0.492859,0.483114,0.436092,0.466829,0.44598,0.467684,0.486227,0.476685,0.499284,0.548693,0.494316,0.460175,0.466815,0.456755,0.528414,0.448331,0.494126,0.486524,0.477335,0.44679,0.471165,0.453231,0.477976,0.511486,0.489458,0.433211,0.476518,0.49075,0.456346,0.479566,0.404909,0.517421,0.477067,0.498872,0.238922,0.504733,0.447699,0.435917,0.472379,0.450609,0.482663,0.462184,0.447799,0.464343,0.530619,0.753738,0.46916,0.482865,0.460146,0.489529,0.486235,0.457387,0.460575,0.472095,0.506805,0.469016,0.454105,0.458717,0.449325,0.456058,0.495336,0.471266,0.440256,0.467249,0.453049,0.488131,0.470352,0.487353,0.445176,0.460139,0.45247,0.469445,0.244583,0.169326,0.451571,0.44315,0.477018,0.44736,0.466333,0.547419,0.402816,0.448315,0.467074,0.4921,0.487628,0.479554,0.489416,0.476411,0.42611,0.437581,0.475609,0.484149,0.461617,0.507637,0.478952,0.447884,0.466818,0.437146,0.446821,0.491235,0.529646,0.499436,0.471622,0.451675,0.471276,0.452826,0.450762,0.484006,0.464328,0.585631,0.45568,0.452579,0.518405,0.467232,0.446822,0.39135,0.464697,0.493513,0.494239,0.609592,0.440514,0.550717,0.617017,0.47375,0.456314,0.405965,0.459913,0.410683,0.432651,0.528895,0.604878,0.502048,0.52597,0.461418,0.467602,0.451857,0.47494,0.482889,0.462033,0.466892,0.443192,0.47522,0.440551,0.49915,0.440109,0.46256,0.451269,0.459332,0.42362,0.454785,0.429994,0.486273,0.443259,0.487732,0.466453,0.436671,0.488789,0.428538,0.430395,0.650086,0.459794,0.485563,0.476398,0.448556,0.445961,0.477572,0.461427,0.490713,0.457607,0.475426,0.510739,0.492226,0.471198,0.468778,0.536845,0.463664,0.479667,0.460984,0.459756,0.449138,0.400535,0.481603,0.573033,0.379099,0.452276,0.508623,0.372783,0.474606,0.454702,0.582488,0.472624,0.38901,0.49034,0.471059,0.453068,0.45951,0.452311,0.516473,0.487529,0.52764,0.469588,0.514558,0.481468,0.469013,0.460126,0.504444,0.419805,0.426563,0.478199,0.472796,0.493762,0.493559,0.448286,0.446668,0.411609,0.475216,0.464951,0.486518,0.465692,0.373902,0.473649,0.46263,0.482328,0.483113,0.485923,0.448114,0.500998,0.483336,0.469564,0.489501,0.492358,0.489632,0.469056,0.393911,0.532511,0.469307,0.496896,0.48274,6.00684};
}

DeepGaze1::DeepGaze1(string net_proto, string net_caffemodel, vector<string> selected_layers, unsigned n_weights)
{
	net = dnn::readNetFromCaffe(net_proto, net_caffemodel);
	layers_names = selected_layers;
	weights = vector<double>(n_weights, 0);
	for(double& i : weights)
	{
		i = (double)(rand() % 10) / 10.0;
	}
}

DeepGaze1::~DeepGaze1(){}

vector<Mat> DeepGaze1::featureMapGenerator(Mat img, Size input_size)
{
	vector<Mat> featureMaps;
	Mat me(input_size, CV_64F, Scalar::all(0.0));
	Mat std(input_size, CV_64F, Scalar::all(0.0));
	Mat temp(input_size, CV_64F, Scalar::all(0.0));


	resize(img, img, input_size, 0, 0, INTER_AREA);//hard coded
	Mat inputBlob = blobFromImage(img);   //Convert Mat to batch of images
	net.setBlob(".data", inputBlob);        //set the network input
	net.forward();                          //compute output
	for(unsigned i = 0; i < layers_names.size(); i++)
	{
		Mat blob_set = net.getBlob(layers_names[i]);
		for(int j = 0; j < blob_set.size[1]; j++)
		{
			Mat m, s, blob = Mat(blob_set.size[2], blob_set.size[3], CV_64F, blob_set.ptr(0, j)).clone();
			resize(blob, blob, input_size, 0, 0, INTER_CUBIC);
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
	Mat centerBias = (Mat_<double>(15, 15) << 0.136605132339619,0.254911885194451,0.354975848201554,0.432118422919642,0.492788960570534,0.537169372900984,0.559893362856313,0.570408994503682,0.558327108322080,0.529592014101891,0.483151700229565,0.421134751334104,0.339516463333039,0.240333898610746,0.119040524151926,0.255057036816424,0.373463164527964,0.471695245848319,0.548331995771048,0.610621682003216,0.652904303445073,0.678084927709815,0.686289988540627,0.675100404076781,0.648787013529810,0.601363163606808,0.539724655027488,0.457897183544702,0.359051684872921,0.237121060715778,0.354996162889476,0.471574664262948,0.569824563418442,0.647926075225057,0.710453524703313,0.751807537518647,0.777982678740224,0.785604558552769,0.773105643206983,0.747143093637550,0.700960041197720,0.637036644172802,0.555488832006236,0.456278688906414,0.336404833368141,0.432114886533253,0.548519754764192,0.647919735424171,0.726252091271344,0.787633820027685,0.830453686804553,0.855159388723024,0.864261146191168,0.851130919200443,0.823782075450280,0.777941301454146,0.714197202377273,0.632042767206447,0.533375632722201,0.413459898633110,0.492846224526104,0.610458063998438,0.710273041291813,0.787934734560166,0.848818377335713,0.892669644305876,0.915962511030596,0.925983774442787,0.914078884852605,0.886340426175551,0.838921069614458,0.776522338759063,0.694910735314247,0.595717337417090,0.474404562107594,0.537258377108661,0.652654274132924,0.751668431422994,0.830557252057301,0.892264877678679,0.934387776714834,0.960294475640559,0.968153776825753,0.957118891968944,0.930719438034762,0.882668936995775,0.818004698514852,0.736335421611659,0.637250235468980,0.518321616011027,0.559912410908083,0.678060440904210,0.778035916610351,0.855337573494623,0.916090961317328,0.960268532575822,0.983151329506359,0.993265865328518,0.981393158586731,0.953452714649956,0.906806535966130,0.844266579567292,0.762471436321223,0.663436280862256,0.542292468493236,0.570471859736804,0.686367557103361,0.785571470982568,0.863984557875191,0.925787582513478,0.968184312606637,0.993160551514547,1.00071969560062,0.990820131015439,0.963788669203278,0.916062100871062,0.851941275823728,0.770061877665113,0.671049079061439,0.551460475500773,0.558354004396870,0.675218828625687,0.773141022132423,0.851023228938218,0.914111180726610,0.957097877210958,0.981435236380489,0.990830375805955,0.979069435484073,0.951285941586670,0.904001536547754,0.841050003441305,0.759279958742245,0.660299547838899,0.539966072338975,0.529794530767286,0.648426268685882,0.747085236556856,0.823784188854240,0.886061506412046,0.930850736147926,0.953129788839557,0.963752221613668,0.951512839210700,0.924189623933934,0.877773421560175,0.814688659956347,0.733050627339970,0.633967523008819,0.512144357629488,0.483094193799984,0.601317535642541,0.700936072455337,0.778084086878178,0.839052826457435,0.882842688790194,0.906345815545706,0.916276056981117,0.904027650739533,0.877827433807473,0.831763094963022,0.767578814024913,0.685653480474771,0.586759596897176,0.465090771371211,0.421344630481700,0.539705161668206,0.637059989497880,0.714213691485248,0.776688317470029,0.818187784047592,0.844251497483870,0.851844570961231,0.841044566248117,0.814778195034621,0.767546270221294,0.705699445037065,0.623983961385159,0.525082406012993,0.403548799412053,0.339352953813616,0.458020160692771,0.555130309401582,0.632633247554868,0.694581915649390,0.736338072460011,0.762425604772228,0.770108192322197,0.759369952472952,0.732901801094476,0.685633947666787,0.623974569677122,0.542296497992531,0.443313473703102,0.321653790809172,0.240452590442239,0.358845767555164,0.456323501276454,0.533311273531990,0.595861650031113,0.637378281509690,0.663434282045727,0.670883353894946,0.660356369332519,0.633940197197224,0.586949913320820,0.525019901253340,0.443239391952182,0.341581133756428,0.221496442250004,0.118883254021697,0.237092401238019,0.336442576365934,0.413042016689593,0.474758090922195,0.518099556977193,0.541973522382581,0.551465525482662,0.539575142928788,0.512205675560351,0.465197135656097,0.403297133574821,0.321504846482331,0.221359680451730,0.100335071938095);
	resize(centerBias, centerBias, input_size, 0, 0, INTER_CUBIC);
	featureMaps.push_back(centerBias);
	return featureMaps;
}

bool DeepGaze1::computeSaliencyImpl(InputArray image, OutputArray saliencyMap)
{
	vector<Mat> featureMaps = featureMapGenerator(image.getMat(), Size(227, 227));
	saliencyMap.assign(softmax(comb(featureMaps, weights)));
	for(double i : weights) cout << i << ",";
	cout << endl;
	return true;
}

Mat DeepGaze1::saliencyMapGenerator(Mat input_image, Size input_size)
{
//raw saliency map generate
	vector<Mat> featureMaps = featureMapGenerator(input_image, input_size);
	return softmax(comb(featureMaps, weights));
}

Mat DeepGaze1::comb(vector<Mat>& featureMaps, vector<double> wei)
{
	Mat res(featureMaps[0].rows, featureMaps[0].cols, CV_64F, Scalar::all(0.0));

	for(unsigned i = 0; i < featureMaps.size(); i++)
	{
		Mat temp = featureMaps[i].clone();
		temp *= wei[i];
		res += temp;
	}
	GaussianBlur(res, res, Size(35, 35), 0, 0, BORDER_CONSTANT);
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



vector<unsigned> DeepGaze1::fixationLoc(Mat img, Size input_size)
{
	vector<unsigned> randIndex;
	resize(img, img, input_size, 0, 0, INTER_AREA);
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
	return res + 0.001 * l1 / sqrt(l2);
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

vector<double> DeepGaze1::evalGrad(vector<Mat>& featureMaps, vector<unsigned>& randIndex, vector<double> wei, Size input_size)
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
		Mat temp(input_size, CV_64F, Scalar::all(0.0));
		temp += 0.0000000001 * featureMaps[i];
		GaussianBlur(temp, temp, Size(35, 35), 0, 0, BORDER_CONSTANT);
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

void DeepGaze1::training(vector<Mat>& images, vector<Mat>& fixMaps, Size input_size, double momentum, double alpha, double decay)
{
  vector<unsigned> randIndex = batchIndex(images.size(), min(100, (int)images.size()));
  vector<unsigned> fixLoc;
  vector<Mat> featureMaps;
  vector<double> grad;
  vector<double> vel(weights.size(), 0);

  unsigned n = 0;
  for(unsigned i : randIndex)
  {
    featureMaps = featureMapGenerator(images[i], input_size);
    fixLoc = fixationLoc(fixMaps[i], input_size);
    grad = evalGrad(featureMaps, fixLoc, weights, input_size);
    for(unsigned j = 0; j < grad.size(); j++)
    {
    	vel[j] = momentum * vel[j] + grad[j];
    	weights[j] -= alpha * vel[j] * exp(-decay * n);
    }
    n++;
    double avgGrad = accumulate(grad.begin(), grad.end(), 0.0) / weights.size();
    double avgWeight = accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
    cout << n << " " << avgGrad << " " << avgWeight << endl;
  }
}

double DeepGaze1::computeAUC(InputArray _saliencyMap, InputArray _fixtionMap)
{
	Mat saliency = _saliencyMap.getMat().clone();
	Mat fixtion = _fixtionMap.getMat().clone();

	if(saliency.empty() || fixtion.empty() || saliency.dims > 2 || fixtion.dims > 2)
	{
		cout << "saliency map and fixtion map must be 1 channel and have same size" << endl;
		return -1;
	}

	resize(fixtion, fixtion, Size(saliency.cols, saliency.rows), 0, 0, INTER_CUBIC);
	double mi = 0, ma = 0;
	minMaxLoc(saliency, &mi, &ma);
	saliency -= mi;
	saliency /= (ma - mi);
	vector<double> threshold_list;
	for(int i = 0;i < saliency.rows;i++)
	{
		for(int j = 0;j < saliency.cols;j++)
		{
			if(fixtion.at<uchar>(i, j) > 0) threshold_list.push_back(saliency.at<double>(i, j));
		}
	}
	sort(threshold_list.begin(), threshold_list.end(), [](double a, double b) {
		return b < a;
	});

	vector<double> tp(1, 0), fp(1, 0);
	for(unsigned i = 0;i < threshold_list.size();i++)
	{
		unsigned aboveth = 0;
		for(int m = 0;m < saliency.rows;m++)
		{
			for(int n = 0;n < saliency.cols;n++)
			{
				if(saliency.at<double>(m, n) >= threshold_list[i]) aboveth++;
			}
		}
		tp.push_back((i + 1.0) / (double)threshold_list.size());
		fp.push_back((aboveth - i - 1.0) / (double)(saliency.rows * saliency.cols - threshold_list.size()));
	}
	tp.push_back(1);
	fp.push_back(1);
	double auc = 0;
	for(unsigned i = 1;i < tp.size();i++)
	{
		auc += (tp[i - 1] + tp[i]) * (fp[i] - fp[i - 1]) / 2;
	}

	return auc;
}

void DeepGaze1::saliencyMapVisualize(InputArray _saliencyMap)
{
	Mat saliency = _saliencyMap.getMat().clone();
	double mi = 0, ma = 0;
	minMaxLoc(saliency, &mi, &ma);
	saliency -= mi;
	saliency /= (ma - mi);
	imshow("saliencyVisual", saliency);
	waitKey(0);
}
}
}
