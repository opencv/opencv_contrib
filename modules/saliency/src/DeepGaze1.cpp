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
	weights = {0.492854,0.492877,0.479609,0.421596,0.480012,0.494546,0.491108,0.439248,0.417933,0.4668,0.486139,0.514718,0.453098,0.472188,0.445103,0.494097,0.529148,0.533347,0.442691,0.489554,0.50217,0.53819,0.531503,0.504056,0.479747,0.461701,0.485831,0.394992,0.570794,0.470482,0.488125,0.471739,0.4398,0.505467,0.550743,0.504148,0.462727,0.459829,0.422992,0.509772,0.553377,0.508442,0.505563,0.491302,0.495632,0.489039,0.499162,0.509256,0.384351,0.491582,0.498799,0.443283,0.14665,0.505998,0.476396,0.441713,0.510864,0.472548,0.475626,0.456787,0.280965,0.518874,0.549786,0.592618,0.471703,0.475373,0.528985,0.526741,0.434502,0.456911,0.491685,0.485708,0.499109,0.4658,0.492202,0.502736,0.472375,0.412582,0.407544,0.494221,0.489164,0.471861,0.47437,0.460959,0.483775,0.501043,0.451273,0.519939,0.508091,0.452396,0.245676,0.0888862,0.595714,0.441607,0.464184,0.29037,0.518533,0.357039,0.437911,0.478792,0.511229,0.595958,0.493729,0.519221,0.523391,0.532277,0.417928,0.475816,0.491772,0.528581,0.475654,0.493478,0.410968,0.520633,0.512556,0.499514,0.498943,0.418418,0.511485,0.446065,0.357794,0.51133,0.209279,0.461075,0.473529,0.465893,0.430571,0.48145,0.546338,0.4401,0.45794,0.483761,0.463332,0.512939,0.502891,0.472128,0.51617,0.333901,0.47877,0.509239,0.510321,0.4077,0.473865,0.445319,0.495594,0.456786,0.518791,0.511876,0.549299,0.498824,0.46347,0.525044,0.47334,0.540368,0.505328,0.409669,0.54626,0.358662,0.46484,0.536226,0.521173,0.519185,0.476368,0.428786,0.432964,0.425856,0.426588,0.457205,0.482571,0.455547,0.48705,0.451839,0.437475,0.542039,0.424395,0.413576,0.51641,0.457871,0.513123,0.48728,0.485663,0.507079,0.526057,0.511003,0.570418,0.514235,0.479556,0.508198,0.498791,0.502174,0.446841,0.447647,0.532451,0.508642,0.496559,0.411333,0.505797,0.420407,0.459041,0.563488,0.469008,0.208642,0.520856,0.519897,0.476584,0.448352,0.451145,0.498658,0.353569,0.401326,0.393847,0.48967,0.522242,0.433495,0.48677,0.497438,0.473391,0.404475,0.490037,0.515052,0.470104,0.434857,0.432513,0.34618,0.479519,0.47566,0.539204,0.455606,0.450167,0.365174,0.527948,0.546035,0.476195,0.549845,0.467285,0.498941,0.480349,0.423689,0.463662,0.103302,0.474756,0.466111,0.474601,0.489604,0.481745,0.500208,0.46845,0.465353,0.481096,0.510841,0.423021,0.279185,0.682107,0.51822,0.492927,0.46524,4.79625};
	//weights = {0.497487,0.502411,0.456505,0.425814,0.497493,0.512262,0.516318,0.44849,0.424749,0.472097,0.494168,0.488127,0.45547,0.476046,0.44338,0.50457,0.55326,0.534006,0.423589,0.498399,0.517197,0.533391,0.518,0.491569,0.491973,0.468363,0.490844,0.387676,0.56501,0.486951,0.498872,0.476382,0.432565,0.492387,0.572339,0.547571,0.481624,0.471914,0.412187,0.457617,0.547966,0.501938,0.517393,0.523375,0.522948,0.486832,0.507451,0.516924,0.424877,0.510275,0.495443,0.472026,0.125719,0.503043,0.506072,0.440315,0.495942,0.477689,0.466947,0.458948,0.346367,0.527349,0.587935,0.58706,0.494051,0.474483,0.539967,0.532128,0.443884,0.462236,0.494746,0.487751,0.488264,0.470903,0.477316,0.498729,0.482376,0.421621,0.439487,0.505218,0.483866,0.471529,0.480059,0.476024,0.496628,0.510684,0.485402,0.521804,0.480137,0.407191,0.231358,0.0521284,0.614009,0.440855,0.47279,0.250036,0.53501,0.413822,0.444445,0.462979,0.518892,0.585769,0.474131,0.508421,0.509128,0.540619,0.412131,0.439915,0.492447,0.522151,0.474511,0.495,0.430954,0.535171,0.518931,0.544447,0.524002,0.421077,0.517055,0.447245,0.251134,0.511714,0.153483,0.490056,0.453425,0.450157,0.442129,0.436315,0.555194,0.46594,0.422647,0.456453,0.472298,0.532641,0.523724,0.471044,0.501304,0.306234,0.485647,0.516084,0.529696,0.318047,0.482182,0.459202,0.524382,0.461817,0.460216,0.518925,0.565913,0.46236,0.397505,0.536967,0.477518,0.526041,0.484764,0.418136,0.546664,0.332771,0.432856,0.474645,0.534784,0.503048,0.460163,0.433259,0.427519,0.291319,0.36069,0.407249,0.48776,0.434639,0.492274,0.471517,0.436101,0.551626,0.41872,0.424817,0.533052,0.457066,0.508638,0.488303,0.501407,0.503762,0.521199,0.500515,0.510584,0.50581,0.466941,0.51722,0.502145,0.447454,0.430834,0.451845,0.541774,0.505495,0.501585,0.434551,0.536647,0.408516,0.47475,0.576825,0.515408,0.278159,0.527432,0.507564,0.461757,0.468113,0.463559,0.511806,0.379718,0.346922,0.395977,0.489966,0.513447,0.450056,0.511192,0.486209,0.481624,0.425786,0.48788,0.514394,0.457541,0.430562,0.438185,0.417461,0.501014,0.49015,0.541213,0.47548,0.471689,0.36902,0.566105,0.573095,0.474797,0.546882,0.492789,0.505544,0.497233,0.418915,0.481781,0.103597,0.474253,0.478368,0.511184,0.503792,0.497224,0.493456,0.450304,0.44323,0.477985,0.508655,0.425664,0.210711,0.541917,0.505914,0.476779,0.417452,2.1771};
	/*weights=vector<double>(257,1.0);
	for(double& i : weights)
	{
		i = (rand() % 10 / 10.0);
			//i = 1;
	}*/
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
	/*Mat centerBias(227, 227, CV_64F, Scalar::all(0.0));
	for(int i = 0;i < centerBias.rows;i++)
	{
		for(int j = 0;j < centerBias.cols;j++)
		{
			double temp = max(abs(i - 227/2), abs(i - 227/2)) * 1.414;
			centerBias.at<double>(i, j) = exp(-0.0001 * temp * temp);
		}
	}*/
	Mat centerBias = (Mat_<double>(15, 15) << 0.136605132339619,0.254911885194451,0.354975848201554,0.432118422919642,0.492788960570534,0.537169372900984,0.559893362856313,0.570408994503682,0.558327108322080,0.529592014101891,0.483151700229565,0.421134751334104,0.339516463333039,0.240333898610746,0.119040524151926,0.255057036816424,0.373463164527964,0.471695245848319,0.548331995771048,0.610621682003216,0.652904303445073,0.678084927709815,0.686289988540627,0.675100404076781,0.648787013529810,0.601363163606808,0.539724655027488,0.457897183544702,0.359051684872921,0.237121060715778,0.354996162889476,0.471574664262948,0.569824563418442,0.647926075225057,0.710453524703313,0.751807537518647,0.777982678740224,0.785604558552769,0.773105643206983,0.747143093637550,0.700960041197720,0.637036644172802,0.555488832006236,0.456278688906414,0.336404833368141,0.432114886533253,0.548519754764192,0.647919735424171,0.726252091271344,0.787633820027685,0.830453686804553,0.855159388723024,0.864261146191168,0.851130919200443,0.823782075450280,0.777941301454146,0.714197202377273,0.632042767206447,0.533375632722201,0.413459898633110,0.492846224526104,0.610458063998438,0.710273041291813,0.787934734560166,0.848818377335713,0.892669644305876,0.915962511030596,0.925983774442787,0.914078884852605,0.886340426175551,0.838921069614458,0.776522338759063,0.694910735314247,0.595717337417090,0.474404562107594,0.537258377108661,0.652654274132924,0.751668431422994,0.830557252057301,0.892264877678679,0.934387776714834,0.960294475640559,0.968153776825753,0.957118891968944,0.930719438034762,0.882668936995775,0.818004698514852,0.736335421611659,0.637250235468980,0.518321616011027,0.559912410908083,0.678060440904210,0.778035916610351,0.855337573494623,0.916090961317328,0.960268532575822,0.983151329506359,0.993265865328518,0.981393158586731,0.953452714649956,0.906806535966130,0.844266579567292,0.762471436321223,0.663436280862256,0.542292468493236,0.570471859736804,0.686367557103361,0.785571470982568,0.863984557875191,0.925787582513478,0.968184312606637,0.993160551514547,1.00071969560062,0.990820131015439,0.963788669203278,0.916062100871062,0.851941275823728,0.770061877665113,0.671049079061439,0.551460475500773,0.558354004396870,0.675218828625687,0.773141022132423,0.851023228938218,0.914111180726610,0.957097877210958,0.981435236380489,0.990830375805955,0.979069435484073,0.951285941586670,0.904001536547754,0.841050003441305,0.759279958742245,0.660299547838899,0.539966072338975,0.529794530767286,0.648426268685882,0.747085236556856,0.823784188854240,0.886061506412046,0.930850736147926,0.953129788839557,0.963752221613668,0.951512839210700,0.924189623933934,0.877773421560175,0.814688659956347,0.733050627339970,0.633967523008819,0.512144357629488,0.483094193799984,0.601317535642541,0.700936072455337,0.778084086878178,0.839052826457435,0.882842688790194,0.906345815545706,0.916276056981117,0.904027650739533,0.877827433807473,0.831763094963022,0.767578814024913,0.685653480474771,0.586759596897176,0.465090771371211,0.421344630481700,0.539705161668206,0.637059989497880,0.714213691485248,0.776688317470029,0.818187784047592,0.844251497483870,0.851844570961231,0.841044566248117,0.814778195034621,0.767546270221294,0.705699445037065,0.623983961385159,0.525082406012993,0.403548799412053,0.339352953813616,0.458020160692771,0.555130309401582,0.632633247554868,0.694581915649390,0.736338072460011,0.762425604772228,0.770108192322197,0.759369952472952,0.732901801094476,0.685633947666787,0.623974569677122,0.542296497992531,0.443313473703102,0.321653790809172,0.240452590442239,0.358845767555164,0.456323501276454,0.533311273531990,0.595861650031113,0.637378281509690,0.663434282045727,0.670883353894946,0.660356369332519,0.633940197197224,0.586949913320820,0.525019901253340,0.443239391952182,0.341581133756428,0.221496442250004,0.118883254021697,0.237092401238019,0.336442576365934,0.413042016689593,0.474758090922195,0.518099556977193,0.541973522382581,0.551465525482662,0.539575142928788,0.512205675560351,0.465197135656097,0.403297133574821,0.321504846482331,0.221359680451730,0.100335071938095);
	resize(centerBias, centerBias, Size(227, 227));
	featureMaps.push_back(centerBias);
	return featureMaps;
}

bool DeepGaze1::computeSaliencyImpl(InputArray image, OutputArray saliencyMap)
{
	vector<Mat> featureMaps = featureMapGenerator(image.getMat());
	saliencyMap.assign(softmax(comb(featureMaps, weights)));
	for(double i : weights) cout << i << ",";
	cout << endl;
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
    	weights[j] -= 0.1 * vel[j] * exp(-0.01 * n);
    }
    n++;
    double avgGrad = accumulate(grad.begin(), grad.end(), 0.0) / weights.size();
    double avgWeight = accumulate(weights.begin(), weights.end(), 0.0) / weights.size();
    cout << n << " " << avgGrad << " " << avgWeight << endl;
  }
}

double DeepGaze1::computeAUC(InputArray _saliencyMap, InputArray _fixtionMap)
{
	Mat saliency = _saliencyMap.getMat();
	Mat fixtion = _fixtionMap.getMat();

	if(saliency.empty() || fixtion.empty() || saliency.dims > 2 || fixtion.dims > 2)
	{
		cout << "saliency map and fixtion map must be 1 channel and have same size" << endl;
		return -1;
	}

	resize(fixtion, fixtion, Size(saliency.cols, saliency.rows));
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
	Mat saliency = _saliencyMap.getMat();
	double mi = 0, ma = 0;
	minMaxLoc(saliency, &mi, &ma);
	saliency -= mi;
	saliency /= (ma - mi);
	imshow("saliencyVisual", saliency);
	waitKey(0);
}
}
}
