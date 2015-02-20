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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2010, Willow Garage Inc., all rights reserved.
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
//
// Copyright(C) 2013 by Tomasz Trzcinski
// tomasz.trzcinski@epfl.ch

//M*/
#include <opencv2/features2d/features2d.hpp>
#include <bitset>
#include <stdlib.h>
#include <cmath>

using namespace std;
using namespace cv;

void BGMDescriptorExtractor::Utils::computeGradientMaps(const Mat& im,
	const Assign& gradAssignType,
	const int orientQuant,
	vector<Mat>& gradMap)
{
	Mat derivx(im.size(), CV_32FC1);
	Mat derivy(im.size(), CV_32FC1);

	Sobel(im, derivx, derivx.depth(), 1, 0);
	Sobel(im, derivy, derivy.depth(), 0, 1);

	for (int i = 0; i < orientQuant; ++i)
		gradMap.push_back(Mat::zeros(im.size(), CV_8UC1));

	//Fill in temp matrices with respones to edge detection
	double binSize = (2 * CV_PI) / orientQuant;
	int index, index2;
	double binCenter, weight;
	const float* pDerivx = derivx.ptr<float>();
	const float* pDerivy = derivy.ptr<float>();
	for (int i = 0; i < im.rows; ++i)
	{
		for (int j = 0; j<im.cols; ++j)
		{
			float gradMagnitude = sqrt((*pDerivx)*(*pDerivx) + (*pDerivy)*(*pDerivy));
			if (gradMagnitude > 20)
			{
				double theta = atan2(*pDerivy, *pDerivx);
				theta = (theta < 0) ? theta + 2 * CV_PI : theta;
				index = int(theta / binSize);
				index = (index == orientQuant) ? 0 : index;

				switch (gradAssignType)
				{
				case ASSIGN_HARD:
					gradMap[index].at<uchar>(i, j) = 1;
					break;
				case ASSIGN_HARD_MAGN:
					gradMap[index].at<uchar>(i, j) = round(gradMagnitude);
					break;
				case ASSIGN_BILINEAR:
					index2 = ceil(theta / binSize);
					index2 = (index2 == orientQuant) ? 0 : index2;
					binCenter = (index + .5)*binSize;
					weight = 1 - std::abs(theta - binCenter) / binSize;
					gradMap[index].at<uchar>(i, j) = (int)round(255 * weight);
					gradMap[index2].at<uchar>(i, j) = (int)round(255 * (1 - weight));
					break;
				case ASSIGN_SOFT:
					for (int binNum = 0; binNum < orientQuant / 2 + 1; ++binNum)
					{
						index2 = (binNum + index + orientQuant - orientQuant / 4) % orientQuant;
						binCenter = (index2 + .5)*binSize;
						weight = cos(theta - binCenter);
						weight = (weight < 0) ? 0 : weight;
						gradMap[index2].at<uchar>(i, j) = (int)round(255 * weight);
					}
					break;
				case ASSIGN_SOFT_MAGN:
					for (int binNum = 0; binNum < orientQuant / 2 + 1; ++binNum)
					{
						index2 = (binNum + index + orientQuant - orientQuant / 4) % orientQuant;
						binCenter = (index2 + .5)*binSize;
						weight = cos(theta - binCenter);
						weight = (weight < 0) ? 0 : weight;
						gradMap[index2].at<uchar>(i, j) = (int)round(gradMagnitude*weight);
					}
					break;
				}
			}
			++pDerivy; ++pDerivx;
		}
	}
}

/**********************************************************************************************/
void BGMDescriptorExtractor::Utils::computeIntegrals(const vector<Mat>& gradMap,
	const int orientQuant,
	vector<Mat>& integralMap)
{
	// Initialize Integral Images
	int rows = gradMap[0].rows;
	int cols = gradMap[0].cols;
	for (int i = 0; i < orientQuant + 1; ++i)
		integralMap.push_back(Mat::zeros(rows + 1, cols + 1, CV_8UC1));

	//Generate corresponding integral images;
	for (int i = 0; i < orientQuant; ++i)
		integral(gradMap[i], integralMap[i]);

	// copy the values from the first quantization bin
	integralMap[0].copyTo(integralMap[orientQuant]);

	int* ptrSum, *ptr;
	for (int k = 1; k < orientQuant; ++k)
	{
		ptr = (int*)integralMap[k].ptr<int>();
		ptrSum = (int*)integralMap[orientQuant].ptr<int>();
		for (int i = 0; i < (rows + 1)*(cols + 1); ++i)
		{
			*ptrSum += *ptr;
			++ptrSum;
			++ptr;
		}
	}
}

/**********************************************************************************************/
float BGMDescriptorExtractor::Utils::computeWLResponse(const WeakLearner& WL,
	const int orientQuant,
	const vector<Mat>& integralMap)
{
	int width = integralMap[0].cols;
	int idx1 = WL.y_min * width + WL.x_min;
	int idx2 = WL.y_min * width + WL.x_max + 1;
	int idx3 = (WL.y_max + 1) * width + WL.x_min;
	int idx4 = (WL.y_max + 1) * width + WL.x_max + 1;

	int A, B, C, D;
	const int* ptr = integralMap[WL.orient].ptr<int>();
	A = ptr[idx1];
	B = ptr[idx2];
	C = ptr[idx3];
	D = ptr[idx4];
	float current = D + A - B - C;

	ptr = integralMap[orientQuant].ptr<int>();
	A = ptr[idx1];
	B = ptr[idx2];
	C = ptr[idx3];
	D = ptr[idx4];
	float total = D + A - B - C;

	return total ? ((current / total) - WL.thresh) : 0.f;
}

/**********************************************************************************************/

void BGMDescriptorExtractor::Utils::rectifyPatch(const Mat& image,
	const KeyPoint& kp,
	const int& patchSize,
	Mat& patch)
{
	float s = 1.5f * (float)kp.size / (float)patchSize; 

	float cosine = (kp.angle >= 0) ? cos(kp.angle*CV_PI / 180) : 1.f;
	float sine = (kp.angle >= 0) ? sin(kp.angle*CV_PI / 180) : 0.f;

	float M_[] = {
		s*cosine, -s*sine, (-s*cosine + s*sine) * patchSize / 2.0f + kp.pt.x,
		s*sine, s*cosine, (-s*sine - s*cosine) * patchSize / 2.0f + kp.pt.y
	};

	warpAffine(image, patch, Mat(2, 3, CV_32FC1, M_), Size(patchSize, patchSize),
		CV_WARP_INVERSE_MAP + CV_INTER_CUBIC + CV_WARP_FILL_OUTLIERS);
}


void BGMDescriptorExtractor::readWLs()
{
	FILE* fid = fopen(wlFile.c_str(), "rt");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
		exit(-3);
	}
	fscanf(fid, "%d\n", &nWLs);
	fscanf(fid, "%d\n", &orientQuant);
	fscanf(fid, "%d\n", &patchSize);
	int iGradAssignType;
	fscanf(fid, "%d\n", &iGradAssignType);
	gradAssignType = (Assign)iGradAssignType;
	pWLs = new WeakLearner[nWLs];
	nDim = nWLs;
	for (int i = 0; i < nWLs; i++)
	{
		fscanf(fid, "%f %d %d %d %d %d %f\n",
			&pWLs[i].thresh,
			&pWLs[i].orient,
			&pWLs[i].y_min,
			&pWLs[i].y_max,
			&pWLs[i].x_min,
			&pWLs[i].x_max,
			&pWLs[i].alpha);
	}
	fclose(fid);
}

/**********************************************************************************************/

void BGMDescriptorExtractor::readWLsBin()
{
	FILE* fid = fopen(wlFile.c_str(), "rb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
		exit(-3);
	}
	fread(&nWLs, sizeof(int), 1, fid);
	fread(&orientQuant, sizeof(int), 1, fid);
	fread(&patchSize, sizeof(int), 1, fid);
	int iGradAssignType;
	fread(&iGradAssignType, sizeof(int), 1, fid);
	gradAssignType = (Assign)iGradAssignType;
	pWLs = new WeakLearner[nWLs];
	nDim = nWLs;
	for (int i = 0; i < nWLs; i++)
	{
		fread(&pWLs[i].thresh, sizeof(float), 1, fid);
		fread(&pWLs[i].orient, sizeof(int), 1, fid);
		fread(&pWLs[i].y_min, sizeof(int), 1, fid);
		fread(&pWLs[i].y_max, sizeof(int), 1, fid);
		fread(&pWLs[i].x_min, sizeof(int), 1, fid);
		fread(&pWLs[i].x_max, sizeof(int), 1, fid);
		fread(&pWLs[i].alpha, sizeof(float), 1, fid);
	}
	fclose(fid);
}

/**********************************************************************************************/

void BGMDescriptorExtractor::saveWLsBin(const string output) const
{
	FILE* fid = fopen(output.c_str(), "wb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", output.c_str());
		exit(-3);
	}
	fwrite(&nWLs, sizeof(int), 1, fid);
	fwrite(&orientQuant, sizeof(int), 1, fid);
	fwrite(&patchSize, sizeof(int), 1, fid);
	fwrite(&gradAssignType, sizeof(int), 1, fid);
	for (int i = 0; i < nWLs; i++)
	{
		fwrite(&pWLs[i].thresh, sizeof(float), 1, fid);
		fwrite(&pWLs[i].orient, sizeof(int), 1, fid);
		fwrite(&pWLs[i].y_min, sizeof(int), 1, fid);
		fwrite(&pWLs[i].y_max, sizeof(int), 1, fid);
		fwrite(&pWLs[i].x_min, sizeof(int), 1, fid);
		fwrite(&pWLs[i].x_max, sizeof(int), 1, fid);
		fwrite(&pWLs[i].alpha, sizeof(float), 1, fid);
	}
	fclose(fid);
}

/**********************************************************************************************/

void BGMDescriptorExtractor::compute(const Mat& image, vector<KeyPoint>& keypoints,
	Mat& descriptors) const
{
	// initialize the variables
	descriptors = Mat::zeros(keypoints.size(), ceil(nDim / 8), descriptorType());
	vector<Mat> gradMap, integralMap;

	// iterate through all the keypoints
	for (unsigned int i = 0; i < keypoints.size(); ++i)
	{
		// rectify the patch around a given keypoint
		Mat patch;
		Utils::rectifyPatch(image, keypoints[i], patchSize, patch);

		// compute gradient maps (and integral gradient maps)
		Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
		Utils::computeIntegrals(gradMap, orientQuant, integralMap);

		// compute the responses of the weak learners
		uchar* desc = descriptors.ptr<uchar>(i);
		for (int j = 0; j < nWLs; ++j)
			desc[j / 8] |= (Utils::computeWLResponse(pWLs[j], orientQuant, integralMap) >= 0) ? binLookUp[j % 8] : 0;

		// clean-up
		patch.release();
		gradMap.clear();
		integralMap.clear();
	}
}

/**********************************************************************************************/

void LBGMDescriptorExtractor::readWLs()
{
	FILE* fid = fopen(wlFile.c_str(), "rt");
	fscanf(fid, "%d\n", &nWLs);
	fscanf(fid, "%d\n", &orientQuant);
	fscanf(fid, "%d\n", &patchSize);
	int iGradAssignType;
	fscanf(fid, "%d\n", &iGradAssignType);
	gradAssignType = (Assign)iGradAssignType;
	pWLs = new WeakLearner[nWLs];
	for (int i = 0; i < nWLs; i++)
	{
		fscanf(fid, "%f %d %d %d %d %d %f\n",
			&pWLs[i].thresh,
			&pWLs[i].orient,
			&pWLs[i].y_min,
			&pWLs[i].y_max,
			&pWLs[i].x_min,
			&pWLs[i].x_max,
			&pWLs[i].alpha);
	}
	fscanf(fid, "%d\n", &nDim);
	betas = new float[nDim*nWLs];
	for (int i = 0; i < nWLs; i++)
	{
		for (int d = 0; d < nDim; ++d)
			fscanf(fid, "%f ", &betas[i*nDim + d]);
		fscanf(fid, "\n");
	}
	fclose(fid);
}

/**********************************************************************************************/

void LBGMDescriptorExtractor::readWLsBin()
{
	FILE* fid = fopen(wlFile.c_str(), "rb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
		exit(-3);
	}
	fread(&nDim, sizeof(int), 1, fid);
	fread(&nWLs, sizeof(int), 1, fid);
	fread(&orientQuant, sizeof(int), 1, fid);
	fread(&patchSize, sizeof(int), 1, fid);
	int iGradAssignType;
	fread(&iGradAssignType, sizeof(int), 1, fid);
	gradAssignType = (Assign)iGradAssignType;
	pWLs = new WeakLearner[nWLs];
	for (int i = 0; i < nWLs; i++)
	{
		fread(&pWLs[i].thresh, sizeof(float), 1, fid);
		fread(&pWLs[i].orient, sizeof(int), 1, fid);
		fread(&pWLs[i].y_min, sizeof(int), 1, fid);
		fread(&pWLs[i].y_max, sizeof(int), 1, fid);
		fread(&pWLs[i].x_min, sizeof(int), 1, fid);
		fread(&pWLs[i].x_max, sizeof(int), 1, fid);
		fread(&pWLs[i].alpha, sizeof(float), 1, fid);
	}
	betas = new float[nDim*nWLs];
	for (int i = 0; i < nWLs; i++)
	for (int d = 0; d < nDim; ++d)
		fread(&betas[i*nDim + d], sizeof(float), 1, fid);
	fclose(fid);
}

/**********************************************************************************************/

void LBGMDescriptorExtractor::saveWLsBin(const string output) const
{
	FILE* fid = fopen(output.c_str(), "wb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", wlFile.c_str());
		exit(-3);
	}
	fwrite(&nDim, sizeof(int), 1, fid);
	fwrite(&nWLs, sizeof(int), 1, fid);
	fwrite(&orientQuant, sizeof(int), 1, fid);
	fwrite(&patchSize, sizeof(int), 1, fid);
	fwrite(&gradAssignType, sizeof(int), 1, fid);
	for (int i = 0; i < nWLs; i++)
	{
		fwrite(&pWLs[i].thresh, sizeof(float), 1, fid);
		fwrite(&pWLs[i].orient, sizeof(int), 1, fid);
		fwrite(&pWLs[i].y_min, sizeof(int), 1, fid);
		fwrite(&pWLs[i].y_max, sizeof(int), 1, fid);
		fwrite(&pWLs[i].x_min, sizeof(int), 1, fid);
		fwrite(&pWLs[i].x_max, sizeof(int), 1, fid);
		fwrite(&pWLs[i].alpha, sizeof(float), 1, fid);
	}
	for (int i = 0; i < nWLs; i++)
	for (int d = 0; d < nDim; ++d)
		fwrite(&betas[i*nDim + d], sizeof(float), 1, fid);
	fclose(fid);
}

/**********************************************************************************************/

void LBGMDescriptorExtractor::compute(const Mat& image, vector<KeyPoint>& keypoints,
	Mat& descriptors) const
{
	// initialize the variables
	descriptors = Mat::zeros(keypoints.size(), nDim, descriptorType());
	vector<Mat> gradMap, integralMap;

	// iterate through all the keypoints
	for (unsigned int i = 0; i < keypoints.size(); ++i)
	{
		// rectify the patch around a given keypoint
		Mat patch;
		Utils::rectifyPatch(image, keypoints[i], patchSize, patch);

		// compute gradient maps (and integral gradient maps)
		Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
		Utils::computeIntegrals(gradMap, orientQuant, integralMap);

		// compute the responses of the weak learners
		std::bitset<512> wlResponses;
		for (int j = 0; j < nWLs; ++j)
			wlResponses[j] = (Utils::computeWLResponse(pWLs[j], orientQuant, integralMap) >= 0) ? 1 : 0;

		float* desc = descriptors.ptr<float>(i);
		for (int d = 0; d < nDim; ++d)
		{
			for (int wl = 0; wl < nWLs; ++wl)
				desc[d] += (wlResponses[wl]) ? betas[wl*nDim + d] : -betas[wl*nDim + d];
		}

		// clean-up
		patch.release();
		gradMap.clear();
		integralMap.clear();
	}
}

/**********************************************************************************************/

void BinBoostDescriptorExtractor::readWLs()
{
	FILE* fid = fopen(wlFile.c_str(), "rt");
	fscanf(fid, "%d\n", &nDim);
	fscanf(fid, "%d\n", &orientQuant);
	fscanf(fid, "%d\n", &patchSize);
	int iGradAssignType;
	fscanf(fid, "%d\n", &iGradAssignType);
	gradAssignType = (Assign)iGradAssignType;
	pWLsArray = new WeakLearner*[nDim];
	for (int d = 0; d < nDim; ++d)
	{
		fscanf(fid, "%d\n", &nWLs);
		pWLsArray[d] = new WeakLearner[nWLs];
		for (int i = 0; i < nWLs; i++)
		{
			fscanf(fid, "%f %d %d %d %d %d %f %f\n",
				&pWLsArray[d][i].thresh,
				&pWLsArray[d][i].orient,
				&pWLsArray[d][i].y_min,
				&pWLsArray[d][i].y_max,
				&pWLsArray[d][i].x_min,
				&pWLsArray[d][i].x_max,
				&pWLsArray[d][i].alpha,
				&pWLsArray[d][i].beta);
		}
	}

	fclose(fid);
}

/**********************************************************************************************/

void BinBoostDescriptorExtractor::readWLsBin()
{
	FILE* fid = fopen(wlFile.c_str(), "rb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot read weak learners from '%s'.\n", wlFile.c_str());
		exit(-3);
	}
	fread(&nDim, sizeof(int), 1, fid);
	fread(&orientQuant, sizeof(int), 1, fid);
	fread(&patchSize, sizeof(int), 1, fid);
	int iGradAssignType;
	fread(&iGradAssignType, sizeof(int), 1, fid);
	gradAssignType = (Assign)iGradAssignType;
	pWLsArray = new WeakLearner*[nDim];
	for (int d = 0; d < nDim; ++d)
	{
		fread(&nWLs, sizeof(int), 1, fid);
		pWLsArray[d] = new WeakLearner[nWLs];
		for (int i = 0; i < nWLs; i++)
		{
			fread(&pWLsArray[d][i].thresh, sizeof(float), 1, fid);
			fread(&pWLsArray[d][i].orient, sizeof(int), 1, fid);
			fread(&pWLsArray[d][i].y_min, sizeof(int), 1, fid);
			fread(&pWLsArray[d][i].y_max, sizeof(int), 1, fid);
			fread(&pWLsArray[d][i].x_min, sizeof(int), 1, fid);
			fread(&pWLsArray[d][i].x_max, sizeof(int), 1, fid);
			fread(&pWLsArray[d][i].alpha, sizeof(float), 1, fid);
			fread(&pWLsArray[d][i].beta, sizeof(float), 1, fid);
		}
	}
	fclose(fid);
}

/**********************************************************************************************/

void BinBoostDescriptorExtractor::saveWLsBin(const string output) const
{
	FILE* fid = fopen(output.c_str(), "wb");
	if (!fid)
	{
		fprintf(stderr, "[ERROR] Cannot save weak learners to '%s'.\n", output.c_str());
		exit(-3);
	}
	fwrite(&nDim, sizeof(int), 1, fid);
	fwrite(&orientQuant, sizeof(int), 1, fid);
	fwrite(&patchSize, sizeof(int), 1, fid);
	fwrite(&gradAssignType, sizeof(int), 1, fid);
	for (int d = 0; d < nDim; ++d)
	{
		fwrite(&nWLs, sizeof(int), 1, fid);
		for (int i = 0; i < nWLs; i++)
		{
			fwrite(&pWLsArray[d][i].thresh, sizeof(float), 1, fid);
			fwrite(&pWLsArray[d][i].orient, sizeof(int), 1, fid);
			fwrite(&pWLsArray[d][i].y_min, sizeof(int), 1, fid);
			fwrite(&pWLsArray[d][i].y_max, sizeof(int), 1, fid);
			fwrite(&pWLsArray[d][i].x_min, sizeof(int), 1, fid);
			fwrite(&pWLsArray[d][i].x_max, sizeof(int), 1, fid);
			fwrite(&pWLsArray[d][i].alpha, sizeof(float), 1, fid);
			fwrite(&pWLsArray[d][i].beta, sizeof(float), 1, fid);
		}
	}
	fclose(fid);
}

/**********************************************************************************************/

void BinBoostDescriptorExtractor::compute(const Mat& image, vector<KeyPoint>& keypoints,
	Mat& descriptors) const
{
	// initialize the variables
	descriptors = Mat::zeros(keypoints.size(), ceil(nDim / 8), descriptorType());
	vector<Mat> gradMap, integralMap;

	// iterate through all the keypoints
	for (unsigned int i = 0; i < keypoints.size(); ++i)
	{
		// rectify the patch around a given keypoint
		Mat patch;
		cv::resize(image, patch, Size(32, 32));	
		Utils::rectifyPatch(image, keypoints[i], patchSize, patch);
		
		// compute gradient maps (and integral gradient maps)
		Utils::computeGradientMaps(patch, gradAssignType, orientQuant, gradMap);
		Utils::computeIntegrals(gradMap, orientQuant, integralMap);

		// compute the responses of the weak learners
		float resp;
		for (int d = 0; d < nDim; ++d)
		{
			uchar* desc = descriptors.ptr<uchar>(i);
			resp = 0;
			for (int wl = 0; wl < nWLs; wl++)
				resp += (Utils::computeWLResponse(pWLsArray[d][wl], orientQuant, integralMap) >= 0) ? pWLsArray[d][wl].beta : -pWLsArray[d][wl].beta;
			desc[d / 8] |= (resp >= 0) ? binLookUp[d % 8] : 0;
		}

		// clean-up
		patch.release();
		gradMap.clear();
		integralMap.clear();
	}
}


