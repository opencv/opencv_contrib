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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "tldDetector.hpp"

namespace cv
{
	namespace tld
	{
		// Calculate offsets for classifiers
		void TLDDetector::prepareClassifiers(int rowstep)
		{
			for (int i = 0; i < (int)classifiers.size(); i++)
				classifiers[i].prepareClassifier(rowstep);
		}

		// Calculate posterior probability, that the patch belongs to the current EC model
		double TLDDetector::ensembleClassifierNum(const uchar* data)
		{
			double p = 0;
			for (int k = 0; k < (int)classifiers.size(); k++)
				p += classifiers[k].posteriorProbabilityFast(data);
			p /= classifiers.size();
			return p;
		}

		// Calculate Relative similarity of the patch (NN-Model)
		double TLDDetector::Sr(const Mat_<uchar>& patch)
		{
			/*
			int64 e1, e2;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;
			for (int i = 0; i < (int)(*positiveExamples).size(); i++)
				splus = std::max(splus, 0.5 * (NCC((*positiveExamples)[i], patch) + 1.0));
			for (int i = 0; i < (int)(*negativeExamples).size(); i++)
				sminus = std::max(sminus, 0.5 * (NCC((*negativeExamples)[i], patch) + 1.0));
			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			printf("Sr: %f\n", t);
			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
			*/
			int64 e1, e2;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;
			Mat_<uchar> modelSample(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			for (int i = 0; i < *posNum; i++)
			{
				modelSample.data = &(posExp->data[i * 225]);
				splus = std::max(splus, 0.5 * (NCC(modelSample, patch) + 1.0));
			}
			for (int i = 0; i < *negNum; i++)
			{
				modelSample.data = &(negExp->data[i * 225]);
				sminus = std::max(sminus, 0.5 * (NCC(modelSample, patch) + 1.0));
			}
			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			printf("Sr CPU: %f\n", t);
			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}

		double TLDDetector::ocl_Sr(const Mat_<uchar>& patch)
		{
			int64 e1, e2, e3, e4;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;
			
			e3 = getTickCount();

			UMat devPatch = patch.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPositiveSamples = posExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegativeSamples = negExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNCC(1, 2*MAX_EXAMPLES_IN_MODEL, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);

			
			ocl::Kernel k;
			ocl::ProgramSource src = ocl::tracking::tldDetector_oclsrc;
			String error;
			ocl::Program prog(src, NULL, error);
			k.create("NCC", prog);
			if (k.empty())
				printf("Kernel create failed!!!\n");
			k.args(
				ocl::KernelArg::PtrReadOnly(devPatch),
				ocl::KernelArg::PtrReadOnly(devPositiveSamples),
				ocl::KernelArg::PtrReadOnly(devNegativeSamples),
				ocl::KernelArg::PtrWriteOnly(devNCC),
				(int)posNum,
				(int)negNum);

			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Mem Cpy GPU: %f\n", t);

			size_t globSize = 1000;
			size_t localSize = 128;		
			e3 = getTickCount();
			if (!k.run(1, &globSize, &localSize, true))
				printf("Kernel Run Error!!!");
			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Kernel Run GPU: %f\n", t);

			e3 = getTickCount();
			Mat resNCC = devNCC.getMat(ACCESS_READ);
			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Read Mem GPU: %f\n", t);
			
			////Compare
			//Mat_<uchar> modelSample(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			//for (int i = 0; i < 200; i+=17)
			//{
			//	modelSample.data = &(posExp->data[i * 225]);
			//	printf("%f\t%f\n\n", resNCC.at<float>(i), NCC(modelSample, patch));
			//}

			//for (int i = 0; i < 200; i+=23)
			//{
			//	modelSample.data = &(negExp->data[i * 225]);
			//	printf("%f\t%f\n", resNCC.at<float>(500+i), NCC(modelSample, patch));
			//}

			
			for (int i = 0; i < *posNum; i++)
				splus = std::max(splus, 0.5 * (resNCC.at<float>(i) + 1.0));

			for (int i = 0; i < *negNum; i++)
				sminus = std::max(sminus, 0.5 * (resNCC.at<float>(i+500) +1.0));

			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			//printf("Sr GPU: %f\n\n", t);

			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}

		// Calculate Conservative similarity of the patch (NN-Model)
		double TLDDetector::Sc(const Mat_<uchar>& patch)
		{
			/*
			int64 e1, e2;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;
			int med = getMedian((*timeStampsPositive));
			for (int i = 0; i < (int)(*positiveExamples).size(); i++)
			{
				if ((int)(*timeStampsPositive)[i] <= med)
					splus = std::max(splus, 0.5 * (NCC((*positiveExamples)[i], patch) + 1.0));
			}
			for (int i = 0; i < (int)(*negativeExamples).size(); i++)
				sminus = std::max(sminus, 0.5 * (NCC((*negativeExamples)[i], patch) + 1.0));
			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			printf("Sc: %f\n", t);
			if (splus + sminus == 0.0)
				return 0.0;
			
			return splus / (sminus + splus);
			*/

			int64 e1, e2;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;
			Mat_<uchar> modelSample(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			int med = getMedian((*timeStampsPositive));
			for (int i = 0; i < *posNum; i++)
			{
				if ((int)(*timeStampsPositive)[i] <= med)
				{
					modelSample.data = &(posExp->data[i * 225]);
					splus = std::max(splus, 0.5 * (NCC(modelSample, patch) + 1.0));
				}
			}
			for (int i = 0; i < *negNum; i++)
			{
				modelSample.data = &(negExp->data[i * 225]);
				sminus = std::max(sminus, 0.5 * (NCC(modelSample, patch) + 1.0));
			}
			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			printf("Sc: %f\n", t);
			if (splus + sminus == 0.0)
				return 0.0;

			return splus / (sminus + splus);
		}

		double TLDDetector::ocl_Sc(const Mat_<uchar>& patch)
		{
			int64 e1, e2, e3, e4;
			float t;
			e1 = getTickCount();
			double splus = 0.0, sminus = 0.0;

			e3 = getTickCount();

			UMat devPatch = patch.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPositiveSamples = posExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegativeSamples = negExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNCC(1, 2 * MAX_EXAMPLES_IN_MODEL, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);


			ocl::Kernel k;
			ocl::ProgramSource src = ocl::tracking::tldDetector_oclsrc;
			String error;
			ocl::Program prog(src, NULL, error);
			k.create("NCC", prog);
			if (k.empty())
				printf("Kernel create failed!!!\n");
			k.args(
				ocl::KernelArg::PtrReadOnly(devPatch),
				ocl::KernelArg::PtrReadOnly(devPositiveSamples),
				ocl::KernelArg::PtrReadOnly(devNegativeSamples),
				ocl::KernelArg::PtrWriteOnly(devNCC),
				(int)posNum,
				(int)negNum);

			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Mem Cpy GPU: %f\n", t);

			size_t globSize = 1000;
			size_t localSize = 128;
			e3 = getTickCount();
			if (!k.run(1, &globSize, &localSize, true))
				printf("Kernel Run Error!!!");
			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Kernel Run GPU: %f\n", t);

			e3 = getTickCount();
			Mat resNCC = devNCC.getMat(ACCESS_READ);
			e4 = getTickCount();
			t = (e4 - e3) / getTickFrequency()*1000.0;
			//printf("Read Mem GPU: %f\n", t);

			////Compare
			//Mat_<uchar> modelSample(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			//for (int i = 0; i < 200; i+=17)
			//{
			//	modelSample.data = &(posExp->data[i * 225]);
			//	printf("%f\t%f\n\n", resNCC.at<float>(i), NCC(modelSample, patch));
			//}

			//for (int i = 0; i < 200; i+=23)
			//{
			//	modelSample.data = &(negExp->data[i * 225]);
			//	printf("%f\t%f\n", resNCC.at<float>(500+i), NCC(modelSample, patch));
			//}

			int med = getMedian((*timeStampsPositive));
			for (int i = 0; i < *posNum; i++)
				if ((int)(*timeStampsPositive)[i] <= med)
					splus = std::max(splus, 0.5 * (resNCC.at<float>(i) +1.0));

			for (int i = 0; i < *negNum; i++)
				sminus = std::max(sminus, 0.5 * (resNCC.at<float>(i + 500) + 1.0));

			e2 = getTickCount();
			t = (e2 - e1) / getTickFrequency()*1000.0;
			//printf("Sc GPU: %f\n\n", t);

			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}

		// Generate Search Windows for detector from aspect ratio of initial BBs
		void TLDDetector::generateScanGrid(int rows, int cols, Size initBox, std::vector<Rect2d>& res, bool withScaling)
		{
			res.clear();
			//Scales step: SCALE_STEP; Translation steps: 10% of width & 10% of height; minSize: 20pix
			for (double h = initBox.height, w = initBox.width; h < cols && w < rows;)
			{
				for (double x = 0; (x + w + 1.0) <= cols; x += (0.1 * w))
				{
					for (double y = 0; (y + h + 1.0) <= rows; y += (0.1 * h))
						res.push_back(Rect2d(x, y, w, h));
				}
				if (withScaling)
				{
					if (h <= initBox.height)
					{
						h /= SCALE_STEP; w /= SCALE_STEP;
						if (h < 20 || w < 20)
						{
							h = initBox.height * SCALE_STEP; w = initBox.width * SCALE_STEP;
							CV_Assert(h > initBox.height || w > initBox.width);
						}
					}
					else
					{
						h *= SCALE_STEP; w *= SCALE_STEP;
					}
				}
				else
				{
					break;
				}
			}
			//dprintf(("%d rects in res\n", (int)res.size()));
		}

		//Detection - returns most probable new target location (Max Sc)
		bool TLDDetector::detect(const Mat& img, const Mat& imgBlurred, Rect2d& res, std::vector<LabeledPatch>& patches, Size initSize)
		{
			patches.clear();

			Mat resized_img, blurred_img;
			Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			img.copyTo(resized_img);
			imgBlurred.copyTo(blurred_img);
			int dx = initSize.width / 10, dy = initSize.height / 10;
			Size2d size = img.size();
			double scale = 1.0;
			int total = 0, pass = 0;
			int npos = 0, nneg = 0;
			double tmp = 0, maxSc = -5.0;
			Rect2d maxScRect;

			//Detection part
			do
			{
				Mat_<double> intImgP, intImgP2;
				computeIntegralImages(resized_img, intImgP, intImgP2);

				prepareClassifiers((int)blurred_img.step[0]);
				for (int i = 0, imax = cvFloor((0.0 + resized_img.cols - initSize.width) / dx); i < imax; i++)
				{
					for (int j = 0, jmax = cvFloor((0.0 + resized_img.rows - initSize.height) / dy); j < jmax; j++)
					{
						LabeledPatch labPatch;
						total++;
						if (!patchVariance(intImgP, intImgP2, originalVariancePtr, Point(dx * i, dy * j), initSize))
							continue;
						if (ensembleClassifierNum(&blurred_img.at<uchar>(dy * j, dx * i)) <= ENSEMBLE_THRESHOLD)
							continue;
						pass++;

						labPatch.rect = Rect2d(dx * i * scale, dy * j * scale, initSize.width * scale, initSize.height * scale);
						resample(resized_img, Rect2d(Point(dx * i, dy * j), initSize), standardPatch);
						
						tmp = ocl_Sr(standardPatch);

						////To fix: Check the paper, probably this cause wrong learning
						//
						labPatch.isObject = tmp > THETA_NN;
						labPatch.shouldBeIntegrated = abs(tmp - THETA_NN) < 0.1;
						patches.push_back(labPatch);
						//

						if (!labPatch.isObject)
						{
							nneg++;
							continue;
						}
						else
						{
							npos++;
						}
						tmp = ocl_Sc(standardPatch);
						if (tmp > maxSc)
						{
							maxSc = tmp;
							maxScRect = labPatch.rect;
						}
					}
				}

				size.width /= SCALE_STEP;
				size.height /= SCALE_STEP;
				scale *= SCALE_STEP;
				resize(img, resized_img, size, 0, 0, DOWNSCALE_MODE);
				GaussianBlur(resized_img, blurred_img, GaussBlurKernelSize, 0.0f);
			} while (size.width >= initSize.width && size.height >= initSize.height);

			if (maxSc < 0)
				return false;
			res = maxScRect;
			return true;
		}

		// Computes the variance of subimage given by box, with the help of two integral
		// images intImgP and intImgP2 (sum of squares), which should be also provided.
		bool TLDDetector::patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, double *originalVariance, Point pt, Size size)
		{
			int x = (pt.x), y = (pt.y), width = (size.width), height = (size.height);
			CV_Assert(0 <= x && (x + width) < intImgP.cols && (x + width) < intImgP2.cols);
			CV_Assert(0 <= y && (y + height) < intImgP.rows && (y + height) < intImgP2.rows);
			double p = 0, p2 = 0;
			double A, B, C, D;

			A = intImgP(y, x);
			B = intImgP(y, x + width);
			C = intImgP(y + height, x);
			D = intImgP(y + height, x + width);
			p = (A + D - B - C) / (width * height);

			A = intImgP2(y, x);
			B = intImgP2(y, x + width);
			C = intImgP2(y + height, x);
			D = intImgP2(y + height, x + width);
			p2 = (A + D - B - C) / (width * height);

			return ((p2 - p * p) > VARIANCE_THRESHOLD * *originalVariance);
		}

	}
}