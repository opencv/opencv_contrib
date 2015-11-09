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

			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}

#ifdef HAVE_OPENCL
		double TLDDetector::ocl_Sr(const Mat_<uchar>& patch)
		{
			double splus = 0.0, sminus = 0.0;


			UMat devPatch = patch.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPositiveSamples = posExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegativeSamples = negExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNCC(1, 2*MAX_EXAMPLES_IN_MODEL, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);


			ocl::Kernel k;
			ocl::ProgramSource src = ocl::tracking::tldDetector_oclsrc;
			String error;
			ocl::Program prog(src, String(), error);
			k.create("NCC", prog);
			if (k.empty())
				printf("Kernel create failed!!!\n");
			k.args(
				ocl::KernelArg::PtrReadOnly(devPatch),
				ocl::KernelArg::PtrReadOnly(devPositiveSamples),
				ocl::KernelArg::PtrReadOnly(devNegativeSamples),
				ocl::KernelArg::PtrWriteOnly(devNCC),
				*posNum,
				*negNum);

			size_t globSize = 1000;

			if (!k.run(1, &globSize, NULL, false))
				printf("Kernel Run Error!!!");

			Mat resNCC = devNCC.getMat(ACCESS_READ);

			for (int i = 0; i < *posNum; i++)
				splus = std::max(splus, 0.5 * (resNCC.at<float>(i) + 1.0));

			for (int i = 0; i < *negNum; i++)
				sminus = std::max(sminus, 0.5 * (resNCC.at<float>(i+500) +1.0));

			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}

		void TLDDetector::ocl_batchSrSc(const Mat_<uchar>& patches, double *resultSr, double *resultSc, int numOfPatches)
		{
			UMat devPatches = patches.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPositiveSamples = posExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegativeSamples = negExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPosNCC(MAX_EXAMPLES_IN_MODEL, numOfPatches, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegNCC(MAX_EXAMPLES_IN_MODEL, numOfPatches, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);

			ocl::Kernel k;
			ocl::ProgramSource src = ocl::tracking::tldDetector_oclsrc;
			String error;
			ocl::Program prog(src, String(), error);
			k.create("batchNCC", prog);
			if (k.empty())
				printf("Kernel create failed!!!\n");
			k.args(
				ocl::KernelArg::PtrReadOnly(devPatches),
				ocl::KernelArg::PtrReadOnly(devPositiveSamples),
				ocl::KernelArg::PtrReadOnly(devNegativeSamples),
				ocl::KernelArg::PtrWriteOnly(devPosNCC),
				ocl::KernelArg::PtrWriteOnly(devNegNCC),
				*posNum,
				*negNum,
				numOfPatches);

			size_t globSize = 2 * numOfPatches*MAX_EXAMPLES_IN_MODEL;

			if (!k.run(1, &globSize, NULL, true))
				printf("Kernel Run Error!!!");

			Mat posNCC = devPosNCC.getMat(ACCESS_READ);
			Mat negNCC = devNegNCC.getMat(ACCESS_READ);

			//Calculate Srs
			for (int id = 0; id < numOfPatches; id++)
			{
				double spr = 0.0, smr = 0.0, spc = 0.0, smc = 0;
				int med = getMedian((*timeStampsPositive));
				for (int i = 0; i < *posNum; i++)
				{
					spr = std::max(spr, 0.5 * (posNCC.at<float>(id * 500 + i) + 1.0));
					if ((int)(*timeStampsPositive)[i] <= med)
						spc = std::max(spr, 0.5 * (posNCC.at<float>(id * 500 + i) + 1.0));
				}
				for (int i = 0; i < *negNum; i++)
					smc = smr = std::max(smr, 0.5 * (negNCC.at<float>(id * 500 + i) + 1.0));

				if (spr + smr == 0.0)
					resultSr[id] = 0.0;
				else
					resultSr[id] = spr / (smr + spr);

				if (spc + smc == 0.0)
					resultSc[id] = 0.0;
				else
					resultSc[id] = spc / (smc + spc);
			}
		}
#endif

		// Calculate Conservative similarity of the patch (NN-Model)
		double TLDDetector::Sc(const Mat_<uchar>& patch)
		{
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

			if (splus + sminus == 0.0)
				return 0.0;

			return splus / (sminus + splus);
		}

#ifdef HAVE_OPENCL
		double TLDDetector::ocl_Sc(const Mat_<uchar>& patch)
		{
			double splus = 0.0, sminus = 0.0;

			UMat devPatch = patch.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devPositiveSamples = posExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNegativeSamples = negExp->getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
			UMat devNCC(1, 2 * MAX_EXAMPLES_IN_MODEL, CV_32FC1, ACCESS_RW, USAGE_ALLOCATE_DEVICE_MEMORY);


			ocl::Kernel k;
			ocl::ProgramSource src = ocl::tracking::tldDetector_oclsrc;
			String error;
			ocl::Program prog(src, String(), error);
			k.create("NCC", prog);
			if (k.empty())
				printf("Kernel create failed!!!\n");
			k.args(
				ocl::KernelArg::PtrReadOnly(devPatch),
				ocl::KernelArg::PtrReadOnly(devPositiveSamples),
				ocl::KernelArg::PtrReadOnly(devNegativeSamples),
				ocl::KernelArg::PtrWriteOnly(devNCC),
				*posNum,
				*negNum);

			size_t globSize = 1000;

			if (!k.run(1, &globSize, NULL, false))
				printf("Kernel Run Error!!!");

			Mat resNCC = devNCC.getMat(ACCESS_READ);

			int med = getMedian((*timeStampsPositive));
			for (int i = 0; i < *posNum; i++)
				if ((int)(*timeStampsPositive)[i] <= med)
					splus = std::max(splus, 0.5 * (resNCC.at<float>(i) +1.0));

			for (int i = 0; i < *negNum; i++)
				sminus = std::max(sminus, 0.5 * (resNCC.at<float>(i + 500) + 1.0));

			if (splus + sminus == 0.0)
				return 0.0;
			return splus / (sminus + splus);
		}
#endif // HAVE_OPENCL

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
		}

		//Detection - returns most probable new target location (Max Sc)

		bool TLDDetector::detect(const Mat& img, const Mat& imgBlurred, Rect2d& res, std::vector<LabeledPatch>& patches, Size initSize)
		{
			patches.clear();
			Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			Mat tmp;
			int dx = initSize.width / 10, dy = initSize.height / 10;
			Size2d size = img.size();
			double scale = 1.0;
			int npos = 0, nneg = 0;
			double maxSc = -5.0;
			Rect2d maxScRect;
			int scaleID;
			std::vector <Mat> resized_imgs, blurred_imgs;
			std::vector <Point> varBuffer, ensBuffer;
			std::vector <int> varScaleIDs, ensScaleIDs;

			//Detection part
			//Generate windows and filter by variance
			scaleID = 0;
			resized_imgs.push_back(img);
			blurred_imgs.push_back(imgBlurred);
			do
			{
				Mat_<double> intImgP, intImgP2;
				computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);
				for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
				{
					for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
					{
						if (!patchVariance(intImgP, intImgP2, originalVariancePtr, Point(dx * i, dy * j), initSize))
							continue;
						varBuffer.push_back(Point(dx * i, dy * j));
						varScaleIDs.push_back(scaleID);
					}
				}
				scaleID++;
				size.width /= SCALE_STEP;
				size.height /= SCALE_STEP;
				scale *= SCALE_STEP;
				resize(img, tmp, size, 0, 0, DOWNSCALE_MODE);
				resized_imgs.push_back(tmp);
				GaussianBlur(resized_imgs[scaleID], tmp, GaussBlurKernelSize, 0.0f);
				blurred_imgs.push_back(tmp);
			} while (size.width >= initSize.width && size.height >= initSize.height);

			//Encsemble classification
			for (int i = 0; i < (int)varBuffer.size(); i++)
			{
				prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[i]].step[0]));
				if (ensembleClassifierNum(&blurred_imgs[varScaleIDs[i]].at<uchar>(varBuffer[i].y, varBuffer[i].x)) <= ENSEMBLE_THRESHOLD)
					continue;
				ensBuffer.push_back(varBuffer[i]);
				ensScaleIDs.push_back(varScaleIDs[i]);
			}

			//NN classification
			for (int i = 0; i < (int)ensBuffer.size(); i++)
			{
				LabeledPatch labPatch;
				double curScale = pow(SCALE_STEP, ensScaleIDs[i]);
				labPatch.rect = Rect2d(ensBuffer[i].x*curScale, ensBuffer[i].y*curScale, initSize.width * curScale, initSize.height * curScale);
				resample(resized_imgs[ensScaleIDs[i]], Rect2d(ensBuffer[i], initSize), standardPatch);

				double srValue, scValue;
				srValue = Sr(standardPatch);

				////To fix: Check the paper, probably this cause wrong learning
				//
				labPatch.isObject = srValue > THETA_NN;
				labPatch.shouldBeIntegrated = abs(srValue - THETA_NN) < 0.1;
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
				scValue = Sc(standardPatch);
				if (scValue > maxSc)
				{
					maxSc = scValue;
					maxScRect = labPatch.rect;
				}
			}

			if (maxSc < 0)
				return false;
			else
			{
				res = maxScRect;
				return true;
			}
		}

#ifdef HAVE_OPENCL
		bool TLDDetector::ocl_detect(const Mat& img, const Mat& imgBlurred, Rect2d& res, std::vector<LabeledPatch>& patches, Size initSize)
		{
			patches.clear();
			Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
			Mat tmp;
			int dx = initSize.width / 10, dy = initSize.height / 10;
			Size2d size = img.size();
			double scale = 1.0;
			int npos = 0, nneg = 0;
			double maxSc = -5.0;
			Rect2d maxScRect;
			int scaleID;
			std::vector <Mat> resized_imgs, blurred_imgs;
			std::vector <Point> varBuffer, ensBuffer;
			std::vector <int> varScaleIDs, ensScaleIDs;

			//Detection part
			//Generate windows and filter by variance
			scaleID = 0;
			resized_imgs.push_back(img);
			blurred_imgs.push_back(imgBlurred);
			do
			{
				Mat_<double> intImgP, intImgP2;
				computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);
				for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
				{
					for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
					{
						if (!patchVariance(intImgP, intImgP2, originalVariancePtr, Point(dx * i, dy * j), initSize))
							continue;
						varBuffer.push_back(Point(dx * i, dy * j));
						varScaleIDs.push_back(scaleID);
					}
				}
				scaleID++;
				size.width /= SCALE_STEP;
				size.height /= SCALE_STEP;
				scale *= SCALE_STEP;
				resize(img, tmp, size, 0, 0, DOWNSCALE_MODE);
				resized_imgs.push_back(tmp);
				GaussianBlur(resized_imgs[scaleID], tmp, GaussBlurKernelSize, 0.0f);
				blurred_imgs.push_back(tmp);
			} while (size.width >= initSize.width && size.height >= initSize.height);

			//Encsemble classification
			for (int i = 0; i < (int)varBuffer.size(); i++)
			{
				prepareClassifiers((int)blurred_imgs[varScaleIDs[i]].step[0]);
				if (ensembleClassifierNum(&blurred_imgs[varScaleIDs[i]].at<uchar>(varBuffer[i].y, varBuffer[i].x)) <= ENSEMBLE_THRESHOLD)
					continue;
				ensBuffer.push_back(varBuffer[i]);
				ensScaleIDs.push_back(varScaleIDs[i]);
			}

			//NN classification
			//Prepare batch of patches
			int numOfPatches = (int)ensBuffer.size();
			Mat_<uchar> stdPatches(numOfPatches, 225);
			double *resultSr = new double[numOfPatches];
			double *resultSc = new double[numOfPatches];

			uchar *patchesData = stdPatches.data;
			for (int i = 0; i < (int)ensBuffer.size(); i++)
			{
				resample(resized_imgs[ensScaleIDs[i]], Rect2d(ensBuffer[i], initSize), standardPatch);
				uchar *stdPatchData = standardPatch.data;
				for (int j = 0; j < 225; j++)
					patchesData[225*i+j] = stdPatchData[j];
			}
			//Calculate Sr and Sc batches
			ocl_batchSrSc(stdPatches, resultSr, resultSc, numOfPatches);


			for (int i = 0; i < (int)ensBuffer.size(); i++)
			{
				LabeledPatch labPatch;
				standardPatch.data = &stdPatches.data[225 * i];
				double curScale = pow(SCALE_STEP, ensScaleIDs[i]);
				labPatch.rect = Rect2d(ensBuffer[i].x*curScale, ensBuffer[i].y*curScale, initSize.width * curScale, initSize.height * curScale);

				double srValue, scValue;

				srValue = resultSr[i];

				////To fix: Check the paper, probably this cause wrong learning
				//
				labPatch.isObject = srValue > THETA_NN;
				labPatch.shouldBeIntegrated = abs(srValue - THETA_NN) < 0.1;
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
				scValue = resultSc[i];
				if (scValue > maxSc)
				{
					maxSc = scValue;
					maxScRect = labPatch.rect;
				}
			}

			if (maxSc < 0)
				return false;
			res = maxScRect;
			return true;
		}
#endif // HAVE_OPENCL

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