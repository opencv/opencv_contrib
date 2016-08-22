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

#include "tldModel.hpp"

#include <opencv2/core/utility.hpp>

namespace cv
{
	namespace tld
	{
		//Constructor
		TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params, const Mat& image, const Rect2d& boundingBox, Size minSize):
			timeStampPositiveNext(0), timeStampNegativeNext(0), minSize_(minSize), params_(params), boundingBox_(boundingBox)
		{
			std::vector<Rect2d> closest, scanGrid;
			Mat scaledImg, blurredImg, image_blurred;

			//Create Detector
			detector = Ptr<TLDDetector>(new TLDDetector());

			//Propagate data to Detector
			posNum = 0;
			negNum = 0;
			posExp = Mat(Size(225, 500), CV_8UC1);
			negExp = Mat(Size(225, 500), CV_8UC1);
			detector->posNum = &posNum;
			detector->negNum = &negNum;
			detector->posExp = &posExp;
			detector->negExp = &negExp;

			detector->positiveExamples = &positiveExamples;
			detector->negativeExamples = &negativeExamples;
			detector->timeStampsPositive = &timeStampsPositive;
			detector->timeStampsNegative = &timeStampsNegative;
			detector->originalVariancePtr = &originalVariance_;

			//Calculate the variance in initial BB
			originalVariance_ = variance(image(boundingBox));
			//Find the scale
			double scale = scaleAndBlur(image, cvRound(log(1.0 * boundingBox.width / (minSize.width)) / log(SCALE_STEP)),
				scaledImg, blurredImg, GaussBlurKernelSize, SCALE_STEP);
			GaussianBlur(image, image_blurred, GaussBlurKernelSize, 0.0);
			TLDDetector::generateScanGrid(image.rows, image.cols, minSize_, scanGrid);
			getClosestN(scanGrid, Rect2d(boundingBox.x / scale, boundingBox.y / scale, boundingBox.width / scale, boundingBox.height / scale), 10, closest);
			Mat_<uchar> blurredPatch(minSize);
			TLDEnsembleClassifier::makeClassifiers(minSize, MEASURES_PER_CLASSIFIER, GRIDSIZE, detector->classifiers);

			//Generate initial positive samples and put them to the model
			positiveExamples.reserve(200);

			for (int i = 0; i < (int)closest.size(); i++)
			{
				for (int j = 0; j < 20; j++)
				{
					Point2f center;
					Size2f size;
					Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
					center.x = (float)(closest[i].x + closest[i].width * (0.5 + rng.uniform(-0.01, 0.01)));
					center.y = (float)(closest[i].y + closest[i].height * (0.5 + rng.uniform(-0.01, 0.01)));
					size.width = (float)(closest[i].width * rng.uniform((double)0.99, (double)1.01));
					size.height = (float)(closest[i].height * rng.uniform((double)0.99, (double)1.01));
					float angle = (float)rng.uniform(-10.0, 10.0);

					resample(scaledImg, RotatedRect(center, size, angle), standardPatch);

					for (int y = 0; y < standardPatch.rows; y++)
					{
						for (int x = 0; x < standardPatch.cols; x++)
						{
							standardPatch(x, y) += (uchar)rng.gaussian(5.0);
						}
					}

#ifdef BLUR_AS_VADIM
					GaussianBlur(standardPatch, blurredPatch, GaussBlurKernelSize, 0.0);
					resize(blurredPatch, blurredPatch, minSize);
#else
					resample(blurredImg, RotatedRect(center, size, angle), blurredPatch);
#endif
					pushIntoModel(standardPatch, true);
					for (int k = 0; k < (int)detector->classifiers.size(); k++)
						detector->classifiers[k].integrate(blurredPatch, true);
				}
			}

			//Generate initial negative samples and put them to the model
			TLDDetector::generateScanGrid(image.rows, image.cols, minSize, scanGrid, true);
			negativeExamples.clear();
			negativeExamples.reserve(NEG_EXAMPLES_IN_INIT_MODEL);
			std::vector<int> indices;
			indices.reserve(NEG_EXAMPLES_IN_INIT_MODEL);
			while ((int)negativeExamples.size() < NEG_EXAMPLES_IN_INIT_MODEL)
			{
				int i = rng.uniform((int)0, (int)scanGrid.size());
				if (std::find(indices.begin(), indices.end(), i) == indices.end() && overlap(boundingBox, scanGrid[i]) < NEXPERT_THRESHOLD)
				{
					Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE);
					resample(image, scanGrid[i], standardPatch);
					pushIntoModel(standardPatch, false);

					resample(image_blurred, scanGrid[i], blurredPatch);
					for (int k = 0; k < (int)detector->classifiers.size(); k++)
						detector->classifiers[k].integrate(blurredPatch, false);
				}
			}
		}


		void TrackerTLDModel::integrateRelabeled(Mat& img, Mat& imgBlurred, const std::vector<TLDDetector::LabeledPatch>& patches)
		{
			Mat_<uchar> standardPatch(STANDARD_PATCH_SIZE, STANDARD_PATCH_SIZE), blurredPatch(minSize_);
			int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
			for (int k = 0; k < (int)patches.size(); k++)
			{
				if (patches[k].shouldBeIntegrated)
				{
					resample(img, patches[k].rect, standardPatch);
					if (patches[k].isObject)
					{
						positiveIntoModel++;
						pushIntoModel(standardPatch, true);
					}
					else
					{
						negativeIntoModel++;
						pushIntoModel(standardPatch, false);
					}
				}

#ifdef CLOSED_LOOP
				if (patches[k].shouldBeIntegrated || !patches[k].isPositive)
#else
				if (patches[k].shouldBeIntegrated)
#endif
				{
					resample(imgBlurred, patches[k].rect, blurredPatch);
					if (patches[k].isObject)
						positiveIntoEnsemble++;
					else
						negativeIntoEnsemble++;
					for (int i = 0; i < (int)detector->classifiers.size(); i++)
						detector->classifiers[i].integrate(blurredPatch, patches[k].isObject);
				}
			}

		}

		class CalcSrParallelLoopBody: public cv::ParallelLoopBody
		{
		public:
			explicit CalcSrParallelLoopBody (TrackerTLDModel * model, const std::vector<Mat_<uchar> >& eForModel):
				modelF (model),
				eForModelF (eForModel)
			{
			}

			virtual void operator () (const cv::Range & r) const
			{
				for (int ind = r.start; ind < r.end; ++ind)
				{
					modelF->srValues[ind] = modelF->detector->Sr (eForModelF[ind]);
				}
			}

			TrackerTLDModel * modelF;
			const std::vector<Mat_<uchar> >& eForModelF;
		private:
			CalcSrParallelLoopBody (const CalcSrParallelLoopBody&);
			CalcSrParallelLoopBody& operator= (const CalcSrParallelLoopBody&);
		};

		void TrackerTLDModel::integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive)
		{
			int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
			if ((int)eForModel.size() == 0) return;

			srValues.resize (eForModel.size ());
			cv::parallel_for_ (cv::Range (0, (int)eForModel.size ()), CalcSrParallelLoopBody (this, eForModel));

			for (int k = 0; k < (int)eForModel.size(); k++)
			{
				const double sr = srValues[k];
				if ((sr > THETA_NN) != isPositive)
				{
					if (isPositive)
					{
						positiveIntoModel++;
						pushIntoModel(eForModel[k], true);
					}
					else
					{
						negativeIntoModel++;
						pushIntoModel(eForModel[k], false);
					}
				}
				double p = 0;
				for (int i = 0; i < (int)detector->classifiers.size(); i++)
					p += detector->classifiers[i].posteriorProbability(eForEnsemble[k].data, (int)eForEnsemble[k].step[0]);
				p /= detector->classifiers.size();
				if ((p > ENSEMBLE_THRESHOLD) != isPositive)
				{
					if (isPositive)
						positiveIntoEnsemble++;
					else
						negativeIntoEnsemble++;
					for (int i = 0; i < (int)detector->classifiers.size(); i++)
						detector->classifiers[i].integrate(eForEnsemble[k], isPositive);
				}
			}
		}

#ifdef HAVE_OPENCL
		void TrackerTLDModel::ocl_integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive)
		{
			int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
			if ((int)eForModel.size() == 0) return;

			//Prepare batch of patches
			int numOfPatches = (int)eForModel.size();
			Mat_<uchar> stdPatches(numOfPatches, 225);
			double *resultSr = new double[numOfPatches];
			double *resultSc = new double[numOfPatches];
			uchar *patchesData = stdPatches.data;
			for (int i = 0; i < numOfPatches; i++)
			{
				uchar *stdPatchData = eForModel[i].data;
				for (int j = 0; j < 225; j++)
					patchesData[225 * i + j] = stdPatchData[j];
			}

			//Calculate Sr and Sc batches
			detector->ocl_batchSrSc(stdPatches, resultSr, resultSc, numOfPatches);

			for (int k = 0; k < (int)eForModel.size(); k++)
			{
				double sr = resultSr[k];
				if ((sr > THETA_NN) != isPositive)
				{
					if (isPositive)
					{
						positiveIntoModel++;
						pushIntoModel(eForModel[k], true);
					}
					else
					{
						negativeIntoModel++;
						pushIntoModel(eForModel[k], false);
					}
				}
				double p = 0;
				for (int i = 0; i < (int)detector->classifiers.size(); i++)
					p += detector->classifiers[i].posteriorProbability(eForEnsemble[k].data, (int)eForEnsemble[k].step[0]);
				p /= detector->classifiers.size();
				if ((p > ENSEMBLE_THRESHOLD) != isPositive)
				{
					if (isPositive)
						positiveIntoEnsemble++;
					else
						negativeIntoEnsemble++;
					for (int i = 0; i < (int)detector->classifiers.size(); i++)
						detector->classifiers[i].integrate(eForEnsemble[k], isPositive);
				}
			}
		}
#endif // HAVE_OPENCL

		//Push the patch to the model
		void TrackerTLDModel::pushIntoModel(const Mat_<uchar>& example, bool positive)
		{
			std::vector<Mat_<uchar> >* proxyV;
			int* proxyN;
			std::vector<int>* proxyT;
			if (positive)
			{
				if (posNum < 500)
				{
					uchar *patchPtr = example.data;
					uchar *modelPtr = posExp.data;
					for (int i = 0; i < STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE; i++)
						modelPtr[posNum*STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE + i] = patchPtr[i];
					posNum++;
				}

				proxyV = &positiveExamples;
				proxyN = &timeStampPositiveNext;
				proxyT = &timeStampsPositive;
			}
			else
			{
				if (negNum < 500)
				{
					uchar *patchPtr = example.data;
					uchar *modelPtr = negExp.data;
					for (int i = 0; i < STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE; i++)
						modelPtr[negNum*STANDARD_PATCH_SIZE*STANDARD_PATCH_SIZE + i] = patchPtr[i];
					negNum++;
				}

				proxyV = &negativeExamples;
				proxyN = &timeStampNegativeNext;
				proxyT = &timeStampsNegative;
			}
			if ((int)proxyV->size() < MAX_EXAMPLES_IN_MODEL)
			{
				proxyV->push_back(example);
				proxyT->push_back(*proxyN);
			}
			else
			{
				int index = rng.uniform((int)0, (int)proxyV->size());
				(*proxyV)[index] = example;
				(*proxyT)[index] = (*proxyN);
			}
			(*proxyN)++;
		}

		void TrackerTLDModel::printme(FILE* port)
		{
			dfprintf((port, "TrackerTLDModel:\n"));
			dfprintf((port, "\tpositiveExamples.size() = %d\n", (int)positiveExamples.size()));
			dfprintf((port, "\tnegativeExamples.size() = %d\n", (int)negativeExamples.size()));
		}
	}
}
