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

#include "precomp.hpp"
#include "multiTracker.hpp"

#include "opencv2/tracking/tracking_legacy.hpp"

namespace cv {
namespace legacy {
inline namespace tracking {

using namespace impl;

	//Multitracker
    bool MultiTracker_Alt::addTarget(InputArray image, const Rect2d& boundingBox, Ptr<Tracker> tracker_algorithm)
	{
        Ptr<Tracker> tracker = tracker_algorithm;
		if (!tracker)
			return false;

		if (!tracker->init(image, boundingBox))
			return false;

		//Add BB of target
		boundingBoxes.push_back(boundingBox);

		//Add Tracker to stack
		trackers.push_back(tracker);

		//Assign a random color to target
		if (targetNum == 1)
			colors.push_back(Scalar(0, 0, 255));
		else
			colors.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));



		//Target counter
		targetNum++;

		return true;
	}

    bool MultiTracker_Alt::update(InputArray image)
	{
		for (int i = 0; i < (int)trackers.size(); i++)
			if (!trackers[i]->update(image, boundingBoxes[i]))
				return false;

		return true;
	}

	//Multitracker TLD
	/*Optimized update method for TLD Multitracker */
    bool MultiTrackerTLD::update_opt(InputArray _image)
	{
        Mat image = _image.getMat();
		//Get parameters from first object
		//TLD Tracker data extraction
		Tracker* trackerPtr = trackers[0];
		tld::TrackerTLDImpl* tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
		//TLD Model Extraction
		tld::TrackerTLDModel* tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));
		Ptr<tld::Data> data = tracker->data;
		double scale = data->getScale();

		Mat image_gray, image_blurred, imageForDetector;
		cvtColor(image, image_gray, COLOR_BGR2GRAY);

		if (scale > 1.0)
            resize(image_gray, imageForDetector, Size(cvRound(image_gray.cols*scale), cvRound(image_gray.rows*scale)), 0, 0, tld::DOWNSCALE_MODE);
		else
			imageForDetector = image_gray;
		GaussianBlur(imageForDetector, image_blurred, tld::GaussBlurKernelSize, 0.0);

		//best overlap around 92%
		Mat_<uchar> standardPatch(tld::STANDARD_PATCH_SIZE, tld::STANDARD_PATCH_SIZE);

		std::vector<std::vector<tld::TLDDetector::LabeledPatch> > detectorResults(targetNum);
		std::vector<std::vector<Rect2d> > candidates(targetNum);
		std::vector<std::vector<double> > candidatesRes(targetNum);
		std::vector<Rect2d> tmpCandidates(targetNum);
		std::vector<bool> detect_flgs(targetNum);
		std::vector<bool> trackerNeedsReInit(targetNum);

		bool DETECT_FLG = false;

		//Detect all
		for (int k = 0; k < targetNum; k++)
			tmpCandidates[k] = boundingBoxes[k];
#ifdef HAVE_OPENCL
		if (ocl::haveOpenCL())
			ocl_detect_all(imageForDetector, image_blurred, tmpCandidates, detectorResults, detect_flgs, trackers);
		else
#endif
			detect_all(imageForDetector, image_blurred, tmpCandidates, detectorResults, detect_flgs, trackers);

		bool success = false;
		for (int k = 0; k < targetNum; k++)
		{
			//TLD Tracker data extraction
			trackerPtr = trackers[k];
			tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
			//TLD Model Extraction
			tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));
			data = tracker->data;

			data->frameNum++;

			for (int i = 0; i < 2; i++)
			{
				Rect2d tmpCandid = boundingBoxes[k];

				//if (i == 1)
				{
					DETECT_FLG = detect_flgs[k];
					tmpCandid = tmpCandidates[k];
				}

				if (((i == 0) && !data->failedLastTime && tracker->trackerProxy->update(image, tmpCandid)) || (DETECT_FLG))
				{
					candidates[k].push_back(tmpCandid);
					if (i == 0)
						tld::resample(image_gray, tmpCandid, standardPatch);
					else
						tld::resample(imageForDetector, tmpCandid, standardPatch);
					candidatesRes[k].push_back(tldModel->detector->Sc(standardPatch));
				}
				else
				{
					if (i == 0)
						trackerNeedsReInit[k] = true;
					else
						trackerNeedsReInit[k] = false;
				}
			}

			std::vector<double>::iterator it = std::max_element(candidatesRes[k].begin(), candidatesRes[k].end());


			if (it == candidatesRes[k].end())
			{

				data->confident = false;
				data->failedLastTime = true;
				continue;
			}
			else
			{
				success = true;
				boundingBoxes[k] = candidates[k][it - candidatesRes[k].begin()];
				data->failedLastTime = false;
				if (trackerNeedsReInit[k] || it != candidatesRes[k].begin())
					tracker->trackerProxy->init(image, boundingBoxes[k]);
			}

#if 1
			if (it != candidatesRes[k].end())
				tld::resample(imageForDetector, candidates[k][it - candidatesRes[k].begin()], standardPatch);
#endif

			if (*it > tld::CORE_THRESHOLD)
				data->confident = true;

			if (data->confident)
			{
				tld::TrackerTLDImpl::Pexpert pExpert(imageForDetector, image_blurred, boundingBoxes[k], tldModel->detector, tracker->params, data->getMinSize());
				tld::TrackerTLDImpl::Nexpert nExpert(imageForDetector, boundingBoxes[k], tldModel->detector, tracker->params);
				std::vector<Mat_<uchar> > examplesForModel, examplesForEnsemble;
				examplesForModel.reserve(100); examplesForEnsemble.reserve(100);
				int negRelabeled = 0;
				for (int i = 0; i < (int)detectorResults[k].size(); i++)
				{
					bool expertResult;
					if (detectorResults[k][i].isObject)
					{
						expertResult = nExpert(detectorResults[k][i].rect);
						if (expertResult != detectorResults[k][i].isObject)
							negRelabeled++;
					}
					else
					{
						expertResult = pExpert(detectorResults[k][i].rect);
					}

					detectorResults[k][i].shouldBeIntegrated = detectorResults[k][i].shouldBeIntegrated || (detectorResults[k][i].isObject != expertResult);
					detectorResults[k][i].isObject = expertResult;
				}
				tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults[k]);
				pExpert.additionalExamples(examplesForModel, examplesForEnsemble);
#ifdef HAVE_OPENCL
				if (ocl::haveOpenCL())
					tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, true);
				else
#endif
					tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, true);
				examplesForModel.clear(); examplesForEnsemble.clear();
				nExpert.additionalExamples(examplesForModel, examplesForEnsemble);

#ifdef HAVE_OPENCL
				if (ocl::haveOpenCL())
					tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, false);
				else
#endif
					tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, false);
			}
			else
			{
#ifdef CLOSED_LOOP
				tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults);
#endif
			}


		}

		return success;
	}

}}  // namespace


inline namespace tracking {
namespace impl {

	void detect_all(const Mat& img, const Mat& imgBlurred, std::vector<Rect2d>& res, std::vector < std::vector < tld::TLDDetector::LabeledPatch > > &patches, std::vector<bool> &detect_flgs,
		std::vector<Ptr<legacy::Tracker> > &trackers)
	{
		//TLD Tracker data extraction
		legacy::Tracker* trackerPtr = trackers[0];
		tld::TrackerTLDImpl* tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
		//TLD Model Extraction
		tld::TrackerTLDModel* tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));
		Size initSize = tldModel->getMinSize();

		for (int k = 0; k < (int)trackers.size(); k++)
			patches[k].clear();

		Mat_<uchar> standardPatch(tld::STANDARD_PATCH_SIZE, tld::STANDARD_PATCH_SIZE);
		Mat tmp;
		int dx = initSize.width / 10, dy = initSize.height / 10;
		Size2d size = img.size();
		double scale = 1.0;
		int npos = 0, nneg = 0;
		double maxSc = -5.0;
		Rect2d maxScRect;
		int scaleID;
		std::vector <Mat> resized_imgs, blurred_imgs;

		std::vector <std::vector <Point> > varBuffer(trackers.size()), ensBuffer(trackers.size());
		std::vector <std::vector <int> > varScaleIDs(trackers.size()), ensScaleIDs(trackers.size());

		std::vector <Point> tmpP;
		std::vector <int> tmpI;

		//Detection part
		//Generate windows and filter by variance
		scaleID = 0;
		resized_imgs.push_back(img);
		blurred_imgs.push_back(imgBlurred);
		do
		{
			Mat_<double> intImgP, intImgP2;
			tld::TLDDetector::computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);
			for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
			{
				for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
				{
					//Optimized variance calculation
					int x = dx * i,
						y = dy * j,
						width = initSize.width,
						height = initSize.height;
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
					double windowVar = p2 - p * p;

					//Loop for on all objects
					for (int k = 0; k < (int)trackers.size(); k++)
					{
						//TLD Tracker data extraction
						trackerPtr = trackers[k];
						tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
						//TLD Model Extraction
						tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));

						//Optimized variance calculation
						bool varPass = (windowVar > tld::VARIANCE_THRESHOLD * *tldModel->detector->originalVariancePtr);

						if (!varPass)
							continue;
						varBuffer[k].push_back(Point(dx * i, dy * j));
						varScaleIDs[k].push_back(scaleID);
					}
				}
			}
			scaleID++;
			size.width /= tld::SCALE_STEP;
			size.height /= tld::SCALE_STEP;
			scale *= tld::SCALE_STEP;
			resize(img, tmp, size, 0, 0, tld::DOWNSCALE_MODE);
			resized_imgs.push_back(tmp);
			GaussianBlur(resized_imgs[scaleID], tmp, tld::GaussBlurKernelSize, 0.0f);
			blurred_imgs.push_back(tmp);
		} while (size.width >= initSize.width && size.height >= initSize.height);

		//Encsemble classification
		for (int k = 0; k < (int)trackers.size(); k++)
		{
			//TLD Tracker data extraction
			trackerPtr = trackers[k];
			tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
			//TLD Model Extraction
			tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));


			for (int i = 0; i < (int)varBuffer[k].size(); i++)
			{
				tldModel->detector->prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[k][i]].step[0]));

				double ensRes = 0;
				uchar* data = &blurred_imgs[varScaleIDs[k][i]].at<uchar>(varBuffer[k][i].y, varBuffer[k][i].x);
				for (int x = 0; x < (int)tldModel->detector->classifiers.size(); x++)
				{
					int position = 0;
					for (int n = 0; n < (int)tldModel->detector->classifiers[x].measurements.size(); n++)
					{
						position = position << 1;
						if (data[tldModel->detector->classifiers[x].offset[n].x] < data[tldModel->detector->classifiers[x].offset[n].y])
							position++;
					}
					double posNum = (double)tldModel->detector->classifiers[x].posAndNeg[position].x;
					double negNum = (double)tldModel->detector->classifiers[x].posAndNeg[position].y;
					if (posNum == 0.0 && negNum == 0.0)
						continue;
					else
						ensRes += posNum / (posNum + negNum);
				}
				ensRes /= tldModel->detector->classifiers.size();
				ensRes = tldModel->detector->ensembleClassifierNum(&blurred_imgs[varScaleIDs[k][i]].at<uchar>(varBuffer[k][i].y, varBuffer[k][i].x));

				if ( ensRes <= tld::ENSEMBLE_THRESHOLD)
					continue;
				ensBuffer[k].push_back(varBuffer[k][i]);
				ensScaleIDs[k].push_back(varScaleIDs[k][i]);
			}
		}

		//NN classification
		for (int k = 0; k < (int)trackers.size(); k++)
		{
			//TLD Tracker data extraction
			trackerPtr = trackers[k];
			tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
			//TLD Model Extraction
			tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));

			npos = 0;
			nneg = 0;
			maxSc = -5.0;

			for (int i = 0; i < (int)ensBuffer[k].size(); i++)
			{
				tld::TLDDetector::LabeledPatch labPatch;
				double curScale = pow(tld::SCALE_STEP, ensScaleIDs[k][i]);
				labPatch.rect = Rect2d(ensBuffer[k][i].x*curScale, ensBuffer[k][i].y*curScale, initSize.width * curScale, initSize.height * curScale);
				tld::resample(resized_imgs[ensScaleIDs[k][i]], Rect2d(ensBuffer[k][i], initSize), standardPatch);

				double srValue, scValue;
				srValue = tldModel->detector->Sr(standardPatch);

				////To fix: Check the paper, probably this cause wrong learning
				//
				labPatch.isObject = srValue > tld::THETA_NN;
                labPatch.shouldBeIntegrated = abs(srValue - tld::THETA_NN) < tld::CLASSIFIER_MARGIN;
				patches[k].push_back(labPatch);
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
				scValue = tldModel->detector->Sc(standardPatch);
				if (scValue > maxSc)
				{
					maxSc = scValue;
					maxScRect = labPatch.rect;
				}
			}



			if (maxSc < 0)
				detect_flgs[k] = false;
			else
			{
				res[k] = maxScRect;
				detect_flgs[k] = true;
			}
		}
	}

#ifdef HAVE_OPENCL
	void ocl_detect_all(const Mat& img, const Mat& imgBlurred, std::vector<Rect2d>& res, std::vector < std::vector < tld::TLDDetector::LabeledPatch > > &patches, std::vector<bool> &detect_flgs,
		std::vector<Ptr<legacy::Tracker> > &trackers)
	{
		//TLD Tracker data extraction
		legacy::Tracker* trackerPtr = trackers[0];
		tld::TrackerTLDImpl* tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
		//TLD Model Extraction
		tld::TrackerTLDModel* tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));
		Size initSize = tldModel->getMinSize();

		for (int k = 0; k < (int)trackers.size(); k++)
			patches[k].clear();

		Mat_<uchar> standardPatch(tld::STANDARD_PATCH_SIZE, tld::STANDARD_PATCH_SIZE);
		Mat tmp;
		int dx = initSize.width / 10, dy = initSize.height / 10;
		Size2d size = img.size();
		double scale = 1.0;
		int npos = 0, nneg = 0;
		double maxSc = -5.0;
		Rect2d maxScRect;
		int scaleID;
		std::vector <Mat> resized_imgs, blurred_imgs;

		std::vector <std::vector <Point> > varBuffer(trackers.size()), ensBuffer(trackers.size());
		std::vector <std::vector <int> > varScaleIDs(trackers.size()), ensScaleIDs(trackers.size());

		std::vector <Point> tmpP;
		std::vector <int> tmpI;

		//Detection part
		//Generate windows and filter by variance
		scaleID = 0;
		resized_imgs.push_back(img);
		blurred_imgs.push_back(imgBlurred);
		do
		{
			Mat_<double> intImgP, intImgP2;
			tld::TLDDetector::computeIntegralImages(resized_imgs[scaleID], intImgP, intImgP2);
			for (int i = 0, imax = cvFloor((0.0 + resized_imgs[scaleID].cols - initSize.width) / dx); i < imax; i++)
			{
				for (int j = 0, jmax = cvFloor((0.0 + resized_imgs[scaleID].rows - initSize.height) / dy); j < jmax; j++)
				{
					//Optimized variance calculation
					int x = dx * i,
						y = dy * j,
						width = initSize.width,
						height = initSize.height;
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
					double windowVar = p2 - p * p;

					//Loop for on all objects
					for (int k = 0; k < (int)trackers.size(); k++)
					{
						//TLD Tracker data extraction
						trackerPtr = trackers[k];
						tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
						//TLD Model Extraction
						tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));

						//Optimized variance calculation
						bool varPass = (windowVar > tld::VARIANCE_THRESHOLD * *tldModel->detector->originalVariancePtr);

						if (!varPass)
							continue;
						varBuffer[k].push_back(Point(dx * i, dy * j));
						varScaleIDs[k].push_back(scaleID);
					}
				}
			}
			scaleID++;
			size.width /= tld::SCALE_STEP;
			size.height /= tld::SCALE_STEP;
			scale *= tld::SCALE_STEP;
			resize(img, tmp, size, 0, 0, tld::DOWNSCALE_MODE);
			resized_imgs.push_back(tmp);
			GaussianBlur(resized_imgs[scaleID], tmp, tld::GaussBlurKernelSize, 0.0f);
			blurred_imgs.push_back(tmp);
		} while (size.width >= initSize.width && size.height >= initSize.height);

		//Encsemble classification
		for (int k = 0; k < (int)trackers.size(); k++)
		{
			//TLD Tracker data extraction
			trackerPtr = trackers[k];
			tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
			//TLD Model Extraction
			tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));


			for (int i = 0; i < (int)varBuffer[k].size(); i++)
			{
				tldModel->detector->prepareClassifiers(static_cast<int> (blurred_imgs[varScaleIDs[k][i]].step[0]));

				double ensRes = 0;
				uchar* data = &blurred_imgs[varScaleIDs[k][i]].at<uchar>(varBuffer[k][i].y, varBuffer[k][i].x);
				for (int x = 0; x < (int)tldModel->detector->classifiers.size(); x++)
				{
					int position = 0;
					for (int n = 0; n < (int)tldModel->detector->classifiers[x].measurements.size(); n++)
					{
						position = position << 1;
						if (data[tldModel->detector->classifiers[x].offset[n].x] < data[tldModel->detector->classifiers[x].offset[n].y])
							position++;
					}
					double posNum = (double)tldModel->detector->classifiers[x].posAndNeg[position].x;
					double negNum = (double)tldModel->detector->classifiers[x].posAndNeg[position].y;
					if (posNum == 0.0 && negNum == 0.0)
						continue;
					else
						ensRes += posNum / (posNum + negNum);
				}
				ensRes /= tldModel->detector->classifiers.size();
				ensRes = tldModel->detector->ensembleClassifierNum(&blurred_imgs[varScaleIDs[k][i]].at<uchar>(varBuffer[k][i].y, varBuffer[k][i].x));

				if (ensRes <= tld::ENSEMBLE_THRESHOLD)
					continue;
				ensBuffer[k].push_back(varBuffer[k][i]);
				ensScaleIDs[k].push_back(varScaleIDs[k][i]);
			}
		}

		//NN classification
		for (int k = 0; k < (int)trackers.size(); k++)
		{
			//TLD Tracker data extraction
			trackerPtr = trackers[k];
			tracker = static_cast<tld::TrackerTLDImpl*>(trackerPtr);
			//TLD Model Extraction
			tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->getModel()));
			npos = 0;
			nneg = 0;
			maxSc = -5.0;

			//Prepare batch of patches
			int numOfPatches = (int)ensBuffer[k].size();
			Mat_<uchar> stdPatches(numOfPatches, 225);
			double *resultSr = new double[numOfPatches];
			double *resultSc = new double[numOfPatches];

			uchar *patchesData = stdPatches.data;
			for (int i = 0; i < (int)ensBuffer.size(); i++)
			{
				tld::resample(resized_imgs[ensScaleIDs[k][i]], Rect2d(ensBuffer[k][i], initSize), standardPatch);
				uchar *stdPatchData = standardPatch.data;
				for (int j = 0; j < 225; j++)
					patchesData[225 * i + j] = stdPatchData[j];
			}
			//Calculate Sr and Sc batches
			tldModel->detector->ocl_batchSrSc(stdPatches, resultSr, resultSc, numOfPatches);

			for (int i = 0; i < (int)ensBuffer[k].size(); i++)
			{
				tld::TLDDetector::LabeledPatch labPatch;
				standardPatch.data = &stdPatches.data[225 * i];
				double curScale = pow(tld::SCALE_STEP, ensScaleIDs[k][i]);
				labPatch.rect = Rect2d(ensBuffer[k][i].x*curScale, ensBuffer[k][i].y*curScale, initSize.width * curScale, initSize.height * curScale);
				tld::resample(resized_imgs[ensScaleIDs[k][i]], Rect2d(ensBuffer[k][i], initSize), standardPatch);

				double srValue, scValue;
				srValue = resultSr[i];

				////To fix: Check the paper, probably this cause wrong learning
				//
				labPatch.isObject = srValue > tld::THETA_NN;
				labPatch.shouldBeIntegrated = abs(srValue - tld::THETA_NN) < 0.1;
				patches[k].push_back(labPatch);
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
				detect_flgs[k] = false;
			else
			{
				res[k] = maxScRect;
				detect_flgs[k] = true;
			}
		}
	}
#endif

}}}  // namespace
