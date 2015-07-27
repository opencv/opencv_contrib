#include "tldTracker.hpp"

namespace cv
{
	//Multitracker
	bool MultiTracker::addTarget(const Mat& image, const Rect2d& boundingBox, char* tracker_algorithm_name)
	{
		Ptr<Tracker> tracker = Tracker::create(tracker_algorithm_name);
		if (tracker == NULL)
			return false;

		if (!tracker->init(image, boundingBox))
			return false;

		//Add BB of target
		boundingBoxes.push_back(boundingBox);

		//Add Tracker to stack
		trackers.push_back(tracker);

		//Assign a random color to target
		colors.push_back(Scalar(rand() % 256, rand() % 256, rand() % 256));

		//Target counter
		targetNum++;

		return true;
	}

	bool MultiTracker::update(const Mat& image)
	{
		for (int i = 0; i < trackers.size(); i++)
			if (!trackers[i]->update(image, boundingBoxes[i]))
				return false;

		return true;
	}
	 
	//Multitracker TLD
	/*Optimized update method for TLD Multitracker */
	bool MultiTrackerTLD::update(const Mat& image)
	{
		
		for (int k = 0; k < trackers.size(); k++)
		{
			//Set current target(tracker) parameters
			Rect2d boundingBox = boundingBoxes[k];
			Ptr<tld::TrackerTLDImpl> tracker = (Ptr<tld::TrackerTLDImpl>)static_cast<Ptr<tld::TrackerTLDImpl>> (trackers[k]);
			tld::TrackerTLDModel* tldModel = ((tld::TrackerTLDModel*)static_cast<TrackerModel*>(tracker->model));
			Ptr<tld::Data> data = tracker->data;
			double scale = data->getScale();


			Mat image_gray, image_blurred, imageForDetector;
			cvtColor(image, image_gray, COLOR_BGR2GRAY);

			if (scale > 1.0)
				resize(image_gray, imageForDetector, Size(cvRound(image.cols*scale), cvRound(image.rows*scale)), 0, 0, tld::DOWNSCALE_MODE);
			else
				imageForDetector = image_gray;
			GaussianBlur(imageForDetector, image_blurred, tld::GaussBlurKernelSize, 0.0);

			data->frameNum++;
			Mat_<uchar> standardPatch(tld::STANDARD_PATCH_SIZE, tld::STANDARD_PATCH_SIZE);
			std::vector<tld::TLDDetector::LabeledPatch> detectorResults;
			//best overlap around 92%

			std::vector<Rect2d> candidates;
			std::vector<double> candidatesRes;
			bool trackerNeedsReInit = false;
			bool DETECT_FLG = false;
			for (int i = 0; i < 2; i++)
			{
				Rect2d tmpCandid = boundingBox;

				if (i == 1)
				{
					if (ocl::haveOpenCL())
						DETECT_FLG = tldModel->detector->ocl_detect(imageForDetector, image_blurred, tmpCandid, detectorResults, tldModel->getMinSize());
					else
						DETECT_FLG = tldModel->detector->detect(imageForDetector, image_blurred, tmpCandid, detectorResults, tldModel->getMinSize());
				}

				if (((i == 0) && !data->failedLastTime && tracker->trackerProxy->update(image, tmpCandid)) || (DETECT_FLG))
				{
					candidates.push_back(tmpCandid);
					if (i == 0)
						tld::resample(image_gray, tmpCandid, standardPatch);
					else
						tld::resample(imageForDetector, tmpCandid, standardPatch);
					candidatesRes.push_back(tldModel->detector->Sc(standardPatch));
				}
				else
				{
					if (i == 0)
						trackerNeedsReInit = true;
				}
			}

			std::vector<double>::iterator it = std::max_element(candidatesRes.begin(), candidatesRes.end());

			//dfprintf((stdout, "scale = %f\n", log(1.0 * boundingBox.width / (data->getMinSize()).width) / log(SCALE_STEP)));
			//for( int i = 0; i < (int)candidatesRes.size(); i++ )
			//dprintf(("\tcandidatesRes[%d] = %f\n", i, candidatesRes[i]));
			//data->printme();
			//tldModel->printme(stdout);

			if (it == candidatesRes.end())
			{
				data->confident = false;
				data->failedLastTime = true;
				return false;
			}
			else
			{
				boundingBox = candidates[it - candidatesRes.begin()];
				boundingBoxes[k] = boundingBox;
				data->failedLastTime = false;
				if (trackerNeedsReInit || it != candidatesRes.begin())
					tracker->trackerProxy->init(image, boundingBox);
			}

#if 1
			if (it != candidatesRes.end())
			{
				tld::resample(imageForDetector, candidates[it - candidatesRes.begin()], standardPatch);
				//dfprintf((stderr, "%d %f %f\n", data->frameNum, tldModel->Sc(standardPatch), tldModel->Sr(standardPatch)));
				//if( candidatesRes.size() == 2 &&  it == (candidatesRes.begin() + 1) )
				//dfprintf((stderr, "detector WON\n"));
			}
			else
			{
				//dfprintf((stderr, "%d x x\n", data->frameNum));
			}
#endif

			if (*it > tld::CORE_THRESHOLD)
				data->confident = true;

			if (data->confident)
			{
				tld::TrackerTLDImpl::Pexpert pExpert(imageForDetector, image_blurred, boundingBox, tldModel->detector, tracker->params, data->getMinSize());
				tld::TrackerTLDImpl::Nexpert nExpert(imageForDetector, boundingBox, tldModel->detector, tracker->params);
				std::vector<Mat_<uchar> > examplesForModel, examplesForEnsemble;
				examplesForModel.reserve(100); examplesForEnsemble.reserve(100);
				int negRelabeled = 0;
				for (int i = 0; i < (int)detectorResults.size(); i++)
				{
					bool expertResult;
					if (detectorResults[i].isObject)
					{
						expertResult = nExpert(detectorResults[i].rect);
						if (expertResult != detectorResults[i].isObject)
							negRelabeled++;
					}
					else
					{
						expertResult = pExpert(detectorResults[i].rect);
					}

					detectorResults[i].shouldBeIntegrated = detectorResults[i].shouldBeIntegrated || (detectorResults[i].isObject != expertResult);
					detectorResults[i].isObject = expertResult;
				}
				tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults);
				//dprintf(("%d relabeled by nExpert\n", negRelabeled));
				pExpert.additionalExamples(examplesForModel, examplesForEnsemble);
				if (ocl::haveOpenCL())
					tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, true);
				else
					tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, true);
				examplesForModel.clear(); examplesForEnsemble.clear();
				nExpert.additionalExamples(examplesForModel, examplesForEnsemble);

				if (ocl::haveOpenCL())
					tldModel->ocl_integrateAdditional(examplesForModel, examplesForEnsemble, false);
				else
					tldModel->integrateAdditional(examplesForModel, examplesForEnsemble, false);
			}
			else
			{
#ifdef CLOSED_LOOP
				tldModel->integrateRelabeled(imageForDetector, image_blurred, detectorResults);
#endif
			}

			
		}

		return true;
	}
}