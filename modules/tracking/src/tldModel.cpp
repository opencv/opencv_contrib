#include "tldModel.hpp"

namespace cv
{
	namespace tld
	{
		//Constructor
		TrackerTLDModel::TrackerTLDModel(TrackerTLD::Params params, const Mat& image, const Rect2d& boundingBox, Size minSize) :minSize_(minSize),
			timeStampPositiveNext(0), timeStampNegativeNext(0), params_(params), boundingBox_(boundingBox)
		{
			std::vector<Rect2d> closest, scanGrid;
			Mat scaledImg, blurredImg, image_blurred;

			//Create Detector
			detector = Ptr<TLDDetector>(new TLDDetector());

			//Propagate data to Detector
			detector->positiveExamples = &positiveExamples;
			detector->negativeExamples = &negativeExamples;
			detector->timeStampsPositive = &timeStampsPositive;
			detector->timeStampsNegative = &timeStampsNegative;
			detector->originalVariance = &originalVariance_;

			//Calculate the variance in initial BB
			originalVariance_ = variance(image(boundingBox));
			
			//Find the scale
			double scale = scaleAndBlur(image, cvRound(log(1.0 * boundingBox.width / (minSize.width)) / log(SCALE_STEP)),
				scaledImg, blurredImg, GaussBlurKernelSize, SCALE_STEP);
			GaussianBlur(image, image_blurred, GaussBlurKernelSize, 0.0);
			TLDDetector::generateScanGrid(image.rows, image.cols, minSize_, scanGrid);
			getClosestN(scanGrid, Rect2d(boundingBox.x / scale, boundingBox.y / scale, boundingBox.width / scale, 
				boundingBox.height / scale), 10, closest);

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
			//dprintf(("positive patches: %d\nnegative patches: %d\n", (int)positiveExamples.size(), (int)negativeExamples.size()));
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
			/*
			if( negativeIntoModel > 0 )
			dfprintf((stdout, "negativeIntoModel = %d ", negativeIntoModel));
			if( positiveIntoModel > 0)
			dfprintf((stdout, "positiveIntoModel = %d ", positiveIntoModel));
			if( negativeIntoEnsemble > 0 )
			dfprintf((stdout, "negativeIntoEnsemble = %d ", negativeIntoEnsemble));
			if( positiveIntoEnsemble > 0 )
			dfprintf((stdout, "positiveIntoEnsemble = %d ", positiveIntoEnsemble));
			dfprintf((stdout, "\n"));*/

		}

		void TrackerTLDModel::integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive)
		{
			int positiveIntoModel = 0, negativeIntoModel = 0, positiveIntoEnsemble = 0, negativeIntoEnsemble = 0;
			for (int k = 0; k < (int)eForModel.size(); k++)
			{
				double sr = detector->Sr(eForModel[k]);
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
			/*
			if( negativeIntoModel > 0 )
			dfprintf((stdout, "negativeIntoModel = %d ", negativeIntoModel));
			if( positiveIntoModel > 0 )
			dfprintf((stdout, "positiveIntoModel = %d ", positiveIntoModel));
			if( negativeIntoEnsemble > 0 )
			dfprintf((stdout, "negativeIntoEnsemble = %d ", negativeIntoEnsemble));
			if( positiveIntoEnsemble > 0 )
			dfprintf((stdout, "positiveIntoEnsemble = %d ", positiveIntoEnsemble));
			dfprintf((stdout, "\n"));*/
		}

		//Push the patch to the model
		void TrackerTLDModel::pushIntoModel(const Mat_<uchar>& example, bool positive)
		{
			std::vector<Mat_<uchar> >* proxyV;
			int* proxyN;
			std::vector<int>* proxyT;
			if (positive)
			{
				proxyV = &positiveExamples;
				proxyN = &timeStampPositiveNext;
				proxyT = &timeStampsPositive;
			}
			else
			{
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

		void TrackerTLDModel::printme(FILE*  port)
		{
			//dfprintf((port, "TrackerTLDModel:\n"));
			//dfprintf((port, "\tpositiveExamples.size() = %d\n", (int)positiveExamples.size()));
			//dfprintf((port, "\tnegativeExamples.size() = %d\n", (int)negativeExamples.size()));
		}


		

	}
}