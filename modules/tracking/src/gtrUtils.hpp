#ifndef OPENCV_GTR_UTILS
#define OPENCV_GTR_UTILS

#include "precomp.hpp"
#include <vector>
#include "opencv2/highgui.hpp"
#include <opencv2/datasets/track_alov.hpp>

namespace cv
{
	namespace gtr
	{
		//Number of samples in batch
		const int samplesInBatch = 50;

		//Number of samples to mine from video frame
		const int samplesInFrame = 10;

		//Number of samples to mine from still image
		const int samplesInImage = 10;

		//Padding coefficients for Target/Search Region
		const double padTarget = 2.0;
		const double padSearch = 2.0;

		//Scale parameters for Lablace distribution for Translation/Scale
		const double bX = 1.0/5;
		const double bY = 1.0 / 5;
		const double bS = 1.0/15;

		//Limits of scale changes
		const double Ymax = 1.4;
		const double Ymin = 0.6;

		//Lower boundary constraints for random samples (sample should include X% of target BB)
		const double minX = 0.5;
		const double minY = 0.5;

		//Structure of sample for training
		struct TrainingSample
		{
			Mat targetPatch;
			Mat searchPatch;
			//Output bounding box on search patch
			Rect2f targetBB;
		};

		//Laplacian distribution
		double generateRandomLaplacian(double b, double m);

		//Convert ALOV300++ anno coordinates to Rectangle BB
		Rect2f anno2rect(vector<Point2f> annoBB);

		//Make a batch for training
		vector <TrainingSample> makeBatch();

		//Gather samples from random video frame
		vector <TrainingSample> gatherFrameSamples(Mat prevFrame, Mat currFrame, Rect2f prevBB, Rect2f currBB);

		//Gather samples from random still image
		vector <TrainingSample> gatherImageSamples();

	}
}

#endif