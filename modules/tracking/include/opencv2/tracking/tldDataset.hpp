#ifndef OPENCV_TLD_DATASET
#define OPENCV_TLD_DATASET

#include "opencv2/highgui.hpp"

namespace cv
{
	namespace tld
	{
		CV_EXPORTS cv::Rect2d tld_InitDataset(int datasetInd, char* rootPath = "TLD_dataset");
		CV_EXPORTS cv::Mat tld_getNextDatasetFrame();
	}
}

#endif