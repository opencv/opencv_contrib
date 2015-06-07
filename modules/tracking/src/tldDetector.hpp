#ifndef OPENCV_TLD_DETECTOR
#define OPENCV_TLD_DETECTOR

#include "precomp.hpp"
#include "tldEnsembleClassifier.hpp"
#include "tldUtils.hpp"

namespace cv
{
	namespace tld
	{
		const int STANDARD_PATCH_SIZE = 15;
		const int NEG_EXAMPLES_IN_INIT_MODEL = 300;
		const int MAX_EXAMPLES_IN_MODEL = 500;
		const int MEASURES_PER_CLASSIFIER = 13;
		const int GRIDSIZE = 15;
		const int DOWNSCALE_MODE = cv::INTER_LINEAR;
		const double THETA_NN = 0.50;
		const double CORE_THRESHOLD = 0.5;
		const double SCALE_STEP = 1.2;
		const double ENSEMBLE_THRESHOLD = 0.5;
		const double VARIANCE_THRESHOLD = 0.5;
		const double NEXPERT_THRESHOLD = 0.2;

		static const cv::Size GaussBlurKernelSize(3, 3);

		class TLDDetector
		{
		public:
			TLDDetector(){}
			~TLDDetector(){}
			
			inline double ensembleClassifierNum(const uchar* data);
			inline void prepareClassifiers(int rowstep);
			double Sr(const Mat_<uchar>& patch);
			double Sc(const Mat_<uchar>& patch);

			std::vector<TLDEnsembleClassifier> classifiers;
			std::vector<Mat_<uchar> > *positiveExamples, *negativeExamples;
			std::vector<int> *timeStampsPositive, *timeStampsNegative;
			double *originalVariance;

			static void generateScanGrid(int rows, int cols, Size initBox, std::vector<Rect2d>& res, bool withScaling = false);
			struct LabeledPatch
			{
				Rect2d rect;
				bool isObject, shouldBeIntegrated;
			};
			bool detect(const Mat& img, const Mat& imgBlurred, Rect2d& res, std::vector<LabeledPatch>& patches, Size initSize);
		protected:
			
			

			friend class MyMouseCallbackDEBUG;
			void computeIntegralImages(const Mat& img, Mat_<double>& intImgP, Mat_<double>& intImgP2){ integral(img, intImgP, intImgP2, CV_64F); }
			inline bool patchVariance(Mat_<double>& intImgP, Mat_<double>& intImgP2, double originalVariance, Point pt, Size size);
		};
	}
}

#endif