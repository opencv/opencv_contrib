#ifndef OPENCV_TLD_MODEL
#define OPENCV_TLD_MODEL

#include "precomp.hpp"
#include "tldDetector.hpp"
#include "tldUtils.hpp"

namespace cv
{
	namespace tld
	{

		


		class TrackerTLDModel : public TrackerModel
		{
		public:
			TrackerTLDModel(TrackerTLD::Params params, const Mat& image, const Rect2d& boundingBox, Size minSize);
			Rect2d getBoundingBox(){ return boundingBox_; }
			void setBoudingBox(Rect2d boundingBox){ boundingBox_ = boundingBox; }
			void integrateRelabeled(Mat& img, Mat& imgBlurred, const std::vector<TLDDetector::LabeledPatch>& patches);
			void integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive);
			Size getMinSize(){ return minSize_; }
			void printme(FILE* port = stdout);
			Ptr<TLDDetector> detector;

			std::vector<Mat_<uchar> > positiveExamples, negativeExamples;
			std::vector<int> timeStampsPositive, timeStampsNegative;
			int timeStampPositiveNext, timeStampNegativeNext;
			double originalVariance_;

			double getOriginalVariance(){ return originalVariance_; }

		protected:
			Size minSize_;
			
			TrackerTLD::Params params_;
			void pushIntoModel(const Mat_<uchar>& example, bool positive);
			void modelEstimationImpl(const std::vector<Mat>& /*responses*/){}
			void modelUpdateImpl(){}
			Rect2d boundingBox_;			
			RNG rng;
			
		};

	}
}

#endif