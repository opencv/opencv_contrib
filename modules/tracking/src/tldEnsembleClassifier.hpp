#include <vector>
#include "precomp.hpp"

namespace cv
{
	namespace tld
	{
		class TLDEnsembleClassifier
		{
		public:
			static int makeClassifiers(Size size, int measurePerClassifier, int gridSize, std::vector<TLDEnsembleClassifier>& classifiers);
			void integrate(const Mat_<uchar>& patch, bool isPositive);
			double posteriorProbability(const uchar* data, int rowstep) const;
			double posteriorProbabilityFast(const uchar* data) const;
			void prepareClassifier(int rowstep);
		private:
			TLDEnsembleClassifier(const std::vector<Vec4b>& meas, int beg, int end);
			static void stepPrefSuff(std::vector<Vec4b> & arr, int pos, int len, int gridSize);
			int code(const uchar* data, int rowstep) const;
			int codeFast(const uchar* data) const;
			std::vector<Point2i> posAndNeg;
			std::vector<Vec4b> measurements;
			std::vector<Point2i> offset;
			int lastStep_;
		};

	
	}
}