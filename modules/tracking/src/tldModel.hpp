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
#ifdef HAVE_OPENCL
			void ocl_integrateAdditional(const std::vector<Mat_<uchar> >& eForModel, const std::vector<Mat_<uchar> >& eForEnsemble, bool isPositive);
#endif
			Size getMinSize(){ return minSize_; }
			void printme(FILE* port = stdout);
			Ptr<TLDDetector> detector;

			std::vector<Mat_<uchar> > positiveExamples, negativeExamples;
			Mat posExp, negExp;
			int posNum, negNum;
			std::vector<int> timeStampsPositive, timeStampsNegative;
			int timeStampPositiveNext, timeStampNegativeNext;
			double originalVariance_;
			std::vector<double> srValues;

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
