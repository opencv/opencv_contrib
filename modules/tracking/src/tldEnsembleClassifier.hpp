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