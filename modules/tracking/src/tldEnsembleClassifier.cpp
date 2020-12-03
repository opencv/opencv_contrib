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
#include "tldEnsembleClassifier.hpp"

namespace cv {
inline namespace tracking {
namespace impl {
namespace tld {

		// Constructor
		TLDEnsembleClassifier::TLDEnsembleClassifier(const std::vector<Vec4b>& meas, int beg, int end) :lastStep_(-1)
		{
			int posSize = 1, mpc = end - beg;
			for (int i = 0; i < mpc; i++)
				posSize *= 2;
			posAndNeg.assign(posSize, Point2i(0, 0));
			measurements.assign(meas.begin() + beg, meas.begin() + end);
			offset.assign(mpc, Point2i(0, 0));
		}
		// Calculate measure locations from 15x15 grid on minSize patches
		void TLDEnsembleClassifier::stepPrefSuff(std::vector<Vec4b>& arr, int pos, int len, int gridSize)
		{
		#if 0
			int step = len / (gridSize - 1), pref = (len - step * (gridSize - 1)) / 2;
			for (int i = 0; i < (int)(sizeof(x1) / sizeof(x1[0])); i++)
				arr[i] = pref + arr[i] * step;
		#else
			int total = len - gridSize;
			int quo = total / (gridSize - 1), rem = total % (gridSize - 1);
			int smallStep = quo, bigStep = quo + 1;
			int bigOnes = rem, smallOnes = gridSize - bigOnes - 1;
			int bigOnes_front = bigOnes / 2, bigOnes_back = bigOnes - bigOnes_front;
			for (int i = 0; i < (int)arr.size(); i++)
			{
				if (arr[i].val[pos] < bigOnes_back)
				{
					arr[i].val[pos] = (uchar)(arr[i].val[pos] * bigStep + arr[i].val[pos]);
					continue;
				}
				if (arr[i].val[pos] < (bigOnes_front + smallOnes))
				{
					arr[i].val[pos] = (uchar)(bigOnes_front * bigStep + (arr[i].val[pos] - bigOnes_front) * smallStep + arr[i].val[pos]);
					continue;
				}
				if (arr[i].val[pos] < (bigOnes_front + smallOnes + bigOnes_back))
				{
					arr[i].val[pos] =
						(uchar)(bigOnes_front * bigStep + smallOnes * smallStep +
						(arr[i].val[pos] - (bigOnes_front + smallOnes)) * bigStep + arr[i].val[pos]);
					continue;
				}
				arr[i].val[pos] = (uchar)(len - 1);
			}
#endif
		}

		// Calculate offsets for classifier
		void TLDEnsembleClassifier::prepareClassifier(int rowstep)
		{
			if (lastStep_ != rowstep)
			{
				lastStep_ = rowstep;
				for (int i = 0; i < (int)offset.size(); i++)
				{
					offset[i].x = rowstep * measurements[i].val[2] + measurements[i].val[0];
					offset[i].y = rowstep * measurements[i].val[3] + measurements[i].val[1];
				}
			}
		}

		// Integrate patch into the Ensemble Classifier model
		void TLDEnsembleClassifier::integrate(const Mat_<uchar>& patch, bool isPositive)
		{
			int position = code(patch.data, (int)patch.step[0]);
			if (isPositive)
				posAndNeg[position].x++;
			else
				posAndNeg[position].y++;
		}

		// Calculate posterior probability on the patch
		double TLDEnsembleClassifier::posteriorProbability(const uchar* data, int rowstep) const
		{
			int position = code(data, rowstep);
			double posNum = (double)posAndNeg[position].x, negNum = (double)posAndNeg[position].y;
			if (posNum == 0.0 && negNum == 0.0)
				return 0.0;
			else
				return posNum / (posNum + negNum);
		}
		double TLDEnsembleClassifier::posteriorProbabilityFast(const uchar* data) const
		{
			int position = codeFast(data);
			double posNum = (double)posAndNeg[position].x, negNum = (double)posAndNeg[position].y;
			if (posNum == 0.0 && negNum == 0.0)
				return 0.0;
			else
				return posNum / (posNum + negNum);
		}

		// Calculate the 13-bit fern index
		int TLDEnsembleClassifier::codeFast(const uchar* data) const
		{
			int position = 0;
			for (int i = 0; i < (int)measurements.size(); i++)
			{
				position = position << 1;
				if (data[offset[i].x] < data[offset[i].y])
					position++;
			}
			return position;
		}
		int TLDEnsembleClassifier::code(const uchar* data, int rowstep) const
		{
			int position = 0;
			for (int i = 0; i < (int)measurements.size(); i++)
			{
				position = position << 1;
				if (*(data + rowstep * measurements[i].val[2] + measurements[i].val[0]) <
					*(data + rowstep * measurements[i].val[3] + measurements[i].val[1]))
				{
					position++;
				}
			}
			return position;
		}

		// Create fern classifiers
		int TLDEnsembleClassifier::makeClassifiers(Size size, int measurePerClassifier, int gridSize,
			std::vector<TLDEnsembleClassifier>& classifiers)
		{

			std::vector<Vec4b> measurements;

			//Generate random measures for 10 ferns x 13 measures
			for (int i = 0; i < 10*measurePerClassifier; i++)
			{
				Vec4b m;
				m.val[0] = rand() % 15;
				m.val[1] = rand() % 15;
				m.val[2] = rand() % 15;
				m.val[3] = rand() % 15;
				measurements.push_back(m);
			}

			//Warp measures to minSize patch coordinates
			stepPrefSuff(measurements, 0, size.width, gridSize);
			stepPrefSuff(measurements, 1, size.width, gridSize);
			stepPrefSuff(measurements, 2, size.height, gridSize);
			stepPrefSuff(measurements, 3, size.height, gridSize);

			//Compile fern classifiers
			for (int i = 0, howMany = (int)measurements.size() / measurePerClassifier; i < howMany; i++)
				classifiers.push_back(TLDEnsembleClassifier(measurements, i * measurePerClassifier, (i + 1) * measurePerClassifier));

			return (int)classifiers.size();
		}

}}}}  // namespace
