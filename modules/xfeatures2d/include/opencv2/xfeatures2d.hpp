/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_XFEATURES2D_HPP__
#define __OPENCV_XFEATURES2D_HPP__

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

/** @defgroup xfeatures2d Extra 2D Features Framework
@{
@defgroup xfeatures2d_experiment Experimental 2D Features Algorithms

This section describes experimental algorithms for 2d feature detection.

@defgroup xfeatures2d_nonfree Non-free 2D Features Algorithms

This section describes two popular algorithms for 2d feature detection, SIFT and SURF, that are
known to be patented. Use them at your own risk.

@}
*/

namespace cv
{
	namespace xfeatures2d
	{

		//! @addtogroup xfeatures2d_experiment
		//! @{

		/** @brief Class implementing the FREAK (*Fast Retina Keypoint*) keypoint descriptor, described in @cite AOV12.

		The algorithm propose a novel keypoint descriptor inspired by the human visual system and more
		precisely the retina, coined Fast Retina Key- point (FREAK). A cascade of binary strings is
		computed by efficiently comparing image intensities over a retinal sampling pattern. FREAKs are in
		general faster to compute with lower memory load and also more robust than SIFT, SURF or BRISK.
		They are competitive alternatives to existing keypoints in particular for embedded applications.

		@note
		-   An example on how to use the FREAK descriptor can be found at
		opencv_source_code/samples/cpp/freak_demo.cpp
		*/
		class CV_EXPORTS FREAK : public Feature2D
		{
		public:

			enum
			{
				NB_SCALES = 64, NB_PAIRS = 512, NB_ORIENPAIRS = 45
			};

			/**
			@param orientationNormalized Enable orientation normalization.
			@param scaleNormalized Enable scale normalization.
			@param patternScale Scaling of the description pattern.
			@param nOctaves Number of octaves covered by the detected keypoints.
			@param selectedPairs (Optional) user defined selected pairs indexes,
			*/
			static Ptr<FREAK> create(bool orientationNormalized = true,
				bool scaleNormalized = true,
				float patternScale = 22.0f,
				int nOctaves = 4,
				const std::vector<int>& selectedPairs = std::vector<int>());
		};


		/** @brief The class implements the keypoint detector introduced by @cite Agrawal08, synonym of StarDetector. :
		 */
		class CV_EXPORTS StarDetector : public FeatureDetector
		{
		public:
			//! the full constructor
			static Ptr<StarDetector> create(int maxSize = 45, int responseThreshold = 30,
				int lineThresholdProjected = 10,
				int lineThresholdBinarized = 8,
				int suppressNonmaxSize = 5);
		};

		/*
		 * BRIEF Descriptor
		 */

		/** @brief Class for computing BRIEF descriptors described in @cite calon2010

		@note
		-   A complete BRIEF extractor sample can be found at
		opencv_source_code/samples/cpp/brief_match_test.cpp

		*/
		class CV_EXPORTS BriefDescriptorExtractor : public DescriptorExtractor
		{
		public:
			static Ptr<BriefDescriptorExtractor> create(int bytes = 32);
		};


		/*
		* BinBoost Descriptor
		* BGM is the base descriptor where each binary dimension is computed as the output of a single weak learner.
		* L-BGM (or FP-Boost from the Pami paper) is the floating point extension where each dimension is computed as
		* a linear combinatioon of the weak learner responses.
		* BinBoost is a binary extension of L-BGM where each bit is computed as a thresholded linear combination of a set of weak learners.

		* BGM-HARD and BGM-Bilinear refers to different type of gradient binning - in the HARD version the gradient
		* is assigned to the nearest orientation bin, in the bilinear version it is assigned to the two neighboring
		* bins and in the soft version it is assigned to 8 nearest bins according to the cosine value between the gradient angle and the bin center.
		*/
		class CV_EXPORTS BGMDescriptorExtractor : public DescriptorExtractor
		{
		public:
			// gradient assignment type
			enum Assign {
				ASSIGN_HARD = 0,
				ASSIGN_BILINEAR = 1,
				ASSIGN_SOFT = 2,
				ASSIGN_HARD_MAGN = 3,
				ASSIGN_SOFT_MAGN = 4
			};

			// struct with weak learners' parameters
			struct WeakLearner{
				float thresh;
				int orient;
				int x_min;
				int x_max;
				int y_min;
				int y_max;
				float alpha;
				float beta;

				WeakLearner() :
					thresh(0.0), orient(0), x_min(0), x_max(0), y_min(0), y_max(0), alpha(0.0), beta(0.0)
				{}
			};
			struct Utils
			{
				// compute images of gradients
				static void computeGradientMaps(const Mat& im,
					const Assign& gradAssignType,
					const int orientQuant,
					std::vector<Mat>& gradMap);

				// compute integral images
				static void computeIntegrals(const std::vector<Mat>& gradMap,
					const int orientQuant,
					std::vector<Mat>& integralMap);

				// compute the response of a weak learner
				static float computeWLResponse(const WeakLearner& WL,
					const int orientQuant,
					const std::vector<Mat>& integralMap);

				// rectify image patch according to the detected interest point
				static void rectifyPatch(const Mat& image,
					const KeyPoint& kp,
					const int& patchSize,
					Mat& patch);
			};

			BGMDescriptorExtractor()
			{
				for (unsigned int i = 0; i < 8; ++i)
					binLookUp[i] = (uchar)1 << i;
			};

			BGMDescriptorExtractor(std::string _wlFile) :
				wlFile(_wlFile)
			{
				for (unsigned int i = 0; i < 8; ++i)
					binLookUp[i] = (uchar)1 << i;
				if (_wlFile.find("txt") != std::string::npos)
					readWLs();
				else
					readWLsBin();
			};

			~BGMDescriptorExtractor()
			{
				delete pWLs;
			};

			// interface methods inherited from cv::DescriptorExtractor
			virtual void compute(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const;
			virtual int descriptorSize() const { return nDim / 8; }
			virtual int descriptorType() const { return CV_8UC1; }
			virtual void computeImpl(const Mat&  image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
			{
				compute(image, keypoints, descriptors);
			}

			virtual void saveWLsBin(const std::string output) const;

			AlgorithmInfo* info() const;
		protected:
			std::string wlFile;
			int nWLs;
			int nDim;
			WeakLearner* pWLs;
			int orientQuant;
			int patchSize;
			Assign gradAssignType;
			uchar binLookUp[8];

			virtual void readWLs();
			virtual void readWLsBin();
		};

		class CV_EXPORTS LBGMDescriptorExtractor : public BGMDescriptorExtractor
		{
		public:
			LBGMDescriptorExtractor(std::string _wlFile)
			{
				wlFile = _wlFile;
				if (wlFile.find("txt") != std::string::npos)
					readWLs();
				else
					readWLsBin();
			}
			~LBGMDescriptorExtractor()
			{
				delete[] betas;

			};

			// interface methods inherited from cv::DescriptorExtractor
			void compute(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const;
			int descriptorSize() const { return nDim*sizeof(float); }
			int descriptorType() const { return CV_32FC1; }
			void computeImpl(const Mat&  image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
			{
				compute(image, keypoints, descriptors);
			}

			void saveWLsBin(const std::string output) const;

		protected:
			float* betas;

			void readWLs();
			void readWLsBin();

		};

		class CV_EXPORTS BinBoostDescriptorExtractor : public BGMDescriptorExtractor
		{
		public:
			BinBoostDescriptorExtractor(std::string _wlFile) :

				BGMDescriptorExtractor(_wlFile)
			{
				if (_wlFile.find("txt") != std::string::npos)
					readWLs();
				else
					readWLsBin();
			}
			~BinBoostDescriptorExtractor()
			{
				delete[] pWLsArray;
			};

			// interface methods inherited from cv::DescriptorExtractor
			void compute(const Mat& image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const;
			int descriptorSize() const { return nDim / 8; }
			int descriptorType() const { return CV_8UC1; }
			void computeImpl(const Mat&  image, std::vector<KeyPoint>& keypoints, Mat& descriptors) const
			{
				compute(image, keypoints, descriptors);
			}

			void saveWLsBin(const std::string output) const;

		protected:
			WeakLearner** pWLsArray;

			void readWLs();
			void readWLsBin();
		};


		//! @}

	}
}

#endif
