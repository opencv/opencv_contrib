////////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2021, Dr Seng Cheong Loke (lokesengcheong@gmail.com)
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
//

 /*
 * BibTeX reference
@article{loke2021accelerated,
  title={Accelerated superpixel image segmentation with a parallelized DBSCAN algorithm},
  author={Loke, Seng Cheong and MacDonald, Bruce A and Parsons, Matthew and W{\"u}nsche, Burkhard Claus},
  journal={Journal of Real-Time Image Processing},
  pages={1--16},
  year={2021},
  publisher={Springer}
}
  */

#ifndef __OPENCV_SCANSEGMENT_HPP__
#define __OPENCV_SCANSEGMENT_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

namespace cv
{
	namespace ximgproc
	{
/** @brief Class implementing the F-DBSCAN (Accelerated superpixel image segmentation with a parallelized DBSCAN algorithm) superpixels
algorithm by Loke SC, et al.
The algorithm uses a parallelised DBSCAN cluster search that is resistant to noise, competitive in segmentation quality, and faster than 
existing superpixel segmentation methods. When tested on the Berkeley Segmentation Dataset, the average processing speed is 175 frames/s 
with a Boundary Recall of 0.797 and an Achievable Segmentation Accuracy of 0.944. The computational complexity is quadratic O(n2) and 
more suited to smaller images, but can still process a 2MP colour image faster than the SEEDS algorithm in OpenCV. The output is deterministic 
when the number of processing threads is fixed, and requires the source image to be in Lab colour format.
 */
		class CV_EXPORTS_W ScanSegment : public Algorithm
		{
		public:
			/** @brief Returns the actual superpixel segmentation from the last image processed using iterate.
			Returns zero if no image has been processed.
			 */
			CV_WRAP virtual int getNumberOfSuperpixels() = 0;

			/** @brief Calculates the superpixel segmentation on a given image with the initialized
			parameters in the ScanSegment object. This function can be called again for other images 
			without the need of initializing the algorithm with createScanSegment(). This save the 
			computational cost of allocating memory	for all the structures of the algorithm.
			@param img Input image. Supported format: CV_8UC3. Image size must match with the initialized 
			image size with the function createScanSegment(). It MUST be in Lab color space.
			 */
			CV_WRAP virtual void iterate(InputArray img) = 0;

			/** @brief Returns the segmentation labeling of the image. Each label represents a superpixel, 
			and each pixel is assigned to one superpixel label.
			@param labels_out Return: A CV_32UC1 integer array containing the labels of the superpixel
			segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
			*/
			CV_WRAP virtual void getLabels(OutputArray labels_out) = 0;

			/** @brief Returns the mask of the superpixel segmentation stored in the ScanSegment object.
			@param image Return: CV_8UC1 image mask where -1 indicates that the pixel is a superpixel border,
			and 0 otherwise.
			@param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border
			are masked.
			The function return the boundaries of the superpixel segmentation.
			*/
			CV_WRAP virtual void getLabelContourMask(OutputArray image, bool thick_line = false) = 0;

			virtual ~ScanSegment() {}
		};

		/** @brief Initializes a ScanSegment object.
		@param image_width Image width.
		@param image_height Image height.
		@param num_superpixels Desired number of superpixels. Note that the actual number may be smaller
		due to restrictions (depending on the image size). Use getNumberOfSuperpixels() to
		get the actual number.
		@param threads Number of processing threads for parallelisation. Default -1 uses the maximum number 
		of threads. In practice, four threads is enough for smaller images and eight threads for larger ones.
		@param merge_small merge small segments to give the desired number of superpixels. Processing is 
		much faster without merging, but many small segments will be left in the image.
		The function initializes a ScanSegment object for the input image. It stores the parameters of
		the image: image_width and image_height. It also sets the parameters of the F-DBSCAN superpixel 
		algorithm, which are: num_superpixels, threads, and merge_small.
		*/
		CV_EXPORTS_W cv::Ptr<ScanSegment> createScanSegment(int image_width, int image_height, int num_superpixels, int threads = -1, bool merge_small = true);
	}
}
#endif
#endif
