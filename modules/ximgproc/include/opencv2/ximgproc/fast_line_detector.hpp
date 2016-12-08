/*********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright (c) 2014, 2015
 * Zhengqin Li <li-zq12 at mails dot tsinghua dot edu dot cn>
 * Jiansheng Chen <jschenthu at mail dot tsinghua dot edu dot cn>
 * Tsinghua University
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holders nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/*

*/

#ifndef __OPENCV_FAST_LINE_DETECTOR_HPP__
#define __OPENCV_FAST_LINE_DETECTOR_HPP__
#ifdef __cplusplus

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_superpixel
//! @{

/** @brief Class implementing the FLD (Fast Line Detector) algorithm described
 * in @cite Lee14.

*/

/** @example fld_lines.cpp
	An example using the FastLineDetector
*/

class CV_EXPORTS_W FastLineDetector : public Algorithm
{
		public:

		/** @brief Finds lines in the input image.

			This is the output of the default parameters of the algorithm on the above shown image.

			![image](pics/building_fld.png)

			@param _image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use:
			`fld_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
			@param _lines A vector of Vec4i or Vec4f elements specifying the beginning and ending point of a line.
			Where Vec4i/Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are directed so that the brighter side is on their left
			@param
			@param
			@param
	*/
		CV_WRAP virtual void detect(InputArray _image, OutputArray _lines) = 0;

		// CV_WRAP virtual void detect(InputArray _image, OutputArray _lines,
		//                     OutputArray width = noArray(), OutputArray prec = noArray(),
		//                     OutputArray nfa = noArray()) = 0;

		/** @brief Draws the line segments on a given image.
			@param _image The image, where the liens will be drawn. Should be bigger or equal to the image,
			where the lines were found.
			@param lines A vector of the lines that needed to be drawn.
			*/
		CV_WRAP virtual void drawSegments(InputOutputArray _image, InputArray lines) = 0;

		/** @brief Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.

			@param size The size of the image, where lines1 and lines2 were found.
			@param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
			@param lines2 The second group of lines. They visualized in red color.
			@param _image Optional image, where the lines will be drawn. The image should be color(3-channel)
			in order for lines1 and lines2 to be drawn in the above mentioned colors.
			*/
		// CV_WRAP virtual int compareSegments(const Size& size, InputArray lines1, InputArray lines2, InputOutputArray _image = noArray()) = 0;

		virtual ~FastLineDetector() { }
};

/** @brief Creates a smart pointer to a FastLineDetector object and initializes it.
*/
CV_EXPORTS_W Ptr<FastLineDetector> createFastLineDetector(
int _legth_threshold = 10, float _distance_threshold = 1.6f);

}
}
#endif
#endif
