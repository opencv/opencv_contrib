// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_FAST_LINE_DETECTOR_HPP__
#define __OPENCV_FAST_LINE_DETECTOR_HPP__

#include <opencv2/core.hpp>

namespace cv
{
namespace ximgproc
{

//! @addtogroup ximgproc_fld
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

          This is the output of the default parameters of the algorithm on the above
          shown image.

          ![image](pics/corridor_fld.png)

          @param _image A grayscale (CV_8UC1) input image. If only a roi needs to be
          selected, use: `fld_ptr-\>detect(image(roi), lines, ...);
          lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
          @param _lines A vector of Vec4i or Vec4f elements specifying the beginning
          and ending point of a line.  Where Vec4i/Vec4f is (x1, y1, x2, y2), point
          1 is the start, point 2 -
          end. Returned lines are directed so that the brighter side is on their
          left
          @param
          @param
          @param
          */
        CV_WRAP virtual void detect(InputArray _image, OutputArray _lines) = 0;

        /** @brief Draws the line segments on a given image.
          @param _image The image, where the liens will be drawn. Should be bigger
          or equal to the image, where the lines were found.
          @param lines A vector of the lines that needed to be drawn.
          @param draw_arrow If true, arrow heads will be drawn.
          */
        CV_WRAP virtual void drawSegments(InputOutputArray _image, InputArray lines, bool draw_arrow = false) = 0;

        /** @brief Draws two groups of lines in blue and red, counting the non
          overlapping (mismatching) pixels.  */
        virtual ~FastLineDetector() { }
};

/** @brief Creates a smart pointer to a FastLineDetector object and initializes it.
*/
CV_EXPORTS_W Ptr<FastLineDetector> createFastLineDetector(
        int _legth_threshold = 10, float _distance_threshold = 1.414213562f, bool _do_merge = false);
}
}
#endif
#endif
