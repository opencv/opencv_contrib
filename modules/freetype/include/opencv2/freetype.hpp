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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2012, Willow Garage Inc., all rights reserved.
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
//################################################################################
//
//                    Created by Kumataro
//
//################################################################################

#ifndef _OPENCV_FREETYPE_H_
#define _OPENCV_FREETYPE_H_
#ifdef __cplusplus

#include <opencv2/core.hpp>

/**
@defgroup freetype Drawing UTF-8 strings with freetype/harfbuzz

This modules is to draw UTF-8 strings with freetype/harfbuzz.

1. Install freetype2 and harfbuzz in your system.
2. Create FreeType2 instance with createFreeType2() function.
3. Load font file with loadFontData() function.
4. Draw text with putText() function.

- If thickness parameter is negative, drawing glyph is filled.
- If thickness parameter is positive, drawing glyph is outlined with thickness.
- If line_type parameter is 16(or CV_AA), drawing glyph is smooth.

*/

namespace cv {
namespace freetype {
//! @addtogroup freetype
//! @{
class CV_EXPORTS_W FreeType2 : public Algorithm
{
public:
/** @brief Load font data.

The function loadFontData loads font data.

@param fontFileName FontFile Name
@param idx face_index to select a font faces in a single file.
*/

    CV_WRAP virtual void loadFontData(String fontFileName, int idx) = 0;

/** @brief Set Split Number from Bezier-curve to line

The function setSplitNumber set the number of split points from bezier-curve to line.
If you want to draw large glyph, large is better.
If you want to draw small glyph, small is better.

@param num number of split points from bezier-curve to line
*/

    CV_WRAP virtual void setSplitNumber( int num ) = 0;

/** @brief Draws a text string.

The function putText renders the specified text string in the image. Symbols that cannot be rendered using the specified font are replaced by "Tofu" or non-drawn.

@param img Image. (Only 8UC1/8UC3/8UC4 2D mat is supported.)
@param text Text string to be drawn.
@param org Bottom-left/Top-left corner of the text string in the image.
@param fontHeight Drawing font size by pixel unit.
@param color Text color.
@param thickness Thickness of the lines used to draw a text when negative, the glyph is filled. Otherwise, the glyph is drawn with this thickness.
@param line_type Line type. See the line for details.
@param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
*/

    CV_WRAP virtual void putText(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    ) = 0;

/** @brief Calculates the width and height of a text string.

The function getTextSize calculates and returns the approximate size of a box that contains the specified text.
That is, the following code renders some text, the tight box surrounding it, and the baseline: :
@code
    String text = "Funny text inside the box";
    int fontHeight = 60;
    int thickness = -1;
    int linestyle = LINE_8;

    Mat img(600, 800, CV_8UC3, Scalar::all(0));

    int baseline=0;

    cv::Ptr<cv::freetype::FreeType2> ft2;
    ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData( "./mplus-1p-regular.ttf", 0 );

    Size textSize = ft2->getTextSize(text,
                                     fontHeight,
                                     thickness,
                                     &baseline);

    if(thickness > 0){
        baseline += thickness;
    }

    // center the text
    Point textOrg((img.cols - textSize.width) / 2,
                  (img.rows + textSize.height) / 2);

    // draw the box
    rectangle(img, textOrg + Point(0, baseline),
              textOrg + Point(textSize.width, -textSize.height),
              Scalar(0,255,0),1,8);

    // ... and the baseline first
    line(img, textOrg + Point(0, thickness),
         textOrg + Point(textSize.width, thickness),
         Scalar(0, 0, 255),1,8);

    // then put the text itself
    ft2->putText(img, text, textOrg, fontHeight,
                 Scalar::all(255), thickness, linestyle, true );
@endcode

@param text Input text string.
@param fontHeight Drawing font size by pixel unit.
@param thickness Thickness of lines used to render the text. See putText for details.
@param[out] baseLine y-coordinate of the baseline relative to the bottom-most text
point.
@return The size of a box that contains the specified text.

@see cv::putText
 */
CV_WRAP virtual Size getTextSize(const String& text,
                            int fontHeight, int thickness,
                            CV_OUT int* baseLine) = 0;

};

/** @brief Create FreeType2 Instance

The function createFreeType2 create instance to draw UTF-8 strings.

*/
    CV_EXPORTS_W Ptr<FreeType2> createFreeType2();

//! @}
} } // namespace freetype

#endif
#endif
