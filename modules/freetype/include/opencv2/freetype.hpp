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

#include <vector>
#include <string>

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftoutln.h>

#include <hb.h>
#include <hb-ft.h>

/**
@defgroup plot Plot function for Mat data
*/

namespace cv {
namespace freetype {
//! @addtogroup freetype
//! @{
class CV_EXPORTS_W FreeType2
{
private:
    FT_Library       mLibrary;
    FT_Face          mFace;
    FT_Outline_Funcs mFn;

    std::vector < Point > mPts;

    Point            mOrg;
    int              mLine_type;
    int              mThickness;
    int              mHeight;
    Scalar           mColor;
    Mat              mImg;
    FT_Vector        mOldP;
    bool             mIsFaceAvailable;
    std::string      mText;
    int              mCtoL;
    hb_font_t        *mHb_font;

    void putTextBitmapMono();
    void putTextBitmapBlend();
    void putTextOutline();

    static int mvFn( const FT_Vector *to, void * user);
    static int lnFn( const FT_Vector *to, void * user);
    static int coFn( const FT_Vector *cnt, 
                     const FT_Vector *to,
                     void * user);
    static int cuFn( const FT_Vector *cnt1, 
                     const FT_Vector *cnt2,
                     const FT_Vector *to,
                     void * user);
    static void readNextCode(FT_Long &c, int &i, const String &text );

    static unsigned int ftd(unsigned int a){ 
        return (unsigned int)(a + (1 << 5)  ) >> 6;
    }

public:
    CV_WRAP FreeType2();
    CV_WRAP ~FreeType2();

/** @brief Load font data.

The function loadFontData loads font data. 

@param fontFileName FontFile Name 
@param id face_index to select a font faces in a single file.
*/

    CV_WRAP void loadFontData(std::string fontFileName, int id);

/** @brief Set Split Number from Bezier-curve to line

The function setSplitNumber set the number of split points from bezier-curve to line.
If you want to draw large glyph, large is better.
If you want to draw small glyph, small is better.

@param num number of split points from bezier-curve to line
*/

    CV_WRAP void setSplitNumber( unsigned int num );

/** @brief Draws a text string.

The function putText renders the specified text string in the image. Symbols that cannot be rendered using the specified font are replaced by "Tofu" or non-drawn. 

@param img Image.
@param text Text string to be drawn.
@param org Bottom-left corner of the text string in the image.
@param fontHeight Scale Font scale factor by pixel unit.
@param color Text color.
@param thickness Thickness of the lines used to draw a text when negative, the glyph is filled. Otherwise, the glyph is drawn with this thickness.
@param lineType Line type. See the line for details.
@param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,
it is at the top-left corner.
*/

    CV_WRAP void putText(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    );
};

} } // namespace freetype

#endif
#endif
