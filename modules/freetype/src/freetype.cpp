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
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H
#include FT_IMAGE_H
#include FT_BBOX_H

#include <hb.h>
#include <hb-ft.h>

namespace cv {
namespace freetype {

using namespace std;

class CV_EXPORTS_W FreeType2Impl CV_FINAL : public FreeType2
{
public:
    FreeType2Impl();
    ~FreeType2Impl();
    void loadFontData(String fontFileName, int idx) CV_OVERRIDE;
    void setSplitNumber( int num ) CV_OVERRIDE;
    void putText(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    ) CV_OVERRIDE;
    Size getTextSize(
        const String& text, int fontHeight, int thickness,
        CV_OUT int* baseLine
    ) CV_OVERRIDE;

private:
    FT_Library       mLibrary;
    FT_Face          mFace;
    FT_Outline_Funcs mFn;

    bool             mIsFaceAvailable;
    int              mCtoL;
    hb_font_t        *mHb_font;

    void putTextBitmapMono(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    );

    void putTextBitmapBlend(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    );

    void putTextOutline(
        InputOutputArray img, const String& text, Point org,
        int fontHeight, Scalar color,
        int thickness, int line_type, bool bottomLeftOrigin
    );

    typedef void (putPixel_mono_fn)( Mat& _dst, const int _py, const int _px, const uint8_t *_col);
    putPixel_mono_fn putPixel_8UC1_mono;
    putPixel_mono_fn putPixel_8UC3_mono;
    putPixel_mono_fn putPixel_8UC4_mono;

    typedef void (putPixel_blend_fn)( Mat& _dst, const int _py, const int _px, const uint8_t *_col, const uint8_t alpha);
    putPixel_blend_fn putPixel_8UC1_blend;
    putPixel_blend_fn putPixel_8UC3_blend;
    putPixel_blend_fn putPixel_8UC4_blend;

    static int mvFn( const FT_Vector *to, void * user);
    static int lnFn( const FT_Vector *to, void * user);
    static int coFn( const FT_Vector *cnt,
                     const FT_Vector *to,
                     void * user);
    static int cuFn( const FT_Vector *cnt1,
                     const FT_Vector *cnt2,
                     const FT_Vector *to,
                     void * user);

    /**
     * Convert from FT_F26Dot6 to int(coodinate of OpenCV)
     * (FT_F26Dot6 is signed 26.6 real)
     */
    static int ftd(FT_F26Dot6 fixedInt){
        if ( fixedInt > 0 ) {
          return ( fixedInt + 32 ) / 64 ;
        }else{
          return ( fixedInt - 32 ) / 64 ;
        }
    }

    class PathUserData{
    private:
    public:
        PathUserData( InputOutputArray _img) : mImg(_img) {};

        InputOutputArray mImg;
        Scalar mColor;
        int    mThickness;
        int    mLine_type;
        FT_Vector        mOldP;
        int              mCtoL;
        std::vector < Point > mPts;
    };
};

FreeType2Impl::FreeType2Impl()
{
    FT_Init_FreeType(&(this->mLibrary) );

    mCtoL        = 16;
    mFn.shift    = 0;
    mFn.delta    = 0;
    mFn.move_to  = FreeType2Impl::mvFn;
    mFn.line_to  = FreeType2Impl::lnFn;
    mFn.cubic_to = FreeType2Impl::cuFn;
    mFn.conic_to = FreeType2Impl::coFn;

    mIsFaceAvailable = false;
}

FreeType2Impl::~FreeType2Impl()
{
    if( mIsFaceAvailable  == true )
    {
        hb_font_destroy (mHb_font);
        CV_Assert(!FT_Done_Face(mFace));
        mIsFaceAvailable = false;
    }
    CV_Assert(!FT_Done_FreeType(mLibrary));
}

void FreeType2Impl::loadFontData(String fontFileName, int idx)
{
    CV_Assert( idx >= 0 );
    if( mIsFaceAvailable  == true )
    {
        hb_font_destroy (mHb_font);
        CV_Assert(!FT_Done_Face(mFace));
    }

    mIsFaceAvailable = false;
    CV_Assert( !FT_New_Face( mLibrary, fontFileName.c_str(), static_cast<FT_Long>(idx), &(mFace) ) );

    mHb_font = hb_ft_font_create (mFace, NULL);
    if ( mHb_font == NULL )
    {
        CV_Assert(!FT_Done_Face(mFace));
        return;
    }
    CV_Assert( mHb_font != NULL );
    mIsFaceAvailable = true;
}

void FreeType2Impl::setSplitNumber(int num ){
    CV_Assert( num > 0 );
    mCtoL        = num;
}

void FreeType2Impl::putText(
    InputOutputArray _img, const String& _text, Point _org,
    int _fontHeight, Scalar _color,
    int _thickness, int _line_type, bool _bottomLeftOrigin
)
{
    CV_Assert  ( mIsFaceAvailable == true );
    CV_Assert  ( _img.empty()    == false );
    CV_Assert  ( _img.isMat()    == true  );
    CV_Assert  ( _img.dims()     == 2     );
    CV_Assert( ( _img.type()     == CV_8UC1 ) ||
               ( _img.type()     == CV_8UC3 ) ||
               ( _img.type()     == CV_8UC4 ) );
    CV_Assert( ( _line_type == LINE_AA) ||
               ( _line_type == LINE_4 ) ||
               ( _line_type == LINE_8 ) );
    CV_Assert  ( _fontHeight >= 0 );

    if ( _text.empty() )
    {
         return;
    }
    if ( _fontHeight == 0 )
    {
         return;
    }

    CV_Assert(!FT_Set_Pixel_Sizes( mFace, _fontHeight, _fontHeight ));

    if( _thickness < 0 ) // CV_FILLED
    {
        if ( _line_type == LINE_AA ) {
            putTextBitmapBlend( _img, _text, _org, _fontHeight, _color,
                _thickness, _line_type, _bottomLeftOrigin );
        }else{
            putTextBitmapMono( _img, _text, _org, _fontHeight, _color,
                _thickness, _line_type, _bottomLeftOrigin );
        }
    }else{
            putTextOutline( _img, _text, _org, _fontHeight, _color,
                _thickness, _line_type, _bottomLeftOrigin );
    }
}

void FreeType2Impl::putTextOutline(
   InputOutputArray _img, const String& _text, Point _org,
   int _fontHeight, Scalar _color,
   int _thickness, int _line_type, bool _bottomLeftOrigin )
{
    hb_buffer_t *hb_buffer = hb_buffer_create ();
    CV_Assert( hb_buffer != NULL );

    hb_buffer_add_utf8 (hb_buffer, _text.c_str(), -1, 0, -1);
    hb_buffer_guess_segment_properties (hb_buffer);
    hb_shape (mHb_font, hb_buffer, NULL, 0);

    unsigned int textLen = 0;
    hb_glyph_info_t *info =
        hb_buffer_get_glyph_infos(hb_buffer,&textLen );
    CV_Assert( info != NULL );

    PathUserData *userData = new PathUserData( _img );
    userData->mColor     = _color;
    userData->mCtoL      = mCtoL;
    userData->mThickness = _thickness;
    userData->mLine_type = _line_type;

    // Initilize currentPosition ( in FreeType coordinates)
    FT_Vector currentPos = {0,0};
    currentPos.x = _org.x * 64;
    currentPos.y = _org.y * 64;

    // Update currentPosition with bottomLeftOrigin ( in FreeType coordinates)
    if( _bottomLeftOrigin != true ){
        currentPos.y += _fontHeight * 64;
    }

    for( unsigned int i = 0 ; i < textLen ; i ++ ){
        CV_Assert(!FT_Load_Glyph(mFace, info[i].codepoint, 0 ));

        FT_GlyphSlot slot  = mFace->glyph;
        FT_Outline outline = slot->outline;

        // Flip ( in FreeType coordinates )
        FT_Matrix mtx = { 1 << 16 , 0 , 0 , -(1 << 16) };
        FT_Outline_Transform(&outline, &mtx);

        // Move to current position ( in FreeType coordinates )
        FT_Outline_Translate(&outline,
                             currentPos.x,
                             currentPos.y);

        // Draw ( in FreeType coordinates )
        CV_Assert( !FT_Outline_Decompose(&outline, &mFn, (void*)userData) );

        // Draw (Last Path) ( in FreeType coordinates )
        mvFn( NULL, (void*)userData );

        // Update current position ( in FreeType coordinates )
        currentPos.x += mFace->glyph->advance.x;
        currentPos.y += mFace->glyph->advance.y;
   }
   delete userData;
   hb_buffer_destroy (hb_buffer);
}

void FreeType2Impl::putPixel_8UC1_mono( Mat& _dst, const int _py, const int _px, const uint8_t *_col)
{
    uint8_t* ptr = _dst.ptr<uint8_t>( _py, _px );
    (*ptr) = _col[0];
}

void FreeType2Impl::putPixel_8UC3_mono ( Mat& _dst, const int _py, const int _px, const uint8_t *_col)
{
    cv::Vec3b* ptr = _dst.ptr<cv::Vec3b>( _py, _px );
    (*ptr)[0] = _col[0];
    (*ptr)[1] = _col[1];
    (*ptr)[2] = _col[2];
}

void FreeType2Impl::putPixel_8UC4_mono( Mat& _dst, const int _py, const int _px, const uint8_t *_col)
{
    cv::Vec4b* ptr = _dst.ptr<cv::Vec4b>( _py, _px );
    (*ptr)[0] = _col[0];
    (*ptr)[1] = _col[1];
    (*ptr)[2] = _col[2];
    (*ptr)[3] = _col[3];
}

void FreeType2Impl::putTextBitmapMono(
   InputOutputArray _img, const String& _text, Point _org,
   int _fontHeight, Scalar _color,
   int _thickness, int _line_type, bool _bottomLeftOrigin )
{
    CV_Assert( _thickness < 0 );
    CV_Assert( _line_type == LINE_4 || _line_type == LINE_8);

    Mat dst = _img.getMat();
    hb_buffer_t *hb_buffer = hb_buffer_create ();
    CV_Assert( hb_buffer != NULL );

    hb_buffer_add_utf8 (hb_buffer, _text.c_str(), -1, 0, -1);
    hb_buffer_guess_segment_properties (hb_buffer);
    hb_shape (mHb_font, hb_buffer, NULL, 0);

    unsigned int textLen = 0;
    hb_glyph_info_t *info =
        hb_buffer_get_glyph_infos(hb_buffer,&textLen );
    CV_Assert( info != NULL );

    _org.y += _fontHeight;
    if( _bottomLeftOrigin == true ){
        _org.y -= _fontHeight;
    }

    const uint8_t _colorUC8n[4] = {
        static_cast<uint8_t>(_color[0]),
        static_cast<uint8_t>(_color[1]),
        static_cast<uint8_t>(_color[2]),
        static_cast<uint8_t>(_color[3]) };

    void (cv::freetype::FreeType2Impl::*putPixel)( Mat&, const int, const int, const uint8_t*) =
        (_img.type() == CV_8UC4)?(&FreeType2Impl::putPixel_8UC4_mono):
        (_img.type() == CV_8UC3)?(&FreeType2Impl::putPixel_8UC3_mono):
                                 (&FreeType2Impl::putPixel_8UC1_mono);

    for( unsigned int i = 0 ; i < textLen ; i ++ ){
        CV_Assert( !FT_Load_Glyph(mFace, info[i].codepoint, 0 ) );
        CV_Assert( !FT_Render_Glyph( mFace->glyph, FT_RENDER_MODE_MONO ) );
        FT_Bitmap    *bmp = &(mFace->glyph->bitmap);

        Point gPos = _org;
        gPos.y -= ( mFace->glyph->metrics.horiBearingY >> 6) ;
        gPos.x += ( mFace->glyph->metrics.horiBearingX >> 6) ;

        for (int row = 0; row < (int)bmp->rows; row ++) {
            if( gPos.y + row < 0 ) {
                continue;
            }
            if( gPos.y + row >= dst.rows ) {
                break;
            }

            for (int col = 0; col < bmp->pitch; col ++) {
                int cl = bmp->buffer[ row * bmp->pitch + col ];
                if ( cl == 0 ) {
                    continue;
                }
                for(int bit = 7; bit >= 0; bit -- ){
                    if( gPos.x + col * 8 + (7 - bit) < 0 )
                    {
                        continue;
                    }
                    if( gPos.x + col * 8 + (7 - bit) >= dst.cols )
                    {
                        break;
                    }

                    if ( ( (cl >> bit) & 0x01 ) == 1 ) {
                        (this->*putPixel)( dst, gPos.y + row, gPos.x + col * 8 + (7 - bit), _colorUC8n );
                    }
                }
            }
        }

        _org.x += ( mFace->glyph->advance.x ) >> 6;
        _org.y += ( mFace->glyph->advance.y ) >> 6;
    }
    hb_buffer_destroy (hb_buffer);
}

// Alpha composite algorithm is porting from imgproc.
// See https://github.com/opencv/opencv/blob/4.6.0/modules/imgproc/src/drawing.cpp
// static void LineAA( Mat& img, Point2l pt1, Point2l pt2, const void* color )
// ICV_PUT_POINT Macro.

void FreeType2Impl::putPixel_8UC1_blend( Mat& _dst, const int _py, const int _px, const uint8_t *_col, const uint8_t alpha)
{
    const int a = alpha;
    const int cb = _col[0];
    uint8_t* tptr = _dst.ptr<uint8_t>( _py, _px );

    int _cb = static_cast<int>(tptr[0]);
    _cb += ((cb - _cb)*a + 127)>> 8;
    _cb += ((cb - _cb)*a + 127)>> 8;

    tptr[0] = static_cast<uint8_t>(_cb);
}

void FreeType2Impl::putPixel_8UC3_blend ( Mat& _dst, const int _py, const int _px, const uint8_t *_col, const uint8_t alpha)
{
    const int a = alpha;
    const int cb = _col[0];
    const int cg = _col[1];
    const int cr = _col[2];
    uint8_t* tptr = _dst.ptr<uint8_t>( _py, _px );

    int _cb = static_cast<int>(tptr[0]);
    _cb += ((cb - _cb)*a + 127)>> 8;
    _cb += ((cb - _cb)*a + 127)>> 8;

    int _cg = static_cast<int>(tptr[1]);
    _cg += ((cg - _cg)*a + 127)>> 8;
    _cg += ((cg - _cg)*a + 127)>> 8;

    int _cr = static_cast<int>(tptr[2]);
    _cr += ((cr - _cr)*a + 127)>> 8;
    _cr += ((cr - _cr)*a + 127)>> 8;

    tptr[0] = static_cast<uint8_t>(_cb);
    tptr[1] = static_cast<uint8_t>(_cg);
    tptr[2] = static_cast<uint8_t>(_cr);
}

void FreeType2Impl::putPixel_8UC4_blend( Mat& _dst, const int _py, const int _px, const uint8_t *_col, const uint8_t alpha)
{
    const uint8_t a = alpha;
    const int cb = _col[0];
    const int cg = _col[1];
    const int cr = _col[2];
    const int ca = _col[3];
    uint8_t* tptr = _dst.ptr<uint8_t>( _py, _px );

    int _cb = static_cast<int>(tptr[0]);
    _cb += ((cb - _cb)*a + 127)>> 8;
    _cb += ((cb - _cb)*a + 127)>> 8;

    int _cg = static_cast<int>(tptr[1]);
    _cg += ((cg - _cg)*a + 127)>> 8;
    _cg += ((cg - _cg)*a + 127)>> 8;

    int _cr = static_cast<int>(tptr[2]);
    _cr += ((cr - _cr)*a + 127)>> 8;
    _cr += ((cr - _cr)*a + 127)>> 8;

    int _ca = static_cast<int>(tptr[3]);
    _ca += ((ca - _ca)*a + 127)>> 8;
    _ca += ((ca - _ca)*a + 127)>> 8;

    tptr[0] = static_cast<uint8_t>(_cb);
    tptr[1] = static_cast<uint8_t>(_cg);
    tptr[2] = static_cast<uint8_t>(_cr);
    tptr[3] = static_cast<uint8_t>(_ca);
}

void FreeType2Impl::putTextBitmapBlend(
   InputOutputArray _img, const String& _text, Point _org,
   int _fontHeight, Scalar _color,
   int _thickness, int _line_type, bool _bottomLeftOrigin )
{

    CV_Assert( _thickness < 0 );
    CV_Assert( _line_type == LINE_AA );

    Mat dst = _img.getMat();
    hb_buffer_t *hb_buffer = hb_buffer_create ();
    CV_Assert( hb_buffer != NULL );

    hb_buffer_add_utf8 (hb_buffer, _text.c_str(), -1, 0, -1);
    hb_buffer_guess_segment_properties (hb_buffer);
    hb_shape (mHb_font, hb_buffer, NULL, 0);

    unsigned int textLen = 0;
    hb_glyph_info_t *info =
        hb_buffer_get_glyph_infos(hb_buffer,&textLen );
    CV_Assert( info != NULL );

    _org.y += _fontHeight;
    if( _bottomLeftOrigin == true ){
        _org.y -= _fontHeight;
    }

    const uint8_t _colorUC8n[4] = {
        static_cast<uint8_t>(_color[0]),
        static_cast<uint8_t>(_color[1]),
        static_cast<uint8_t>(_color[2]),
        static_cast<uint8_t>(_color[3]) };

    void (cv::freetype::FreeType2Impl::*putPixel)( Mat&, const int, const int, const uint8_t*, const uint8_t) =
        (_img.type() == CV_8UC4)?(&FreeType2Impl::putPixel_8UC4_blend):
        (_img.type() == CV_8UC3)?(&FreeType2Impl::putPixel_8UC3_blend):
                                 (&FreeType2Impl::putPixel_8UC1_blend);

    for( unsigned int i = 0 ; i < textLen ; i ++ ){
        CV_Assert( !FT_Load_Glyph(mFace, info[i].codepoint, 0 ) );
        CV_Assert( !FT_Render_Glyph( mFace->glyph, FT_RENDER_MODE_NORMAL ) );
        FT_Bitmap    *bmp = &(mFace->glyph->bitmap);

        Point gPos = _org;
        gPos.y -= ( mFace->glyph->metrics.horiBearingY >> 6) ;
        gPos.x += ( mFace->glyph->metrics.horiBearingX >> 6) ;

        for (int row = 0; row < (int)bmp->rows; row ++) {
            if( gPos.y + row < 0 ) {
                continue;
            }
            if( gPos.y + row >= dst.rows ) {
                break;
            }

            for (int col = 0; col < bmp->pitch; col ++) {
                uint8_t cl = bmp->buffer[ row * bmp->pitch + col ];
                if ( cl == 0 ) {
                    continue;
                }
                if( gPos.x + col < 0 )
                {
                    continue;
                }
                if( gPos.x + col >= dst.cols )
                {
                    break;
                }

                (this->*putPixel)( dst, gPos.y + row, gPos.x + col, _colorUC8n, cl );
            }
        }
        _org.x += ( mFace->glyph->advance.x ) >> 6;
        _org.y += ( mFace->glyph->advance.y ) >> 6;
    }
    hb_buffer_destroy (hb_buffer);
}

Size FreeType2Impl::getTextSize(
    const String& _text,
    int _fontHeight,
    int _thickness,
    CV_OUT int* _baseLine)
{
    if ( _text.empty() )
    {
         return Size(0,0);
    }

    CV_Assert( _fontHeight >= 0 ) ;
    if ( _fontHeight == 0 )
    {
         return Size(0,0);
    }

    CV_Assert(!FT_Set_Pixel_Sizes( mFace, _fontHeight, _fontHeight ));

    hb_buffer_t *hb_buffer = hb_buffer_create ();
    CV_Assert( hb_buffer != NULL );
    FT_Vector currentPos = {0,0};

    hb_buffer_add_utf8 (hb_buffer, _text.c_str(), -1, 0, -1);
    hb_buffer_guess_segment_properties (hb_buffer);
    hb_shape (mHb_font, hb_buffer, NULL, 0);

    unsigned int textLen = 0;
    hb_glyph_info_t *info =
        hb_buffer_get_glyph_infos(hb_buffer,&textLen );
    CV_Assert( info != NULL );

    // Initilize BoundaryBox ( in OpenCV coordinates )
    int xMin = INT_MAX, yMin = INT_MAX;
    int xMax = INT_MIN, yMax = INT_MIN;

    for( unsigned int i = 0 ; i < textLen ; i ++ ){
        CV_Assert(!FT_Load_Glyph(mFace, info[i].codepoint, 0 ));

        FT_GlyphSlot slot  = mFace->glyph;
        FT_Outline outline = slot->outline;
        FT_BBox bbox ;

        // Flip ( in FreeType coordinates )
        FT_Matrix mtx = { 1 << 16 , 0 , 0 , -(1 << 16) };
        FT_Outline_Transform(&outline, &mtx);

        // Move to current position ( in FreeType coordinates )
        FT_Outline_Translate(&outline,
                             currentPos.x,
                             currentPos.y );

        // Get BoundaryBox ( in FreeType coordinatrs )
        CV_Assert( !FT_Outline_Get_BBox( &outline, &bbox ) );

        // If codepoint is space(0x20), it has no glyph.
        // A dummy boundary box is needed when last code is space.
        if(
            (bbox.xMin == 0 ) && (bbox.xMax == 0 ) &&
            (bbox.yMin == 0 ) && (bbox.yMax == 0 )
        ){
            bbox.xMin = currentPos.x ;
            bbox.xMax = currentPos.x + ( mFace->glyph->advance.x );
            bbox.yMin = yMin;
            bbox.yMax = yMax;
        }

        // Update current position ( in FreeType coordinates )
        currentPos.x += mFace->glyph->advance.x;
        currentPos.y += mFace->glyph->advance.y;

        // Update BoundaryBox ( in OpenCV coordinates )
        xMin = cv::min ( xMin, ftd(bbox.xMin) );
        xMax = cv::max ( xMax, ftd(bbox.xMax) );
        yMin = cv::min ( yMin, ftd(bbox.yMin) );
        yMax = cv::max ( yMax, ftd(bbox.yMax) );
    }

    hb_buffer_destroy (hb_buffer);

    // Calcurate width/height/baseline ( in OpenCV coordinates )
    int width  = xMax - xMin ;
    int height = -yMin ;

    if ( _thickness > 0 ) {
        width  = cvRound(width  + _thickness * 2);
        height = cvRound(height + _thickness * 1);
    }else{
        width  = cvRound(width  + 1);
        height = cvRound(height + 1);
    }

    if ( _baseLine ) {
        *_baseLine = yMax;
    }

    return Size( width, height );
}

int FreeType2Impl::mvFn( const FT_Vector *to, void * user)
{
    if(user == NULL ) { return 1; }
    PathUserData *p = (PathUserData*)user;

    // Draw polylines( in OpenCV coordinates ).
    if( p->mPts.size() > 0 ){
        Mat dst = p->mImg.getMat();
        const Point *ptsList[] = { &(p->mPts[0]) };
        int npt[1]; npt[0] = p->mPts.size();
        polylines(
            dst,
            ptsList,
            npt,
            1,
            false,
            p->mColor,
            p->mThickness,
            p->mLine_type,
            0
        );
    }

    p->mPts.clear();

    if( to == NULL ) { return 1; }

    // Store points to draw( in OpenCV coordinates ).
    p->mPts.push_back( Point ( ftd(to->x), ftd(to->y) ) );
    p->mOldP = *to;
    return 0;
}

int FreeType2Impl::lnFn( const FT_Vector *to, void * user)
{
    if(to   == NULL ) { return 1; }
    if(user == NULL ) { return 1; }

    PathUserData *p = (PathUserData *)user;

    // Store points to draw( in OpenCV coordinates ).
    p->mPts.push_back( Point ( ftd(to->x), ftd(to->y) ) );
    p->mOldP = *to;
    return 0;
}

int FreeType2Impl::coFn( const FT_Vector *cnt,
                     const FT_Vector *to,
                     void * user)
{
    if(cnt  == NULL ) { return 1; }
    if(to   == NULL ) { return 1; }
    if(user == NULL ) { return 1; }

    PathUserData *p = (PathUserData *)user;

    // Bezier to Line
    for(int i = 0;i <= p->mCtoL; i++){
        // Split Bezier to lines ( in FreeType coordinates ).
        double u = (double)i * 1.0 / (p->mCtoL) ;
        double nu = 1.0 - u;
        double p0 =                  nu * nu;
        double p1 = 2.0 * u *        nu;
        double p2 =       u * u;

        double X = (p->mOldP.x) * p0 + cnt->x * p1 + to->x * p2;
        double Y = (p->mOldP.y) * p0 + cnt->y * p1 + to->y * p2;

        // Store points to draw( in OpenCV coordinates ).
        p->mPts.push_back( Point ( ftd(X), ftd(Y) ) );
    }
    p->mOldP = *to;
    return 0;
}

int FreeType2Impl::cuFn( const FT_Vector *cnt1,
                     const FT_Vector *cnt2,
                     const FT_Vector *to,
                     void * user)
{
    if(cnt1 == NULL ) { return 1; }
    if(cnt2 == NULL ) { return 1; }
    if(to   == NULL ) { return 1; }
    if(user == NULL ) { return 1; }

    PathUserData *p = (PathUserData *)user;

    // Bezier to Line
    for(int i = 0; i <= p->mCtoL ;i++){
        // Split Bezier to lines ( in FreeType coordinates ).
        double u = (double)i * 1.0 / (p->mCtoL) ;
        double nu = 1.0 - u;
        double p0 =                  nu * nu * nu;
        double p1 = 3.0 * u *        nu * nu;
        double p2 = 3.0 * u * u *    nu;
        double p3 =       u * u * u;

        double X = (p->mOldP.x) * p0 + (cnt1->x)    * p1 +
                   (cnt2->x   ) * p2 + (to->x  )    * p3;
        double Y = (p->mOldP.y) * p0 + (cnt1->y)    * p1 +
                   (cnt2->y   ) * p2 + (to->y  )    * p3;

        // Store points to draw( in OpenCV coordinates ).
        p->mPts.push_back( Point ( ftd(X), ftd(Y) ) );
    }
    p->mOldP = *to;
    return 0;
}

CV_EXPORTS_W Ptr<FreeType2> createFreeType2()
{
    return Ptr<FreeType2Impl> (new FreeType2Impl () );
}


}} // namespace freetype2
