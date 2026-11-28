// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_OPENCV_V4D_NVG_HPP_
#define SRC_OPENCV_V4D_NVG_HPP_

#include "opencv2/v4d/v4d.hpp"
#include <stdio.h>
#include <opencv2/core.hpp>
#include "nanovg.h"
struct NVGcontext;

namespace cv {
namespace v4d {
/*!
 * In general please refer to https://github.com/memononen/nanovg/blob/master/src/nanovg.h for reference.
 */
namespace nvg {
/*!
 * Equivalent of a NVGtextRow.
 */
struct CV_EXPORTS TextRow: public NVGtextRow {
};

/*!
 * Equivalent of a NVGglyphPosition.
 */
struct CV_EXPORTS GlyphPosition: public NVGglyphPosition {
};

/*!
 * Equivalent of a NVGPaint. Converts back and forth between the two representations (Paint/NVGPaint).
 */
struct CV_EXPORTS Paint {
    Paint() {
    }
    Paint(const NVGpaint& np);
    NVGpaint toNVGpaint();

    float xform[6];
    float extent[2];
    float radius = 0;
    float feather = 0;
    cv::Scalar innerColor;
    cv::Scalar outerColor;
    int image = 0;
};

/*!
 * Internals of the NanoVG wrapper
 */
namespace detail {
/*!
 * Internal NanoVG singleton that wraps all NanoVG functions.
 */
class NVG {
private:
    friend class V4D;
    static thread_local NVG* nvg_instance_;
    NVGcontext* ctx_ = nullptr;
    NVG(NVGcontext* ctx) :
            ctx_(ctx) {
    }
public:
    /*!
     * Initialize the current NVG object;
     * @param ctx The NVGcontext to create the NVG object from.
     */
    static void initializeContext(NVGcontext* ctx);
    /*!
     * Get the current NVGcontext.
     * @return The current NVGcontext context.
     */
    static NVG* getCurrentContext();

    /*!
     * Get the underlying NVGcontext.
     * @return The underlying NVGcontext.
     */
    NVGcontext* getContext() {
        assert(ctx_ != nullptr);
        return ctx_;
    }
public:
    int createFont(const char* name, const char* filename);
    int createFontMem(const char* name, unsigned char* data, int ndata, int freeData);
    int findFont(const char* name);
    int addFallbackFontId(int baseFont, int fallbackFont);
    int addFallbackFont(const char* baseFont, const char* fallbackFont);
    void fontSize(float size);
    void fontBlur(float blur);
    void textLetterSpacing(float spacing);
    void textLineHeight(float lineHeight);
    void textAlign(int align);
    void fontFaceId(int font);
    void fontFace(const char* font);
    float text(float x, float y, const char* string, const char* end);
    void textBox(float x, float y, float breakRowWidth, const char* string, const char* end);
    float textBounds(float x, float y, const char* string, const char* end, float* bounds);
    void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end,
            float* bounds);
    int textGlyphPositions(float x, float y, const char* string, const char* end,
            GlyphPosition* positions, int maxPositions);
    void textMetrics(float* ascender, float* descender, float* lineh);
    int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows,
            int maxRows);

    void save();
    void restore();
    void reset();

//    void shapeAntiAlias(int enabled);
    void strokeColor(const cv::Scalar& bgra);
    void strokePaint(Paint paint);
    void fillColor(const cv::Scalar& bgra);
    void fillPaint(Paint paint);
    void miterLimit(float limit);
    void strokeWidth(float size);
    void lineCap(int cap);
    void lineJoin(int join);
    void globalAlpha(float alpha);

    void resetTransform();
    void transform(float a, float b, float c, float d, float e, float f);
    void translate(float x, float y);
    void rotate(float angle);
    void skewX(float angle);
    void skewY(float angle);
    void scale(float x, float y);
    void currentTransform(float* xform);
    void transformIdentity(float* dst);
    void transformTranslate(float* dst, float tx, float ty);
    void transformScale(float* dst, float sx, float sy);
    void transformRotate(float* dst, float a);
    void transformSkewX(float* dst, float a);
    void transformSkewY(float* dst, float a);
    void transformMultiply(float* dst, const float* src);
    void transformPremultiply(float* dst, const float* src);
    int transformInverse(float* dst, const float* src);
    void transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy);

    float degToRad(float deg);
    float radToDeg(float rad);

    int createImage(const char* filename, int imageFlags);
    int createImageMem(int imageFlags, unsigned char* data, int ndata);
    int createImageRGBA(int w, int h, int imageFlags, const unsigned char* data);
    void updateImage(int image, const unsigned char* data);
    void imageSize(int image, int* w, int* h);
    void deleteImage(int image);

    void beginPath();
    void moveTo(float x, float y);
    void lineTo(float x, float y);
    void bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y);
    void quadTo(float cx, float cy, float x, float y);
    void arcTo(float x1, float y1, float x2, float y2, float radius);
    void closePath();
    void pathWinding(int dir);
    void arc(float cx, float cy, float r, float a0, float a1, int dir);
    void rect(float x, float y, float w, float h);
    void roundedRect(float x, float y, float w, float h, float r);
    void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight,
            float radBottomRight, float radBottomLeft);
    void ellipse(float cx, float cy, float rx, float ry);
    void circle(float cx, float cy, float r);
    void fill();
    void stroke();

    Paint linearGradient(float sx, float sy, float ex, float ey, const cv::Scalar& icol,
            const cv::Scalar& ocol);
    Paint boxGradient(float x, float y, float w, float h, float r, float f, const cv::Scalar& icol,
            const cv::Scalar& ocol);
    Paint radialGradient(float cx, float cy, float inr, float outr, const cv::Scalar& icol,
            const cv::Scalar& ocol);
    Paint imagePattern(float ox, float oy, float ex, float ey, float angle, int image, float alpha);
    void scissor(float x, float y, float w, float h);
    void intersectScissor(float x, float y, float w, float h);
    void resetScissor();
};
} // namespace detail

/*!
 * A forward to nvgCreateFont. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int createFont(const char* name, const char* filename);
/*!
 * A forward to nvgCreateFontMem. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int createFontMem(const char* name, unsigned char* data, int ndata, int freeData);
/*!
 * A forward to nvgFindFont. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int findFont(const char* name);
/*!
 * A forward to nvgAddFallbackFontId. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int addFallbackFontId(int baseFont, int fallbackFont);
/*!
 * A forward to nvgAddFallbackFont. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int addFallbackFont(const char* baseFont, const char* fallbackFont);
/*!
 * A forward to nvgFontSize. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fontSize(float size);
/*!
 * A forward to nvgFontBlur. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fontBlur(float blur);
/*!
 * A forward to nvgTextLetterSpacing. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textLetterSpacing(float spacing);
/*!
 * A forward to nvgTextLineHeight. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textLineHeight(float lineHeight);
/*!
 * A forward to nvgTextAlign. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textAlign(int align);
/*!
 * A forward to nvgFontFaceId. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fontFaceId(int font);
/*!
 * A forward to nvgFontFace. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fontFace(const char* font);
/*!
 * A forward to nvgText. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS float text(float x, float y, const char* string, const char* end);
/*!
 * A forward to nvgTextBox. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textBox(float x, float y, float breakRowWidth, const char* string, const char* end);
/*!
 * A forward to nvgTextBounds. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS float textBounds(float x, float y, const char* string, const char* end, float* bounds);
/*!
 * A forward to nvgTextBoxBounds. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end,
        float* bounds);
/*!
 * A forward to nvgTextGlyphPositions. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int textGlyphPositions(float x, float y, const char* string, const char* end,
        GlyphPosition* positions, int maxPositions);
/*!
 * A forward to nvgTextMetrics. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void textMetrics(float* ascender, float* descender, float* lineh);
/*!
 * A forward to nvgTextBreakLines. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows,
        int maxRows);
/*!
 * A forward to nvgSave. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void save();
/*!
 * A forward to nvgRestore. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void restore();
/*!
 * A forward to nvgReset. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void reset();
///*!
// * A forward to nvgShapeAntiAlias. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
// */
//CV_EXPORTS void shapeAntiAlias(int enabled);
/*!
 * A forward to nvgStrokeColor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void strokeColor(const cv::Scalar& bgra);
/*!
 * A forward to nvgStrokePaint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void strokePaint(Paint paint);
/*!
 * A forward to nvgFillColor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fillColor(const cv::Scalar& color);
/*!
 * A forward to nvgFillPaint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fillPaint(Paint paint);
/*!
 * A forward to nvgMiterLimit. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void miterLimit(float limit);
/*!
 * A forward to nvgStrokeWidth. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void strokeWidth(float size);
/*!
 * A forward to nvgLineCap. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void lineCap(int cap);
/*!
 * A forward to nvgLineJoin. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void lineJoin(int join);
/*!
 * A forward to nvgGlobalAlpha. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void globalAlpha(float alpha);

/*!
 * A forward to nvgResetTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void resetTransform();
/*!
 * A forward to nvgTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transform(float a, float b, float c, float d, float e, float f);
/*!
 * A forward to nvgTranslate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void translate(float x, float y);
/*!
 * A forward to nvgRotate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void rotate(float angle);
/*!
 * A forward to nvgSkewX. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void skewX(float angle);
/*!
 * A forward to nvgSkewY. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void skewY(float angle);
/*!
 * A forward to nvgScale. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void scale(float x, float y);
/*!
 * A forward to nvgCurrentTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void currentTransform(float* xform);
/*!
 * A forward to nvgTransformIdentity. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformIdentity(float* dst);
/*!
 * A forward to nvgTransformTranslate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformTranslate(float* dst, float tx, float ty);
/*!
 * A forward to nvgTransformScale. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformScale(float* dst, float sx, float sy);
/*!
 * A forward to nvgTransformRotate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformRotate(float* dst, float a);
/*!
 * A forward to nvgTransformSkewX. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformSkewX(float* dst, float a);
/*!
 * A forward to nvgTransformSkewY. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformSkewY(float* dst, float a);
/*!
 * A forward to nvgTransformMultiply. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformMultiply(float* dst, const float* src);
/*!
 * A forward to nvgTransformPremultiply. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformPremultiply(float* dst, const float* src);
/*!
 * A forward to nvgTransformInverse. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS int transformInverse(float* dst, const float* src);
/*!
 * A forward to nvgTransformPoint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy);

/*!
 * A forward to nvgDegToRad. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS float degToRad(float deg);
/*!
 * A forward to nvgRadToDeg. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS float radToDeg(float rad);

CV_EXPORTS int createImage(const char* filename, int imageFlags);
CV_EXPORTS int createImageMem(int imageFlags, unsigned char* data, int ndata);
CV_EXPORTS int createImageRGBA(int w, int h, int imageFlags, const unsigned char* data);
CV_EXPORTS void updateImage(int image, const unsigned char* data);
CV_EXPORTS void imageSize(int image, int* w, int* h);
CV_EXPORTS void deleteImage(int image);

/*!
 * A forward to nvgBeginPath. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void beginPath();
/*!
 * A forward to nvgMoveTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void moveTo(float x, float y);
/*!
 * A forward to nvgLineTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void lineTo(float x, float y);
/*!
 * A forward to nvgBezierTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y);
/*!
 * A forward to nvgQuadTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void quadTo(float cx, float cy, float x, float y);
/*!
 * A forward to nvgArcTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void arcTo(float x1, float y1, float x2, float y2, float radius);
/*!
 * A forward to nvgClosePath. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void closePath();
/*!
 * A forward to nvgPathWinding. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void pathWinding(int dir);
/*!
 * A forward to nvgArc. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void arc(float cx, float cy, float r, float a0, float a1, int dir);
/*!
 * A forward to nvgRect. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void rect(float x, float y, float w, float h);
/*!
 * A forward to nvgRoundedRect. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void roundedRect(float x, float y, float w, float h, float r);
/*!
 * A forward to nvgRoundedRectVarying. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight,
        float radBottomRight, float radBottomLeft);
/*!
 * A forward to nvgEllipse. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void ellipse(float cx, float cy, float rx, float ry);
/*!
 * A forward to nvgCircle. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void circle(float cx, float cy, float r);
/*!
 * A forward to nvgFill. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void fill();
/*!
 * A forward to nvgStroke. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void stroke();

/*!
 * A forward to nvgLinearGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS Paint linearGradient(float sx, float sy, float ex, float ey, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgBoxGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS Paint boxGradient(float x, float y, float w, float h, float r, float f, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgRadialGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS Paint radialGradient(float cx, float cy, float inr, float outr, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgImagePattern. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS Paint imagePattern(float ox, float oy, float ex, float ey, float angle, int image, float alpha);
/*!
 * A forward to nvgScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void scissor(float x, float y, float w, float h);
/*!
 * A forward to nvgIntersectScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void intersectScissor(float x, float y, float w, float h);
/*!
 * A forward to nvgRresetScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
CV_EXPORTS void resetScissor();

CV_EXPORTS void clear(const cv::Scalar& bgra = cv::Scalar(0, 0, 0, 255));
}
}
}

#endif /* SRC_OPENCV_V4D_NVG_HPP_ */
