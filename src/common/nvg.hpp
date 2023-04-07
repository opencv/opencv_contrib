// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#ifndef SRC_COMMON_NVG_HPP_
#define SRC_COMMON_NVG_HPP_

#include "viz2d.hpp"
#ifndef __EMSCRIPTEN__
#define NANOGUI_USE_OPENGL
#else
#define NANOGUI_USE_GLES
#define NANOGUI_GLES_VERSION 3
#endif
#include <nanogui/opengl.h>

namespace cv {
namespace viz {
/*!
 * In general please refere to https://github.com/memononen/nanovg/blob/master/src/nanovg.h for reference.
 */
namespace nvg {

/*!
 * Equivalent of a NVGtextRow.
 */
struct TextRow: public NVGtextRow {
};

/*!
 * Equivalent of a NVGglyphPosition.
 */
struct GlyphPosition: public NVGglyphPosition {
};

/*!
 * Equivalent of a NVGPaint. Converts back and forth between the two representations (Paint/NVGPaint).
 */
struct Paint {
    Paint() {
    }
    Paint(const NVGpaint& np) {
        memcpy(this->xform, np.xform, sizeof(this->xform));
        memcpy(this->extent, np.extent, sizeof(this->extent));
        this->radius = np.radius;
        this->feather = np.feather;
        this->innerColor = cv::Scalar(np.innerColor.rgba[2] * 255, np.innerColor.rgba[1] * 255,
                np.innerColor.rgba[0] * 255, np.innerColor.rgba[3] * 255);
        this->outerColor = cv::Scalar(np.outerColor.rgba[2] * 255, np.outerColor.rgba[1] * 255,
                np.outerColor.rgba[0] * 255, np.outerColor.rgba[3] * 255);
        this->image = np.image;
    }

    NVGpaint toNVGpaint() {
        NVGpaint np;
        memcpy(np.xform, this->xform, sizeof(this->xform));
        memcpy(np.extent, this->extent, sizeof(this->extent));
        np.radius = this->radius;
        np.feather = this->feather;
        np.innerColor = nvgRGBA(this->innerColor[2], this->innerColor[1], this->innerColor[0],
                this->innerColor[3]);
        np.outerColor = nvgRGBA(this->outerColor[2], this->outerColor[1], this->outerColor[0],
                this->outerColor[3]);
        np.image = this->image;
        return np;
    }

    float xform[6];
    float extent[2];
    float radius = 0;
    float feather = 0;
    cv::Scalar innerColor;
    cv::Scalar outerColor;
    int image = 0;
};

namespace detail {
/*!
 * Internal NanoVG singleton that wraps all NanoVG functions.
 */
class NVG {
private:
    friend class Viz2D;
    static NVG* nvg_instance_;
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

    void shapeAntiAlias(int enabled);
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
int createFont(const char* name, const char* filename);
/*!
 * A forward to nvgCreateFontMem. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int createFontMem(const char* name, unsigned char* data, int ndata, int freeData);
/*!
 * A forward to nvgFindFont. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int findFont(const char* name);
/*!
 * A forward to nvgAddFallbackFontId. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int addFallbackFontId(int baseFont, int fallbackFont);
/*!
 * A forward to nvgAddFallbackFont. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int addFallbackFont(const char* baseFont, const char* fallbackFont);
/*!
 * A forward to nvgFontSize. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fontSize(float size);
/*!
 * A forward to nvgFontBlur. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fontBlur(float blur);
/*!
 * A forward to nvgTextLetterSpacing. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textLetterSpacing(float spacing);
/*!
 * A forward to nvgTextLineHeight. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textLineHeight(float lineHeight);
/*!
 * A forward to nvgTextAlign. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textAlign(int align);
/*!
 * A forward to nvgFontFaceId. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fontFaceId(int font);
/*!
 * A forward to nvgFontFace. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fontFace(const char* font);
/*!
 * A forward to nvgText. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
float text(float x, float y, const char* string, const char* end);
/*!
 * A forward to nvgTextBox. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textBox(float x, float y, float breakRowWidth, const char* string, const char* end);
/*!
 * A forward to nvgTextBounds. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
float textBounds(float x, float y, const char* string, const char* end, float* bounds);
/*!
 * A forward to nvgTextBoxBounds. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end,
        float* bounds);
/*!
 * A forward to nvgTextGlyphPositions. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int textGlyphPositions(float x, float y, const char* string, const char* end,
        GlyphPosition* positions, int maxPositions);
/*!
 * A forward to nvgTextMetrics. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void textMetrics(float* ascender, float* descender, float* lineh);
/*!
 * A forward to nvgTextBreakLines. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows,
        int maxRows);
/*!
 * A forward to nvgSave. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void save();
/*!
 * A forward to nvgRestore. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void restore();
/*!
 * A forward to nvgReset. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void reset();
/*!
 * A forward to nvgShapeAntiAlias. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void shapeAntiAlias(int enabled);
/*!
 * A forward to nvgStrokeColor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void strokeColor(const cv::Scalar& bgra);
/*!
 * A forward to nvgStrokePaint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void strokePaint(Paint paint);
/*!
 * A forward to nvgFillColor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fillColor(const cv::Scalar& color);
/*!
 * A forward to nvgFillPaint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fillPaint(Paint paint);
/*!
 * A forward to nvgMiterLimit. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void miterLimit(float limit);
/*!
 * A forward to nvgStrokeWidth. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void strokeWidth(float size);
/*!
 * A forward to nvgLineCap. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void lineCap(int cap);
/*!
 * A forward to nvgLineJoin. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void lineJoin(int join);
/*!
 * A forward to nvgGlobalAlpha. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void globalAlpha(float alpha);

/*!
 * A forward to nvgResetTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void resetTransform();
/*!
 * A forward to nvgTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transform(float a, float b, float c, float d, float e, float f);
/*!
 * A forward to nvgTranslate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void translate(float x, float y);
/*!
 * A forward to nvgRotate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void rotate(float angle);
/*!
 * A forward to nvgSkewX. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void skewX(float angle);
/*!
 * A forward to nvgSkewY. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void skewY(float angle);
/*!
 * A forward to nvgScale. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void scale(float x, float y);
/*!
 * A forward to nvgCurrentTransform. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void currentTransform(float* xform);
/*!
 * A forward to nvgTransformIdentity. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformIdentity(float* dst);
/*!
 * A forward to nvgTransformTranslate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformTranslate(float* dst, float tx, float ty);
/*!
 * A forward to nvgTransformScale. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformScale(float* dst, float sx, float sy);
/*!
 * A forward to nvgTransformRotate. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformRotate(float* dst, float a);
/*!
 * A forward to nvgTransformSkewX. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformSkewX(float* dst, float a);
/*!
 * A forward to nvgTransformSkewY. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformSkewY(float* dst, float a);
/*!
 * A forward to nvgTransformMultiply. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformMultiply(float* dst, const float* src);
/*!
 * A forward to nvgTransformPremultiply. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformPremultiply(float* dst, const float* src);
/*!
 * A forward to nvgTransformInverse. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
int transformInverse(float* dst, const float* src);
/*!
 * A forward to nvgTransformPoint. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy);

/*!
 * A forward to nvgDegToRad. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
float degToRad(float deg);
/*!
 * A forward to nvgRadToDeg. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
float radToDeg(float rad);

/*!
 * A forward to nvgBeginPath. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void beginPath();
/*!
 * A forward to nvgMoveTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void moveTo(float x, float y);
/*!
 * A forward to nvgLineTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void lineTo(float x, float y);
/*!
 * A forward to nvgBezierTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y);
/*!
 * A forward to nvgQuadTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void quadTo(float cx, float cy, float x, float y);
/*!
 * A forward to nvgArcTo. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void arcTo(float x1, float y1, float x2, float y2, float radius);
/*!
 * A forward to nvgClosePath. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void closePath();
/*!
 * A forward to nvgPathWinding. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void pathWinding(int dir);
/*!
 * A forward to nvgArc. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void arc(float cx, float cy, float r, float a0, float a1, int dir);
/*!
 * A forward to nvgRect. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void rect(float x, float y, float w, float h);
/*!
 * A forward to nvgRoundedRect. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void roundedRect(float x, float y, float w, float h, float r);
/*!
 * A forward to nvgRoundedRectVarying. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight,
        float radBottomRight, float radBottomLeft);
/*!
 * A forward to nvgEllipse. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void ellipse(float cx, float cy, float rx, float ry);
/*!
 * A forward to nvgCircle. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void circle(float cx, float cy, float r);
/*!
 * A forward to nvgFill. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void fill();
/*!
 * A forward to nvgStroke. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void stroke();

/*!
 * A forward to nvgLinearGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
Paint linearGradient(float sx, float sy, float ex, float ey, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgBoxGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
Paint boxGradient(float x, float y, float w, float h, float r, float f, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgRadialGradient. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
Paint radialGradient(float cx, float cy, float inr, float outr, const cv::Scalar& icol,
        const cv::Scalar& ocol);
/*!
 * A forward to nvgImagePattern. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
Paint imagePattern(float ox, float oy, float ex, float ey, float angle, int image, float alpha);
/*!
 * A forward to nvgScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void scissor(float x, float y, float w, float h);
/*!
 * A forward to nvgIntersectScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void intersectScissor(float x, float y, float w, float h);
/*!
 * A forward to nvgRresetScissor. See https://github.com/memononen/nanovg/blob/master/src/nanovg.h
 */
void resetScissor();
}
}
}

#endif /* SRC_COMMON_NVG_HPP_ */
