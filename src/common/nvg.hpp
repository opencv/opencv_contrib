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
namespace nvg {

struct TextRow: public NVGtextRow {
};

struct GlyphPosition: public NVGglyphPosition {
};

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

class NVG {
    friend class Viz2D;
    static NVG* nvg_instance_;
    NVGcontext* ctx_ = nullptr;

public:
    NVG(NVGcontext* ctx) :
            ctx_(ctx) {
    }

    static void setCurrentContext(NVGcontext* ctx);
    static NVG* getCurrentContext();

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
void fillColor(const cv::Scalar& color);
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
}
}
}

#endif /* SRC_COMMON_NVG_HPP_ */
