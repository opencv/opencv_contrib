#ifndef SRC_COMMON_NVG_HPP_
#define SRC_COMMON_NVG_HPP_

#include "viz2d.hpp"
#define NANOGUI_USE_OPENGL
#include <nanogui/opengl.h>

namespace kb {
namespace nvg {

struct TextRow {
    const char* start;  // Pointer to the input text where the row starts.
    const char* end;    // Pointer to the input text where the row ends (one past the last character).
    const char* next;   // Pointer to the beginning of the next row.
    float width;        // Logical width of the row.
    float minx, maxx;   // Actual bounds of the row. Logical with and bounds can differ because of kerning and some parts over extending.
};

struct GlyphPosition {
    const char* str;    // Position of the glyph in the input string.
    float x;            // The x-coordinate of the logical glyph position.
    float minx, maxx;   // The bounds of the glyph shape.
};

struct Paint {
    float xform[6];
    float extent[2];
    float radius;
    float feather;
    cv::Scalar innerColor;
    cv::Scalar outerColor;
    int image;
};

namespace detail {

class NVG {
    friend class Viz2D;
    NVGcontext* ctx_;
public:
    NVG(NVGcontext* ctx) : ctx_(ctx) {
    }

    NVGcontext* getContext() {
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
void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end, float* bounds);
int textGlyphPositions(float x, float y, const char* string, const char* end, GlyphPosition* positions, int maxPositions);
void textMetrics(float* ascender, float* descender, float* lineh);
int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows, int maxRows);

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
void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight, float radBottomRight, float radBottomLeft);
void ellipse(float cx, float cy, float rx, float ry);
void circle(float cx, float cy, float r);
void fill();
void stroke();
};

static NVG* nvg_instance;

void set_current_context(NVGcontext* ctx);
NVG* get_current_context();
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
void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end, float* bounds);
int textGlyphPositions(float x, float y, const char* string, const char* end, GlyphPosition* positions, int maxPositions);
void textMetrics(float* ascender, float* descender, float* lineh);
int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows, int maxRows);

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
void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight, float radBottomRight, float radBottomLeft);
void ellipse(float cx, float cy, float rx, float ry);
void circle(float cx, float cy, float r);
void fill();
void stroke();
} // namespace nvg
} // namespace kb



#endif /* SRC_COMMON_NVG_HPP_ */
