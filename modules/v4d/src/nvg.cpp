// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>

#include "opencv2/v4d/nvg.hpp"
#include "opencv2/v4d/v4d.hpp"

namespace cv {
namespace v4d {
namespace nvg {
namespace detail {
class NVG;

thread_local NVG* NVG::nvg_instance_ = nullptr;

void NVG::initializeContext(NVGcontext* ctx) {
    if (nvg_instance_ != nullptr)
        delete nvg_instance_;
    nvg_instance_ = new NVG(ctx);
}

NVG* NVG::getCurrentContext() {
    assert(nvg_instance_ != nullptr);
    return nvg_instance_;
}

int NVG::createFont(const char* name, const char* filename) {
    return nvgCreateFont(getContext(), name, filename);
}

int NVG::createFontMem(const char* name, unsigned char* data, int ndata, int freeData) {
    return nvgCreateFontMem(getContext(), name, data, ndata, freeData);
}

int NVG::findFont(const char* name) {
    return nvgFindFont(getContext(), name);
}

int NVG::addFallbackFontId(int baseFont, int fallbackFont) {
    return nvgAddFallbackFontId(getContext(), baseFont, fallbackFont);
}

int NVG::addFallbackFont(const char* baseFont, const char* fallbackFont) {
    return nvgAddFallbackFont(getContext(), baseFont, fallbackFont);
}

void NVG::fontSize(float size) {
    nvgFontSize(getContext(), size);
}

void NVG::fontBlur(float blur) {
    nvgFontBlur(getContext(), blur);
}

void NVG::textLetterSpacing(float spacing) {
    nvgTextLetterSpacing(getContext(), spacing);
}

void NVG::textLineHeight(float lineHeight) {
    nvgTextLineHeight(getContext(), lineHeight);
}

void NVG::textAlign(int align) {
    nvgTextAlign(getContext(), align);
}

void NVG::fontFaceId(int font) {
    nvgFontFaceId(getContext(), font);
}

void NVG::fontFace(const char* font) {
    nvgFontFace(getContext(), font);
}

float NVG::text(float x, float y, const char* string, const char* end) {
    return nvgText(getContext(), x, y, string, end);
}

void NVG::textBox(float x, float y, float breakRowWidth, const char* string, const char* end) {
    nvgTextBox(getContext(), x, y, breakRowWidth, string, end);
}

float NVG::textBounds(float x, float y, const char* string, const char* end, float* bounds) {
    return nvgTextBounds(getContext(), x, y, string, end, bounds);
}

void NVG::textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end,
        float* bounds) {
    nvgTextBoxBounds(getContext(), x, y, breakRowWidth, string, end, bounds);
}

int NVG::textGlyphPositions(float x, float y, const char* string, const char* end,
        GlyphPosition* positions, int maxPositions) {
    return nvgTextGlyphPositions(getContext(), x, y, string, end, positions, maxPositions);
}

void NVG::textMetrics(float* ascender, float* descender, float* lineh) {
    nvgTextMetrics(getContext(), ascender, descender, lineh);
}

int NVG::textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows,
        int maxRows) {
    return nvgTextBreakLines(getContext(), string, end, breakRowWidth, rows, maxRows);
}

void NVG::save() {
    nvgSave(getContext());
}

void NVG::restore() {
    nvgRestore(getContext());
}

void NVG::reset() {
    nvgReset(getContext());
}

//void NVG::shapeAntiAlias(int enabled) {
//    nvgShapeAntiAlias(getContext(), enabled);
//}

void NVG::strokeColor(const cv::Scalar& bgra) {
    nvgStrokeColor(getContext(), nvgRGBA(bgra[2], bgra[1], bgra[0], bgra[3]));
}

void NVG::strokePaint(Paint paint) {
    NVGpaint np = paint.toNVGpaint();
    nvgStrokePaint(getContext(), np);
}

void NVG::fillColor(const cv::Scalar& rgba) {
    nvgFillColor(getContext(), nvgRGBA(rgba[0], rgba[1], rgba[2], rgba[3]));
}

void NVG::fillPaint(Paint paint) {
    NVGpaint np = paint.toNVGpaint();
    nvgFillPaint(getContext(), np);
}

void NVG::miterLimit(float limit) {
    nvgMiterLimit(getContext(), limit);
}

void NVG::strokeWidth(float size) {
    nvgStrokeWidth(getContext(), size);
}

void NVG::lineCap(int cap) {
    nvgLineCap(getContext(), cap);
}

void NVG::lineJoin(int join) {
    nvgLineJoin(getContext(), join);
}

void NVG::globalAlpha(float alpha) {
    nvgGlobalAlpha(getContext(), alpha);
}

void NVG::resetTransform() {
    nvgResetTransform(getContext());
}

void NVG::transform(float a, float b, float c, float d, float e, float f) {
    nvgTransform(getContext(), a, b, c, d, e, f);
}

void NVG::translate(float x, float y) {
    nvgTranslate(getContext(), x, y);
}

void NVG::rotate(float angle) {
    nvgRotate(getContext(), angle);
}

void NVG::skewX(float angle) {
    nvgSkewX(getContext(), angle);
}

void NVG::skewY(float angle) {
    nvgSkewY(getContext(), angle);
}

void NVG::scale(float x, float y) {
    nvgScale(getContext(), x, y);
}

void NVG::currentTransform(float* xform) {
    nvgCurrentTransform(getContext(), xform);
}

void NVG::transformIdentity(float* dst) {
    nvgTransformIdentity(dst);
}

void NVG::transformTranslate(float* dst, float tx, float ty) {
    nvgTransformTranslate(dst, tx, ty);
}

void NVG::transformScale(float* dst, float sx, float sy) {
    nvgTransformScale(dst, sx, sy);
}

void NVG::transformRotate(float* dst, float a) {
    nvgTransformRotate(dst, a);
}

void NVG::transformSkewX(float* dst, float a) {
    nvgTransformSkewX(dst, a);
}

void NVG::transformSkewY(float* dst, float a) {
    nvgTransformSkewY(dst, a);
}

void NVG::transformMultiply(float* dst, const float* src) {
    nvgTransformMultiply(dst, src);
}

void NVG::transformPremultiply(float* dst, const float* src) {
    nvgTransformPremultiply(dst, src);
}

int NVG::transformInverse(float* dst, const float* src) {
    return nvgTransformInverse(dst, src);
}

void NVG::transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy) {
    nvgTransformPoint(dstx, dsty, xform, srcx, srcy);
}

float NVG::degToRad(float deg) {
    return nvgDegToRad(deg);
}

float NVG::radToDeg(float rad) {
    return nvgRadToDeg(rad);
}

int NVG::createImage(const char* filename, int imageFlags) {
    return nvgCreateImage(getContext(), filename, imageFlags);
}

int NVG::createImageMem(int imageFlags, unsigned char* data, int ndata) {
    return nvgCreateImageMem(getContext(), imageFlags, data, ndata);
}

int NVG::createImageRGBA(int w, int h, int imageFlags, const unsigned char* data) {
    return nvgCreateImageRGBA(getContext(), w, h, imageFlags, data);
}

void NVG::updateImage(int image, const unsigned char* data) {
    nvgUpdateImage(getContext(), image, data);
}

void NVG::imageSize(int image, int* w, int* h) {
    nvgImageSize(getContext(), image, w, h);
}

void NVG::deleteImage(int image) {
    nvgDeleteImage(getContext(), image);
}

void NVG::beginPath() {
    nvgBeginPath(getContext());
}

void NVG::moveTo(float x, float y) {
    nvgMoveTo(getContext(), x, y);
}

void NVG::lineTo(float x, float y) {
    nvgLineTo(getContext(), x, y);
}

void NVG::bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y) {
    nvgBezierTo(getContext(), c1x, c1y, c2x, c2y, x, y);
}

void NVG::quadTo(float cx, float cy, float x, float y) {
    nvgQuadTo(getContext(), cx, cy, x, y);
}

void NVG::arcTo(float x1, float y1, float x2, float y2, float radius) {
    nvgArcTo(getContext(), x1, y1, x2, y2, radius);
}

void NVG::closePath() {
    nvgClosePath(getContext());
}

void NVG::pathWinding(int dir) {
    nvgPathWinding(getContext(), dir);
}

void NVG::arc(float cx, float cy, float r, float a0, float a1, int dir) {
    nvgArc(getContext(), cx, cy, r, a0, a1, dir);
}

void NVG::rect(float x, float y, float w, float h) {
    nvgRect(getContext(), x, y, w, h);
}

void NVG::roundedRect(float x, float y, float w, float h, float r) {
    nvgRoundedRect(getContext(), x, y, w, h, r);
}

void NVG::roundedRectVarying(float x, float y, float w, float h, float radTopLeft,
        float radTopRight, float radBottomRight, float radBottomLeft) {
    nvgRoundedRectVarying(getContext(), x, y, w, h, radTopLeft, radTopRight, radBottomRight,
            radBottomLeft);
}

void NVG::ellipse(float cx, float cy, float rx, float ry) {
    nvgEllipse(getContext(), cx, cy, rx, ry);
}

void NVG::circle(float cx, float cy, float r) {
    nvgCircle(getContext(), cx, cy, r);
}

void NVG::fill() {
    nvgFill(getContext());
}

void NVG::stroke() {
    nvgStroke(getContext());
}

Paint NVG::linearGradient(float sx, float sy, float ex, float ey, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    NVGpaint np = nvgLinearGradient(getContext(), sx, sy, ex, ey,
            nvgRGBA(icol[2], icol[1], icol[0], icol[3]),
            nvgRGBA(ocol[2], ocol[1], ocol[0], ocol[3]));
    return Paint(np);
}

Paint NVG::boxGradient(float x, float y, float w, float h, float r, float f, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    NVGpaint np = nvgBoxGradient(getContext(), x, y, w, h, r, f,
            nvgRGBA(icol[2], icol[1], icol[0], icol[3]),
            nvgRGBA(ocol[2], ocol[1], ocol[0], ocol[3]));
    return Paint(np);
}

Paint NVG::radialGradient(float cx, float cy, float inr, float outr, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    NVGpaint np = nvgRadialGradient(getContext(), cx, cy, inr, outr,
            nvgRGBA(icol[2], icol[1], icol[0], icol[3]),
            nvgRGBA(ocol[2], ocol[1], ocol[0], ocol[3]));
    return Paint(np);
}

Paint NVG::imagePattern(float ox, float oy, float ex, float ey, float angle, int image,
        float alpha) {
    NVGpaint np = nvgImagePattern(getContext(), ox, oy, ex, ey, angle, image, alpha);
    return Paint(np);
}

void NVG::scissor(float x, float y, float w, float h) {
    nvgScissor(getContext(), x, y, w, h);
}

void NVG::intersectScissor(float x, float y, float w, float h) {
    nvgIntersectScissor(getContext(), x, y, w, h);
}

void NVG::resetScissor() {
    nvgResetScissor(getContext());
}
}

int createFont(const char* name, const char* filename) {
    return detail::NVG::getCurrentContext()->createFont(name, filename);
}

int createFontMem(const char* name, unsigned char* data, int ndata, int freeData) {
    return detail::NVG::getCurrentContext()->createFontMem(name, data, ndata, freeData);
}

int findFont(const char* name) {
    return detail::NVG::getCurrentContext()->findFont(name);
}

int addFallbackFontId(int baseFont, int fallbackFont) {
    return detail::NVG::getCurrentContext()->addFallbackFontId(baseFont, fallbackFont);
}
int addFallbackFont(const char* baseFont, const char* fallbackFont) {
    return detail::NVG::getCurrentContext()->addFallbackFont(baseFont, fallbackFont);
}

void fontSize(float size) {
    detail::NVG::getCurrentContext()->fontSize(size);
}

void fontBlur(float blur) {
    detail::NVG::getCurrentContext()->fontBlur(blur);
}

void textLetterSpacing(float spacing) {
    detail::NVG::getCurrentContext()->textLetterSpacing(spacing);
}

void textLineHeight(float lineHeight) {
    detail::NVG::getCurrentContext()->textLineHeight(lineHeight);
}

void textAlign(int align) {
    detail::NVG::getCurrentContext()->textAlign(align);
}

void fontFaceId(int font) {
    detail::NVG::getCurrentContext()->fontFaceId(font);
}

void fontFace(const char* font) {
    detail::NVG::getCurrentContext()->fontFace(font);
}

float text(float x, float y, const char* string, const char* end) {
    return detail::NVG::getCurrentContext()->text(x, y, string, end);
}

void textBox(float x, float y, float breakRowWidth, const char* string, const char* end) {
    detail::NVG::getCurrentContext()->textBox(x, y, breakRowWidth, string, end);
}

float textBounds(float x, float y, const char* string, const char* end, float* bounds) {
    return detail::NVG::getCurrentContext()->textBounds(x, y, string, end, bounds);
}

void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end,
        float* bounds) {
    detail::NVG::getCurrentContext()->textBoxBounds(x, y, breakRowWidth, string, end, bounds);
}

int textGlyphPositions(float x, float y, const char* string, const char* end,
        GlyphPosition* positions, int maxPositions) {
    return detail::NVG::getCurrentContext()->textGlyphPositions(x, y, string, end, positions,
            maxPositions);
}

void textMetrics(float* ascender, float* descender, float* lineh) {
    detail::NVG::getCurrentContext()->textMetrics(ascender, descender, lineh);
}

int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows,
        int maxRows) {
    return detail::NVG::getCurrentContext()->textBreakLines(string, end, breakRowWidth, rows,
            maxRows);
}

void save() {
    detail::NVG::getCurrentContext()->save();
}

void restore() {
    detail::NVG::getCurrentContext()->restore();
}

void reset() {
    detail::NVG::getCurrentContext()->reset();
}

//void shapeAntiAlias(int enabled) {
//    detail::NVG::getCurrentContext()->strokeColor(enabled);
//}

void strokeColor(const cv::Scalar& bgra) {
    detail::NVG::getCurrentContext()->strokeColor(bgra);
}

void strokePaint(Paint paint) {
    detail::NVG::getCurrentContext()->strokePaint(paint);
}

void fillColor(const cv::Scalar& color) {
    detail::NVG::getCurrentContext()->fillColor(color);
}

void fillPaint(Paint paint) {
    detail::NVG::getCurrentContext()->fillPaint(paint);
}

void miterLimit(float limit) {
    detail::NVG::getCurrentContext()->miterLimit(limit);
}

void strokeWidth(float size) {
    detail::NVG::getCurrentContext()->strokeWidth(size);
}

void lineCap(int cap) {
    detail::NVG::getCurrentContext()->lineCap(cap);
}

void lineJoin(int join) {
    detail::NVG::getCurrentContext()->lineJoin(join);
}

void globalAlpha(float alpha) {
    detail::NVG::getCurrentContext()->globalAlpha(alpha);
}

void resetTransform() {
    detail::NVG::getCurrentContext()->resetTransform();
}

void transform(float a, float b, float c, float d, float e, float f) {
    detail::NVG::getCurrentContext()->transform(a, b, c, d, e, f);
}

void translate(float x, float y) {
    detail::NVG::getCurrentContext()->translate(x, y);
}

void rotate(float angle) {
    detail::NVG::getCurrentContext()->rotate(angle);
}

void skewX(float angle) {
    detail::NVG::getCurrentContext()->skewX(angle);
}

void skewY(float angle) {
    detail::NVG::getCurrentContext()->skewY(angle);
}

void scale(float x, float y) {
    detail::NVG::getCurrentContext()->scale(x, y);
}

void currentTransform(float* xform) {
    detail::NVG::getCurrentContext()->currentTransform(xform);
}

void transformIdentity(float* dst) {
    detail::NVG::getCurrentContext()->transformIdentity(dst);
}

void transformTranslate(float* dst, float tx, float ty) {
    detail::NVG::getCurrentContext()->transformTranslate(dst, tx, ty);
}

void transformScale(float* dst, float sx, float sy) {
    detail::NVG::getCurrentContext()->transformScale(dst, sx, sy);
}

void transformRotate(float* dst, float a) {
    detail::NVG::getCurrentContext()->transformRotate(dst, a);
}

void transformSkewX(float* dst, float a) {
    detail::NVG::getCurrentContext()->transformSkewX(dst, a);
}

void transformSkewY(float* dst, float a) {
    detail::NVG::getCurrentContext()->transformSkewY(dst, a);
}

void transformMultiply(float* dst, const float* src) {
    detail::NVG::getCurrentContext()->transformMultiply(dst, src);
}

void transformPremultiply(float* dst, const float* src) {
    detail::NVG::getCurrentContext()->transformPremultiply(dst, src);
}

int transformInverse(float* dst, const float* src) {
    return detail::NVG::getCurrentContext()->transformInverse(dst, src);
}

void transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy) {
    return detail::NVG::getCurrentContext()->transformPoint(dstx, dsty, xform, srcx, srcy);
}

float degToRad(float deg) {
    return detail::NVG::getCurrentContext()->degToRad(deg);
}

float radToDeg(float rad) {
    return detail::NVG::getCurrentContext()->radToDeg(rad);
}

int createImage(const char* filename, int imageFlags) {
    return detail::NVG::getCurrentContext()->createImage(filename, imageFlags);
}

int createImageMem(int imageFlags, unsigned char* data, int ndata) {
    return detail::NVG::getCurrentContext()->createImageMem(imageFlags, data, ndata);
}

int createImageRGBA(int w, int h, int imageFlags, const unsigned char* data) {
    return detail::NVG::getCurrentContext()->createImageRGBA(w, h, imageFlags, data);
}

void updateImage(int image, const unsigned char* data) {
    detail::NVG::getCurrentContext()->updateImage(image, data);
}

void imageSize(int image, int* w, int* h) {
    detail::NVG::getCurrentContext()->imageSize(image, w, h);
}

void deleteImage(int image) {
    detail::NVG::getCurrentContext()->deleteImage(image);
}

void beginPath() {
    detail::NVG::getCurrentContext()->beginPath();
}
void moveTo(float x, float y) {
    detail::NVG::getCurrentContext()->moveTo(x, y);
}

void lineTo(float x, float y) {
    detail::NVG::getCurrentContext()->lineTo(x, y);
}

void bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y) {
    detail::NVG::getCurrentContext()->bezierTo(c1x, c1y, c2x, c2y, x, y);
}

void quadTo(float cx, float cy, float x, float y) {
    detail::NVG::getCurrentContext()->quadTo(cx, cy, x, y);
}

void arcTo(float x1, float y1, float x2, float y2, float radius) {
    detail::NVG::getCurrentContext()->arcTo(x1, y1, x2, y2, radius);
}

void closePath() {
    detail::NVG::getCurrentContext()->closePath();
}

void pathWinding(int dir) {
    detail::NVG::getCurrentContext()->pathWinding(dir);
}

void arc(float cx, float cy, float r, float a0, float a1, int dir) {
    detail::NVG::getCurrentContext()->arc(cx, cy, r, a0, a1, dir);
}

void rect(float x, float y, float w, float h) {
    detail::NVG::getCurrentContext()->rect(x, y, w, h);
}

void roundedRect(float x, float y, float w, float h, float r) {
    detail::NVG::getCurrentContext()->roundedRect(x, y, w, h, r);
}

void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight,
        float radBottomRight, float radBottomLeft) {
    detail::NVG::getCurrentContext()->roundedRectVarying(x, y, w, h, radTopLeft, radTopRight,
            radBottomRight, radBottomLeft);
}

void ellipse(float cx, float cy, float rx, float ry) {
    detail::NVG::getCurrentContext()->ellipse(cx, cy, rx, ry);
}

void circle(float cx, float cy, float r) {
    detail::NVG::getCurrentContext()->circle(cx, cy, r);
}

void fill() {
    detail::NVG::getCurrentContext()->fill();
}

void stroke() {
    detail::NVG::getCurrentContext()->stroke();
}

Paint linearGradient(float sx, float sy, float ex, float ey, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    return detail::NVG::getCurrentContext()->linearGradient(sx, sy, ex, ey, icol, ocol);
}

Paint boxGradient(float x, float y, float w, float h, float r, float f, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    return detail::NVG::getCurrentContext()->boxGradient(x, y, w, h, r, f, icol, ocol);
}

Paint radialGradient(float cx, float cy, float inr, float outr, const cv::Scalar& icol,
        const cv::Scalar& ocol) {
    return detail::NVG::getCurrentContext()->radialGradient(cx, cy, inr, outr, icol, ocol);
}

Paint imagePattern(float ox, float oy, float ex, float ey, float angle, int image, float alpha) {
    return detail::NVG::getCurrentContext()->imagePattern(ox, oy, ex, ey, angle, image, alpha);
}

void scissor(float x, float y, float w, float h) {
    detail::NVG::getCurrentContext()->scissor(x, y, w, h);
}

void intersectScissor(float x, float y, float w, float h) {
    detail::NVG::getCurrentContext()->intersectScissor(x, y, w, h);
}

void resetScissor() {
    detail::NVG::getCurrentContext()->resetScissor();
}

void clear(const cv::Scalar& bgra) {
    const float& b = bgra[0] / 255.0f;
    const float& g = bgra[1] / 255.0f;
    const float& r = bgra[2] / 255.0f;
    const float& a = bgra[3] / 255.0f;
    GL_CHECK(glClearColor(r, g, b, a));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
}
}
}
}
