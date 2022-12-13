#include "nvg.hpp"

namespace kb {
namespace nvg {
namespace detail {
class NVG;

void set_current_context(NVGcontext* ctx) {
    if(nvg_instance != nullptr)
        delete nvg_instance;
    nvg_instance = new NVG(ctx);
}

NVG* get_current_context() {
    assert(nvg_instance != nullptr);
    return nvg_instance;
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

void NVG::textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end, float* bounds) {
    nvgTextBoxBounds(getContext(), x, y, breakRowWidth, string, end, bounds);
}

int NVG::textGlyphPositions(float x, float y, const char* string, const char* end, GlyphPosition* positions, int maxPositions) {
    std::vector<NVGglyphPosition> gp(maxPositions);
    int result = nvgTextGlyphPositions(getContext(), x, y, string, end, gp.data(), maxPositions);
    for(int i = 0; i < maxPositions; ++i) {
        positions[i].str = gp[i].str;
        positions[i].x = gp[i].x;
        positions[i].minx = gp[i].minx;
        positions[i].maxx = gp[i].maxx;
    }
    return result;
}

void NVG::textMetrics(float* ascender, float* descender, float* lineh) {
    nvgTextMetrics(getContext(), ascender, descender, lineh);
}

int NVG::textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows, int maxRows) {
    NVGtextRow tr[maxRows];
    int result = nvgTextBreakLines(getContext(),string, end, breakRowWidth, tr, maxRows);
    for(int i = 0; i < maxRows; ++i) {
        rows[i].start = tr[i].start;
        rows[i].end = tr[i].end;
        rows[i].next = tr[i].next;
        rows[i].width = tr[i].width;
        rows[i].minx = tr[i].minx;
        rows[i].maxx = tr[i].maxx;
    }
    return result;
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

void NVG::shapeAntiAlias(int enabled) {
    nvgShapeAntiAlias(getContext(), enabled);
}

void NVG::strokeColor(const cv::Scalar& bgra) {
    nvgStrokeColor(getContext(), nvgRGBA(bgra[2],bgra[1],bgra[0],bgra[3]));
}

void NVG::strokePaint(Paint paint) {
    NVGpaint np;
    memcpy(paint.xform, np.xform, 6);
    memcpy(paint.extent, np.extent, 2);
    np.radius = paint.radius;
    np.feather = paint.feather;
    np.innerColor = nvgRGBA(paint.innerColor[2],paint.innerColor[1],paint.innerColor[0],paint.innerColor[3]);
    np.outerColor = nvgRGBA(paint.outerColor[2],paint.outerColor[1],paint.outerColor[0],paint.outerColor[3]);;
    np.image = paint.image;

    nvgStrokePaint(getContext(), np);
}

void NVG::fillColor(const cv::Scalar& bgra) {
    nvgFillColor(getContext(), nvgRGBA(bgra[2],bgra[1],bgra[0],bgra[3]));
}

void NVG::fillPaint(Paint paint) {
    NVGpaint np;
    memcpy(paint.xform, np.xform, 6);
    memcpy(paint.extent, np.extent, 2);
    np.radius = paint.radius;
    np.feather = paint.feather;
    np.innerColor = nvgRGBA(paint.innerColor[2],paint.innerColor[1],paint.innerColor[0],paint.innerColor[3]);
    np.outerColor = nvgRGBA(paint.outerColor[2],paint.outerColor[1],paint.outerColor[0],paint.outerColor[3]);;
    np.image = paint.image;

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

void NVG::roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight, float radBottomRight, float radBottomLeft) {
    nvgRoundedRectVarying(getContext(), x, y, w, h, radTopLeft, radTopRight, radBottomRight, radBottomLeft);
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
} //namespace detail

int createFont(const char* name, const char* filename) {
    return detail::get_current_context()->createFont(name,filename);
}

int createFontMem(const char* name, unsigned char* data, int ndata, int freeData) {
    return detail::get_current_context()->createFontMem(name, data, ndata, freeData);
}

int findFont(const char* name) {
    return detail::get_current_context()->findFont(name);
}

int addFallbackFontId(int baseFont, int fallbackFont) {
    return detail::get_current_context()->addFallbackFontId(baseFont, fallbackFont);
}
int addFallbackFont(const char* baseFont, const char* fallbackFont) {
    return detail::get_current_context()->addFallbackFont(baseFont, fallbackFont);
}

void fontSize(float size) {
    detail::get_current_context()->fontSize(size);
}

void fontBlur(float blur) {
    detail::get_current_context()->fontBlur(blur);
}

void textLetterSpacing(float spacing) {
    detail::get_current_context()->textLetterSpacing(spacing);
}

void textLineHeight(float lineHeight) {
    detail::get_current_context()->textLineHeight(lineHeight);
}

void textAlign(int align) {
    detail::get_current_context()->textAlign(align);
}

void fontFaceId(int font) {
    detail::get_current_context()->fontFaceId(font);
}

void fontFace(const char* font) {
    detail::get_current_context()->fontFace(font);
}

float text(float x, float y, const char* string, const char* end) {
    return detail::get_current_context()->text(x, y, string, end);
}

void textBox(float x, float y, float breakRowWidth, const char* string, const char* end) {
    detail::get_current_context()->textBox(x, y, breakRowWidth, string, end);
}

float textBounds(float x, float y, const char* string, const char* end, float* bounds) {
    return detail::get_current_context()->textBounds(x, y, string, end, bounds);
}

void textBoxBounds(float x, float y, float breakRowWidth, const char* string, const char* end, float* bounds) {
    detail::get_current_context()->textBoxBounds(x, y, breakRowWidth, string, end, bounds);
}

int textGlyphPositions(float x, float y, const char* string, const char* end, GlyphPosition* positions, int maxPositions) {
    return detail::get_current_context()->textGlyphPositions(x, y, string, end, positions, maxPositions);
}

void textMetrics(float* ascender, float* descender, float* lineh) {
    detail::get_current_context()->textMetrics(ascender, descender, lineh);
}

int textBreakLines(const char* string, const char* end, float breakRowWidth, TextRow* rows, int maxRows) {
    return detail::get_current_context()->textBreakLines(string, end, breakRowWidth, rows, maxRows);
}

void save() {
    detail::get_current_context()->save();
}

void restore() {
    detail::get_current_context()->restore();
}

void reset() {
    detail::get_current_context()->reset();
}

void shapeAntiAlias(int enabled) {
    detail::get_current_context()->strokeColor(enabled);
}

void strokeColor(const cv::Scalar& bgra) {
    detail::get_current_context()->strokeColor(bgra);
}

void strokePaint(Paint paint) {
    detail::get_current_context()->strokePaint(paint);
}

void fillColor(const cv::Scalar& color) {
    detail::get_current_context()->fillColor(color);
}

void fillPaint(Paint paint) {
    detail::get_current_context()->fillPaint(paint);
}

void miterLimit(float limit) {
    detail::get_current_context()->miterLimit(limit);
}

void strokeWidth(float size) {
    detail::get_current_context()->strokeWidth(size);
}

void lineCap(int cap) {
    detail::get_current_context()->lineCap(cap);
}

void lineJoin(int join) {
    detail::get_current_context()->lineJoin(join);
}

void globalAlpha(float alpha) {
    detail::get_current_context()->globalAlpha(alpha);
}

void resetTransform() {
    detail::get_current_context()->resetTransform();
}

void transform(float a, float b, float c, float d, float e, float f) {
    detail::get_current_context()->transform(a, b, c, d, e, f);
}

void translate(float x, float y) {
    detail::get_current_context()->translate(x, y);
}

void rotate(float angle) {
    detail::get_current_context()->rotate(angle);
}

void skewX(float angle) {
    detail::get_current_context()->skewX(angle);
}

void skewY(float angle) {
    detail::get_current_context()->skewY(angle);
}

void scale(float x, float y) {
    detail::get_current_context()->scale(x, y);
}

void currentTransform(float* xform) {
    detail::get_current_context()->currentTransform(xform);
}

void transformIdentity(float* dst) {
    detail::get_current_context()->transformIdentity(dst);
}

void transformTranslate(float* dst, float tx, float ty) {
    detail::get_current_context()->transformTranslate(dst, tx, ty);
}

void transformScale(float* dst, float sx, float sy) {
    detail::get_current_context()->transformScale(dst, sx, sy);
}

void transformRotate(float* dst, float a) {
    detail::get_current_context()->transformRotate(dst, a);
}

void transformSkewX(float* dst, float a) {
    detail::get_current_context()->transformSkewX(dst, a);
}

void transformSkewY(float* dst, float a) {
    detail::get_current_context()->transformSkewY(dst, a);
}

void transformMultiply(float* dst, const float* src) {
    detail::get_current_context()->transformMultiply(dst, src);
}

void transformPremultiply(float* dst, const float* src) {
    detail::get_current_context()->transformPremultiply(dst, src);
}

int transformInverse(float* dst, const float* src) {
    return detail::get_current_context()->transformInverse(dst, src);
}

void transformPoint(float* dstx, float* dsty, const float* xform, float srcx, float srcy) {
    return detail::get_current_context()->transformPoint(dstx, dsty, xform, srcx, srcy);
}

float degToRad(float deg) {
    return detail::get_current_context()->degToRad(deg);
}

float radToDeg(float rad) {
    return detail::get_current_context()->radToDeg(rad);
}

void beginPath() {
    detail::get_current_context()->beginPath();
}
void moveTo(float x, float y) {
    detail::get_current_context()->moveTo(x, y);
}

void lineTo(float x, float y) {
    detail::get_current_context()->lineTo(x, y);
}

void bezierTo(float c1x, float c1y, float c2x, float c2y, float x, float y) {
    detail::get_current_context()->bezierTo(c1x, c1y, c2x, c2y, x, y);
}

void quadTo(float cx, float cy, float x, float y) {
    detail::get_current_context()->quadTo(cx, cy, x, y);
}

void arcTo(float x1, float y1, float x2, float y2, float radius) {
    detail::get_current_context()->arcTo(x1, y1, x2, y2, radius);
}

void closePath() {
    detail::get_current_context()->closePath();
}

void pathWinding(int dir) {
    detail::get_current_context()->pathWinding(dir);
}

void arc(float cx, float cy, float r, float a0, float a1, int dir) {
    detail::get_current_context()->arc(cx, cy, r, a0, a1, dir);
}

void rect(float x, float y, float w, float h) {
    detail::get_current_context()->rect(x, y, w, h);
}

void roundedRect(float x, float y, float w, float h, float r) {
    detail::get_current_context()->roundedRect(x, y, w, h, r);
}

void roundedRectVarying(float x, float y, float w, float h, float radTopLeft, float radTopRight, float radBottomRight, float radBottomLeft) {
    detail::get_current_context()->roundedRectVarying(x, y, w, h, radTopLeft, radTopRight, radBottomRight, radBottomLeft);
}

void ellipse(float cx, float cy, float rx, float ry) {
    detail::get_current_context()->ellipse(cx, cy, rx, ry);
}

void circle(float cx, float cy, float r) {
    detail::get_current_context()->circle(cx, cy, r);
}

void fill() {
    detail::get_current_context()->fill();
}

void stroke() {
    detail::get_current_context()->stroke();
}

} //namespace nvg
} //namespace kb
