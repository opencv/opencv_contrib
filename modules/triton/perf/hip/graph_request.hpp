// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
// Private JSON request parser for perf_graph.cpp. Validate types, dimensions
// and byte-count products before the helper allocates GPU buffers.
#ifndef OPENCV_TRITON_GRAPH_REQUEST_HPP
#define OPENCV_TRITON_GRAPH_REQUEST_HPP
#include "../../src/runtime/compile_protocol.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <set>
#include <sstream>
#include <iomanip>
namespace triton_graph
{
using cv_triton_runtime::Json;
inline void require(bool value, const std::string &why)
{
    if (!value)
        throw std::runtime_error("invalid graph request: " + why);
}
inline void fields(const Json &node, const std::set<std::string> &allowed)
{
    require(node.kind == Json::Object, "object required");
    std::set<std::string> seen;
    for (const auto &entry : node.obj)
    {
        require(allowed.count(entry.first) != 0, "unsupported field " + entry.first);
        require(seen.insert(entry.first).second, "duplicate field " + entry.first);
    }
}
inline std::string text(const Json &j, const std::string &key, const std::string &fallback = "")
{
    const auto *value = j.get(key);
    if (!value)
        return fallback;
    require(value->kind == Json::String && value->str.find('\0') == std::string::npos,
            key + " must be a string without NUL");
    return value->str;
}
inline double finite(const Json &value, const std::string &key)
{
    require(value.kind == Json::Number && std::isfinite(value.num),
            key + " must be a finite number");
    return value.num;
}
inline int number(const Json &j, const std::string &key, int fallback = 0)
{
    const auto *value = j.get(key);
    if (!value)
        return fallback;
    const double n = finite(*value, key);
    require(std::trunc(n) == n && n >= std::numeric_limits<int>::min() &&
                n <= std::numeric_limits<int>::max(),
            key + " must be an int32 integer");
    return static_cast<int>(n);
}
inline bool boolean(const Json &j, const std::string &key, bool fallback)
{
    const auto *value = j.get(key);
    if (!value)
        return fallback;
    require(value->kind == Json::Bool, key + " must be boolean");
    return value->b;
}
// Reject size_t overflow before allocation or input-file length comparisons.
inline std::size_t product(std::size_t a, std::size_t b)
{
    require(!b || a <= std::numeric_limits<std::size_t>::max() / b, "size multiplication overflow");
    return a * b;
}
struct Request
{
    std::string op, dtype, mode, border, code;
    int rows, cols, channels, height, width, calls, samples;
    bool roi, inverse;
    double matrix[6] = {1, 0, 0, 0, 1, 0};
    std::string input, inputHash, changed, changedHash;
};
inline Request parse(const Json &root)
{
    fields(root, {"op", "case", "captured_calls", "samples", "roi", "input_path", "input_sha256",
                  "changed_path", "changed_sha256"});
    const auto *item = root.get("case");
    require(item != nullptr, "missing case");
    fields(*item, {"src_h", "src_w", "channels", "dst_h", "dst_w", "dtype", "mode", "code", "M",
                   "inverse", "border", "borderValue", "id"});
    const Json &c = *item;
    Request r;
    (void)text(c, "id");
    r.op = text(root, "op");
    require(r.op == "resize" || r.op == "warpaffine" || r.op == "cvtcolor", "unknown operator");
    // This helper supports the default zero border value only.
    // Reject even zero/null values rather than claim to measure an ignored field.
    require(!c.get("borderValue"), "borderValue is unsupported");
    r.rows = number(c, "src_h");
    r.cols = number(c, "src_w");
    r.channels = number(c, "channels", 1);
    r.height = number(c, "dst_h", r.rows);
    r.width = number(c, "dst_w", r.cols);
    for (int n : {r.rows, r.cols, r.height, r.width})
        require(n > 0 && n <= 4096, "dimensions outside 1..4096");
    require(r.channels == 1 || r.channels == 3 || r.channels == 4 ||
                (r.op == "cvtcolor" && r.channels == 2),
            "unsupported channel count");
    require(r.op != "resize" || r.rows != r.height || r.cols != r.width,
            "same-size resize copy has no compiled-kernel identity in this graph tool");
    r.dtype = text(c, "dtype", "uint8");
    require(r.dtype == "uint8" || r.dtype == "uint16" || r.dtype == "float32", "unknown dtype");
    require(r.op != "resize" || r.dtype != "uint16", "resize uint16 unsupported");
    r.calls = number(root, "captured_calls", 250);
    r.samples = number(root, "samples", 10);
    require(r.calls >= 1 && r.calls <= 250 && r.samples == 10,
            "frozen budget: calls 1..250, samples 10");
    r.mode = text(c, "mode", "INTER_LINEAR");
    const std::set<std::string> modes = {
        "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC", "INTER_AREA", "NEAREST", "LINEAR", "CUBIC"};
    require(modes.count(r.mode), "unknown mode");
    if (r.mode.find("INTER_") != 0)
        r.mode = "INTER_" + r.mode;
    r.inverse = boolean(c, "inverse", false);
    r.roi = true;
    if (const auto *roi = root.get("roi"))
    {
        if (roi->kind == Json::Bool)
            r.roi = roi->b;
        else
        {
            int n = number(root, "roi");
            require(n == 0 || n == 1, "roi must be boolean or 0/1");
            r.roi = n != 0;
        }
    }
    r.border = text(c, "border", "BORDER_REPLICATE");
    require(r.border == "BORDER_REPLICATE" || r.border == "BORDER_CONSTANT" ||
                r.border == "BORDER_REFLECT_101" || r.border == "BORDER_WRAP",
            "unsupported border");
    if (const auto *matrix = c.get("M"))
    {
        require(matrix->kind == Json::Array && matrix->arr.size() == 6, "M must have six numbers");
        for (int i = 0; i < 6; ++i)
            r.matrix[i] = finite(matrix->arr[i], "M");
    }
    if (r.op != "warpaffine")
        require(!c.get("M") && !c.get("inverse") && !c.get("border"), "warp-only fields");
    r.code = text(c, "code");
    if (r.op == "cvtcolor")
    {
        require(!r.code.empty(), "missing conversion code");
        require(!c.get("mode"), "cvtcolor has no interpolation mode");
    }
    else
        require(!c.get("code"), "code is cvtcolor-only");
    r.input = text(root, "input_path");
    r.inputHash = text(root, "input_sha256");
    r.changed = text(root, "changed_path");
    r.changedHash = text(root, "changed_sha256");
    require(!r.input.empty() && !r.changed.empty(), "input paths required");
    for (const auto &hash : {r.inputHash, r.changedHash})
        require(hash.size() == 64 &&
                    hash.find_first_not_of("0123456789abcdef") == std::string::npos,
                "invalid SHA256");
    const std::size_t bytes = r.dtype == "float32" ? 4 : r.dtype == "uint16" ? 2 : 1;
    // Cover dimensions, ROI padding and element-stride int32 limits before host/GPU allocation.
    const auto row = product(product(static_cast<std::size_t>(r.cols) + 7, r.channels), bytes);
    require(product(row, static_cast<std::size_t>(r.rows) + 4) <= std::numeric_limits<int>::max(),
            "source allocation too large");
    const auto outputRow = product(product(static_cast<std::size_t>(r.width) + 7, 4), bytes);
    require(product(outputRow, static_cast<std::size_t>(r.height) + 4) <=
                std::numeric_limits<int>::max(),
            "output allocation too large");
    return r;
}
inline std::string executionJson(const Request &r)
{
    using cv_triton_runtime::json_quote;
    std::ostringstream out;
    out << std::setprecision(17) << "{\"op\":" << json_quote(r.op) << ",\"src_h\":" << r.rows
        << ",\"src_w\":" << r.cols << ",\"dst_h\":" << r.height << ",\"dst_w\":" << r.width
        << ",\"channels\":" << r.channels << ",\"dtype\":" << json_quote(r.dtype)
        << ",\"roi\":" << (r.roi ? "true" : "false");
    if (r.op == "cvtcolor")
        out << ",\"code\":" << json_quote(r.code);
    else
        out << ",\"mode\":" << json_quote(r.mode);
    if (r.op == "warpaffine")
    {
        out << ",\"border\":" << json_quote(r.border)
            << ",\"inverse\":" << (r.inverse ? "true" : "false") << ",\"M\":[";
        for (int i = 0; i < 6; ++i)
            out << (i ? "," : "") << r.matrix[i];
        out << "],\"borderValue\":[0,0,0,0]";
    }
    out << ",\"captured_calls\":" << r.calls << ",\"samples\":" << r.samples << "}";
    return out.str();
}

} // namespace triton_graph
#endif
