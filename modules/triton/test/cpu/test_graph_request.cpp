// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
// CPU-only controls for the internal graph helper's request parser and execution identity.
// These checks allocate no GPU buffers and do not prove graph capture/replay support.
#include "../../perf/hip/graph_request.hpp"
#include <cassert>
#include <functional>
#include <iostream>
using namespace cv_triton_runtime;
static Json &field(Json &j, const std::string &key)
{
    for (auto &p : j.obj)
        if (p.first == key)
            return p.second;
    j.obj.emplace_back(key, Json());
    return j.obj.back().second;
}
static Json num(double n)
{
    Json j;
    j.kind = Json::Number;
    j.num = n;
    return j;
}
static Json str(const char *s)
{
    Json j;
    j.kind = Json::String;
    j.str = s;
    return j;
}
int main()
{
    Json good;
    std::string error;
    const std::string hash(64, 'a');
    assert(json_parse(
        "{\"op\":\"warpaffine\",\"case\":{\"id\":\"sample\",\"src_h\":32,\"src_w\":32,\"channels\":3,\"dst_h\":16,\"dst_w\":16},\"input_path\":\"in\",\"changed_path\":\"other\",\"input_sha256\":\"" +
            hash + "\",\"changed_sha256\":\"" + hash + "\"}",
        good, error));
    auto normal = triton_graph::parse(good);
    assert(normal.rows == 32 && normal.height == 16 && normal.mode == "INTER_LINEAR" &&
           !normal.inverse && normal.roi);
    int rejected = 0;
    auto reject = [&](std::function<void(Json &)> mutate)
    {
        Json test = good;
        mutate(test);
        try
        {
            (void)triton_graph::parse(test);
            assert(false);
        }
        catch (const std::runtime_error &)
        {
            ++rejected;
        }
    };
    for (double n : {32.9, 0.0, -1.0, 4097.0, 2147483648.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
        reject(
            [&](Json &j)
            {
                field(field(j, "case"), "src_h") = num(n);
            });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "src_w") = str("32");
        });
    reject(
        [](Json &j)
        {
            field(j, "case") = num(1);
        });
    reject(
        [](Json &j)
        {
            field(j, "op") = num(1);
        });
    reject(
        [](Json &j)
        {
            field(j, "samples") = num(10.5);
        });
    reject(
        [](Json &j)
        {
            field(j, "samples") = num(0);
        });
    reject(
        [](Json &j)
        {
            field(j, "captured_calls") = num(251);
        });
    reject(
        [](Json &j)
        {
            field(j, "captured_calls") = num(1e100);
        });
    reject(
        [](Json &j)
        {
            field(j, "roi") = num(2);
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "inverse") = num(1);
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "mode") = str("bad_NEAREST");
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "border") = str("BORDER_TRANSPARENT");
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "borderValue") = num(0);
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "M") = str("123456");
        });
    reject(
        [](Json &j)
        {
            Json a;
            a.kind = Json::Array;
            a.arr = {num(1), num(0)};
            field(field(j, "case"), "M") = a;
        });
    for (auto invalid : {str("1"), num(std::numeric_limits<double>::infinity()), Json()})
        reject(
            [&](Json &j)
            {
                Json a;
                a.kind = Json::Array;
                a.arr = {num(1), num(0), num(0), num(0), num(1), invalid};
                field(field(j, "case"), "M") = a;
            });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "id") = num(0);
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "channels") = num(2);
        });
    reject(
        [](Json &j)
        {
            field(j, "case").obj.emplace_back("src_h", num(32));
        });
    reject(
        [](Json &j)
        {
            field(field(j, "case"), "ignored_option") = num(1);
        });
    reject(
        [](Json &j)
        {
            field(j, "input_path") = str("");
        });
    reject(
        [](Json &j)
        {
            field(j, "input_sha256") = str("invalid");
        });
    reject(
        [](Json &j)
        {
            field(j, "op") = str("resize");
            field(field(j, "case"), "dst_h") = num(32);
            field(field(j, "case"), "dst_w") = num(32);
        });
    for (const std::string invalid : {"NaN", "Infinity", "1e9999"})
    {
        Json raw;
        std::string why;
        const std::string data = "{\"src_h\":" + invalid + "}";
        if (json_parse(data, raw, why))
        {
            try
            {
                (void)triton_graph::number(raw, "src_h");
                assert(false);
            }
            catch (const std::runtime_error &)
            {
                ++rejected;
            }
        }
        else
            ++rejected;
    }
    Json legal = good;
    field(field(legal, "case"), "mode") = str("CUBIC");
    auto normalized = triton_graph::parse(legal);
    Json identity;
    assert(json_parse(triton_graph::executionJson(normalized), identity, error));
    assert(identity.get("mode")->str == "INTER_CUBIC" && identity.get("src_h")->num == 32);
    assert(identity.get("M")->arr.size() == 6 && identity.get("borderValue")->arr[0].num == 0);
    assert(normalized.mode == "INTER_CUBIC" && normalized.matrix[0] == 1 &&
           normalized.matrix[5] == 0);
    try
    {
        (void)triton_graph::product(std::numeric_limits<std::size_t>::max(), 2);
        assert(false);
    }
    catch (const std::runtime_error &)
    {
        ++rejected;
    }
    assert(triton_graph::product(32, 4) == 128);
    std::cout << "PASS: graph schema legal request and " << rejected
              << " malformed/overflow controls before any GPU allocation\n";
}
