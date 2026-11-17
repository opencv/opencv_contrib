// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Internal HIP graph timing of the Triton implementation. No Torch, alternate kernel or
// dispatch table: every recorded operation uses the public production DeviceImage
// entry. All graph-bound buffers/streams/modules/tables outlive completed replays.
#include "opencv2/triton.hpp"
#include "opencv2/triton/device.hpp"
#include "opencv2/imgproc.hpp"
#include "../../src/runtime/runtime.hpp"
#include "../../src/runtime/compile_protocol.hpp"
#include "../../src/runtime/cvtcolor.hpp"
#include "../../test/resize_reference.hpp"
#include "../../validation/numerics.hpp"
#include "graph_request.hpp"
#include "../../src/entry.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace J = cv_triton_runtime;
namespace D = cv::triton::detail;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
static void checked(hipError_t e)
{
    if (e != hipSuccess)
        throw std::runtime_error(std::string("HIP_API_ERROR: ") + hipGetErrorString(e));
}
static void require(bool value, const std::string &why)
{
    if (!value)
        throw std::runtime_error(why);
}
using triton_graph::text;
static std::string hashFile(const std::string &p)
{
    std::string h;
    require(J::sha256_file(p, h), "missing file: " + p);
    return h;
}

// Error policy: do not free potentially in-flight resources after an API error.
// main returns immediately to a supervised process exit; no reset or retry.
struct Resources
{
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    void *source = nullptr;
    void *output = nullptr;
    bool clean = false;
    ~Resources()
    {
        if (!clean)
            return;
        // Explicit cleanup completed before clean is set. No throwing teardown.
    }
    void finish()
    {
        checked(hipStreamSynchronize(stream));
        if (exec)
        {
            checked(hipGraphExecDestroy(exec));
            exec = nullptr;
        }
        if (graph)
        {
            checked(hipGraphDestroy(graph));
            graph = nullptr;
        }
        cv::triton::releaseGraphResources();
        D::reapCompleted();
        require(D::runtimeStats().capture_pin_holds == 0 && D::runtimeStats().pending == 0,
                "runtime resources still pending/pinned");
        if (start)
        {
            checked(hipEventDestroy(start));
            start = nullptr;
        }
        if (stop)
        {
            checked(hipEventDestroy(stop));
            stop = nullptr;
        }
        if (source)
        {
            checked(hipFree(source));
            source = nullptr;
        }
        if (output)
        {
            checked(hipFree(output));
            output = nullptr;
        }
        checked(hipStreamDestroy(stream));
        stream = nullptr;
        clean = true;
    }
};

struct Pitched
{
    cv::triton::DeviceImage image;
    size_t parentRows = 0, parentBytes = 0, offset = 0;
    void *base = nullptr;
    static constexpr unsigned char guard = 0xA5;
    void allocate(Resources &r, bool src, cv::Size size, int type, bool roi)
    {
        const size_t pixel = CV_ELEM_SIZE(type), width = (size.width + (roi ? 7 : 0)) * pixel;
        size_t pitch = 0;
        parentRows = size.height + (roi ? 4 : 0);
        checked(hipMallocPitch(&base, &pitch, width, parentRows));
        if (src)
            r.source = base;
        else
            r.output = base;
        parentBytes = pitch * parentRows;
        offset = roi ? pitch + 2 * pixel : 0;
        image = {reinterpret_cast<std::uintptr_t>(base) + offset, size.height, size.width, type,
                 pitch};
        checked(hipMemsetAsync(base, guard, parentBytes, r.stream));
    }
    void upload(const cv::Mat &host, hipStream_t stream) const
    {
        checked(hipMemcpy2DAsync(reinterpret_cast<void *>(image.ptr), image.step, host.data,
                                 host.step, host.cols * host.elemSize(), host.rows,
                                 hipMemcpyHostToDevice, stream));
    }
    void poison(hipStream_t stream, unsigned char value = guard) const
    {
        checked(hipMemsetAsync(base, guard, parentBytes, stream));
        if (value != guard)
            checked(hipMemset2DAsync(reinterpret_cast<void *>(image.ptr), image.step, value,
                                     image.cols * CV_ELEM_SIZE(image.type), image.rows, stream));
    }
    cv::Mat downloadAndCheck(hipStream_t stream) const
    {
        checked(hipStreamSynchronize(stream));
        std::vector<unsigned char> parent(parentBytes);
        checked(hipMemcpy(parent.data(), base, parentBytes, hipMemcpyDeviceToHost));
        const size_t firstRow = offset / image.step, firstByte = offset % image.step;
        const size_t used = image.cols * CV_ELEM_SIZE(image.type);
        for (size_t y = 0; y < parentRows; ++y)
            for (size_t x = 0; x < image.step; ++x)
            {
                bool inside = y >= firstRow && y < firstRow + static_cast<size_t>(image.rows) &&
                              x >= firstByte && x < firstByte + used;
                if (!inside && parent[y * image.step + x] != guard)
                    throw std::runtime_error("output/input canary changed");
            }
        cv::Mat view(image.rows, image.cols, image.type, parent.data() + offset, image.step);
        return view.clone();
    }
};

// This helper does not cache a graph across requests. The captured binding is
// checked before every launch so a changed size, pitch, ptr/alignment, type or
// immutable parameter fingerprint never reaches a stale executable.
struct Binding
{
    cv::triton::DeviceImage src, dst;
    std::string parameters;
    static bool same(const cv::triton::DeviceImage &a, const cv::triton::DeviceImage &b)
    {
        return a.ptr == b.ptr && a.rows == b.rows && a.cols == b.cols && a.type == b.type &&
               a.step == b.step;
    }
    bool matches(const cv::triton::DeviceImage &a, const cv::triton::DeviceImage &b,
                 const std::string &p) const
    {
        return same(src, a) && same(dst, b) && parameters == p;
    }
    int negativeControls() const
    {
        int n = 0;
        auto changed = src;
        changed.cols++;
        require(!matches(changed, dst, parameters), "shape mismatch accepted");
        ++n;
        changed = src;
        changed.step += CV_ELEM_SIZE(src.type);
        require(!matches(changed, dst, parameters), "pitch mismatch accepted");
        ++n;
        changed = src;
        changed.ptr++;
        require(!matches(changed, dst, parameters), "alignment mismatch accepted");
        ++n;
        changed = src;
        changed.type = CV_64FC1;
        require(!matches(changed, dst, parameters), "dtype mismatch accepted");
        ++n;
        require(!matches(src, dst, parameters + "changed-config"), "parameter mismatch accepted");
        return n + 1;
    }
};

struct Operation
{
    std::string op;
    int interpolation = 0, code = 0, flags = 0, border = 0;
    cv::Size size;
    cv::Mat matrix;
    cv::Scalar borderValue;
    void call(const Pitched &src, const Pitched &dst, hipStream_t stream) const
    {
        auto handle = reinterpret_cast<std::uintptr_t>(stream);
        if (op == "resize")
            cv::triton::resize(src.image, dst.image, size, interpolation, 0, handle);
        else if (op == "cvtcolor")
            cv::triton::cvtColor(src.image, dst.image, code, 0, handle);
        else
            cv::triton::warpAffine(src.image, dst.image, matrix.ptr<double>(), flags, border,
                                   borderValue, 0, handle);
    }
    cv::Mat reference(const cv::Mat &host) const
    {
        cv::Mat result;
        if (op == "resize")
        {
            result.create(size, host.type());
            const auto z =
                cv_triton_resize::resolve_size(host.rows, host.cols, size.width, size.height, 0, 0);
            cv_triton_resize::ImageRef a{const_cast<unsigned char *>(host.data),
                                         host.rows,
                                         host.cols,
                                         host.channels(),
                                         host.step,
                                         host.depth() == CV_32F};
            cv_triton_resize::ImageRef b{result.data,       result.rows, result.cols,
                                         result.channels(), result.step, result.depth() == CV_32F};
            cv_triton_resize::resize(a, b, z.inv_y, z.inv_x, interpolation);
        }
        else if (op == "cvtcolor")
            cv::cvtColor(host, result, code);
        else
        {
            cv::Size padded = (interpolation == cv::INTER_CUBIC)
                                  ? size
                                  : cv::Size((size.width + 127) / 128 * 128, size.height);
            cv::Mat wide;
            cv::warpAffine(host, wide, matrix, padded, flags, border, borderValue);
            result = wide(cv::Rect(0, 0, size.width, size.height)).clone();
        }
        return result;
    }
    double verify(const cv::Mat &got, const cv::Mat &expected, const cv::Mat &input) const
    {
        require(got.size() == expected.size() && got.type() == expected.type(),
                "output size/type mismatch");
        double worst = 0;
        // Only the supported float32 gray/YUV/YCrCb arithmetic families use the
        // per-pixel scaled bound; permutation/copy paths require exact values.
        // See validation/numerics.hpp and src/runtime/cvtcolor.hpp.
        bool arithmetic = op == "cvtcolor" && input.depth() == CV_32F &&
                          (cv_triton_color::find(code)->op == cv_triton_color::Op::Gray ||
                           cv_triton_color::find(code)->op == cv_triton_color::Op::Yuv ||
                           cv_triton_color::find(code)->op == cv_triton_color::Op::YCrCb);
        for (int y = 0; y < got.rows; ++y)
            for (int x = 0; x < got.cols; ++x)
            {
                double scale = 0;
                if (arithmetic)
                    for (int c = 0; c < input.channels(); ++c)
                        scale = std::max(
                            scale, std::abs(double(input.ptr<float>(y)[x * input.channels() + c])));
                for (int c = 0; c < got.channels(); ++c)
                {
                    int i = x * got.channels() + c;
                    double a, b, tol = 0;
                    if (got.depth() == CV_8U)
                    {
                        a = got.ptr<unsigned char>(y)[i];
                        b = expected.ptr<unsigned char>(y)[i];
                    }
                    else if (got.depth() == CV_16U)
                    {
                        a = got.ptr<unsigned short>(y)[i];
                        b = expected.ptr<unsigned short>(y)[i];
                        tol = op == "warpaffine" ? 1 : 0;
                    }
                    else
                    {
                        a = got.ptr<float>(y)[i];
                        b = expected.ptr<float>(y)[i];
                        if (op == "resize")
                            tol = triton_validation::numeric::resizeF32Atol(interpolation);
                        else if (op == "warpaffine")
                            tol = triton_validation::numeric::warpF32Tolerance(b);
                        else if (arithmetic)
                            tol = triton_validation::numeric::cvtF32Tolerance(
                                b, scale, cv_triton_color::float_cpu_separate4());
                    }
                    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tol,
                            "independent numerical reference mismatch");
                    worst = std::max(worst, std::abs(a - b));
                }
            }
        return worst;
    }
};

static cv::Mat readInput(const J::Json &request, const std::string &pathName,
                         const std::string &hashName, int rows, int cols, int type)
{
    cv::Mat host(rows, cols, type);
    std::string path = text(request, pathName);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(in && in.tellg() == static_cast<std::streamoff>(host.total() * host.elemSize()),
            "input file size mismatch");
    in.seekg(0);
    in.read(reinterpret_cast<char *>(host.data), host.total() * host.elemSize());
    require(bool(in) &&
                J::sha256_hex(host.data, host.total() * host.elemSize()) == text(request, hashName),
            "input SHA256 mismatch");
    return host;
}

static std::string nodeInventory(hipGraph_t graph, int expectedCalls, const std::string &op)
{
    size_t count = 0;
    checked(hipGraphGetNodes(graph, nullptr, &count));
    require(count > 0, "empty graph");
    std::vector<hipGraphNode_t> nodes(count);
    checked(hipGraphGetNodes(graph, nodes.data(), &count));
    size_t kernels = 0, copies = 0, memsets = 0;
    std::map<std::string, int> launches;
    for (auto node : nodes)
    {
        hipGraphNodeType type;
        checked(hipGraphNodeGetType(node, &type));
        if (type == hipGraphNodeTypeKernel)
        {
            ++kernels;
            hipKernelNodeParams p{};
            checked(hipGraphKernelNodeGetParams(node, &p));
            std::ostringstream k;
            k << "fn=" << p.func << " grid=" << p.gridDim.x << "," << p.gridDim.y << ","
              << p.gridDim.z << " block=" << p.blockDim.x << "," << p.blockDim.y << ","
              << p.blockDim.z << " shared=" << p.sharedMemBytes;
            launches[k.str()]++;
        }
        else if (type == hipGraphNodeTypeMemcpy)
            ++copies;
        else if (type == hipGraphNodeTypeMemset)
            ++memsets;
    }
    // Require kernel work for every recorded call and keep destination fills outside capture.
    // A call may contain multiple kernels; node counts are reported for verification.
    require(kernels >= static_cast<size_t>(expectedCalls) && memsets == 0,
            "missing kernel work or poison accidentally captured");
    (void)op;
    std::ostringstream o;
    o << "{\"nodes\":" << count << ",\"kernel_nodes\":" << kernels << ",\"memcpy_nodes\":" << copies
      << ",\"memset_nodes\":" << memsets << ",\"launches\":[";
    bool first = true;
    for (const auto &entry : launches)
    {
        if (!first)
            o << ",";
        first = false;
        o << "{\"geometry\":" << J::json_quote(entry.first) << ",\"count\":" << entry.second << "}";
    }
    o << "]}";
    return o.str();
}

static void primitiveControl()
{
    Resources r;
    checked(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking));
    checked(hipMalloc(&r.output, 64));
    checked(hipMemsetAsync(r.output, 0, 64, r.stream));
    checked(hipStreamSynchronize(r.stream));
    checked(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal));
    checked(hipMemsetAsync(r.output, 0x3C, 64, r.stream));
    checked(hipStreamEndCapture(r.stream, &r.graph));
    size_t nodes = 0;
    checked(hipGraphGetNodes(r.graph, nullptr, &nodes));
    require(nodes == 1, "primitive graph not one real memset node");
    checked(hipGraphInstantiate(&r.exec, r.graph, nullptr, nullptr, 0));
    for (int i = 0; i < 3; ++i)
    {
        checked(hipMemsetAsync(r.output, 0, 64, r.stream));
        checked(hipGraphLaunch(r.exec, r.stream));
        checked(hipStreamSynchronize(r.stream));
        unsigned char data[64];
        checked(hipMemcpy(data, r.output, 64, hipMemcpyDeviceToHost));
        for (auto value : data)
            require(value == 0x3C, "primitive replay missing writes");
    }
    r.finish();
    std::cout << "{\"status\":\"PASS\",\"primitive_graph_nodes\":1,\"replays\":3}" << std::endl;
}

int main(int argc, char **argv)
{
    try
    {
        cv::setNumThreads(2);
        if (argc == 2 && std::string(argv[1]) == "--primitive-control")
        {
            primitiveControl();
            return 0;
        }
        require(argc == 2, "usage: perf_graph REQUEST.json | --primitive-control");
        std::string raw;
        require(J::read_file(argv[1], raw), "missing request");
        J::Json request;
        std::string why;
        require(J::json_parse(raw, request, why), why);
        const auto parsed = triton_graph::parse(request); // Before host/GPU allocation.
        const auto &c = *request.get("case");
        const int rows = parsed.rows, cols = parsed.cols, cn = parsed.channels;
        const int dh = parsed.height, dw = parsed.width;
        const int depth = parsed.dtype == "uint8"    ? CV_8U
                          : parsed.dtype == "uint16" ? CV_16U
                                                     : CV_32F;
        const int calls = parsed.calls, samples = parsed.samples;
        Operation op;
        op.op = parsed.op;
        op.size = {dw, dh};
        const std::string &mode = parsed.mode;
        op.interpolation = mode == "INTER_NEAREST" ? cv::INTER_NEAREST
                           : mode == "INTER_CUBIC" ? cv::INTER_CUBIC
                           : mode == "INTER_AREA"  ? cv::INTER_AREA
                                                   : cv::INTER_LINEAR;
        if (op.op == "cvtcolor")
        {
            const auto &table = cv_triton_color::all();
            auto found = std::find_if(table.begin(), table.end(),
                                      [&](const cv_triton_color::CodeSpec &spec)
                                      {
                                          return parsed.code == spec.name;
                                      });
            require(found != table.end(), "unknown conversion code");
            op.code = found->code;
            J::DeviceView view;
            view.ptr = 4096;
            view.rows = rows;
            view.cols = cols;
            view.channels = cn;
            view.step = static_cast<std::size_t>(cols) * cn * CV_ELEM_SIZE1(CV_MAKETYPE(depth, cn));
            view.type = depth == CV_8U    ? J::ElemType::U8
                        : depth == CV_16U ? J::ElemType::U16
                                          : J::ElemType::F32;
            const auto out = D::cvtColorOutput(view, op.code);
            require(dw == out.size.width && dh == out.size.height,
                    "declared destination differs from conversion output");
        }
        op.matrix = cv::Mat(2, 3, CV_64F, const_cast<double *>(parsed.matrix)).clone();
        op.flags = op.interpolation | (parsed.inverse ? cv::WARP_INVERSE_MAP : 0);
        op.border = parsed.border == "BORDER_CONSTANT"      ? cv::BORDER_CONSTANT
                    : parsed.border == "BORDER_REFLECT_101" ? cv::BORDER_REFLECT_101
                    : parsed.border == "BORDER_WRAP"        ? cv::BORDER_WRAP
                                                            : cv::BORDER_REPLICATE;
        if (op.op == "warpaffine")
        {
            J::DeviceView view;
            view.ptr = 4096;
            view.rows = rows;
            view.cols = cols;
            view.channels = cn;
            view.step = static_cast<std::size_t>(cols) * cn * CV_ELEM_SIZE1(CV_MAKETYPE(depth, cn));
            view.type = depth == CV_8U    ? J::ElemType::U8
                        : depth == CV_16U ? J::ElemType::U16
                                          : J::ElemType::F32;
            const auto warp = D::warpRequest(view, op.matrix, op.flags, op.border, op.borderValue);
            (void)D::warpOutput(warp, op.size); // Pure source/output admission before allocation.
        }
        cv::Mat host =
            readInput(request, "input_path", "input_sha256", rows, cols, CV_MAKETYPE(depth, cn));
        cv::Mat changed =
            readInput(request, "changed_path", "changed_sha256", rows, cols, host.type());
        if (depth == CV_32F)
            require(cv::checkRange(host) && cv::checkRange(changed), "nonfinite input data");
        const cv::Mat reference = op.reference(host), changedRef = op.reference(changed);
        require(cv::norm(reference, changedRef, cv::NORM_INF) > 0,
                "input mutation did not change expected output");
        Resources r;
        checked(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking));
        checked(hipEventCreate(&r.start));
        checked(hipEventCreate(&r.stop));
        Pitched src, dst;
        bool roi = parsed.roi;
        src.allocate(r, true, host.size(), host.type(), roi);
        dst.allocate(r, false, reference.size(), reference.type(), roi);
        src.upload(host, r.stream);
        checked(hipStreamSynchronize(r.stream));
        auto first = Clock::now();
        op.call(src, dst, r.stream);
        checked(hipStreamSynchronize(r.stream));
        const double firstMs = ms(first);
        cv::Mat normal = dst.downloadAndCheck(r.stream);
        op.verify(normal, reference, host);
        auto warm = Clock::now();
        for (int i = 0; i < 20; ++i)
            op.call(src, dst, r.stream);
        checked(hipStreamSynchronize(r.stream));
        D::reapCompleted();
        double warmMs = ms(warm);
        const auto identity = D::runtimeStats();
        require(!identity.last_key.empty() && !identity.last_artifact.empty(),
                "no actual production dispatch identity");
        const std::string manifest = identity.last_artifact + "/manifest.json";
        std::string manifestRaw;
        require(J::read_file(manifest, manifestRaw), "missing actual manifest");
        J::Json man;
        require(J::json_parse(manifestRaw, man, why), why);
        require(text(man, "key_text") == identity.last_key, "manifest/key identity mismatch");
        const auto *bin = man.get("binary");
        require(bin, "manifest binary absent");
        std::string binary = identity.last_artifact + "/" + text(*bin, "file");
        require(hashFile(binary) == text(*bin, "sha256"), "compiled binary hash mismatch");
        const std::string parameterHash = J::sha256_hex(raw);
        const Binding binding{src.image, dst.image, parameterHash};
        const int refusals = binding.negativeControls();
        std::vector<double> perCall, graphTotal;
        std::vector<double> wallBlocks, eventBlocks;
        double captureMs = 0, instantiateMs = 0, ordinaryMs = 0, ordinaryWall = 0;
        std::string inventory;
        long long maxPins = 0;
        double worst = 0;
        // Two independent construct/destroy lifecycles; the second is a correctness
        // control after module/table eviction, not another timing sample population.
        for (int generation = 0; generation < 2; ++generation)
        {
            src.upload(host, r.stream);
            dst.poison(r.stream);
            checked(hipStreamSynchronize(r.stream));
            op.call(src, dst, r.stream);
            checked(hipStreamSynchronize(r.stream));
            D::reapCompleted();
            const auto before = D::runtimeStats();
            auto start = Clock::now();
            checked(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal));
            for (int i = 0; i < calls; ++i)
                op.call(src, dst, r.stream);
            checked(hipStreamEndCapture(r.stream, &r.graph));
            const double captured = ms(start);
            auto after = D::runtimeStats();
            require(after.helper_compiles == before.helper_compiles &&
                        after.module_loads == before.module_loads &&
                        after.table_uploads == before.table_uploads,
                    "JIT/load/upload occurred inside capture");
            require(after.capture_launches - before.capture_launches == calls &&
                        after.capture_pin_holds > 0,
                    "runtime capture work/pins missing");
            maxPins = std::max(maxPins, after.capture_pin_holds);
            auto nodes = nodeInventory(r.graph, calls, op.op);
            if (generation == 0)
                inventory = nodes;
            start = Clock::now();
            checked(hipGraphInstantiate(&r.exec, r.graph, nullptr, nullptr, 0));
            const double instantiated = ms(start);
            if (generation == 0)
            {
                captureMs = captured;
                instantiateMs = instantiated;
            }
            auto admission = [&]()
            {
                require(binding.matches(src.image, dst.image, parameterHash),
                        "graph binding changed");
            };
            auto replay = [&]()
            {
                checked(hipGraphLaunch(r.exec, r.stream));
            };
            // Full poison and complete reference comparison after replay; canaries
            // catch unexpected writes, changed input prevents stale/empty work passes.
            for (int trial = 0; trial < 3; ++trial)
            {
                if (trial == 1)
                {
                    src.upload(changed, r.stream);
                    D::setCacheLimits(0, 0);
                }
                dst.poison(r.stream, trial == 2 ? 0x5A : Pitched::guard);
                admission();
                replay();
                checked(hipStreamSynchronize(r.stream));
                auto got = dst.downloadAndCheck(r.stream);
                const auto &expected = trial ? changedRef : reference;
                const auto &source = trial ? changed : host;
                worst = std::max(worst, op.verify(got, expected, source));
                if (!trial)
                    require(cv::norm(got, normal, cv::NORM_INF) == 0,
                            "graph differs from ordinary production output");
                const auto uploaded = src.downloadAndCheck(r.stream);
                require(cv::norm(uploaded, source, cv::NORM_INF) == 0, "source altered by graph");
            }
            if (generation == 0)
            {
                for (int sample = 0; sample < samples; ++sample)
                {
                    admission();
                    checked(hipEventRecord(r.start, r.stream));
                    replay();
                    checked(hipEventRecord(r.stop, r.stream));
                    checked(hipEventSynchronize(r.stop));
                    float elapsed = 0;
                    checked(hipEventElapsedTime(&elapsed, r.start, r.stop));
                    require(elapsed > 0 && std::isfinite(elapsed), "invalid graph elapsed time");
                    graphTotal.push_back(elapsed);
                    perCall.push_back(elapsed / calls);
                }
                worst =
                    std::max(worst, op.verify(dst.downloadAndCheck(r.stream), changedRef, changed));
            }
            checked(hipStreamSynchronize(r.stream));
            checked(hipGraphExecDestroy(r.exec));
            r.exec = nullptr;
            checked(hipGraphDestroy(r.graph));
            r.graph = nullptr;
            cv::triton::releaseGraphResources();
            D::reapCompleted();
            require(D::runtimeStats().capture_pin_holds == 0 && D::runtimeStats().pending == 0,
                    "pins/pending survive graph destruction");
            D::setCacheLimits(64, 256);
        }
        // Ordinary production-call cost remains separate, with JIT re-warmed after eviction.
        op.call(src, dst, r.stream);
        checked(hipStreamSynchronize(r.stream));
        for (int sample = 0; sample < 7; ++sample)
        {
            auto wallStart = Clock::now();
            checked(hipEventRecord(r.start, r.stream));
            for (int i = 0; i < 250; ++i)
                op.call(src, dst, r.stream);
            checked(hipEventRecord(r.stop, r.stream));
            checked(hipEventSynchronize(r.stop));
            wallBlocks.push_back(ms(wallStart) / 250);
            float elapsed = 0;
            checked(hipEventElapsedTime(&elapsed, r.start, r.stop));
            eventBlocks.push_back(elapsed / 250);
        }
        auto sortedWall = wallBlocks, sortedEvent = eventBlocks;
        std::sort(sortedWall.begin(), sortedWall.end());
        std::sort(sortedEvent.begin(), sortedEvent.end());
        ordinaryWall = sortedWall[3];
        ordinaryMs = sortedEvent[3];
        worst = std::max(worst, op.verify(dst.downloadAndCheck(r.stream), changedRef, changed));
        D::reapCompleted();
        require(hashFile(manifest) == J::sha256_hex(manifestRaw) &&
                    hashFile(binary) == text(*bin, "sha256"),
                "artifact changed during benchmark");
        r.finish();
        std::vector<double> sorted = perCall;
        std::sort(sorted.begin(), sorted.end());
        auto quantile = [&](double q)
        {
            double pos = q * (sorted.size() - 1);
            size_t low = static_cast<size_t>(pos), high = std::min(low + 1, sorted.size() - 1);
            return sorted[low] * (high - pos) + sorted[high] * (pos - low);
        };
        // Retain raw timing samples alongside the per-call summary.
        std::cout.precision(17);
        std::cout
            << "{\"case_id\":" << J::json_quote(text(c, "id"))
            << ",\"metric\":\"graph_replay\",\"status\":\"PASS\",\"baseline_status\":\"BASELINE_UNAVAILABLE\",\"candidate_ms\":"
            << quantile(.5) << ",\"p10_ms\":" << quantile(.1) << ",\"p90_ms\":" << quantile(.9)
            << ",\"min_ms\":" << sorted.front() << ",\"max_ms\":" << sorted.back()
            << ",\"captured_calls\":" << calls << ",\"timed_replays\":" << samples
            << ",\"raw_total_graph_ms\":[";
        for (size_t i = 0; i < graphTotal.size(); ++i)
            std::cout << (i ? "," : "") << graphTotal[i];
        std::cout << "],\"raw_per_call_ms\":[";
        for (size_t i = 0; i < perCall.size(); ++i)
            std::cout << (i ? "," : "") << perCall[i];
        std::cout << "],\"ordinary_wall_blocks_ms\":[";
        for (size_t i = 0; i < wallBlocks.size(); ++i)
            std::cout << (i ? "," : "") << wallBlocks[i];
        std::cout << "],\"ordinary_event_blocks_ms\":[";
        for (size_t i = 0; i < eventBlocks.size(); ++i)
            std::cout << (i ? "," : "") << eventBlocks[i];
        std::cout << "],\"first_compile_call_wall_ms\":" << firstMs
                  << ",\"warmup20_wall_ms\":" << warmMs << ",\"capture_wall_ms\":" << captureMs
                  << ",\"instantiate_wall_ms\":" << instantiateMs
                  << ",\"ordinary_event_ms\":" << ordinaryMs
                  << ",\"ordinary_completion_wall_ms\":" << ordinaryWall
                  << ",\"max_abs_error\":" << worst << ",\"binding_refusal_controls\":" << refusals
                  << ",\"construct_destroy_cycles\":2,\"max_resource_pins\":" << maxPins
                  << ",\"remaining_pins\":0,\"remaining_pending\":0,\"graph_work\":" << inventory
                  << ",\"production_route\":" << J::json_quote(identity.last_route)
                  << ",\"production_key\":" << J::json_quote(identity.last_key)
                  << ",\"artifact\":" << J::json_quote(identity.last_artifact)
                  << ",\"manifest_sha256\":" << J::json_quote(J::sha256_hex(manifestRaw))
                  << ",\"binary_sha256\":" << J::json_quote(text(*bin, "sha256"))
                  << ",\"request_sha256\":" << J::json_quote(J::sha256_hex(raw))
                  << ",\"executed_parameters\":" << triton_graph::executionJson(parsed)
                  << ",\"input_sha256\":" << J::json_quote(text(request, "input_sha256"))
                  << ",\"changed_sha256\":" << J::json_quote(text(request, "changed_sha256")) << "}"
                  << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
