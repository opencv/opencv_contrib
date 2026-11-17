#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

# Copyright 2026, The FlagOS Contributors.
"""Derive src/runtime/hip_runtime.cpp from src/runtime/cuda_runtime.cpp.

    port_hip_runtime.py            write src/runtime/hip_runtime.cpp
    port_hip_runtime.py --check    exit 1 if the checked-in file is not what this would write

The CUDA file is the common source. This generator owns the HIP differences:
driver signatures, target queries, context identification and guarded colour dispatch.
Checked replacement anchors/counts and the final CUDA-token scan detect source changes
that need porting updates. Run --check after editing either owner; do not edit the
generated runtime directly. This check proves reproducible derivation, not GPU behavior.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
src = (HERE / "src/runtime/cuda_runtime.cpp").read_text()
out = src

def sub(old, new, count=None):
    global out
    n = out.count(old)
    if count is not None and n != count:
        sys.exit(f"expected {count} x {old!r}, found {n}")
    if n == 0:
        sys.exit(f"pattern not found: {old!r}")
    out = out.replace(old, new)

# ---- 1. header: what this file is, and why it differs ----------------------------------
sub('// Runtime JIT for the CUDA driver API. See runtime.hpp for execution and ownership rules.\n',
'''// Runtime JIT for the HIP driver API (ROCm/DTK). See runtime.hpp for execution and ownership rules.
//
// GENERATED from cuda_runtime.cpp by tools/port_hip_runtime.py -- do not edit; edit the CUDA
// file and re-run the port. Exactly one of the two is compiled into a build (CMakeLists.txt).
//
// What differs from the CUDA file, and why -- everything else is the same code:
//   * Target: backend "hip", numeric gfx architecture and warp size queried from the device.
//     BW1000 uses gfx936/wave64; W7900D uses gfx1100/wave32. Letter-suffix targets such as
//     gfx90a cannot be represented by this key and are rejected.
//   * Kernels: kernels/<op>_cpu_hip.py where the CUDA source calls a libdevice primitive
//     the HIP compiler lacks; tools/make_kernels_cpu_hip.py derives resize, cvtColor
//     and warpAffine sources and rejects stale copies.
//   * Binary: manifest format "hsaco", loaded with hipModuleLoadData.
//   * Compile helper environment: ROCM_PATH is forwarded (the hip compiler finds clang and its
//     device libraries through it), and the GPU is hidden from the helper with the HIP and ROCR
//     visibility variables as CUDA_VISIBLE_DEVICES= does for cuda.
//   * Dynamic shared memory above the 64 KiB LDS is refused rather than requested through an
//     attribute call, which HIP defines for host function pointers, not module functions.
//   * Context identity uses DTK's hipCtxGetId when available, otherwise retained primary
//     contexts on ROCm. Deprecation warnings for the required hipCtx* API are local to this file.
''', 1)
sub('#include <cuda.h>\n', '#include <hip/hip_runtime_api.h>\n', 1)
sub('#include <sys/stat.h>\n', '#include <sys/stat.h>\n\n#pragma GCC diagnostic ignored "-Wdeprecated-declarations"\n', 1)
sub('#include <algorithm>\n', '#include <algorithm>\n#include <cctype>\n', 1)

# ---- 2. configuration: hip kernels where needed, ROCM_PATH, helper env -----------------
sub('    std::string resources, kernels, helper, cache, python, pythonpath, compiler, source_sha256, error;\n',
    '    std::string resources, kernels, helper, cache, python, pythonpath, compiler, rocm_path, source_sha256, error;\n', 1)
sub('    return isFile(d + "/kernels/resize_cpu.py") && isFile(d + "/tools/compile_helper.py");\n',
    '    return isFile(d + "/kernels/resize_cpu_hip.py") && isFile(d + "/tools/compile_helper.py");\n', 1)
sub('        k.kernels = r + "/kernels/resize_cpu.py";\n', '        k.kernels = r + "/kernels/resize_cpu_hip.py";\n', 1)
sub('            k.error = "kernels/resize_cpu.py and tools/compile_helper.py not found under " + r;\n',
    '            k.error = "kernels/resize_cpu_hip.py and tools/compile_helper.py not found under " + r;\n', 1)
# Float cvtColor uses explicit libdevice rounding; derive its HIP source.
sub('        k.cvt_kernels = r + "/kernels/cvtcolor_cpu.py";\n',
    '        k.cvt_kernels = r + "/kernels/cvtcolor_cpu_hip.py";\n', 1)
sub('        k.wa_kernels = r + "/kernels/warpaffine_cpu.py";\n',
    '        k.wa_kernels = r + "/kernels/warpaffine_cpu_hip.py";\n', 1)
sub('        k.compiler = envStr("OPENCV_TRITON_COMPILER");\n',
    '        k.rocm_path = envStr("ROCM_PATH");\n'
    '        if (k.rocm_path.empty()) k.rocm_path = OPENCV_TRITON_ROCM_PATH_DEFAULT;\n'
    '        k.compiler = envStr("OPENCV_TRITON_COMPILER");\n', 1)
sub('             "PYTHONPATH=" + c.pythonpath, "CUDA_VISIBLE_DEVICES=", "LC_ALL=C"};   // the helper needs no device\n',
    '             "PYTHONPATH=" + c.pythonpath,\n'
    '             "TMPDIR=" + c.cache,\n'
    '             "ROCM_PATH=" + c.rocm_path,   // the hip compiler finds clang and its device libraries through this\n'
    '             "HIP_VISIBLE_DEVICES=", "ROCR_VISIBLE_DEVICES=", "CUDA_VISIBLE_DEVICES=", "LC_ALL=C"};   // no device\n', 1)

# ---- 3. error text -------------------------------------------------------------------
sub('''// --------------------------------------------------------------------- CUDA

std::string cuErr(CUresult r)
{
    const char* s = nullptr;
    cuGetErrorName(r, &s);
    return s ? std::string(s) : ("CUresult " + std::to_string(static_cast<int>(r)));
}''', '''// ---------------------------------------------------------------------- HIP

std::string hipErr(hipError_t r)
{
    const char* s = hipGetErrorName(r);
    return s ? std::string(s) : ("hipError_t " + std::to_string(static_cast<int>(r)));
}''', 1)
out = re.sub(r'\bcuErr\(', 'hipErr(', out)

# ---- 4. calls whose signature differs (before the token renames) -----------------------
# CUdeviceptr is an integer, hipDeviceptr_t a void*: the resources keep integers, the calls cast.
sub('    CUdeviceptr base = 0;\n', '    std::uintptr_t base = 0;\n', None)
sub('const CUresult result = stream_ordered ? cuMemFreeAsync(base, nullptr) : cuMemFree(base);',
    'const hipError_t result = stream_ordered ? hipFreeAsync(reinterpret_cast<void*>(base), nullptr) : hipFree(reinterpret_cast<void*>(base));', 1)
alloc_old = '''    CUresult r = t->stream_ordered ? cuMemAllocAsync(&t->base, t->bytes, stream) : cuMemAlloc(&t->base, t->bytes);
    if (r == CUDA_SUCCESS && poison) r = cuMemsetD8Async(t->base, 0xFF, t->bytes, stream);
    if (r == CUDA_SUCCESS) { t->upload_started = true; r = cuMemcpyHtoDAsync(t->base, t->host.data(), t->bytes, stream); }
    if (r == CUDA_SUCCESS) r = cuEventCreate(&t->uploaded, CU_EVENT_DISABLE_TIMING);
    if (r == CUDA_SUCCESS) { r = cuEventRecord(t->uploaded, stream); t->upload_recorded = r == CUDA_SUCCESS; }'''
alloc_new = '''    void* dptr = nullptr;
    hipError_t r = t->stream_ordered ? hipMallocAsync(&dptr, t->bytes, stream) : hipMalloc(&dptr, t->bytes);
    t->base = reinterpret_cast<std::uintptr_t>(dptr);
    if (r == hipSuccess && poison) r = hipMemsetD8Async(dptr, 0xFF, t->bytes, stream);
    if (r == hipSuccess) { t->upload_started = true; r = hipMemcpyHtoDAsync(dptr, t->host.data(), t->bytes, stream); }
    if (r == hipSuccess) r = hipEventCreateWithFlags(&t->uploaded, hipEventDisableTiming);
    if (r == hipSuccess) { r = hipEventRecord(t->uploaded, stream); t->upload_recorded = r == hipSuccess; }'''
n_alloc = out.count(alloc_old)
if n_alloc < 2: sys.exit(f"table-upload block: expected >=2, found {n_alloc}")
out = out.replace(alloc_old, alloc_new)

copy_old = re.compile(r'''        CUDA_MEMCPY2D cp;
        std::memset\(&cp, 0, sizeof cp\);
        cp\.srcMemoryType = CU_MEMORYTYPE_DEVICE; cp\.srcDevice = (?P<s>[\w.]+)\.ptr; cp\.srcPitch = (?P=s)\.step;
        cp\.dstMemoryType = CU_MEMORYTYPE_DEVICE; cp\.dstDevice = (?P<d>[\w.]+)\.ptr; cp\.dstPitch = (?P=d)\.step;
        cp\.WidthInBytes = (?P<w>[^;]+); cp\.Height = (?P<h>[^;]+);
        const CUresult r = cuMemcpy2DAsync\(&cp, stream\);''')
def copy_new(m):
    return (f'''        const hipError_t r = hipMemcpy2DAsync(reinterpret_cast<void*>({m["d"]}.ptr), {m["d"]}.step,
                                              reinterpret_cast<const void*>({m["s"]}.ptr), {m["s"]}.step,
                                              {m["w"]}, {m["h"]},
                                              hipMemcpyDeviceToDevice, stream);''')
out, n_copy = copy_old.subn(copy_new, out)
if n_copy < 1: sys.exit("2-D copy block not found")

load_old = '''    CUresult r = cuModuleLoadData(&m->mod, bin.data());
    if (r == CUDA_SUCCESS) r = cuModuleGetFunction(&m->fn, m->mod, man.function.c_str());
    if (r == CUDA_SUCCESS && man.shared_bytes > 48 * 1024)
        r = cuFuncSetAttribute(m->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, man.shared_bytes);'''
load_new = '''    if (man.format != "hsaco") { st = error(Reason::ManifestInvalid, "binary format '" + man.format + "' is not hsaco"); return nullptr; }
    if (man.shared_bytes > 64 * 1024)
    {
        st = error(Reason::LoadFailed, "kernel asks for " + std::to_string(man.shared_bytes) + " bytes of dynamic shared memory; the LDS is 64 KiB");
        return nullptr;
    }
    hipError_t r = hipModuleLoadData(&m->mod, bin.data());
    if (r == hipSuccess) r = hipModuleGetFunction(&m->fn, m->mod, man.function.c_str());'''
sub(load_old, load_new, 1)

sub('''    const CUresult r = cuLaunchKernel(m.fn, grid[0], grid[1], grid[2], static_cast<unsigned>(m.man.threads_per_block), 1, 1,
                                      static_cast<unsigned>(m.man.shared_bytes), stream, params.data(), nullptr);
    if (r != CUDA_SUCCESS) return error(Reason::LaunchFailed, "cuLaunchKernel: " + hipErr(r));''',
'''    const hipError_t r = hipModuleLaunchKernel(m.fn, grid[0], grid[1], grid[2], static_cast<unsigned>(m.man.threads_per_block), 1, 1,
                                               static_cast<unsigned>(m.man.shared_bytes), stream, params.data(), nullptr);
    if (r != hipSuccess) return error(Reason::LaunchFailed, "hipModuleLaunchKernel: " + hipErr(r));''', 1)

sub('''    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    return cuStreamIsCapturing(stream, &st) == CUDA_SUCCESS && st != CU_STREAM_CAPTURE_STATUS_NONE;''',
'''    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &st) == hipSuccess && st != hipStreamCaptureStatusNone;''', 1)

# ---- 5. target identity: three identical query blocks -> one cached device query --------
tgt = re.compile(r'''    CUdevice dev;
    cuCtxGetDevice\(&dev\);
    int major = 0, minor = 0, warp = 32;
    cuDeviceGetAttribute\(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev\);
    cuDeviceGetAttribute\(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev\);
    cuDeviceGetAttribute\(&warp, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev\);
''')
tgt_new = '''    DevInfo dev;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        auto it = s.devices.find(ex.ctx.device);
        if (it == s.devices.end()) it = s.devices.emplace(ex.ctx.device, queryDevice(ex.ctx.device)).first;
        dev = it->second;
    }
    if (!dev.ok) return error(Reason::ToolchainUnavailable, dev.why);
'''
out, n_tgt = tgt.subn(tgt_new, out)
if n_tgt != 3: sys.exit(f"target query blocks: expected 3, found {n_tgt}")
out, n_a = re.subn(r'(\b(?:key\.)?target)\.backend = "cuda";\n(\s+)\1\.arch = major \* 10 \+ minor;\n\s+\1\.warp_size = warp;',
                   lambda m: f'{m[1]}.backend = "hip";\n{m[2]}{m[1]}.arch = dev.arch;\n{m[2]}{m[1]}.warp_size = dev.warp;', out)
if n_a != 3: sys.exit(f"target assignments: expected 3, found {n_a}")

# DevInfo + its query live beside State, which owns the per-device cache.
sub('''struct State
{''', '''// The device facts the compile key names, read once per ordinal: the gfx number (the key's
// integer arch) and the wave size.
struct DevInfo
{
    bool ok = false;
    int arch = 0;                // numeric gfx suffix, e.g. 936 or 1100
    int warp = 64;
    std::string arch_name, why;
};

DevInfo queryDevice(int device)
{
    DevInfo d;
    hipDeviceProp_t p;
    std::memset(&p, 0, sizeof p);
    const hipError_t r = hipGetDeviceProperties(&p, device);
    if (r != hipSuccess) { d.why = "hipGetDeviceProperties: " + hipErr(r); return d; }
    d.arch_name = p.gcnArchName;
    d.warp = p.warpSize > 0 ? p.warpSize : 64;
    const std::string& n = d.arch_name;
    std::size_t i = 3;
    long v = 0;
    if (n.compare(0, 3, "gfx") != 0) { d.why = "unrecognised architecture name '" + n + "'"; return d; }
    for (; i < n.size() && std::isdigit(static_cast<unsigned char>(n[i])); ++i) v = v * 10 + (n[i] - '0');
    if (i == 3 || (i < n.size() && n[i] != ':'))
    {
        d.why = "architecture '" + n + "' has no integer gfx number, so the compile key cannot name it";
        return d;
    }
    d.arch = static_cast<int>(v);
    d.ok = true;
    return d;
}

struct State
{
    std::map<int, DevInfo> devices;''', 1)

# ROCm does not provide DTK's hipCtxGetId extension. Preserve it when exported;
# otherwise support retained primary contexts only. No pointer-derived fake
# generation: the retained context is pinned for this process's cache lifetime.
# Device reset and primary-context reset are unsupported while this runtime is in use.
context_helper = r'''
hipError_t contextGeneration(hipCtx_t context, unsigned long long* generation, bool registerPrimary = true)
{
    using GetId = hipError_t (*)(hipCtx_t, unsigned long long*);
    static const GetId get_id = reinterpret_cast<GetId>(dlsym(RTLD_DEFAULT, "hipCtxGetId"));
    if (get_id) return get_id(context, generation);
    if (!context || !generation) return hipErrorInvalidContext;
    struct Primary { hipCtx_t context; int device; unsigned long long generation; };
    struct Registry { std::mutex mutex; std::vector<Primary> primaries; };
    static auto* registry = new Registry();
    std::lock_guard<std::mutex> lock(registry->mutex);
    for (const auto& primary : registry->primaries)
        if (primary.context == context)
        {
            unsigned int flags = 0; int active = 0;
            const hipError_t result = hipDevicePrimaryCtxGetState(primary.device, &flags, &active);
            if (result != hipSuccess) return result;
            if (!active) return hipErrorInvalidContext;
            *generation = primary.generation;
            return hipSuccess;
        }
    // Cleanup must not register/retain a new context or switch a destroyed one.
    if (!registerPrimary) return hipErrorInvalidValue;
    hipCtx_t previous = nullptr;
    hipError_t result = hipCtxGetCurrent(&previous);
    if (result != hipSuccess) return result;
    if (previous != context) return hipErrorInvalidContext;
    hipDevice_t device;
    result = hipCtxGetDevice(&device);
    if (result != hipSuccess) return result;
    hipCtx_t primary = nullptr;
    result = primaryContext(static_cast<int>(device), primary);
    // ROCm/DTK retain can change the current context: restore even when ownership
    // acquisition succeeded but this request names a distinct private context.
    const hipError_t restored = hipCtxSetCurrent(previous);
    if (result != hipSuccess) return result;
    if (restored != hipSuccess) return restored;
    if (context != primary) return hipErrorInvalidContext;
    const auto id = static_cast<unsigned long long>(registry->primaries.size()) + 1;
    registry->primaries.push_back({primary, static_cast<int>(device), id});
    *generation = id;
    return hipSuccess;
}

'''
# Declare primaryContext before context identification so hipCtxGetId and the
# primary-context fallback share one process-lifetime reference per device.
sub('ContextLife contextLife(CUcontext c, std::uint64_t gen)\n',
    'CUresult primaryContext(int device, CUcontext& context);\n' + context_helper +
    'ContextLife contextLife(CUcontext c, std::uint64_t gen)\n', 1)
sub('cuCtxGetId(c, &id)', 'contextGeneration(c, &id, false)', 1)
sub('cuCtxGetId(ctx, &id)', 'contextGeneration(ctx, &id)', 1)

# ---- 6. plain token renames -----------------------------------------------------------
pairs = [
    ("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED", "hipErrorStreamCaptureUnsupported"),
    ("CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED", "hipDeviceAttributeMemoryPoolsSupported"),
    ("CU_EVENT_DISABLE_TIMING", "hipEventDisableTiming"),
    ("CUDA_ERROR_NOT_READY", "hipErrorNotReady"),
    ("CU_STREAM_CAPTURE_STATUS_NONE", "hipStreamCaptureStatusNone"),
    ("CUstreamCaptureStatus", "hipStreamCaptureStatus"),
    ("cuStreamIsCapturing", "hipStreamIsCapturing"),
    ("CUDA_ERROR_INVALID_CONTEXT", "hipErrorInvalidContext"),
    ("CUDA_ERROR_CONTEXT_IS_DESTROYED", "hipErrorContextIsDestroyed"),
    ("CUDA_ERROR_DEINITIALIZED", "hipErrorDeinitialized"),
    ("CUDA_SUCCESS", "hipSuccess"),
    ("CUcontext", "hipCtx_t"), ("CUstream", "hipStream_t"), ("CUresult", "hipError_t"),
    ("CUdevice", "hipDevice_t"), ("CUevent", "hipEvent_t"), ("CUmodule", "hipModule_t"),
    ("CUfunction", "hipFunction_t"), ("CUdeviceptr", "std::uintptr_t"),
    ("cuDevicePrimaryCtxRetain", "hipDevicePrimaryCtxRetain"),
    ("cuDeviceGetAttribute", "hipDeviceGetAttribute"), ("cuDeviceGetCount", "hipGetDeviceCount"),
    ("cuDeviceGet", "hipDeviceGet"), ("cuCtxGetCurrent", "hipCtxGetCurrent"),
    ("cuCtxSetCurrent", "hipCtxSetCurrent"), ("cuCtxPushCurrent", "hipCtxPushCurrent"),
    ("cuCtxPopCurrent", "hipCtxPopCurrent"), ("cuCtxGetDevice", "hipCtxGetDevice"),
    ("cuCtxGetId", "hipCtxGetId"), ("cuStreamWaitEvent", "hipStreamWaitEvent"),
    ("cuEventRecord", "hipEventRecord"), ("cuEventQuery", "hipEventQuery"),
    ("cuEventCreate", "hipEventCreateWithFlags"), ("cuEventDestroy", "hipEventDestroy"),
    ("cuModuleUnload", "hipModuleUnload"), ("cuInit", "hipInit"),
]
for a, b in pairs:
    out = re.sub(rf"\b{a}\b", b, out)
# messages that name the API
out = out.replace('"cuInit failed"', '"hipInit failed"').replace('"hipCtxGetId failed"', '"hipCtxGetId failed"')
out = out.replace('return "cuda-jit";', 'return "hip-jit";')
out = out.replace('// never destroyed: process exit may follow CUDA teardown', '// never destroyed: process exit may follow HIP teardown')

# Driver API spellings only: cuXxx( calls, CU_ macros, CUDA_ codes, CUxxx types. A bare
# all-caps word such as CUBIC (an interpolation) is not one.
# HIP declares these [[nodiscard]]; the CUDA file ignores their results on purpose (a
# best-effort query, a teardown call whose failure has no recovery). Say so with (void),
# rather than silencing -Wunused-result for the whole file.
for call in ("hipCtxGetCurrent(", "hipCtxPopCurrent(", "hipModuleUnload(", "hipStreamWaitEvent(nullptr",
             "hipEventDestroy(", "hipDeviceGetAttribute("):
    fixed = []
    for line in out.split("\n"):
        i = line.find(call)
        while i != -1:
            before = line[:i].rstrip()
            # A bare statement: the call starts the line, or follows `{`, `;` or an if-condition's `)`.
            if (before == "" or before.endswith(("{", ";", ")"))) and not before.endswith("(void)"):
                line = line[:i] + "(void)" + line[i:]
                i += len("(void)")
            i = line.find(call, i + len(call))
        fixed.append(line)
    out = "\n".join(fixed)
left = sorted(set(re.findall(r"\bcu[A-Z][A-Za-z0-9_]*\(|\bCU_[A-Z_]+\b|\bCUDA_[A-Z_]+\b|\bCU[a-z][A-Za-z]*\b", out)))
# CUDA_VISIBLE_DEVICES is intentionally forwarded (empty) and comments may name CUDA errors.
code_left = [t for t in left if t != "CUDA_VISIBLE_DEVICES" and
             any(t in line and not line.lstrip().startswith("//") for line in out.splitlines())]
if code_left:
    sys.exit(f"CUDA tokens left in code: {code_left}")

# HIP colour vectorization keeps the CUDA runtime's generic route unchanged.
sub('''    std::sort(key.constexprs.begin(), key.constexprs.end());

    std::shared_ptr<ModuleRes> mod = moduleFor(key, ctx, gen, st, c.cvt_kernels);''',
'''    const bool packed = dev.arch == 936 && !n.u16 && !n.f32 && n.cols >= 1024 && n.cols % 4 == 0 &&
        req.src.ptr % 4 == 0 && req.dst.ptr % 4 == 0 &&
        req.src.step % 4 == 0 && req.dst.step % 4 == 0 &&
        spec.code == 6 && n.n >= (1LL << 19);
    if (packed)
    {
        key.kernel = "_cvtcolor_hip_packed";
        key.signature = "src:*u8, dst:*u8, cols:i32, src_rs:i32, dst_rs:i32";
        key.cfg.block_w = n.n < (1LL << 22) ? 1024 : 512;
        key.cfg.num_warps = n.n < (1LL << 22) ? 4 : 2;
        key.constexprs = {{"BLOCK", key.cfg.block_w}};
    }
    std::sort(key.constexprs.begin(), key.constexprs.end());

    std::shared_ptr<ModuleRes> mod = moduleFor(key, ctx, gen, st, c.cvt_kernels);''', 1)
sub('''    cv_triton_color::grid(n, key.cfg.block_w, grid);''',
'''    cv_triton_color::grid(n, key.cfg.block_w, grid);
    if (packed)
    {
        // A lane covers four complete pixels; row pitches stay independent.
        grid[1] = static_cast<unsigned>((n.cols / 4 + key.cfg.block_w - 1) / key.cfg.block_w);
    }''', 1)

dst = HERE / "src/runtime/hip_runtime.cpp"
if "--check" in sys.argv[1:]:
    if not dst.is_file() or dst.read_text() != out:
        print("src/runtime/hip_runtime.cpp is stale or missing; run tools/port_hip_runtime.py", file=sys.stderr)
        sys.exit(1)
    print("src/runtime/hip_runtime.cpp is current")
    sys.exit(0)
dst.write_text(out)
print(f"wrote src/runtime/hip_runtime.cpp ({out.count(chr(10))} lines)")
