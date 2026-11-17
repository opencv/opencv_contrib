// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Private compile-helper protocol, C++ side.
//
// The runtime never embeds Python. On a cache miss it writes a structured request, runs the
// Torch-free compile helper (tools/compile_helper.py) as a separate process
// under a deadline, and accepts only an artifact it can verify itself:
//
//   request  {protocol, key{kernel, signature, constexprs, num_warps, num_stages, target,
//             compiler, source_sha256}, key_text, source_path, cache_dir}
//   response {protocol, request_sha256, key_digest, status compiled|cached|error, error{kind,
//             message}, artifact_dir, manifest_sha256}
//   artifact <cache_dir>/<key_digest>/{manifest.json, kernel.cubin | kernel.hsaco}
//
// key_text is KernelKey::text() and key_digest = sha256(key_text): the cache entry, the
// manifest and the compile key name one another. An entry is valid only if its manifest parses,
// names this protocol and this key, and the binary matches the manifest's hash; anything
// else (a half-published directory, a corrupted file) is a miss, never a hit.
#ifndef OPENCV_TRITON_RUNTIME_COMPILE_PROTOCOL_HPP
#define OPENCV_TRITON_RUNTIME_COMPILE_PROTOCOL_HPP
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "provider.hpp"
namespace cv_triton_runtime
{

constexpr const char *COMPILE_PROTOCOL = "opencv-triton-compile/1";

// ------------------------------------------------------------------ utilities

std::string sha256_hex(const void *data, std::size_t n);
std::string sha256_hex(const std::string &s);
bool sha256_file(const std::string &path, std::string &hex, std::size_t *bytes = nullptr);
bool read_file(const std::string &path, std::string &out);

// A small JSON value: enough for this protocol, no third-party dependency.
struct Json
{
    enum Kind
    {
        Null,
        Bool,
        Number,
        String,
        Array,
        Object
    } kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;
    const Json *get(const std::string &k) const; // object member or nullptr
    bool is_int() const;
    long long as_int() const
    {
        return static_cast<long long>(num);
    }
};
bool json_parse(const std::string &text, Json &out, std::string &err);
std::string json_quote(const std::string &s);

// --------------------------------------------------------------------- request

// The request for one key. source_path is the kernels file the helper compiles; its
// sha256 must equal key.toolchain.source_sha256 or the helper refuses.
std::string make_request(const KernelKey &key, const std::string &source_path,
                         const std::string &cache_dir);
std::string key_digest(const KernelKey &key); // sha256(key.text())

// --------------------------------------------------------------------- runner

struct HelperCommand
{
    std::string python;           // the interpreter the module was configured with
    std::string helper;           // path to compile_helper.py
    std::vector<std::string> env; // full environment for the child, "K=V"
    double timeout_s = 120;
};

struct CompileOutcome
{
    Status status; // Prepared on success; Error otherwise
    std::string
        kind; // compiled | cached | timeout | helper_crashed | bad_response | <helper error kind>
    std::string response; // raw response text (may be empty)
    std::string artifact_dir;
    int exit_code = 0; // child exit status, -signal, or INT_MIN if unavailable
};

// Runs the helper in its own process group with --request <file>; stdin is /dev/null.
// One monotonic deadline covers stdout and child exit independently. Timeout kills the
// whole group (SIGKILL), then allows at most one second to confirm reap; an unconfirmed
// reap reports helper_cleanup_failed, never a successful timeout recovery. A killed or
// crashed helper can leave a temporary directory that no reader accepts. The response
// is verified (protocol,
// request_sha256, key_digest, manifest_sha256) and so is the artifact before success.
CompileOutcome compile_key(const HelperCommand &cmd, const KernelKey &key,
                           const std::string &source_path, const std::string &cache_dir);

// The same for a request already serialised (the runtime uses make_request; tests send
// tampered ones).
CompileOutcome run_request(const HelperCommand &cmd, const std::string &request);

// ------------------------------------------------------------ artifact checks

struct ManifestParam
{
    std::string name, type, ptx;
    bool null_pointer = false;
};
struct Manifest
{
    std::string protocol, key_text, key_digest, function, binary_file, binary_sha256, format;
    std::size_t binary_bytes = 0;
    std::vector<ManifestParam> abi; // every PTX entry parameter, in order
    int threads_per_block = 0, num_warps = 0, warp_size = 0, num_ctas = 0, shared_bytes = 0;
    // -1 means absent/unknown metadata; only an explicit zero permits a null scratch pointer.
    long long global_scratch_bytes = -1, profile_scratch_bytes = -1;
    std::string compiler; // key.compiler: whose ABI the parameter list follows
};

// Compilers whose entry ABI has been validated against the manifest's parameter list: the
// runtime arguments, then exactly two trailing pointers (global_scratch, profile_scratch)
// that the launcher may pass as null when the manifest records zero scratch bytes.
// Keyed by (compiler, binary format): "cubin" (PTX entry parameters) and "hsaco" (AMDHSA
// kernel-argument metadata) are validated separately, because two Triton distributions can
// report the same version string. For hsaco the manifest's `ptx` field carries the scalar
// class derived from that metadata (the field name is the protocol's, kept for stability).
// launch_abi_ok() is the check the launcher runs before every first launch of a module.
bool compiler_abi_validated(const std::string &compiler, const std::string &format);
bool launch_abi_ok(const Manifest &m, std::string &why);

// Verifies <dir>/manifest.json and the binary against an expected key_text: protocol, key_text
// equal to the manifest's own structured key re-serialised, digest, binary hash and size,
// ABI = the key's signature in order plus only null pointer parameters after it, and
// threads = num_warps * warp_size. Returns false with a reason on the first mismatch.
bool verify_artifact(const std::string &dir, const std::string &expected_key_text, Manifest &out,
                     std::string &why);

// Verifies a response against the request bytes it answers. On success, artifact_dir names
// a verified artifact.
bool verify_response(const std::string &request, const std::string &response,
                     std::string &artifact_dir, std::string &status, std::string &why);

// Rebuilds KernelKey::text() from the protocol's structured key object.
bool key_text_from_json(const Json &key, std::string &text, std::string &why);

} // namespace cv_triton_runtime
#endif // OPENCV_TRITON_RUNTIME_COMPILE_PROTOCOL_HPP
