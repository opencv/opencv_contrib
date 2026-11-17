// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Compile protocol, C++ side. See compile_protocol.hpp. Requires POSIX process
// support (fork/exec); file locking is handled by the compile helper.
// Standalone CPU controls and test support do not use library-private headers.
#ifdef __OPENCV_BUILD
#include "../precomp.hpp"
#endif
#include "compile_protocol.hpp"
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
namespace cv_triton_runtime
{

// ------------------------------------------------------------------ sha256
// Standard SHA-256, implemented locally without a crypto-library dependency.
// test/cpu/test_protocol.cpp checks a known digest; the hashes bind cache
// content, not an authenticated publisher or a trusted signature.

namespace
{

struct Sha256
{
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char buf[64];
    std::size_t len = 0;
    uint64_t total = 0;
    static uint32_t rotr(uint32_t x, int n)
    {
        return (x >> n) | (x << (32 - n));
    }
    void block(const unsigned char *p)
    {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) |
                   (uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
        for (int i = 16; i < 64; ++i)
        {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i)
        {
            uint32_t t1 =
                hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }
    void update(const void *data, std::size_t n)
    {
        const unsigned char *p = static_cast<const unsigned char *>(data);
        total += n;
        while (n)
        {
            std::size_t k = std::min(n, sizeof(buf) - len);
            std::memcpy(buf + len, p, k);
            len += k;
            p += k;
            n -= k;
            if (len == 64)
            {
                block(buf);
                len = 0;
            }
        }
    }
    std::string hex()
    {
        uint64_t bits = total * 8;
        unsigned char pad = 0x80, zero = 0;
        update(&pad, 1);
        while (len != 56)
            update(&zero, 1);
        unsigned char lb[8];
        for (int i = 0; i < 8; ++i)
            lb[i] = static_cast<unsigned char>(bits >> (56 - 8 * i));
        update(lb, 8);
        char out[65];
        for (int i = 0; i < 8; ++i)
            std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

} // namespace

std::string sha256_hex(const void *data, std::size_t n)
{
    Sha256 s;
    s.update(data, n);
    return s.hex();
}
std::string sha256_hex(const std::string &s)
{
    return sha256_hex(s.data(), s.size());
}

bool read_file(const std::string &path, std::string &out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::ostringstream o;
    o << f.rdbuf();
    out = o.str();
    return true;
}

bool sha256_file(const std::string &path, std::string &hex, std::size_t *bytes)
{
    std::string d;
    if (!read_file(path, d))
        return false;
    hex = sha256_hex(d);
    if (bytes)
        *bytes = d.size();
    return true;
}

// -------------------------------------------------------------------- json

const Json *Json::get(const std::string &k) const
{
    if (kind != Object)
        return nullptr;
    for (const auto &kv : obj)
        if (kv.first == k)
            return &kv.second;
    return nullptr;
}

bool Json::is_int() const
{
    return kind == Number && std::floor(num) == num && std::fabs(num) < 9e15;
}

namespace
{

struct Parser
{
    const std::string &s;
    std::size_t i = 0;
    std::string err;
    explicit Parser(const std::string &t) : s(t)
    {
    }
    void ws()
    {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t'))
            ++i;
    }
    bool fail(const char *m)
    {
        if (err.empty())
            err = std::string(m) + " at offset " + std::to_string(i);
        return false;
    }
    bool lit(const char *w)
    {
        std::size_t n = std::strlen(w);
        if (s.compare(i, n, w) != 0)
            return fail("bad literal");
        i += n;
        return true;
    }
    bool str(std::string &out)
    {
        if (i >= s.size() || s[i] != '"')
            return fail("expected string");
        ++i;
        while (i < s.size() && s[i] != '"')
        {
            char c = s[i++];
            if (static_cast<unsigned char>(c) < 0x20)
                return fail("control character in string");
            if (c != '\\')
            {
                out += c;
                continue;
            }
            if (i >= s.size())
                return fail("bad escape");
            char e = s[i++];
            switch (e)
            {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u':
            {
                if (i + 4 > s.size())
                    return fail("bad \\u escape");
                unsigned cp = static_cast<unsigned>(std::stoul(s.substr(i, 4), nullptr, 16));
                i += 4;
                if (cp < 0x80)
                    out += static_cast<char>(cp);
                else if (cp < 0x800)
                {
                    out += static_cast<char>(0xC0 | (cp >> 6));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                else
                {
                    out += static_cast<char>(0xE0 | (cp >> 12));
                    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (cp & 0x3F));
                }
                break;
            }
            default:
                return fail("bad escape");
            }
        }
        if (i >= s.size())
            return fail("unterminated string");
        ++i;
        return true;
    }
    bool value(Json &v, int depth)
    {
        if (depth > 64)
            return fail("nesting too deep");
        ws();
        if (i >= s.size())
            return fail("unexpected end");
        char c = s[i];
        if (c == '{')
        {
            v.kind = Json::Object;
            ++i;
            ws();
            if (i < s.size() && s[i] == '}')
            {
                ++i;
                return true;
            }
            while (true)
            {
                ws();
                std::string k;
                if (!str(k))
                    return false;
                ws();
                if (i >= s.size() || s[i] != ':')
                    return fail("expected ':'");
                ++i;
                Json m;
                if (!value(m, depth + 1))
                    return false;
                v.obj.emplace_back(std::move(k), std::move(m));
                ws();
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    return true;
                }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[')
        {
            v.kind = Json::Array;
            ++i;
            ws();
            if (i < s.size() && s[i] == ']')
            {
                ++i;
                return true;
            }
            while (true)
            {
                Json m;
                if (!value(m, depth + 1))
                    return false;
                v.arr.push_back(std::move(m));
                ws();
                if (i < s.size() && s[i] == ',')
                {
                    ++i;
                    continue;
                }
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    return true;
                }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"')
        {
            v.kind = Json::String;
            return str(v.str);
        }
        if (c == 't')
        {
            v.kind = Json::Bool;
            v.b = true;
            return lit("true");
        }
        if (c == 'f')
        {
            v.kind = Json::Bool;
            v.b = false;
            return lit("false");
        }
        if (c == 'n')
        {
            v.kind = Json::Null;
            return lit("null");
        }
        std::size_t start = i;
        if (s[i] == '-')
            ++i;
        while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' ||
                                s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-'))
            ++i;
        if (start == i)
            return fail("unexpected character");
        try
        {
            v.num = std::stod(s.substr(start, i - start));
        }
        catch (...)
        {
            return fail("bad number");
        }
        v.kind = Json::Number;
        return true;
    }
};

const std::string *get_str(const Json &j, const char *k)
{
    const Json *v = j.get(k);
    return v && v->kind == Json::String ? &v->str : nullptr;
}
bool get_int(const Json &j, const char *k, long long &out)
{
    const Json *v = j.get(k);
    if (!v || !v->is_int())
        return false;
    out = v->as_int();
    return true;
}

std::vector<std::pair<std::string, std::string>> split_signature(const std::string &sig)
{
    std::vector<std::pair<std::string, std::string>> out;
    std::size_t p = 0;
    while (p < sig.size())
    {
        std::size_t q = sig.find(", ", p);
        std::string item = sig.substr(p, q == std::string::npos ? std::string::npos : q - p);
        std::size_t c = item.find(':');
        out.emplace_back(item.substr(0, c), c == std::string::npos ? "" : item.substr(c + 1));
        if (q == std::string::npos)
            break;
        p = q + 2;
    }
    return out;
}

} // namespace

bool json_parse(const std::string &text, Json &out, std::string &err)
{
    Parser p(text);
    out = Json();
    if (!p.value(out, 0))
    {
        err = p.err;
        return false;
    }
    p.ws();
    if (p.i != text.size())
    {
        err = "trailing characters at offset " + std::to_string(p.i);
        return false;
    }
    return true;
}

std::string json_quote(const std::string &s)
{
    std::string o = "\"";
    for (char c : s)
    {
        switch (c)
        {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                o += b;
            }
            else
                o += c;
        }
    }
    return o + "\"";
}

// ------------------------------------------------------------------ request

std::string key_digest(const KernelKey &key)
{
    return sha256_hex(key.text());
}

std::string make_request(const KernelKey &k, const std::string &source_path,
                         const std::string &cache_dir)
{
    std::ostringstream o;
    o << "{\"protocol\":" << json_quote(COMPILE_PROTOCOL)
      << ",\"key\":{\"kernel\":" << json_quote(k.kernel) << ",\"signature\":[";
    const auto sig = split_signature(k.signature);
    for (std::size_t i = 0; i < sig.size(); ++i)
        o << (i ? "," : "") << "[" << json_quote(sig[i].first) << "," << json_quote(sig[i].second)
          << "]";
    o << "],\"constexprs\":[";
    for (std::size_t i = 0; i < k.constexprs.size(); ++i)
        o << (i ? "," : "") << "[" << json_quote(k.constexprs[i].first) << ","
          << k.constexprs[i].second << "]";
    o << "],\"num_warps\":" << k.cfg.num_warps << ",\"num_stages\":" << k.cfg.num_stages
      << ",\"target\":{\"backend\":" << json_quote(k.target.backend)
      << ",\"arch\":" << k.target.arch << ",\"warp_size\":" << k.target.warp_size
      << "},\"compiler\":" << json_quote(k.toolchain.compiler)
      << ",\"source_sha256\":" << json_quote(k.toolchain.source_sha256)
      << "},\"key_text\":" << json_quote(k.text()) << ",\"source_path\":" << json_quote(source_path)
      << ",\"cache_dir\":" << json_quote(cache_dir) << "}";
    return o.str();
}

bool key_text_from_json(const Json &key, std::string &text, std::string &why)
{
    const std::string *kernel = get_str(key, "kernel"), *compiler = get_str(key, "compiler"),
                      *src = get_str(key, "source_sha256");
    const Json *sig = key.get("signature"), *cx = key.get("constexprs"), *tg = key.get("target");
    long long nw, ns, arch, ws;
    if (!kernel || !compiler || !src || !sig || sig->kind != Json::Array || !cx ||
        cx->kind != Json::Array || !tg || !get_int(key, "num_warps", nw) ||
        !get_int(key, "num_stages", ns) || !get_int(*tg, "arch", arch) ||
        !get_int(*tg, "warp_size", ws) || !get_str(*tg, "backend"))
    {
        why = "key: missing or mistyped field";
        return false;
    }
    std::ostringstream o;
    o << "kernel=" << *kernel << ";sig=";
    for (std::size_t i = 0; i < sig->arr.size(); ++i)
    {
        const Json &p = sig->arr[i];
        if (p.kind != Json::Array || p.arr.size() != 2 || p.arr[0].kind != Json::String ||
            p.arr[1].kind != Json::String)
        {
            why = "key.signature: expected [name, type] pairs";
            return false;
        }
        o << (i ? ", " : "") << p.arr[0].str << ":" << p.arr[1].str;
    }
    std::vector<std::pair<std::string, long long>> c;
    for (const Json &p : cx->arr)
    {
        if (p.kind != Json::Array || p.arr.size() != 2 || p.arr[0].kind != Json::String ||
            !p.arr[1].is_int())
        {
            why = "key.constexprs: expected [name, int] pairs";
            return false;
        }
        c.emplace_back(p.arr[0].str, p.arr[1].as_int());
    }
    std::sort(c.begin(), c.end());
    o << ";cx=";
    for (std::size_t i = 0; i < c.size(); ++i)
        o << (i ? "," : "") << c[i].first << "=" << c[i].second;
    o << ";nw=" << nw << ";ns=" << ns << ";target=" << *get_str(*tg, "backend") << ":" << arch
      << ":" << ws << ";compiler=" << *compiler << ";source=" << *src;
    text = o.str();
    return true;
}

// ------------------------------------------------------------ artifact check

namespace
{

bool ptx_compatible(const std::string &type, const std::string &ptx)
{
    if (!type.empty() && type[0] == '*')
        return ptx == "u64";
    if (type == "i32")
        return ptx == "u32" || ptx == "s32";
    if (type == "fp32")
        return ptx == "f32";
    return false;
}

} // namespace

bool verify_artifact(const std::string &dir, const std::string &expected_key_text, Manifest &m,
                     std::string &why)
{
    std::string text, err;
    if (!read_file(dir + "/manifest.json", text))
    {
        why = "no manifest.json";
        return false;
    }
    Json j;
    if (!json_parse(text, j, err))
    {
        why = "manifest.json does not parse: " + err;
        return false;
    }
    // Validate protocol and reconstruct the exact requested compile identity.
    const std::string *p = get_str(j, "protocol");
    if (!p || *p != COMPILE_PROTOCOL)
    {
        why = "manifest protocol is not " + std::string(COMPILE_PROTOCOL);
        return false;
    }
    m.protocol = *p;
    const std::string *kt = get_str(j, "key_text");
    const std::string *kd = get_str(j, "key_digest");
    const std::string *fn = get_str(j, "function");
    const Json *key = j.get("key"), *bin = j.get("binary"), *abi = j.get("abi"),
               *launch = j.get("launch");
    if (!kt || !kd || !fn || !key || !bin || !abi || abi->kind != Json::Array || !launch)
    {
        why = "manifest: missing field";
        return false;
    }
    if (*kt != expected_key_text)
    {
        why = "manifest names another key";
        return false;
    }
    std::string rebuilt;
    if (!key_text_from_json(*key, rebuilt, why))
    {
        why = "manifest " + why;
        return false;
    }
    if (rebuilt != *kt)
    {
        why = "manifest key fields do not serialise to its key_text";
        return false;
    }
    if (*kd != sha256_hex(*kt))
    {
        why = "key_digest is not sha256(key_text)";
        return false;
    }
    const std::size_t slash = dir.find_last_of('/');
    if (dir.substr(slash == std::string::npos ? 0 : slash + 1) != *kd)
    {
        why = "artifact directory is not named by the key digest";
        return false;
    }
    m.key_text = *kt;
    m.key_digest = *kd;
    m.function = *fn;

    // Bind the selected binary bytes and size to the manifest.
    const std::string *bf = get_str(*bin, "file"), *bh = get_str(*bin, "sha256"),
                      *fmt = get_str(*bin, "format");
    long long bytes;
    if (!bf || !bh || !fmt || !get_int(*bin, "bytes", bytes) || bf->find('/') != std::string::npos)
    {
        why = "manifest.binary: bad field";
        return false;
    }
    std::string got;
    std::size_t got_bytes = 0;
    if (!sha256_file(dir + "/" + *bf, got, &got_bytes))
    {
        why = "binary file missing";
        return false;
    }
    if (got != *bh || static_cast<long long>(got_bytes) != bytes)
    {
        why = "binary does not match the manifest's hash";
        return false;
    }
    m.binary_file = *bf;
    m.binary_sha256 = *bh;
    m.binary_bytes = got_bytes;
    m.format = *fmt;

    // Match ordered ABI arguments to the signature; extra arguments must be
    // explicitly recorded null pointers. launch_abi_ok checks launcher support.
    const Json *sig = key->get("signature");
    std::size_t i = 0;
    for (const Json &a : abi->arr)
    {
        const std::string *n = get_str(a, "name"), *t = get_str(a, "type"), *x = get_str(a, "ptx");
        const Json *np = a.get("null_pointer");
        if (!n || !t || !x || !np || np->kind != Json::Bool)
        {
            why = "manifest.abi: bad entry";
            return false;
        }
        ManifestParam mp{*n, *t, *x, np->b};
        if (!ptx_compatible(mp.type, mp.ptx))
        {
            why = "abi " + mp.name + ": " + mp.type + " is not passed as ." + mp.ptx;
            return false;
        }
        if (i < sig->arr.size())
        {
            if (mp.null_pointer || mp.name != sig->arr[i].arr[0].str ||
                mp.type != sig->arr[i].arr[1].str)
            {
                why = "abi parameter " + std::to_string(i) + " is not signature argument " +
                      sig->arr[i].arr[0].str;
                return false;
            }
        }
        else if (!mp.null_pointer || mp.type[0] != '*')
        {
            why = "abi parameter after the signature that is not a null pointer: " + mp.name;
            return false;
        }
        m.abi.push_back(mp);
        ++i;
    }
    if (i < sig->arr.size())
    {
        why = "abi shorter than the signature";
        return false;
    }

    // Launch geometry must agree with the key; retain scratch requirements for
    // the separate launcher-compatibility check.
    long long tpb, nw, ws, nc, sh, knw, kws;
    if (!get_int(*launch, "threads_per_block", tpb) || !get_int(*launch, "num_warps", nw) ||
        !get_int(*launch, "warp_size", ws) || !get_int(*launch, "num_ctas", nc) ||
        !get_int(*launch, "shared_bytes", sh) || !get_int(*key, "num_warps", knw) ||
        !get_int(*key->get("target"), "warp_size", kws))
    {
        why = "manifest.launch: bad field";
        return false;
    }
    if (nw != knw || ws != kws || tpb != nw * ws || nc != 1 || sh < 0)
    {
        why = "manifest.launch disagrees with the key";
        return false;
    }
    long long gs, ps;
    if (!get_int(*launch, "global_scratch_bytes", gs) ||
        !get_int(*launch, "profile_scratch_bytes", ps))
    {
        why = "manifest.launch: scratch sizes missing";
        return false;
    }
    m.global_scratch_bytes = gs;
    m.profile_scratch_bytes = ps;
    const std::string *comp = get_str(*key, "compiler");
    m.compiler = comp ? *comp : "";
    m.threads_per_block = static_cast<int>(tpb);
    m.num_warps = static_cast<int>(nw);
    m.warp_size = static_cast<int>(ws);
    m.num_ctas = static_cast<int>(nc);
    m.shared_bytes = static_cast<int>(sh);
    return true;
}

bool compiler_abi_validated(const std::string &compiler, const std::string &format)
{
    // Admit a compiler only after validating its entry parameters and scratch requirements.
    // CPU protocol controls are in test/cpu/test_protocol.cpp.
    //   cubin: the PTX .entry parameters of upstream triton 3.6.0.
    //   hsaco: AMDHSA kernel-argument metadata from the HIP compiler, read by
    //          compile_helper.py's hip_abi(). flagtree reports the same version string "3.6.0",
    //          so the compiler string cannot tell the two distributions apart: the binary
    //          format keys the entry, and a hip artifact is not validated by the cuda check.
    if (format == "cubin")
        return compiler == "triton-3.6.0";
    if (format == "hsaco")
        return compiler == "triton-3.6.0";
    return false;
}

bool launch_abi_ok(const Manifest &m, std::string &why)
{
    if (!compiler_abi_validated(m.compiler, m.format))
    {
        why = "compiler " + m.compiler + " (" + m.format + "): entry ABI not validated";
        return false;
    }
    if (m.global_scratch_bytes != 0 || m.profile_scratch_bytes != 0)
    {
        why = "kernel needs scratch memory; null scratch is not valid";
        return false;
    }
    std::size_t nulls = 0;
    for (const ManifestParam &p : m.abi)
        nulls += p.null_pointer;
    if (nulls != 2 || m.abi.size() < 2 || m.abi[m.abi.size() - 2].name != "global_scratch" ||
        m.abi.back().name != "profile_scratch" || !m.abi[m.abi.size() - 2].null_pointer ||
        !m.abi.back().null_pointer)
    {
        why = "trailing parameters are not exactly (global_scratch, profile_scratch)";
        return false;
    }
    return true;
}

bool verify_response(const std::string &request, const std::string &response,
                     std::string &artifact_dir, std::string &status, std::string &why)
{
    Json q, r;
    std::string err;
    if (!json_parse(request, q, err))
    {
        why = "request does not parse: " + err;
        return false;
    }
    if (!json_parse(response, r, err))
    {
        why = "response does not parse: " + err;
        return false;
    }
    const std::string *p = get_str(r, "protocol");
    if (!p || *p != COMPILE_PROTOCOL)
    {
        why = "response protocol is not " + std::string(COMPILE_PROTOCOL);
        return false;
    }
    const std::string *rs = get_str(r, "request_sha256");
    if (!rs || *rs != sha256_hex(request))
    {
        why = "response answers another request";
        return false;
    }
    // An error response only has to answer this request: it may be about a request whose
    // key is itself malformed. Everything below applies to a success.
    const std::string *st = get_str(r, "status");
    if (!st)
    {
        why = "response has no status";
        return false;
    }
    if (*st == "error")
    {
        status = *st;
        const Json *e = r.get("error");
        const std::string *kind = e ? get_str(*e, "kind") : nullptr;
        why = kind ? *kind : "error without a kind";
        return false;
    }
    if (*st != "compiled" && *st != "cached")
    {
        why = "unknown status " + *st;
        return false;
    }
    const std::string *kt = get_str(q, "key_text");
    const Json *key = q.get("key");
    std::string rebuilt;
    if (!kt || !key || !key_text_from_json(*key, rebuilt, why) || rebuilt != *kt)
    {
        if (why.empty())
            why = "request key fields do not serialise to its key_text";
        return false;
    }
    const std::string *kd = get_str(r, "key_digest");
    if (!kd || *kd != sha256_hex(*kt))
    {
        why = "response key_digest is not the request's key";
        return false;
    }
    status = *st;
    const std::string *dir = get_str(r, "artifact_dir"), *ms = get_str(r, "manifest_sha256"),
                      *cd = get_str(q, "cache_dir");
    if (!dir || !ms || !cd || *dir != *cd + "/" + *kd)
    {
        why = "artifact_dir is not <cache_dir>/<key_digest>";
        return false;
    }
    std::string mh;
    if (!sha256_file(*dir + "/manifest.json", mh) || mh != *ms)
    {
        why = "manifest is not the one the response names";
        return false;
    }
    Manifest m;
    if (!verify_artifact(*dir, *kt, m, why))
        return false;
    artifact_dir = *dir;
    return true;
}

// ------------------------------------------------------------------- runner

namespace
{

CompileOutcome failed(Reason r, const std::string &kind, const std::string &msg)
{
    CompileOutcome o;
    o.status.code = Code::Error;
    o.status.reason = r;
    o.status.message = kind + ": " + msg;
    o.kind = kind;
    return o;
}

Reason reason_for(const std::string &kind)
{
    if (kind == "timeout")
        return Reason::CompileTimeout;
    if (kind == "toolchain_unavailable" || kind == "compiler_mismatch")
        return Reason::ToolchainUnavailable;
    if (kind == "bad_response" || kind == "abi_mismatch" || kind == "manifest_invalid")
        return Reason::ManifestInvalid;
    return Reason::CompileFailed;
}

} // namespace

CompileOutcome compile_key(const HelperCommand &cmd, const KernelKey &key,
                           const std::string &source_path, const std::string &cache_dir)
{
    return run_request(cmd, make_request(key, source_path, cache_dir));
}

CompileOutcome run_request(const HelperCommand &cmd, const std::string &request)
{
    Json q;
    std::string err;
    if (!json_parse(request, q, err) || !get_str(q, "cache_dir"))
        return failed(Reason::CompileFailed, "invalid_request", err);
    const std::string cache = *get_str(q, "cache_dir");
    ::mkdir(cache.c_str(), 0755);
    std::random_device rd;
    const std::string req_path =
        cache + "/.req-" + std::to_string(::getpid()) + "-" + std::to_string(rd()) + ".json";
    {
        std::ofstream f(req_path, std::ios::binary);
        if (!(f << request))
            return failed(Reason::CompileFailed, "invalid_request", "cannot write " + req_path);
    }
    int out[2];
    if (::pipe(out) != 0)
    {
        ::unlink(req_path.c_str());
        return failed(Reason::CompileFailed, "internal", "pipe");
    }
    std::vector<std::string> args = {cmd.python, cmd.helper, "--request", req_path};
    std::vector<char *> argv, envp;
    for (auto &a : args)
        argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    for (auto &e : cmd.env)
        envp.push_back(const_cast<char *>(e.c_str()));
    envp.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0)
    {
        ::close(out[0]);
        ::close(out[1]);
        ::unlink(req_path.c_str());
        return failed(Reason::CompileFailed, "internal", "fork");
    }
    if (pid == 0)
    {
        ::setpgid(0, 0);
        ::dup2(out[1], 1);
        const int devnull = ::open("/dev/null", O_RDONLY);
        if (devnull >= 0)
            ::dup2(devnull, 0);
        ::close(out[0]);
        ::close(out[1]);
        ::execve(cmd.python.c_str(), argv.data(), envp.data());
        ::_exit(127);
    }
    ::setpgid(pid, pid);
    ::close(out[1]);
    std::string text;
    bool timed_out = false, stdout_closed = false, reaped = false;
    int st = 0, wait_error = 0, io_error = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(cmd.timeout_s);
    char buf[4096];
    // Pipe EOF and child exit are independent. One monotonic deadline covers both,
    // including descendants that inherit stdout after the direct helper exits.
    while (!stdout_closed || !reaped)
    {
        if (!reaped)
        {
            const pid_t done = ::waitpid(pid, &st, WNOHANG);
            if (done == pid)
                reaped = true;
            else if (done < 0 && errno != EINTR)
            {
                wait_error = errno;
                break;
            }
        }
        if (stdout_closed && reaped)
            break;
        const auto left =
            std::chrono::duration<double, std::milli>(deadline - std::chrono::steady_clock::now())
                .count();
        if (left <= 0)
        {
            timed_out = true;
            break;
        }
        // After EOF there is no descriptor to wait on: sleep briefly between
        // nonblocking exit checks rather than spinning or doing a blocking waitpid.
        const int pause = static_cast<int>(std::min(left, stdout_closed ? 10.0 : 1000.0)) + 1;
        pollfd pf{out[0], POLLIN, 0};
        const int result = ::poll(stdout_closed ? nullptr : &pf, stdout_closed ? 0 : 1, pause);
        if (result < 0)
        {
            if (errno == EINTR)
                continue;
            io_error = errno;
            break;
        }
        if (result == 0 || stdout_closed)
            continue;
        if (pf.revents & POLLNVAL)
        {
            io_error = EBADF;
            break;
        }
        const ssize_t n = ::read(out[0], buf, sizeof buf);
        if (n == 0)
            stdout_closed = true; // writer closed the pipe; it may still be running
        else if (n > 0)
            text.append(buf, static_cast<std::size_t>(n));
        else if (errno != EINTR)
        {
            io_error = errno;
            break;
        }
    }
    ::close(out[0]);
    if (timed_out || wait_error || io_error)
    {
        // The group includes compiler descendants. Also kill the direct child if
        // process-group creation failed. Do not signal a pid already reaped/reused.
        if (wait_error != ECHILD)
        {
            const int group_result = ::kill(-pid, SIGKILL);
            if (group_result < 0 && !reaped)
                (void)::kill(pid, SIGKILL);
        }
        // SIGKILL normally permits immediate reap. Bound cleanup separately so an
        // abnormal kernel/child-wait condition cannot turn a timeout into a hang.
        const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!reaped && wait_error != ECHILD &&
               std::chrono::steady_clock::now() < cleanup_deadline)
        {
            const pid_t done = ::waitpid(pid, &st, WNOHANG);
            if (done == pid)
            {
                reaped = true;
                break;
            }
            if (done < 0 && errno != EINTR)
            {
                wait_error = errno;
                break;
            }
            (void)::poll(nullptr, 0, 10);
        }
    }
    ::unlink(req_path.c_str());

    if (!reaped)
    {
        CompileOutcome f =
            failed(Reason::CompileFailed, "helper_cleanup_failed",
                   "cannot confirm helper reap for pid " + std::to_string(pid) +
                       " within bounded cleanup (wait errno " + std::to_string(wait_error) + ")");
        f.exit_code = INT_MIN; // no exit status was collected; do not report a successful exit
        f.response = text;
        return f;
    }
    CompileOutcome o;
    o.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : -(WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    if (timed_out || wait_error || io_error)
    {
        CompileOutcome f =
            timed_out ? failed(Reason::CompileTimeout, "timeout",
                               "helper output/exit deadline exceeded after " +
                                   std::to_string(cmd.timeout_s) + " s")
                      : failed(Reason::CompileFailed, "helper_wait_failed",
                               "helper pipe/wait failed (errno " +
                                   std::to_string(wait_error ? wait_error : io_error) + ")");
        f.exit_code = o.exit_code;
        f.response = text;
        return f;
    }
    if (o.exit_code == 127)
        return failed(Reason::ToolchainUnavailable, "toolchain_unavailable",
                      "cannot execute " + cmd.python);
    // Compiler diagnostics may precede the response. The helper writes its JSON
    // response last; preceding stdout lines are not parsed as protocol messages.
    std::string last;
    std::istringstream lines(text);
    for (std::string l; std::getline(lines, l);)
        if (!l.empty())
            last = l;
    o.response = last;
    std::string dir, status, why;
    if (last.empty())
    {
        CompileOutcome f = failed(Reason::CompileFailed, "helper_crashed",
                                  "exit " + std::to_string(o.exit_code) + ", no response");
        f.exit_code = o.exit_code;
        return f;
    }
    if (!verify_response(request, last, dir, status, why))
    {
        const std::string kind =
            status == "error"
                ? why
                : (o.exit_code != 0 && status.empty() ? "helper_crashed" : "bad_response");
        CompileOutcome f = failed(reason_for(kind), kind, status == "error" ? last : why);
        f.response = last;
        f.exit_code = o.exit_code;
        return f;
    }
    o.status.code = Code::Prepared;
    o.status.reason = Reason::None;
    o.kind = status;
    o.artifact_dir = dir;
    return o;
}

} // namespace cv_triton_runtime

// ---------------------------------------------------------------- C ABI (tests)
