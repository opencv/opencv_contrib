#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.

"""Compile helper for the Triton module runtime JIT (protocol opencv-triton-compile/1).

The module never embeds Python. On a cache miss its C++ runtime (compile_protocol.cpp) runs
this file as a separate process:

    python compile_helper.py --request <request.json>

It prints exactly one JSON response line on stdout. Exit 0 = compiled or cached, 2 = error
response; any other exit, or no response line, is a crash, which the caller handles.

Torch-free and device-free by construction. `torch` is blocked before anything is imported;
the target comes from the request, never from a device query. Line info is disabled so the
binary does not depend on where the kernels source is installed, and no bytecode is written
next to it: the install tree is read-only to this process.

Publication is crash-safe:
  * one flock per key (<cache>/<digest>.lock) serialises concurrent requests for a key,
    across processes;
  * the artifact is built in <cache>/.tmp-<digest>-<pid>-*, manifest last, every file and
    the directory fsynced, then renamed to <cache>/<digest> in one step;
  * an existing entry is reused only if it verifies (manifest, key, binary hash); otherwise
    it is moved to .bad-<digest>-<ms> and rebuilt. Under the lock, any .tmp-<digest>-* left
    by a killed helper is removed.
So a reader that only trusts <cache>/<digest> with a verifying manifest never sees a half
artifact.

OPENCV_TRITON_HELPER_TEST_FAULT is a test-only fault-injection hook:
  hang, slow (sleep after taking the lock), crash_mid_publish (exit 3 after the binary,
  before the manifest), kill_before_rename (SIGKILL after the manifest, before the rename).
"""
import os
import sys

sys.modules["torch"] = None            # any `import torch` now raises: Torch-free by construction
sys.dont_write_bytecode = True         # never write __pycache__ next to the (installed) kernels source
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"

import fcntl          # noqa: E402
import glob           # noqa: E402
import hashlib        # noqa: E402
import importlib.util  # noqa: E402
import json           # noqa: E402
import re             # noqa: E402
import shutil         # noqa: E402
import signal         # noqa: E402
import tempfile       # noqa: E402
import time           # noqa: E402

PROTOCOL = "opencv-triton-compile/1"
BINARY = "kernel.cubin"
HIP_BINARY = "kernel.hsaco"
T0 = time.time()


class HelperError(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fault(point: str) -> None:
    f = os.environ.get("OPENCV_TRITON_HELPER_TEST_FAULT", "")
    if not f:
        return
    if point == "after_lock" and f == "hang":
        while True:
            time.sleep(60)
    if point == "after_lock" and f == "slow":
        time.sleep(2.0)
    if point == "mid_publish" and f == "crash_mid_publish":
        os._exit(3)
    if point == "before_rename" and f == "kill_before_rename":
        os.kill(os.getpid(), signal.SIGKILL)


def key_text(key: dict) -> str:
    """KernelKey::text() (provider.cpp), from the protocol's structured key."""
    sig = ", ".join(f"{n}:{t}" for n, t in key["signature"])
    cx = ",".join(f"{n}={int(v)}" for n, v in sorted((n, v) for n, v in key["constexprs"]))
    t = key["target"]
    return (f"kernel={key['kernel']};sig={sig};cx={cx};nw={key['num_warps']};ns={key['num_stages']};"
            f"target={t['backend']}:{t['arch']}:{t['warp_size']};compiler={key['compiler']};source={key['source_sha256']}")


def check_request(req: dict) -> tuple[dict, str, str]:
    if not isinstance(req, dict) or req.get("protocol") != PROTOCOL:
        raise HelperError("protocol_mismatch", f"request protocol is not {PROTOCOL}")
    try:
        key, text = req["key"], req["key_text"]
        src_path, cache = req["source_path"], req["cache_dir"]
        # Check signature/constexpr element types, launch/target integer fields
        # and compiler/source/path strings before serialization or compilation.
        # Canonical key equality and absolute paths are checked separately below.
        ok = (isinstance(key["kernel"], str) and all(isinstance(n, str) and isinstance(t, str) for n, t in key["signature"])
              and all(isinstance(n, str) and isinstance(v, int) and not isinstance(v, bool) for n, v in key["constexprs"])
              and all(isinstance(key[f], int) for f in ("num_warps", "num_stages"))
              and isinstance(key["target"]["backend"], str) and isinstance(key["target"]["arch"], int)
              and isinstance(key["target"]["warp_size"], int) and isinstance(key["compiler"], str)
              and isinstance(key["source_sha256"], str) and isinstance(src_path, str) and isinstance(cache, str))
    except (KeyError, TypeError, ValueError) as exc:
        raise HelperError("invalid_request", f"malformed request: {exc!r}")
    if not ok:
        raise HelperError("invalid_request", "malformed request: a field has the wrong type")
    if key_text(key) != text:
        raise HelperError("key_mismatch", "key_text is not the serialisation of key")
    if not os.path.isabs(cache) or not os.path.isabs(src_path):
        raise HelperError("invalid_request", "source_path and cache_dir must be absolute")
    return key, text, sha(text.encode())


def entry_valid(d: str, text: str, digest: str) -> tuple[bool, str]:
    """The helper's own check before it hands out an existing entry."""
    try:
        man_bytes = open(os.path.join(d, "manifest.json"), "rb").read()
        m = json.loads(man_bytes)
        b = open(os.path.join(d, m["binary"]["file"]), "rb").read()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"unreadable: {exc!r}"
    if m.get("protocol") != PROTOCOL or m.get("key_text") != text or m.get("key_digest") != digest:
        return False, "manifest names another protocol or key"
    if sha(b) != m["binary"].get("sha256") or len(b) != m["binary"].get("bytes"):
        return False, "binary does not match its manifest"
    return True, sha(man_bytes)


def ptx_params(ptx: str) -> tuple[str, list[str]]:
    # Inspect compiled ABI order, including compiler-added parameters; do not infer it
    # from the source signature. The C++ loader independently checks this manifest.
    m = re.search(r"\.entry\s+(\w+)\s*\((.*?)\)", ptx, re.S)
    if not m:
        raise HelperError("abi_mismatch", "no .entry in the PTX")
    types = []
    for p in m.group(2).split(","):
        t = re.match(r"\s*\.param\s+\.(\w+)", p)
        if not t:
            raise HelperError("abi_mismatch", f"unparsed PTX parameter {p.strip()!r}")
        types.append(t.group(1))
    return m.group(1), types


def compatible(ty: str, ptx: str) -> bool:
    if ty.startswith("*"):
        return ptx == "u64"
    return {"i32": ptx in ("u32", "s32"), "fp32": ptx == "f32"}.get(ty, False)


def amdhsa_params(amdgcn: str) -> tuple[str, list[dict]]:
    """The hip counterpart of ptx_params(): (kernel name, explicit kernel arguments) read from
    the AMDHSA metadata the compiler embeds in the code object. Only `.args` is read:
    `.unfolded_args` repeats it further down. Exactly one kernel entry is expected."""
    i = amdgcn.find("amdhsa.kernels:")
    if i < 0:
        raise HelperError("abi_mismatch", "no amdhsa.kernels metadata in the AMDGCN")
    name, args, kernels, section = None, [], 0, None
    for line in amdgcn[i:].splitlines()[1:]:
        if line and not line.startswith(" "):
            break                                        # the next top-level key (amdhsa.target, ...)
        if not line.strip():
            continue
        m = re.match(r"^  - (\.\w+):\s*(.*)$", line)     # a kernel entry opens with its first key
        if m:
            kernels += 1
        else:
            m = re.match(r"^    (\.\w+):\s*(.*)$", line)   # a further key of the same kernel
        if m:
            section = m.group(1)
            if section == ".name":
                name = m.group(2).strip()
        elif kernels == 1 and section == ".args":         # a line inside the .args list
            body = line.strip()
            if body.startswith("- "):
                args.append({})
                body = body[2:]
            kv = re.match(r"^(\.\w+):\s*(.*)$", body)
            if kv and args:
                args[-1][kv.group(1)] = kv.group(2).strip()
    if kernels != 1 or not name:
        raise HelperError("abi_mismatch", f"expected one amdhsa kernel entry with a name, found {kernels}")
    return name, args


def hip_scalar_class(ty: str, kind: str, size: int):
    """The scalar class stored in the manifest's `ptx` field for a HIP argument,
    or None if the compiler passes it some other way.
    The AMDHSA metadata separates pointers from scalars and gives the size, but not
    signed-vs-float, so i32 and fp32 are told apart by the key's own type string."""
    if ty.startswith("*"):
        return "u64" if kind == "global_buffer" and size == 8 else None
    if kind == "by_value" and size == 4:
        return {"i32": "s32", "fp32": "f32"}.get(ty)
    return None


def hip_abi(key: dict, sig_names: list[str], ck):
    """(entry, abi) for a hip kernel. Every kernel argument must be an explicit one
    (global_buffer / by_value): a hidden_* argument would need the launcher to supply it, and
    launching without it would be silent corruption, so it is refused here instead."""
    entry, args = amdhsa_params(ck.asm["amdgcn"])
    if entry != ck.name:
        raise HelperError("abi_mismatch", f"metadata names kernel {entry!r}, the compiler produced {ck.name!r}")
    parsed = []
    for i, a in enumerate(args):
        try:
            kind, size = a[".value_kind"], int(a[".size"])
        except (KeyError, ValueError):
            raise HelperError("abi_mismatch", f"kernel argument {i} has no .value_kind/.size: {a}")
        if kind not in ("global_buffer", "by_value"):
            raise HelperError("abi_mismatch", f"kernel argument {i} is {kind!r}; the launcher passes explicit arguments only")
        parsed.append((kind, size))
    if len(parsed) < len(sig_names):
        raise HelperError("abi_mismatch", f"AMDHSA entry has {len(parsed)} arguments for {len(sig_names)} runtime arguments")
    extra_names = ["global_scratch", "profile_scratch"]
    abi = []
    for i, (kind, size) in enumerate(parsed):
        if i < len(sig_names):
            n, t = key["signature"][i]
            cls = hip_scalar_class(t, kind, size)
            if cls is None:
                raise HelperError("abi_mismatch", f"argument {n}:{t} is passed as {kind}/{size}")
            abi.append({"name": n, "type": t, "ptx": cls, "null_pointer": False, "abi_source": "amdhsa"})
        else:
            if kind != "global_buffer" or size != 8:
                raise HelperError("abi_mismatch", f"trailing argument {i} is {kind}/{size}, not a pointer")
            j = i - len(sig_names)
            abi.append({"name": extra_names[j] if j < len(extra_names) else f"extra_{j}", "type": "*i8", "ptx": "u64",
                        "null_pointer": True, "abi_source": "amdhsa"})
    return entry, abi


def build(key: dict, src_path: str):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    if f"triton-{triton.__version__}" != key["compiler"]:
        raise HelperError("compiler_mismatch", f"this helper runs triton-{triton.__version__}, the key names {key['compiler']}")
    backend_name = key["target"]["backend"]
    if backend_name not in ("cuda", "hip"):
        raise HelperError("invalid_request", f"backend {backend_name!r}: this helper builds cuda and hip only")
    spec = importlib.util.spec_from_file_location(f"opencv_triton_kernels_{key['source_sha256'][:12]}", src_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, key["kernel"], None)
    if fn is None or not hasattr(fn, "arg_names"):
        raise HelperError("invalid_request", f"{key['kernel']} is not a jit function of the source")
    cx_names = {p.name for p in fn.params if p.is_constexpr}
    runtime = [a for a in fn.arg_names if a not in cx_names]
    sig_names = [n for n, _ in key["signature"]]
    if runtime != sig_names:
        raise HelperError("invalid_request", f"signature {sig_names} is not the kernel's runtime arguments {runtime}")
    if {n for n, _ in key["constexprs"]} != cx_names:
        raise HelperError("invalid_request", f"constexprs {sorted(n for n, _ in key['constexprs'])} != the kernel's {sorted(cx_names)}")
    full = {a: "constexpr" for a in fn.arg_names}
    full.update({n: t for n, t in key["signature"]})
    src = ASTSource(fn, full, {n: v for n, v in key["constexprs"]}, {})   # no implicit specialisation
    # cuda: arch is the integer compute capability (90 = sm_90). hip: it is the gfx number
    # (936 = gfx936); an arch with a letter suffix (gfx90a) has no integer form, so the key
    # cannot name it and the runtime reports it unsupported rather than guessing.
    arch = key["target"]["arch"] if backend_name == "cuda" else f"gfx{key['target']['arch']}"
    target = GPUTarget(backend_name, arch, key["target"]["warp_size"])
    backend = triton.compiler.make_backend(target)
    opts = backend.parse_options({"num_warps": key["num_warps"], "num_stages": key["num_stages"]})
    try:
        ck = triton.compile(src, target=target, options=opts.__dict__)
    except Exception as exc:  # noqa: BLE001 -- any compiler failure is a structured error
        raise HelperError("compile_failed", compile_error_text(exc))
    md = ck.metadata
    if getattr(md, "global_scratch_size", 0) or getattr(md, "profile_scratch_size", 0):
        raise HelperError("unsupported_scratch", "the kernel needs global/profile scratch memory; the launcher passes null")
    launch = {"threads_per_block": md.num_warps * md.warp_size, "num_warps": md.num_warps, "warp_size": md.warp_size,
              "num_ctas": getattr(md, "num_ctas", 1), "shared_bytes": md.shared,
              "global_scratch_bytes": int(getattr(md, "global_scratch_size", 0) or 0),
              "profile_scratch_bytes": int(getattr(md, "profile_scratch_size", 0) or 0)}
    if backend_name == "hip":
        entry, abi = hip_abi(key, sig_names, ck)
        return entry, HIP_BINARY, "hsaco", ck.asm["hsaco"], abi, launch, triton.__version__
    entry, types = ptx_params(ck.asm["ptx"])
    if len(types) < len(sig_names):
        raise HelperError("abi_mismatch", f"PTX entry has {len(types)} parameters for {len(sig_names)} arguments")
    extra_names = ["global_scratch", "profile_scratch"]
    abi = []
    for i, ptx in enumerate(types):
        if i < len(sig_names):
            n, t = key["signature"][i]
            if not compatible(t, ptx):
                raise HelperError("abi_mismatch", f"argument {n}:{t} is passed as .{ptx}")
            abi.append({"name": n, "type": t, "ptx": ptx, "null_pointer": False})
        else:
            if ptx != "u64":
                raise HelperError("abi_mismatch", f"trailing PTX parameter {i} is .{ptx}, not a pointer")
            j = i - len(sig_names)
            abi.append({"name": extra_names[j] if j < len(extra_names) else f"extra_{j}", "type": "*i8", "ptx": ptx,
                        "null_pointer": True})
    return entry, BINARY, "cubin", ck.asm["cubin"], abi, launch, triton.__version__


def compile_error_text(exc: BaseException) -> str:
    """The cause, not the call site. Triton wraps a failure once per jit-function frame;
    the deepest CompilationError.error_message is what went wrong, the outermost frame is
    where in the kernel. Both are reported, capped."""
    chain, e = [], exc
    while e is not None and len(chain) < 32:
        chain.append(e)
        e = e.__cause__ or e.__context__
    cause = next((str(getattr(x, "error_message", "") or "").strip() for x in reversed(chain)
                  if str(getattr(x, "error_message", "") or "").strip()), "")
    if not cause:
        lines = [l for l in str(chain[-1]).splitlines() if l.strip() and l.strip() != "^"]
        cause = lines[-1].strip() if lines else ""
    where = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(chain[-1]).__name__}: {cause} ({where})"[:2000]


def write_synced(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def serve(raw: bytes) -> dict:
    try:
        req = json.loads(raw)
    except ValueError as exc:
        raise HelperError("invalid_request", f"request is not JSON: {exc}")
    key, text, digest = check_request(req)
    out = {"key_digest": digest}
    src_path, cache = req["source_path"], req["cache_dir"]
    try:
        src_bytes = open(src_path, "rb").read()
    except OSError as exc:
        raise HelperError("source_mismatch", f"cannot read the kernels source: {exc}")
    if sha(src_bytes) != key["source_sha256"]:
        raise HelperError("source_mismatch", "the kernels source does not have the key's sha256")
    os.makedirs(cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = os.path.join(cache, ".triton")   # private, never the user's
    final = os.path.join(cache, digest)
    lock = os.open(os.path.join(cache, f"{digest}.lock"), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        # One publisher per key. Readers accept only a complete manifest/binary pair;
        # syncing the temporary directory before rename avoids publishing partial files.
        fcntl.flock(lock, fcntl.LOCK_EX)
        fault("after_lock")
        for stale in glob.glob(os.path.join(cache, f".tmp-{digest}-*")):   # left by a killed helper
            shutil.rmtree(stale, ignore_errors=True)
        if os.path.isdir(final):
            ok, info = entry_valid(final, text, digest)
            if ok:
                return dict(out, status="cached", artifact_dir=final, manifest_sha256=info)
            os.rename(final, os.path.join(cache, f".bad-{digest}-{int(time.time() * 1000)}"))
            out["replaced_invalid_entry"] = info
        entry, binfile, binfmt, binary, abi, launch, tver = build(key, src_path)
        tmp = tempfile.mkdtemp(prefix=f".tmp-{digest}-{os.getpid()}-", dir=cache)
        write_synced(os.path.join(tmp, binfile), binary)
        fault("mid_publish")
        manifest = {"protocol": PROTOCOL, "key": key, "key_text": text, "key_digest": digest, "function": entry,
                    "binary": {"file": binfile, "format": binfmt, "sha256": sha(binary), "bytes": len(binary)},
                    "abi": abi, "launch": launch,
                    "build": {"triton_version": tver, "line_info": False, "python": sys.version.split()[0],
                              "torch_loaded": False, "created_unix": round(time.time(), 3)}}
        man_bytes = (json.dumps(manifest, indent=1, sort_keys=False) + "\n").encode()
        write_synced(os.path.join(tmp, "manifest.json"), man_bytes)
        fsync_dir(tmp)
        fault("before_rename")
        os.rename(tmp, final)                       # the one step that publishes
        fsync_dir(cache)
        return dict(out, status="compiled", artifact_dir=final, manifest_sha256=sha(man_bytes))
    finally:
        os.close(lock)                              # releases the flock; a killed helper releases it too


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--request":
        print("usage: compile_helper.py --request <request.json>", file=sys.stderr)
        return 64
    raw = open(sys.argv[2], "rb").read()
    resp = {"protocol": PROTOCOL, "request_sha256": sha(raw), "key_digest": None, "status": "error", "error": None,
            "artifact_dir": None, "manifest_sha256": None}
    code = 0
    try:
        resp.update(serve(raw))
    except HelperError as exc:
        resp.update(status="error", error={"kind": exc.kind, "message": str(exc)})
        try:
            resp["key_digest"] = sha(json.loads(raw)["key_text"].encode())
        except Exception:  # noqa: BLE001
            pass
        code = 2
    resp["elapsed_s"] = round(time.time() - T0, 3)
    resp["helper"] = {"torch_blocked": sys.modules.get("torch", 0) is None,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    sys.stdout.write(json.dumps(resp, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return code


if __name__ == "__main__":
    sys.exit(main())
