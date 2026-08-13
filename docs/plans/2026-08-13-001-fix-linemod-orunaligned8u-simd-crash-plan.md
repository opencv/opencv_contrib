---
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
execution: code
product_contract_source: ce-plan-bootstrap
---

# fix: Prevent SIGSEGV in linemod::orUnaligned8u from unconditionally-aligned dst access

**Target repo:** opencv_contrib (fork clone at `SomSamantray/opencv_contrib`, upstream `opencv/opencv_contrib`)
**Origin issue:** [opencv/opencv#29559](https://github.com/opencv/opencv/issues/29559) — "linemod::match crashes with SIGSEGV due to misaligned store in orUnaligned8u"

---

## Summary

`cv::linemod::Detector::match()` can crash with SIGSEGV inside the SSE-optimized helper `orUnaligned8u()` in `modules/rgbd/src/linemod.cpp`. The function checks whether its `src` pointer is 16-byte aligned to pick a load strategy, but never checks or guarantees alignment of its `dst` pointer — every branch still performs a plain `__m128i*` dereference on `dst`, which the compiler lowers to an aligned `movdqa` load/store. On misaligned `dst`, that instruction faults. The fix replaces the unconditional-aligned `dst` access with unaligned load/store intrinsics in all three SIMD branches, leaving the already-correct `src`-side branching untouched.

## Problem Frame

`orUnaligned8u(src, src_stride, dst, dst_stride, width, height)` computes `dst[i] |= src[i]` over a 2D region, with an SSE2/SSE3 fast path for 16-byte chunks. It is called from `spread()` (same file) in a nested loop over sub-pixel offsets `r, c` in `[0, T)`, passing `&src.at<uchar>(r, c)` as `src` — a pointer that is frequently *not* 16-byte aligned by construction, which is exactly why the function branches on `src_aligned` at all.

The bug: all three SIMD branches — the aligned-load path, the SSE3 `_mm_lddqu_si128` path, and the SSE2 `_mm_loadu_si128` fallback — write the result back via:

```cpp
__m128i* dst_ptr = reinterpret_cast<__m128i*>(dst + c);
*dst_ptr = _mm_or_si128(*dst_ptr, val);
```

`__m128i` carries a 16-byte alignment attribute, so both the read (`*dst_ptr` on the right-hand side) and the write (`*dst_ptr =` on the left-hand side) compile to aligned SSE instructions (`movdqa`), regardless of which branch executed or what `src_aligned` was. `dst`'s actual alignment is never checked. On inputs/platforms where the destination buffer or its row/column offset isn't 16-byte aligned, the aligned instruction faults with SIGSEGV — reported via `linemod::match()`, which calls `spread()` internally.

One related open issue describes the same crash from a different call site and is almost certainly fixed by the same change (out of scope to verify here, but worth noting for the PR description):
- opencv/opencv#29576 — `cv2.linemod.match()` segfault via Python bindings on Python 3.13

**Actual misalignment mechanism (confirmed by re-reading the call site, and by building and running a standalone reproduction harness against both the buggy and fixed function bodies — see U1 Verification):** `spread()` always passes `dst.ptr()` — the Mat's unoffset, allocator-aligned base pointer — as `dst` (only `src` is offset by `(r, c)`). `orUnaligned8u` then advances `src += src_stride` and `dst += dst_stride` per row. If either stride is not itself a multiple of 16 (which happens whenever the quantized image's row width isn't 16-byte-aligned), later rows' pointer drifts out of 16-byte alignment even though row 0 was aligned.

**This drift affects `src`, not only `dst`, which is a deeper defect than the "dst is never checked" framing above suggests.** `orUnaligned8u` computes `src_aligned` **once, before the row loop**, from the function's initial `src` pointer, and reuses that single boolean to pick a branch for *every* row. Once `src` (or `dst`) drifts out of alignment on a later row via non-16-aligned stride, the branch decision made at row 0 is stale — if row 0 was aligned, the loop keeps taking the "aligned src" branch (`*src_ptr` plain dereference) for all subsequent rows regardless of their actual drifted alignment, faulting on `src` reads exactly as it does on `dst` writes. A standalone reproduction (see U1 Verification) confirmed that patching only the `dst` access, per the original diagnosis above, still crashes — the `src`-side aligned dereference in the same branch faults independently on drifted rows. The correct fix removes the one-time `src_aligned` fast path entirely rather than trying to fix it: since it cannot be evaluated per-row without recomputing it every iteration (defeating its own purpose), and unaligned SSE2/SSE3 loads/stores have negligible cost on the hardware OpenCV 5.x targets (see KTD1), both `src` and `dst` are always accessed with unaligned intrinsics, unconditionally, in every row.

## Requirements

- **R1**: `orUnaligned8u()` must not assume `dst` is 16-byte aligned in any SIMD branch. All `dst` reads and writes in the SIMD paths must use unaligned load/store intrinsics.
- **R2**: `orUnaligned8u()` must not assume `src` is 16-byte aligned either, for any row — including the once-per-call `src_aligned` fast-path branch, which is unsound because it is computed from the initial pointer only and does not account for per-row drift when `src_stride` isn't a multiple of 16. The `src_aligned` fast path must be removed; `src` reads in the SIMD paths must use unaligned load intrinsics (`_mm_lddqu_si128`/`_mm_loadu_si128`) unconditionally. The remaining branch (SSE3 LDDQU vs. SSE2 `loadu` fallback) is a hardware-capability choice, not an alignment optimization, and is preserved.
- **R3**: The scalar tail loop (`for (; c < width; ++c) dst[c] |= src[c];`) is already alignment-safe and must not be touched.
- **R4**: The fix must not change the function's signature, calling convention, or behavior for already-working (aligned-dst) inputs — output values must be identical, only the crash is eliminated.
- **R5**: A regression test must exist that exercises `orUnaligned8u()`'s dst-alignment fix — preferably via `spread()`/`Detector` (its real caller path), or via a direct unit test of the function if that path proves impractical — using a scenario where the destination pointer's *row-stride advancement* drifts out of 16-byte alignment (not merely an initially-offset base pointer), matching the actual production trigger. Either path satisfies this requirement as long as the alignment-drift mechanism is genuinely exercised.

## Scope Boundaries

**In scope:** the three SIMD branches inside `orUnaligned8u()` in `modules/rgbd/src/linemod.cpp`, plus a new regression test in `modules/rgbd/test/`.

**Out of scope:**
- Any other function in `linemod.cpp` (e.g. `computeResponseMaps`, similarity-map computation) — not implicated by this issue.
- Investigating or fixing opencv/opencv#29576 directly — noted as likely-related but not verified or claimed fixed by this PR.
- Full CMake build / `opencv_test_rgbd` execution — the rgbd module depends on Eigen and the full OpenCV core build; building the entire project to run one test is impractical for this fix's scope and turnaround. See Verification Contract below for the substitute approach.

### Deferred to Follow-Up Work
- None identified — this is a narrowly-scoped bug fix.

## Key Technical Decisions

**KTD1: Remove alignment-based branching entirely; always use unaligned SSE load/store intrinsics for both src and dst.**
Rationale: the original code's `src_aligned` check is computed once per call from the initial `src` pointer and reused for every row, which is unsound whenever `src_stride`/`dst_stride` isn't a multiple of 16 — later rows drift out of the alignment state that was checked at row 0 (confirmed by reproduction — see Problem Frame and U1 Verification). A per-row alignment recheck would fix this correctly but adds a branch evaluation to every iteration of a hot inner loop to save, at best, one aligned-vs-unaligned instruction variant — and unaligned SSE2/SSE3 loads/stores on modern x86 (post-Nehalem, i.e. everything OpenCV 5.x targets) have negligible overhead versus aligned ones. Given the function's name (`orUnaligned8u`) and purpose (OR-ing buffers whose alignment isn't controlled by the caller), removing the alignment fast path entirely — always-unaligned access for both src and dst — is the simplest correct fix and matches what the function was clearly always intended to guarantee. The remaining SSE3-vs-SSE2 branch is retained because it reflects genuine hardware capability, not a false alignment assumption.
Alternatives considered and rejected: (a) adding a `dst_aligned` check mirroring `src_aligned` — rejected, since `src_aligned` itself is the unsound pattern being removed, not a model to extend; (b) recomputing alignment per row instead of removing the fast path — rejected as unjustified complexity for negligible perf gain per the above.

**KTD2: Verify via a standalone, isolated compile-and-run harness rather than the full CMake build.**
Rationale: `modules/rgbd` depends on `opencv_core`, `opencv_calib3d`, `opencv_imgproc`, and optionally Eigen; building the full dependency chain from a fresh clone is a multi-hour, multi-GB undertaking that provides no additional confidence over directly compiling the fixed/unfixed function body with the same compiler and flags OpenCV would use (`g++`, `-msse2 -msse3`), against a synthetic misaligned buffer. The standalone harness reproduces the exact SIGSEGV on the pre-fix code and passes on the post-fix code, which is what the issue is actually about (a fault, not a subtle numerical difference). A `modules/rgbd/test/test_linemod.cpp` regression test is still added to the module for permanent CI coverage (R5), but is not itself run as part of this fix's local verification, since doing so requires the full module build.
Alternative considered and rejected: attempting a full `cmake --build` of `opencv` + `opencv_contrib` — rejected as impractical for the time budget; the standalone harness gives equivalent confidence for this specific bug (a hardware fault triggered by specific instruction encoding, independent of the rest of OpenCV).

## Implementation Units

### U1. Fix unconditional-aligned dst access in orUnaligned8u

**Goal:** Eliminate the SIGSEGV by making every `dst` read/write in `orUnaligned8u()`'s SIMD branches alignment-independent.

**Requirements:** R1, R2, R3, R4

**Dependencies:** None

**Files:**
- `modules/rgbd/src/linemod.cpp` (modify `orUnaligned8u`, approx. lines 916-978)

**Approach:**
- Delete the once-per-call `src_aligned` computation and the "aligned src" branch entirely (per R2/KTD1) — it is unsound regardless of the dst fix, since it goes stale on any row where `src` has drifted out of alignment via non-16-aligned stride.
- Collapse the remaining two branches to: `if (haveSSE3) { ... } else if (haveSSE2) { ... }` — hardware-capability selection only, no alignment condition.
- In the SSE3 branch: keep the `_mm_lddqu_si128` src load (already unaligned-safe); replace the dst-side plain dereference with `_mm_loadu_si128(dst_ptr)` for the read and `_mm_storeu_si128(dst_ptr, result)` for the write.
- In the SSE2 fallback branch: keep the `_mm_loadu_si128` src load (already unaligned-safe); apply the identical dst-side replacement.
- Do not modify the `for` loop bounds, the `haveSSE2`/`haveSSE3` hardware-capability checks, the scalar tail loop, or the row-advance (`src += src_stride; dst += dst_stride;`) logic.
- Add a short comment at the top of the function (near the `haveSSE2`/`haveSSE3` declarations) recording *why* no alignment branching exists — future maintainers must not reintroduce a per-call alignment fast path, since that is exactly the defect being fixed.

**Patterns to follow:** OpenCV's universal intrinsics and raw-SSE code elsewhere in the codebase consistently use `_mm_loadu_si128`/`_mm_storeu_si128` (not pointer dereference) whenever a buffer's alignment isn't statically guaranteed — mirror that convention here.

**Test scenarios:**
- Happy path: `dst` and `src` both 16-byte aligned, width a multiple of 16 — result matches the pre-fix (already-correct) output bit-for-bit.
- **Primary regression scenario (matches the real production trigger):** `dst` base pointer 16-byte aligned, but called across multiple rows with a `dst_stride`/`src_stride` that is *not* a multiple of 16 (e.g., row width 17 or 33 bytes) — so later rows' `dst` pointer drifts out of alignment purely through `dst += dst_stride` advancement, exactly as `spread()` triggers it via `dst.step1()`. Pre-fix code segfaults (or is flagged unsafe) on a drifted row; post-fix code completes and produces the bitwise-correct OR result on every row.
- Edge case: `dst` additionally given an initially-offset base address (e.g., allocate 16 extra bytes and start the buffer at `base + 1`) on top of the stride-drift scenario — a secondary, synthetic misalignment case, not the primary one; confirms the fix isn't merely accidentally-correct for the stride-drift case alone.
- Edge case: both `src` and `dst` unaligned, various `width` values including one not a multiple of 16 (exercises the scalar tail loop combined with the SIMD chunks).
- Edge case: `width` smaller than 16 (SIMD loop body never executes; only the scalar tail runs) — confirms no regression to the already-correct tail path.

**Verification:** Compile a standalone `.cpp` containing the fixed `orUnaligned8u` function body plus a small `main()` that allocates buffers and drives the primary stride-drift scenario above (plus the other scenarios), and checks the OR result against a scalar reference implementation. `orUnaligned8u` calls `cv::checkHardwareSupport(CPU_SSE2)`/`CPU_SSE3`, which require linking `opencv_core`; since this is a standalone harness (KTD2), stub both calls to return `true` (compiling with `-msse2 -msse3` already guarantees both are available on the build machine) — and additionally parameterize the stub per test run (SSE2-only vs SSE2+SSE3) so all three SIMD branches (aligned-load, SSE3 LDDQU, SSE2 MOVDQU fallback) are each exercised at least once, not just whichever branch the real hardware would pick by default. Compile with `g++ -O2 -msse2 -msse3` and run it. Confirm: (a) running the *pre-fix* version of the harness against the stride-drift scenario reproduces a crash (or is flagged via `-fsanitize=address` as an unaligned access, whichever is observable on this machine/OS), and (b) the *post-fix* version completes successfully with correct output for all test scenarios above, across all three stubbed SIMD branches.

### U2. Add regression test to modules/rgbd/test/

**Goal:** Give the fix permanent CI coverage inside the OpenCV test suite, following existing module conventions, so this cannot silently regress.

**Requirements:** R5

**Dependencies:** U1

**Files:**
- `modules/rgbd/test/test_linemod.cpp` (new file)

**Approach:**
- Follow the structure of sibling test files in `modules/rgbd/test/` (e.g. `test_normal.cpp`, `test_odometry.cpp`): start with the standard OpenCV license header, `#include "test_precomp.hpp"`, wrap tests in `namespace opencv_test { namespace { ... } }`, and use the module's actual `TEST(Rgbd_Linemod, <case_name>)` gtest naming convention — PascalCase group name (matching sibling files' `Rgbd_Normals`, `Rgbd_Plane`, `RGBD_Odometry_Rgbd`), not all-lowercase. OpenCV's test runner auto-discovers `test_*.cpp` files in a module's `test/` directory — no CMakeLists changes needed.
- Since `orUnaligned8u` is a private `static` function in `linemod.cpp` and not part of the public API, exercise it indirectly through `cv::linemod::Detector`'s public `addTemplate`/`match` path (which internally calls `spread()` → `orUnaligned8u()`). Construct the quantized template so its row width in bytes is *not* a multiple of 16 (per the Problem Frame's confirmed mechanism: this makes `dst.step1()` non-16-aligned, so `dst += dst_stride` drifts `dst` out of alignment on later rows of `spread()` — the actual production trigger, not an initially-offset pointer) and use `T` (spread sampling step) and image height large enough that `spread()`'s row loop runs multiple iterations past the first drifted row. If reaching this reliably through the public API proves impractical without deeper internals access, fall back to a `#include`-based unit test of `orUnaligned8u` directly (the file can be included via a relative path for test purposes) — either path satisfies R5 as long as the stride-drift mechanism is genuinely exercised; note in the PR description which approach was used and why.
- Do not attempt to build or run this test locally per KTD2 — it is added for CI, and its correctness is validated logically against the U1 standalone harness's reference behavior, not by running it in this environment.

**Test scenarios:**
- `Detector::match()` (or the chosen direct-call path) completes without crashing and returns expected match results for an input constructed to exercise `spread()`'s unaligned-offset code path.
- Test expectation: none beyond the above — this unit *is* the test; there is no separate implementation to further test.

**Verification:** Code review against sibling test files' structure and against U1's confirmed-correct behavior. Full execution deferred to OpenCV's own CI once the PR is opened (per KTD2), since this environment cannot practically build `opencv_test_rgbd`.

## Verification Contract

- U1 is verified locally via a standalone, non-CMake compile-and-run harness using the same compiler/flags family OpenCV uses for SSE2/SSE3 code, proving the fix eliminates the fault and preserves correct output (see U1 Verification).
- U2 is verified by structural review against existing sibling tests; actual execution happens in OpenCV's CI after the PR is opened, not in this local environment.
- No full `opencv_contrib`/`opencv` CMake build is performed as part of this fix, per KTD2.

## Definition of Done

- [ ] `orUnaligned8u()` in `modules/rgbd/src/linemod.cpp` uses unaligned SSE load/store intrinsics for every `src` and `dst` access, with the once-per-call `src_aligned` fast path removed entirely (R1-R4).
- [ ] Standalone harness confirms: pre-fix code crashes on stride-drift and offset-misalignment scenarios; post-fix code produces correct, bit-identical-to-reference output across all U1 test scenarios, including the primary stride-drift scenario and both SSE3/SSE2-fallback branches.
- [ ] `modules/rgbd/test/test_linemod.cpp` added, following module test conventions (R5).
- [ ] No other code in `linemod.cpp` or elsewhere is modified.
- [ ] Changes committed with a message referencing opencv/opencv#29559; PR description notes the likely-related opencv/opencv#29576.
