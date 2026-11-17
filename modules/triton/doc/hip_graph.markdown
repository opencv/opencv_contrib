# Internal HIP graph metric {#triton_hip_graph_metric}

This optional helper measures warmed production `DeviceImage` calls captured
repeatedly on one fixed nondefault stream. It uses production dispatch and
kernels, with no Torch dependency. It measures the Triton implementation only:
`baseline_status: BASELINE_UNAVAILABLE` means there is no baseline latency or
speedup ratio. Graph timing does not establish ordinary-call performance or
application-level performance.

## Build and request

For an enabled HIP module build, set `BUILD_PERF_TESTS=ON` and
`TRITON_BUILD_HIP_GRAPH_BENCH=ON` to build `opencv_perf_triton_hip_graph`.
The latter option is OFF by default and respects the selected perf module list.

Each JSON request supplies:

| Field | Meaning |
|---|---|
| `op`, `case` | Operator and its dimensions, depth, channels and supported options |
| `input_path`, `input_sha256` | Contiguous input bytes and their file hash |
| `changed_path`, `changed_sha256` | A second input whose expected output differs |
| `captured_calls` | Calls per graph, from 1 to 250; default 250 |
| `samples` | Timed replays, fixed at 10 |
| `roi` | Use a pitched ROI; boolean or exact integer 0/1 |

The helper allocates its own pitched device buffers. It never accepts runtime
pointers from requests. Dimensions, allocation products and options are validated
before GPU allocation. Validation also checks:

- Strings, booleans and all six matrix entries have the required types; numeric
  values are finite and integer fields fit their declared ranges.
- Duplicate and unknown fields are refused. `case.id` is an optional string.
- Only the parser's listed interpolation aliases and borders are accepted.
  Colour conversions have no interpolation mode; their declared output size
  must match the actual conversion output.
- Any `case.borderValue` field is refused, including zero or null. This helper
  uses the default zero border value; production operator support is unchanged.
- Same-size resize copies are refused here because they have no compiled-kernel
  manifest identity. Production copy support is unchanged.
- Float input files contain finite values, and both input hashes match their bytes.

`request_sha256` identifies the accepted JSON. `executed_parameters` records the
normalized options and defaults, so an omitted field cannot hide a changed
matrix, interpolation, shape or border setting.

## Prepare and capture

1. Confirm the selected device is idle and follow the host's device-use rules.
   Use the current build; do not infer capture support from another platform.
2. Run `--primitive-control`. It captures a 64-byte asynchronous fill and checks
   three replays, overwriting the destination before each replay.
3. Validate a tiny production request before increasing scope.
4. Warm JIT compilation, module load and table uploads with 20 calls on the
   capture stream, then synchronize.
5. Capture N identical production calls, end capture, inspect nodes and instantiate.
   Compilation, module load and table upload must not occur inside capture.

The helper records the actual production key/route and verifies hashes of the
compiled binary and metadata. Kernel-node parameters record actual functions,
grid/block/shared-memory geometry and work counts. It does not select a binary
by filename or bypass the public entry.

## Correctness and lifetime controls

Each request checks:

- Ordinary output against the independent numerical reference, then full
  ordinary-versus-graph equality.
- Changed input against a newly computed reference, with two destination fill
  patterns and repeated replays to detect empty or stale work.
- Unchanged input and output guard bytes around the valid views.
- Buffer addresses, alignment, types, shapes, pitches and parameter identity
  before submission; changed bindings are rejected.
- Replay after entries are evicted from the runtime's least-recently-used cache.
  Retained references must keep graph modules and tables alive.
- A second construct/replay/destroy cycle, separate from the timing population.

Graphs are not cached across requests. CPU parser tests establish request
validation, not HIP capture/replay support.

## Timing

For the default workload, N is 250 production calls per graph. Smaller controls
may request 1..250 calls. The helper collects 10 samples:

1. Record a start event on the graph's explicit stream.
2. Launch the graph once and record a stop event on the same stream.
3. Wait for the stop event and compute `hipEventElapsedTime(start, stop) / N`.
4. Retain both total-graph and per-call samples; report median, p10, p90 and range.

No minimum selection or L2 flush is inserted. Buffers are reused, so this is a
warm repeated-buffer metric, not ordinary submission or isolated kernel latency.
Host-side binding checks are outside the timed bracket. Graph launch/device
scheduling costs remain inside it.

Other measurements remain separate:

| Measurement | Boundary |
|---|---|
| First production call | Host wall, including conditional JIT/cache lookup/load |
| Warmup | Host wall for 20 calls and completion |
| Capture and instantiation | Separate host wall costs |
| Ordinary calls | Seven blocks of 250 calls, GPU-event time and batch-end completion wall per call |

`ordinary_wall_blocks_ms` and `ordinary_event_blocks_ms` retain all per-call
block averages; scalar ordinary fields report their medians. This is amortized
synchronized batch cost, not separately synchronized individual-call latency.

The raw field `first_compile_call_wall_ms` names the first production request
in that process. It is not isolated compiler time and does not prove compilation
occurred. Use binary creation records to distinguish compilation from cache reuse.

## Cleanup and errors

After successful work:

1. Synchronize the stream.
2. Destroy the graph executable, then the graph.
3. Release graph-retained module/table references only after all graphs are destroyed.
4. Reclaim resources from completed launches.
5. Destroy events, free storage and destroy the stream.

Buffers, stream and context stay live throughout. API/device errors stop the
process without reset or blind retries. Unknown completion never permits early
release: the helper leaves potentially in-flight resources to supervised process
exit rather than retrying unsafe cleanup.

Run `tools/check_source.py` for CPU request-validation and error-handling controls.
Check actual graph execution on the intended HIP device; a result on one
architecture does not establish support on another. Results use `graph_replay`
and preserve raw observations for each case.
