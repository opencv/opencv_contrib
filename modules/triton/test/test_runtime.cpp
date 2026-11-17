// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright 2026, The FlagOS Contributors.
// Runtime behaviour behind cv::triton::resize: non-default streams, context isolation,
// table-upload ordering, in-flight resource retention, cache eviction, artifact integrity and the
// condition under which the two scratch arguments may be passed as null.
//
// The ordering and retention tests are deterministic: a host callback blocks one stream, so the
// upload and the launch behind it are known to be pending when the next call is made, and new
// table buffers are poisoned, so reading a table too early produces garbage, not plausible output.
#include "test_helpers.hpp"
#include "../src/runtime/compile_protocol.hpp"
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cuda.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
namespace opencv_test
{
namespace
{

using namespace triton_test;

static void CUDA_CB hostSleep(void *ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(*static_cast<int *>(ms)));
}

static void blockStream(cuda::Stream &s, int *ms)
{
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(cuda::StreamAccessor::getStream(s), hostSleep, ms));
}

// A bounded gate keeps pending work a state, not a guess about elapsed time.
// The watchdog only releases a broken scenario so a failed test cannot hang
// the context; a watchdog release is an explicit test failure.
class HostGate
{
    std::mutex mutex_;
    std::condition_variable changed_;
    bool entered_ = false, released_ = false, timedOut_ = false, finished_ = false;
    std::thread watchdog_;

  public:
    HostGate()
        : watchdog_(
              [this]
              {
                  std::unique_lock<std::mutex> lock(mutex_);
                  if (!changed_.wait_for(lock, std::chrono::seconds(30),
                                         [this]
                                         {
                                             return released_;
                                         }))
                  {
                      timedOut_ = released_ = true;
                      changed_.notify_all();
                  }
              })
    {
    }
    ~HostGate()
    {
        release();
        watchdog_.join();
        // An exception after submission must not destroy the callback's state
        // while it is still leaving its wait.
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock,
                      [this]
                      {
                          return !entered_ || finished_;
                      });
    }
    void release()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        changed_.notify_all();
    }
    bool waitEntered()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock,
                      [this]
                      {
                          return entered_ || timedOut_;
                      });
        return entered_ && !timedOut_;
    }
    bool timedOut()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return timedOut_;
    }
    static void CUDA_CB callback(void *ptr)
    {
        HostGate &gate = *static_cast<HostGate *>(ptr);
        std::unique_lock<std::mutex> lock(gate.mutex_);
        gate.entered_ = true;
        gate.changed_.notify_all();
        gate.changed_.wait(lock,
                           [&gate]
                           {
                               return gate.released_;
                           });
        gate.finished_ = true;
        gate.changed_.notify_all();
    }
};

static std::string configField(const std::string &key)
{
    const std::string c = td::runtimeConfig();
    const std::size_t a = c.find(key + "=");
    if (a == std::string::npos)
        return std::string();
    const std::size_t b = c.find(';', a);
    return c.substr(a + key.size() + 1,
                    b == std::string::npos ? std::string::npos : b - a - key.size() - 1);
}

struct Restore
{
    // Pins are process-global and outlive the caches, so a test that captures must not leave them
    // behind for the next test in the binary.
    ~Restore()
    {
        td::setCacheLimits(64, 256);
        td::setPoisonNewTables(false);
        td::releaseCapturePins();
    }
};

static void runOne(const Mat &host, Size dsize, int interp, cuda::Stream &stream, cuda::GpuMat &dst)
{
    cuda::GpuMat src(host);
    cv::triton::resize(src, dst, dsize, 0, 0, interp, stream);
    stream.waitForCompletion();
}

// The installed module must find its resources in the install tree, not in a source or build tree.
TEST(Triton_Runtime, ResourcesResolved)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    const std::string res = configField("resources");
    EXPECT_FALSE(res.empty());
    EXPECT_EQ(std::string::npos, td::runtimeConfig().find(";error=")) << td::runtimeConfig();
    const char *want = std::getenv("OPENCV_TRITON_TEST_EXPECT_RESOURCES_PREFIX");
    if (want)
        EXPECT_EQ(0u, res.find(want)) << "resources " << res << " are not under " << want;
    std::cout << "runtime: " << td::runtimeConfig() << std::endl;
}

// Two non-default streams in flight at once, different data, the same key and tables.
TEST(Triton_Runtime, ConcurrentStreams)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    const Size ssz(321, 203), dsz(149, 87);
    for (int round = 0; round < 8; ++round)
    {
        const Mat ha = randomImage(ssz, CV_8UC3, 1000 + round),
                  hb = randomImage(ssz, CV_8UC3, 2000 + round);
        cuda::GpuMat sa(ha), sb(hb), da, db;
        cv::triton::resize(sa, da, dsz, 0, 0, INTER_LINEAR, a);
        cv::triton::resize(sb, db, dsz, 0, 0, INTER_LINEAR, b);
        a.waitForCompletion();
        b.waitForCompletion();
        Mat ga, gb;
        da.download(ga);
        db.download(gb);
        EXPECT_TRUE(meetsContract(ga, oracle(ha, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
        EXPECT_TRUE(meetsContract(gb, oracle(hb, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
    }
}

// Tables uploaded on stream A, whose upload is still behind a blocked host callback; a call on
// stream B with the same shape must wait for that upload through its event.
TEST(Triton_Runtime, TableUploadOrderingAcrossStreams)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    td::setPoisonNewTables(true);
    const Size ssz(211, 97), dsz(131, 61); // shape unique to this test: its tables are new
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    { // compile/load the binary beforehand, with other tables: the wait below is the upload's
        cuda::GpuMat d;
        runOne(randomImage(Size(53, 41), CV_8UC3, 1), Size(23, 19), INTER_LINEAR, a, d);
    }
    td::reapCompleted();
    td::resetRuntimeStats();
    const Mat ha = randomImage(ssz, CV_8UC3, 11), hb = randomImage(ssz, CV_8UC3, 12);
    cuda::GpuMat sa(ha), sb(hb), da(dsz, CV_8UC3), db(dsz, CV_8UC3);
    int ms = 400;
    const auto t0 = std::chrono::steady_clock::now();
    blockStream(a, &ms);
    cv::triton::resize(sa, da, dsz, 0, 0, INTER_LINEAR, a);
    const double returnedA =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cv::triton::resize(sb, db, dsz, 0, 0, INTER_LINEAR, b);
    const double returnedB =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    b.waitForCompletion();
    const double waited =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    a.waitForCompletion();
    const td::RuntimeStats st = td::runtimeStats();
    EXPECT_EQ(1, st.table_uploads);
    EXPECT_EQ(1, st.table_hits);
    EXPECT_EQ(1, st.cross_stream_waits);
    // Neither call stalled the host (the module was loaded beforehand), so B's wait is the event wait.
    EXPECT_LT(returnedA, 0.5 * ms) << "the first call stalled the host: the scenario tests nothing";
    EXPECT_LT(returnedB, 0.5 * ms)
        << "the second call stalled the host: the scenario tests nothing";
    EXPECT_GE(waited, 0.8 * ms) << "stream B did not wait for the upload behind stream A's block";
    Mat ga, gb;
    da.download(ga);
    db.download(gb);
    EXPECT_TRUE(meetsContract(ga, oracle(ha, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
    EXPECT_TRUE(meetsContract(gb, oracle(hb, dsz, 0, 0, INTER_LINEAR), INTER_LINEAR));
}

// Eviction must retain resources referenced by a known-pending launch. All
// resources are warmed before the gate: new allocations and blocking waits in
// another stream may drain the context under compute-sanitizer. Only cached
// submission and host cache operations occur while the gate is closed.
TEST(Triton_Runtime, InFlightRetentionAndEviction)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    const Size sa(141, 83), da(67, 59), sb(123, 77), db(71, 43), sc(119, 73), dc(61, 47);
    cuda::Stream a(cudaStreamNonBlocking), b(cudaStreamNonBlocking);
    const Mat ha = randomImage(sa, CV_8UC1, 21), hb = randomImage(sb, CV_8UC1, 22),
              hc = randomImage(sc, CV_8UC1, 23);
    cuda::GpuMat ga(ha), gb(hb), gc(hc), oa(da, CV_8UC1), ob(db, CV_8UC1), oc(dc, CV_8UC1);
    td::setCacheLimits(2, 3);
    td::setPoisonNewTables(true);
    cv::triton::resize(ga, oa, da, 0, 0, INTER_NEAREST, a);
    cv::triton::resize(gb, ob, db, 0, 0, INTER_NEAREST, b);
    cv::triton::resize(gc, oc, dc, 0, 0, INTER_LINEAR, b);
    a.waitForCompletion();
    b.waitForCompletion();
    td::reapCompleted();
    td::resetRuntimeStats();

    CUevent pending = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuEventCreate(&pending, CU_EVENT_DISABLE_TIMING));
    HostGate gate;
    const CUstream rawA = cuda::StreamAccessor::getStream(a);
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(rawA, HostGate::callback, &gate));
    EXPECT_TRUE(gate.waitEntered()) << "host gate did not become active";
    cv::triton::resize(ga, oa, da, 0, 0, INTER_NEAREST, a);
    EXPECT_EQ(CUDA_SUCCESS, cuEventRecord(pending, rawA));
    cv::triton::resize(gb, ob, db, 0, 0, INTER_NEAREST, b);
    cv::triton::resize(gc, oc, dc, 0, 0, INTER_LINEAR, b);
    // A's table and nearest module are now least recent. No resource needs
    // allocation, upload or unload merely to evict them from the caches.
    td::setCacheLimits(1, 2);
    td::reapCompleted();
    td::RuntimeStats st = td::runtimeStats();
    EXPECT_FALSE(gate.timedOut()) << "the watchdog released the gate: pending scenario failed";
    EXPECT_EQ(CUDA_ERROR_NOT_READY, cuEventQuery(pending)) << "A must still be pending";
    EXPECT_EQ(0, st.module_loads);
    EXPECT_EQ(0, st.table_uploads);
    EXPECT_EQ(1, st.table_evictions);
    EXPECT_EQ(1, st.module_evictions);
    EXPECT_GE(st.deferred, 2);
    EXPECT_EQ(0, st.module_frees) << "a module in use was unloaded";
    EXPECT_EQ(0, st.table_frees) << "an evicted table in use was freed";
    EXPECT_GE(st.pending, 1);

    gate.release();
    a.waitForCompletion();
    b.waitForCompletion();
    EXPECT_EQ(CUDA_SUCCESS, cuEventQuery(pending));
    EXPECT_EQ(CUDA_SUCCESS, cuEventDestroy(pending));
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_GE(st.module_frees, 1) << "the evicted module was not released after completion";
    EXPECT_EQ(st.table_evictions, st.table_frees)
        << "evicted tables were not released after completion";
    EXPECT_EQ(0, st.pending);
    Mat ra, rb, rc;
    oa.download(ra);
    ob.download(rb);
    oc.download(rc);
    EXPECT_TRUE(meetsContract(ra, oracle(ha, da, 0, 0, INTER_NEAREST), INTER_NEAREST));
    EXPECT_TRUE(meetsContract(rb, oracle(hb, db, 0, 0, INTER_NEAREST), INTER_NEAREST));
    EXPECT_TRUE(meetsContract(rc, oracle(hc, dc, 0, 0, INTER_LINEAR), INTER_LINEAR));

    // Evicted is not lost: the same call reloads from disk, re-uploads, compiles nothing.
    const long long compiles = st.helper_compiles, loads = st.module_loads,
                    uploads = st.table_uploads, disk = st.disk_hits;
    cuda::GpuMat oa2;
    runOne(ha, da, INTER_NEAREST, a, oa2);
    st = td::runtimeStats();
    EXPECT_EQ(compiles, st.helper_compiles);
    EXPECT_EQ(loads + 1, st.module_loads);
    EXPECT_EQ(disk + 1, st.disk_hits);
    EXPECT_EQ(uploads + 1, st.table_uploads);
    oa2.download(ra);
    EXPECT_TRUE(meetsContract(ra, oracle(ha, da, 0, 0, INTER_NEAREST), INTER_NEAREST));
}

// Modules and tables belong to one context: another context on the same device loads and
// uploads its own, and a destroyed context's resources are never reused, even if its handle
// value comes back.
TEST(Triton_Runtime, ContextIsolation)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    const Size ssz(90, 70), dsz(47, 35);
    const Mat host = randomImage(ssz, CV_8UC3, 31);
    const Mat want = oracle(host, dsz, 0, 0, INTER_LINEAR);
    { // the primary context has this key and these tables
        cuda::GpuMat d;
        cuda::Stream s;
        runOne(host, dsz, INTER_LINEAR, s, d);
    }
    CUdevice dev;
    ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, cv::cuda::getDevice()));
    CUcontext primary = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuCtxGetCurrent(&primary));

    for (int round = 0; round < 2; ++round)
    {
        SCOPED_TRACE(cv::format("private context %d", round));
        CUcontext ctx = nullptr;
        ASSERT_EQ(CUDA_SUCCESS, cuCtxCreate(&ctx, 0, dev)); // now current
        const std::size_t sstep = ssz.width * 3, dstep = dsz.width * 3;
        CUdeviceptr ps = 0, pd = 0;
        ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&ps, sstep * ssz.height));
        ASSERT_EQ(CUDA_SUCCESS, cuMemAlloc(&pd, dstep * dsz.height));
        Mat dense = host.clone();
        ASSERT_EQ(CUDA_SUCCESS, cuMemcpyHtoD(ps, dense.data, sstep * ssz.height));
        td::reapCompleted();
        td::resetRuntimeStats();
        {
            cuda::GpuMat gs(ssz.height, ssz.width, CV_8UC3, reinterpret_cast<void *>(ps), sstep);
            cuda::GpuMat gd(dsz.height, dsz.width, CV_8UC3, reinterpret_cast<void *>(pd), dstep);
            cv::triton::resize(gs, gd, dsz, 0, 0, INTER_LINEAR);
            ASSERT_EQ(CUDA_SUCCESS, cuCtxSynchronize());
        }
        const td::RuntimeStats st = td::runtimeStats();
        EXPECT_EQ(1, st.module_loads) << "the module of another context was reused";
        EXPECT_EQ(1, st.table_uploads) << "the tables of another context were reused";
        EXPECT_EQ(0, st.helper_compiles)
            << "the binary is the same; only its loading is per context";
        Mat got(dsz, CV_8UC3);
        ASSERT_EQ(CUDA_SUCCESS, cuMemcpyDtoH(got.data, pd, dstep * dsz.height));
        EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
        td::reapCompleted();
        EXPECT_EQ(0, td::runtimeStats().pending);
        cuMemFree(ps);
        cuMemFree(pd);
        ASSERT_EQ(CUDA_SUCCESS, cuCtxDestroy(ctx)); // its cache entries are now stale
        ASSERT_EQ(CUDA_SUCCESS, cuCtxSetCurrent(primary));
    }
    // Stale entries are dropped without touching the destroyed context, and the primary still works.
    td::setCacheLimits(1, 1);
    td::setCacheLimits(64, 256);
    cuda::GpuMat d;
    cuda::Stream s;
    runOne(host, dsz, INTER_LINEAR, s, d);
    Mat got;
    d.download(got);
    EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
}

TEST(Triton_Runtime, SecondDevice)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    if (cv::cuda::getCudaEnabledDeviceCount() < 2)
        throw SkipTestException("one visible device");
    const int home = cv::cuda::getDevice();
    const Mat host = randomImage(Size(77, 55), CV_32FC3, 41);
    const Mat want = oracle(host, Size(39, 28), 0, 0, INTER_LINEAR);
    td::reapCompleted();
    td::resetRuntimeStats();
    cv::cuda::setDevice(home == 0 ? 1 : 0);
    {
        cuda::GpuMat d;
        cuda::Stream s;
        runOne(host, Size(39, 28), INTER_LINEAR, s, d);
        Mat got;
        d.download(got);
        EXPECT_TRUE(meetsContract(got, want, INTER_LINEAR));
    }
    EXPECT_GE(td::runtimeStats().module_loads, 1);
    td::reapCompleted();
    cv::cuda::setDevice(home);
}

// What the launcher runs is what the manifest says: every loaded artifact verifies independently
// of the runtime, and a corrupted one on disk is rejected and rebuilt, not loaded.
TEST(Triton_Runtime, ArtifactIntegrity)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    const Mat host = randomImage(Size(93, 67), CV_32FC1, 51);
    const Size dsz(41, 37);
    cuda::Stream s;
    cuda::GpuMat d;
    runOne(host, dsz, INTER_NEAREST, s, d);
    td::RuntimeStats st = td::runtimeStats();
    const std::string dir = st.last_artifact, key = st.last_key;
    ASSERT_FALSE(dir.empty());
    cv_triton_runtime::Manifest m;
    std::string why;
    ASSERT_TRUE(cv_triton_runtime::verify_artifact(dir, key, m, why)) << why;
    EXPECT_TRUE(cv_triton_runtime::launch_abi_ok(m, why)) << why;
    EXPECT_EQ("_resize_cpu_nearest_kernel", m.function);
    EXPECT_EQ(0, m.global_scratch_bytes);
    EXPECT_EQ(0, m.profile_scratch_bytes);

    // Corrupt it on disk, push it out of the in-process cache, ask again.
    td::setCacheLimits(1, 1);
    {
        cuda::GpuMat other;
        runOne(randomImage(Size(31, 29), CV_32FC1, 52), Size(13, 11), INTER_LINEAR, s, other);
    }
    td::reapCompleted();
    {
        std::fstream f(dir + "/" + m.binary_file, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(f.good());
        f.seekg(static_cast<std::streamoff>(m.binary_bytes / 2));
        char c = 0;
        f.read(&c, 1);
        c = static_cast<char>(c ^ 0xFF);
        f.seekp(static_cast<std::streamoff>(m.binary_bytes / 2));
        f.write(&c, 1);
    }
    cv_triton_runtime::Manifest bad;
    EXPECT_FALSE(cv_triton_runtime::verify_artifact(dir, key, bad, why));
    td::resetRuntimeStats();
    cuda::GpuMat d2;
    runOne(host, dsz, INTER_NEAREST, s, d2);
    st = td::runtimeStats();
    EXPECT_EQ(1, st.disk_rejected);
    EXPECT_EQ(1, st.helper_compiles);
    cv_triton_runtime::Manifest again;
    EXPECT_TRUE(cv_triton_runtime::verify_artifact(dir, key, again, why)) << why;
    Mat got;
    d2.download(got);
    EXPECT_TRUE(meetsContract(got, oracle(host, dsz, 0, 0, INTER_NEAREST), INTER_NEAREST));
}

// A call must be capturable, and a replayed graph must compute the same answer as a direct call.
//
// Event queries during capture can invalidate the capture
// (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED). Reaping must avoid event queries and
// completion-event recording while the caller's stream is capturing.
//
// The supported pattern is the one used here: warm on the stream you will capture on, so the
// tables are already resident. Capturing on a stream that never saw the table upload still fails
// (the capture cannot wait on an event recorded outside it) -- that is a documented limit, not
// asserted here.
// Capture, then evict everything, then replay: the graph must still read live memory.
//
// A graph recorded under capture refers to the module
// binary and the table buffers by handle, and it is launched later -- possibly after the caches
// have evicted both. The runtime therefore pins them at capture time (finishLaunch) and releases
// them through the public releaseGraphResources() call after graph use. The scenario below
// forces the eviction explicitly:
//
//   capture -> shrink the caches to 1/1 and load other keys so the entries are really gone
//           -> replay twice with DIFFERENT inputs and check both answers
//           -> destroy the graph -> release the pins -> confirm the resources are freed
//
// Change input and overwrite output between replays so an empty or inactive
// graph cannot pass by retaining the warmup result. Release through the public
// API, since applications cannot access private runtime resource functions.
TEST(Triton_Runtime, PublicGraphResourceRelease)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    cv::triton::releaseGraphResources();
    td::reapCompleted();

    const Size sz(129, 97);
    const Mat host = randomImage(sz, CV_8UC1, 73);
    cuda::GpuMat src(host), dst(sz, CV_8UC1);
    cuda::Stream s(cudaStreamNonBlocking);
    for (int i = 0; i < 3; ++i)
        cv::triton::warpAffine(src, dst, cv::Mat::eye(2, 3, CV_64F), sz, INTER_LINEAR,
                               BORDER_REPLICATE, Scalar(), s);
    s.waitForCompletion();

    CUstream raw = cuda::StreamAccessor::getStream(s);
    CUgraph graph = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuStreamBeginCapture(raw, CU_STREAM_CAPTURE_MODE_GLOBAL));
    ASSERT_NO_THROW(cv::triton::warpAffine(src, dst, cv::Mat::eye(2, 3, CV_64F), sz, INTER_LINEAR,
                                           BORDER_REPLICATE, Scalar(), s));
    ASSERT_EQ(CUDA_SUCCESS, cuStreamEndCapture(raw, &graph));
    ASSERT_NE(nullptr, graph);

    td::RuntimeStats st = td::runtimeStats();
    EXPECT_GE(st.capture_pin_holds, 1) << "a captured warpAffine pinned nothing";

    cuGraphDestroy(graph);

    // The documented call, not the internal one.
    cv::triton::releaseGraphResources();
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_EQ(0, st.capture_pin_holds) << "the public entry did not release the pin";

    // Still usable afterwards: releasing must not poison the runtime.
    cv::triton::warpAffine(src, dst, cv::Mat::eye(2, 3, CV_64F), sz, INTER_LINEAR, BORDER_REPLICATE,
                           Scalar(), s);
    s.waitForCompletion();
    Mat got, want;
    dst.download(got);
    cv::warpAffine(host, want, cv::Mat::eye(2, 3, CV_64F), sz, INTER_LINEAR, BORDER_REPLICATE);
    EXPECT_EQ(0.0, cvtest::norm(got, want, NORM_INF));
}

TEST(Triton_Runtime, CaptureSurvivesEviction)
{
    if (!ready())
        throw SkipTestException("no Triton backend");
    Restore restore;
    td::releaseCapturePins();
    td::reapCompleted();

    const Size sz(257, 131); // width not a multiple of the 256 block: exercises masking
    cuda::Stream s(cudaStreamNonBlocking);
    const Mat hostA = randomImage(sz, CV_8UC3, 71);
    const Mat hostB = randomImage(sz, CV_8UC3, 72);
    Mat wantB; // filled below, after the capture
    cuda::GpuMat srcA(hostA), srcB(hostB);
    cuda::GpuMat dst(sz, CV_8UC1);

    for (int i = 0; i < 3; ++i)
        cv::triton::cvtColor(srcA, dst, COLOR_BGR2GRAY, s);
    s.waitForCompletion();

    CUstream raw = cuda::StreamAccessor::getStream(s);
    // DELTA, not an absolute count: capture_launches is process-global, and any earlier test in
    // this binary that captured (PublicGraphResourceRelease does) would make an absolute
    // expectation depend on test order. The invariant here is "this capture added one launch".
    const long long capturesBefore = td::runtimeStats().capture_launches;
    CUgraph graph = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuStreamBeginCapture(raw, CU_STREAM_CAPTURE_MODE_GLOBAL));
    ASSERT_NO_THROW(cv::triton::cvtColor(srcA, dst, COLOR_BGR2GRAY, s));
    ASSERT_EQ(CUDA_SUCCESS, cuStreamEndCapture(raw, &graph));
    ASSERT_NE(nullptr, graph);

    td::RuntimeStats st = td::runtimeStats();
    EXPECT_EQ(1, st.capture_launches - capturesBefore);
    EXPECT_GE(st.capture_pin_holds, 1) << "the captured graph pinned nothing: a later eviction "
                                          "would free what it reads";

    // Evict. Shrinking alone is not enough -- the entries have to be replaced by other keys, so
    // other codes are run until the module of this graph is out of the cache.
    td::setCacheLimits(1, 1);
    {
        // A SEPARATE destination: cvtColor resizes dst to the code's channel count, so running
        // BGR2YUV into `dst` would leave it 3-channel and the later comparison would fail on
        // type rather than on values.
        cuda::GpuMat other(sz, CV_8UC3);
        cv::triton::cvtColor(srcA, other, COLOR_BGR2YUV, s);
        cv::triton::cvtColor(srcA, other, COLOR_BGR2HSV, s);
        s.waitForCompletion();
    }
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_GE(st.module_evictions + st.table_evictions, 1) << "nothing was evicted: the test "
                                                              "would not exercise the pin";
    // `module_frees == 0` would be the wrong assertion here, because the eviction traffic above
    // loads and drops modules of its own, and those are free to be released. The evidence that
    // the PINNED resource survived is `deferred`: a resource evicted while its use count is
    // still above one. That can only be the graph's own module or tables.
    EXPECT_GE(st.deferred, 1) << "an eviction did not find the captured graph's resources held, "
                                 "so nothing pinned them";

    CUgraphExec exec = nullptr;
    ASSERT_EQ(CUDA_SUCCESS, cuGraphInstantiate(&exec, graph, 0));

    // Replay 1: input A. dst is polluted first so a no-op cannot pass.
    dst.setTo(Scalar::all(0), s);
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, raw));
    ASSERT_EQ(CUDA_SUCCESS, cuStreamSynchronize(raw));
    Mat gotA;
    Mat wantA;
    dst.download(gotA);
    cv::cvtColor(hostA, wantA, COLOR_BGR2GRAY);
    EXPECT_EQ(0.0, cvtest::norm(gotA, wantA, NORM_INF)) << "the replay after eviction is wrong";

    // Replay 2: DIFFERENT input. The graph recorded the address of srcA, so overwriting srcA's
    // bytes changes what the replayed graph must compute. This is what separates "the graph ran"
    // from "dst still held the previous answer": a no-op, an empty graph, or a graph that never
    // executed would leave dst equal to replay 1's output and fail here.
    srcA.upload(hostB, s);
    cv::cvtColor(hostB, wantB, COLOR_BGR2GRAY);
    ASSERT_GT(cvtest::norm(wantA, wantB, NORM_INF), 0.0)
        << "the two inputs are identical: the replay check proves nothing";
    dst.setTo(Scalar::all(0), s);
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, raw));
    ASSERT_EQ(CUDA_SUCCESS, cuStreamSynchronize(raw));
    Mat gotB;
    dst.download(gotB);
    EXPECT_EQ(0.0, cvtest::norm(gotB, wantB, NORM_INF))
        << "the replay did not recompute from the new input; it may not have executed at all";

    // Replay 3: back to A, so the pair is two different inputs in both directions rather than
    // one change that could have been a stale buffer.
    srcA.upload(hostA, s);
    dst.setTo(Scalar::all(0), s);
    ASSERT_EQ(CUDA_SUCCESS, cuGraphLaunch(exec, raw));
    ASSERT_EQ(CUDA_SUCCESS, cuStreamSynchronize(raw));
    Mat gotA2;
    dst.download(gotA2);
    EXPECT_EQ(0.0, cvtest::norm(gotA2, wantA, NORM_INF)) << "third replay disagrees with the first";

    cuGraphExecDestroy(exec);
    cuGraphDestroy(graph);

    // Only now may the resources go.
    st = td::runtimeStats();
    const long long freedBefore = st.module_frees + st.table_frees;
    td::releaseCapturePins();
    td::reapCompleted();
    st = td::runtimeStats();
    EXPECT_EQ(0, st.capture_pin_holds);
    EXPECT_GE(st.module_frees + st.table_frees, freedBefore) << "release did not drop the pin";

    // And the module still works afterwards (it reloads if it was freed).
    cv::triton::cvtColor(srcA, dst, COLOR_BGR2GRAY, s);
    s.waitForCompletion();
    Mat got;
    dst.download(got);
    EXPECT_EQ(0.0, cvtest::norm(got, wantA, NORM_INF));
}

// The two scratch arguments are null only under a validated compiler ABI and zero scratch bytes.
TEST(Triton_Runtime, ScratchArgumentsAreConditional)
{
    using cv_triton_runtime::Manifest;
    Manifest ok;
    ok.compiler = "triton-3.6.0";
    // The ABI check is keyed by (compiler, binary format): a HIP build can report the same
    // version string as the CUDA one, so the format is what tells them apart. A manifest
    // without it is an unvalidated ABI, not an implicitly-good one.
    ok.format = "cubin";
    ok.global_scratch_bytes = 0;
    ok.profile_scratch_bytes = 0;
    ok.abi = {{"src", "*u8", "u64", false},
              {"W", "i32", "u32", false},
              {"global_scratch", "*i8", "u64", true},
              {"profile_scratch", "*i8", "u64", true}};
    std::string why;
    EXPECT_TRUE(cv_triton_runtime::launch_abi_ok(ok, why)) << why;
    Manifest m = ok;
    m.compiler = "triton-9.9.9";
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why))
        << "a compiler whose ABI was not validated";
    // The format is a key PART, not a decoration: "hsaco" is a legitimately validated ABI for
    // the same version string (flagtree's HIP build), while an empty or unknown format is not
    // validated at all -- and must not pass by falling back to the compiler string alone.
    m = ok;
    m.format.clear();
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why))
        << "no binary format cannot be validated";
    m = ok;
    m.format = "exe";
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why))
        << "an unknown binary format is not validated";
    m = ok;
    m.format = "hsaco";
    EXPECT_TRUE(cv_triton_runtime::launch_abi_ok(m, why))
        << "the HIP format is validated for the same ABI";
    m = ok;
    m.global_scratch_bytes = 1024;
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why)) << "the kernel needs scratch memory";
    m = ok;
    m.profile_scratch_bytes = -1;
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why)) << "scratch size unknown";
    m = ok;
    m.abi.pop_back();
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why)) << "only one trailing pointer";
    m = ok;
    m.abi[2].name = "something_else";
    EXPECT_FALSE(cv_triton_runtime::launch_abi_ok(m, why)) << "not the expected trailing pointers";
}

}
} // namespace
