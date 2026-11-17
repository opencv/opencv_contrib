#!/usr/bin/env python3
# This file is part of OpenCV project.
# It is subject to the license terms in the LICENSE file found in the top-level directory
# of this distribution and at http://opencv.org/license.html.
"""Compile selected backend lifetime/context functions against injected CPU driver calls.

Reads the module sources and uses a temporary C++ harness to check completion,
cleanup failures and primary-context ownership. No SDK or GPU fault is needed.
Prints control results and exits nonzero on a failed compile/assertion; this does
not establish real-driver or hardware qualification.
"""
import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile

PRELUDE = r'''
#include "compile_protocol.hpp"
#include "lifetime.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <memory>
#include <list>
using namespace cv_triton_runtime;
using NativeContext = void*;
thread_local NativeContext activeContext = nullptr;
std::atomic<int> retains{0}, sets{0}, failSets{0}, idQueries{0}, invalidId{0};
bool nativeId = false;
int init(unsigned) { return 0; }
int getCurrent(NativeContext* c) { *c = activeContext; return 0; }
int setCurrent(NativeContext c) { ++sets; if (failSets > 0) { --failSets; return 5; } activeContext = c; return 0; }
int deviceGet(int* d, int n) { *d = n; return 0; }
int getDevice(int* d) { *d = activeContext == reinterpret_cast<void*>(0x2000) ? 1 : 0; return 0; }
int retain(NativeContext* c, int d) { ++retains; *c = reinterpret_cast<void*>(0x1000 + d * 0x1000); return 0; }
int getId(NativeContext c, unsigned long long* g) { ++idQueries; if (invalidId) return invalidId == 1 ? 2 : 5; *g = reinterpret_cast<std::uintptr_t>(c); return 0; }
int getState(int, unsigned* flags, int* active) { *flags = 0; *active = !invalidId; return invalidId == 2 ? 5 : 0; }
void* fakeDlsym(void*, const char*) { return nativeId ? reinterpret_cast<void*>(&getId) : nullptr; }
#define dlsym fakeDlsym
#define RTLD_DEFAULT nullptr
bool driverReady() { return true; }
Status error(Reason r, const std::string& m) { Status s; s.code = Code::Error; s.reason = r; s.message = m; return s; }
std::string driverError(int r) { return std::to_string(r); }
'''
ALIASES = r'''
using CUcontext = NativeContext; using CUresult = int; using CUdevice = int;
constexpr int CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_CONTEXT = 2, CUDA_ERROR_CONTEXT_IS_DESTROYED = 3,
              CUDA_ERROR_DEINITIALIZED = 4;
#define cuCtxGetCurrent getCurrent
#define cuCtxSetCurrent setCurrent
#define cuCtxGetDevice getDevice
#define cuCtxGetId getId
#define cuDeviceGet deviceGet
#define cuDevicePrimaryCtxRetain retain
#define cuErr driverError
using hipCtx_t = NativeContext; using hipError_t = int; using hipDevice_t = int;
constexpr int hipSuccess=0, hipErrorInvalidContext=2, hipErrorContextIsDestroyed=3,
              hipErrorDeinitialized=4, hipErrorInvalidValue=6;
#define hipCtxGetCurrent getCurrent
#define hipCtxSetCurrent setCurrent
#define hipCtxGetDevice getDevice
#define hipDeviceGet deviceGet
#define hipDevicePrimaryCtxRetain retain
#define hipDevicePrimaryCtxGetState getState
#define hipErr driverError
'''

REAPER_PRELUDE = r'''
using CUevent = void*; using hipEvent_t = void*;
using CUstream = void*; using hipStream_t = void*;
using CUmodule = void*; using CUfunction = void*; using CUdeviceptr = std::uintptr_t;
using hipModule_t = void*; using hipFunction_t = void*;
std::atomic<long long> g_module_frees{0}, g_table_frees{0};
int freeStatus = 0, waitStatus = 0, freeCalls = 0, unloadCalls = 0;
int freeMemory(std::uintptr_t) { ++freeCalls; return freeStatus; }
int freeMemory(void*) { ++freeCalls; return freeStatus; }
int freeAsync(std::uintptr_t p, void*) { return freeMemory(p); }
int freeAsync(void* p, void*) { return freeMemory(p); }
int unloadModule(void*) { ++unloadCalls; return freeStatus; }
int waitEvent(void*, void*, unsigned) { return waitStatus; }
#define cuMemFree freeMemory
#define hipFree freeMemory
#define cuMemFreeAsync freeAsync
#define hipFreeAsync freeAsync
#define cuModuleUnload unloadModule
#define hipModuleUnload unloadModule
#define cuStreamWaitEvent waitEvent
#define hipStreamWaitEvent waitEvent
struct CtxGuard { bool valid = true; explicit CtxGuard(NativeContext) {} };
std::atomic<int> eventQueries{0}, eventDestroys{0}, diagnostics{0};
Completion completion = Completion::NotReady;
Completion eventCompletion(void*) { ++eventQueries; return completion; }
void cleanupError(const std::string&) { ++diagnostics; }
int destroyStatus=0;
int destroyEvent(void*) { ++eventDestroys; return destroyStatus; }
#define cuEventDestroy destroyEvent
#define hipEventDestroy destroyEvent
struct Pending {
    NativeContext ctx; std::uint64_t gen; void* done;
    bool query_error_reported = false;
    std::vector<std::shared_ptr<void>> keep;
};
struct State { std::list<Pending> pending, failed_events; struct { long long pending = 0; } st; };
'''
REAPER_TEST = r'''
void exerciseReap(NativeContext context, std::uint64_t generation) {
    State state; auto owner=std::make_shared<int>(7);
    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    std::vector<std::shared_ptr<void>> released;
    assert(reapLocked(state,released,true).ok() && eventQueries==0 && state.pending.size()==1);
    invalidId=2; const int unknownQueries=eventQueries;
    assert(!reapLocked(state,released).ok() && state.pending.size()==1 && eventQueries==unknownQueries);
    invalidId=0; diagnostics=0; state.pending.front().query_error_reported=false;
    completion=Completion::NotReady;
    assert(reapLocked(state,released).ok() && state.pending.size()==1 && released.empty());
    completion=Completion::Error;
    assert(!reapLocked(state,released).ok() && state.pending.size()==1 && released.empty() && diagnostics==1);
    assert(!reapLocked(state,released).ok() && diagnostics==1); // observable, not log spam
    completion=Completion::Complete;
    assert(reapLocked(state,released).ok() && state.pending.empty() && released.size()==1 && eventDestroys==1);
    assert(reapLocked(state,released).ok() && eventDestroys==1);
    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    destroyStatus=7;
    assert(!reapLocked(state,released).ok() && state.pending.empty() && state.failed_events.size()==1);
    const int failedDestroys=eventDestroys;
    assert(reapLocked(state,released).ok() && eventDestroys==failedDestroys && state.failed_events.size()==1);
    invalidId=1;
    assert(reapLocked(state,released).ok() && state.failed_events.empty() && eventDestroys==failedDestroys);
    invalidId=0;destroyStatus=0;

    state.pending.push_back({context,generation,reinterpret_cast<void*>(9),false,{owner}});
    invalidId=1; const int queries=eventQueries, destroys=eventDestroys;
    assert(reapLocked(state,released).ok() && state.pending.empty());
    assert(eventQueries==queries && eventDestroys==destroys); // known destroyed context
    invalidId=0;
}
'''

RESOURCE_TEST = r'''
void exerciseResources(NativeContext context, std::uint64_t generation) {
    Retirement<TableRes> tables;
    auto* table=new TableRes();table->ctx=context;table->gen=generation;
    table->base=256;table->uploaded=reinterpret_cast<void*>(9);
    table->upload_started=table->upload_recorded=true;table->host.resize(64,17);
    completion=Completion::NotReady;waitStatus=5;
    tables.dispose(table);
    assert(tables.size()==1 && table->host[0]==17 && freeCalls==0);
    completion=Completion::Error;tables.retry(false);
    assert(tables.size()==1 && table->host.size()==64 && freeCalls==0);
    completion=Completion::Complete;waitStatus=0;freeStatus=7;
    tables.retry(false);
    assert(tables.size()==1 && freeCalls==1 && table->cleanup_failed && tables.hasFailure());
    tables.retry(false);
    assert(freeCalls==1); // unknown irreversible outcome must not double-free
    invalidId=1;tables.retry(false);invalidId=0;
    assert(tables.size()==0 && freeCalls==1 && !tables.hasFailure());
    Retirement<ModuleRes> modules;
    auto* module=new ModuleRes();module->ctx=context;module->gen=generation;
    module->mod=reinterpret_cast<void*>(8);
    modules.dispose(module);
    assert(modules.size()==1 && unloadCalls==1);
    modules.retry(false);assert(unloadCalls==1);
    invalidId=1;modules.retry(false);invalidId=0;
    assert(modules.size()==0 && unloadCalls==1);
    freeStatus=0;
}
'''
MAIN = r'''
int main(int argc, char** argv) {
    nativeId = argc > 1 && std::string(argv[1]) == "id";
    NativeContext context = nullptr; std::uint64_t generation = 0;
    failSets = 1;
    assert(!currentContext(0, context, generation).ok());
    assert(retains == 1); // failed SetCurrent did not leak an additional thread retain
    assert(currentContext(0, context, generation).ok());
    assert(retains == 1 && context == reinterpret_cast<void*>(0x1000));
    const auto first = generation;
    exerciseReap(context,generation);
    exerciseResources(context,generation);
    for (int i = 0; i < 8; ++i) assert(currentContext(0, context, generation).ok() && generation == first);
    std::vector<std::thread> threads;
    for (int i = 0; i < 64; ++i) threads.emplace_back([&]() {
        NativeContext c=nullptr; std::uint64_t g=0;
        assert(currentContext(0,c,g).ok() && g==first);
        activeContext=nullptr; // host thread exit does not drop runtime's cache owner
    });
    for(auto& t:threads)t.join();
    assert(retains == 1);
    activeContext=nullptr;
    assert(currentContext(1,context,generation).ok());
    assert(retains == 2 && generation != first);
    invalidId = 1;
    assert(contextLife(context,generation)==ContextLife::Gone);
    invalidId = 0;
    std::cout << "PASS: actual context acquisition + lifetime, 64 threads, SetCurrent failure, two devices, native-ID=" << nativeId << ", retains=" << retains << "\n";
}
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[2] / 'src' / 'runtime'
    with tempfile.TemporaryDirectory(prefix='opencv-triton-native-lifetime-') as directory:
        for backend in ('cuda', 'hip'):
            text = (runtime / (backend + '_runtime.cpp')).read_text()
            start = text.index('ContextLife contextLife') if backend == 'cuda' else text.index('hipError_t primaryContext(int device, hipCtx_t& context);')
            end = text.index('struct CleanupDiagnostics', start)
            contexts = text[start:end]
            start = text.index('Status currentContext(')
            end = text.index('// ---------------------------------------------------------------- modules', start)
            reap_start = text.index("Status reapLocked(")
            reap_end = text.index("Status currentContext(", reap_start)
            resource_start = text.index("struct ModuleRes")
            resource_end = text.index("thread_local bool recordingCapture", resource_start)
            code = PRELUDE + ALIASES + contexts + text[start:end] + REAPER_PRELUDE + text[resource_start:resource_end] + text[reap_start:reap_end] + REAPER_TEST + RESOURCE_TEST + MAIN
            path = Path(directory) / (backend + '.cpp'); path.write_text(code)
            executable = path.with_suffix('')
            subprocess.run(shlex.split(args.cxx) + ['-std=c++17', '-pthread', '-O2', '-Wall', '-Wextra',
                           '-I' + str(runtime), str(path), '-o', str(executable)], check=True)
            for mode in ('id', 'fallback') if backend == 'hip' else ('id',):
                subprocess.run([str(executable), mode], check=True)

if __name__ == '__main__':
    main()
