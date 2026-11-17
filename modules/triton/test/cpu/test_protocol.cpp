// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Standalone CPU control, deliberately outside the native test target's source list.
// Build/run with tools/check_source.py; no OpenCV, GPU SDK or Triton dependency.
#include "compile_protocol.hpp"
#include <cerrno>
#include <chrono>
#include <climits>
#include <csignal>
#include <dirent.h>
#include <fstream>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <iostream>
#include <stdexcept>
namespace
{
volatile sig_atomic_t interruptions = 0;
void interruptWait(int)
{
    ++interruptions;
}
void require(bool value, const char *what)
{
    if (!value)
        throw std::runtime_error(what);
}
}

int main(int argc, char **argv)
try
{
    using namespace cv_triton_runtime;
    require(argc == 3, "expected Python executable and temporary directory");
    require(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 known-answer vector");
    Json value;
    std::string why;
    require(json_parse("{\"x\":[1,true,\"quote\\\"\"]}", value, why), "valid JSON");
    require(!json_parse("{\"x\":[1,}", value, why), "malformed JSON must fail");

    KernelKey key;
    key.kernel = "test_kernel";
    key.signature = "src:*u8,W:i32";
    key.constexprs = {{"CN", 3}};
    key.cfg = {1, 256, 4, 3, 0};
    key.toolchain.source_sha256 = sha256_hex("source");
    const std::string text = key.text();
    const std::string request = make_request(key, "/source.py", argv[2]);
    require(json_parse(request, value, why), "request JSON");
    std::string rebuilt;
    require(value.get("key") && key_text_from_json(*value.get("key"), rebuilt, why) &&
                rebuilt == text,
            "C++ key/request round trip");
    KernelKey changed = key;
    changed.toolchain.source_sha256 = sha256_hex("new source");
    require(changed.text() != text, "source identity changes key");
    changed = key;
    changed.target = {"hip", 936, 64};
    require(changed.text() != text, "backend/target changes key");

    Manifest manifest;
    manifest.compiler = "triton-3.6.0";
    manifest.format = "cubin";
    manifest.global_scratch_bytes = manifest.profile_scratch_bytes = 0;
    manifest.abi = {{"src", "*u8", "u64", false},
                    {"global_scratch", "*i8", "u64", true},
                    {"profile_scratch", "*i8", "u64", true}};
    require(launch_abi_ok(manifest, why), "validated CUDA scratch ABI");
    manifest.format = "hsaco";
    require(launch_abi_ok(manifest, why), "validated HIP scratch ABI");
    manifest.global_scratch_bytes = 4;
    require(!launch_abi_ok(manifest, why), "nonzero scratch refused");
    manifest.global_scratch_bytes = 0;
    manifest.compiler = "unvalidated";
    require(!launch_abi_ok(manifest, why), "unvalidated compiler refused");
    manifest.compiler = "triton-3.6.0";
    manifest.abi.back().name = "wrong";
    require(!launch_abi_ok(manifest, why), "unexpected trailing parameter refused");

    std::string dir, status;
    const std::string response =
        "{\"protocol\":" + json_quote(COMPILE_PROTOCOL) +
        ",\"request_sha256\":" + json_quote(sha256_hex(request)) +
        ",\"status\":\"error\",\"error\":{\"kind\":\"toolchain_unavailable\"}}";
    require(!verify_response(request, response, dir, status, why) && why == "toolchain_unavailable",
            "helper error retains cause");
    require(!verify_response(request + " ", response, dir, status, why) &&
                why == "response answers another request",
            "response bound to exact request bytes");
    Manifest absent;
    require(!verify_artifact(argv[2], text, absent, why), "incomplete artifact refused");
    HelperCommand command;
    command.python = "/nonexistent/triton-test-python";
    command.helper = "/nonexistent/helper.py";
    command.timeout_s = 1;
    require(!run_request(command, request).status.ok(), "missing toolchain must fail");
    command.python = argv[1];
    const std::string helper = std::string(argv[2]) + "/process-control.py";
    const std::string pidfile = std::string(argv[2]) + "/process-control.pid";
    command.helper = helper;
    auto control = [&](const char *name, const std::string &body, double budget, const char *kind,
                       int exitCode)
    {
        ::unlink(pidfile.c_str());
        std::ofstream(helper) << "import os,sys,time,json,hashlib,signal\n"
                              << "with open(" << json_quote(pidfile)
                              << ", 'w') as f: f.write(str(os.getpid()))\n"
                              << body;
        command.timeout_s = budget;
        const auto start = std::chrono::steady_clock::now();
        const CompileOutcome outcome = run_request(command, request);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "control=" << name << " elapsed_s=" << elapsed << " kind=" << outcome.kind
                  << " exit=" << outcome.exit_code << " response=" << outcome.response << std::endl;
        require(outcome.kind == kind, name);
        require(outcome.exit_code == exitCode, "actual helper exit status");
        // A deliberately loose wall limit covers scheduler delays and the bounded
        // post-kill cleanup. The old EOF path fails the timeout-kind check as well.
        require(elapsed < budget + 2.0, "bounded helper return including cleanup");
        int child = -1;
        std::ifstream(pidfile) >> child;
        require(child > 0, "helper pid receipt");
        errno = 0;
        require(::kill(child, 0) == -1 && errno == ESRCH, "helper no longer running or zombie");
        int status_ = 0;
        errno = 0;
        require(::waitpid(child, &status_, WNOHANG) == -1 && errno == ECHILD,
                "direct helper already reaped");
        DIR *directory = ::opendir(argv[2]);
        require(directory != nullptr, "open control directory");
        bool leftover = false;
        for (dirent *entry; (entry = ::readdir(directory));)
            if (std::string(entry->d_name).find(".req-") == 0)
                leftover = true;
        ::closedir(directory);
        require(!leftover, "request files removed");
        return outcome;
    };
    const std::string normal =
        "assert sys.argv[1] == '--request'\nassert sys.stdin.buffer.read() == b''\n"
        "raw = open(sys.argv[2], 'rb').read()\n"
        "print(json.dumps({'protocol': 'opencv-triton-compile/1', 'request_sha256': hashlib.sha256(raw).hexdigest(), "
        "'status': 'error', 'error': {'kind': 'cpu_control', 'message': 'response received'}}), flush=True)\n";
    control("normal-request-file", normal, 2.0, "cpu_control", 0);
    control("response-before-delayed-exit", normal + "os.close(1)\ntime.sleep(.05)\n", 2.0,
            "cpu_control", 0);
    control("stdout-open-timeout", "time.sleep(1.2)\n", .15, "timeout", -SIGKILL);
    control("stdout-closed-timeout", "os.close(1)\ntime.sleep(1.2)\n", .15, "timeout", -SIGKILL);
    control("abnormal-exit", "sys.exit(7)\n", 2.0, "helper_crashed", 7);
    control("signal-exit", "os.kill(os.getpid(), signal.SIGTERM)\n", 2.0, "helper_crashed",
            -SIGTERM);

    // A host SIGCHLD policy may reap children externally. Preserve that policy,
    // refuse an unavailable wait status and never signal a potentially reused pid.
    struct sigaction noChildWait{}, oldChildAction{};
    noChildWait.sa_handler = SIG_DFL;
    noChildWait.sa_flags = SA_NOCLDWAIT;
    ::sigemptyset(&noChildWait.sa_mask);
    require(::sigaction(SIGCHLD, &noChildWait, &oldChildAction) == 0,
            "install no-child-wait fixture");
    control("external-reap-status-unavailable", "sys.exit(0)\n", 2.0, "helper_cleanup_failed",
            INT_MIN);
    require(::sigaction(SIGCHLD, &oldChildAction, nullptr) == 0, "restore child wait policy");

    struct sigaction action{}, previous{};
    action.sa_handler = interruptWait;
    ::sigemptyset(&action.sa_mask);
    require(::sigaction(SIGALRM, &action, &previous) == 0, "install CPU interrupt control");
    itimerval timer{};
    timer.it_value.tv_usec = timer.it_interval.tv_usec = 10000;
    require(::setitimer(ITIMER_REAL, &timer, nullptr) == 0, "start interrupt timer");
    control("interrupted-EOF-timeout", "os.close(1)\ntime.sleep(1.2)\n", .15, "timeout", -SIGKILL);
    timer = {};
    require(::setitimer(ITIMER_REAL, &timer, nullptr) == 0, "stop interrupt timer");
    require(::sigaction(SIGALRM, &previous, nullptr) == 0, "restore interrupt handler");
    require(interruptions > 0, "poll was interrupted by the CPU timer");
#ifdef __linux__
    // Adopt the fixture's grandchild in this standalone control only, so the
    // test can reap it after the real runner kills the inherited-pipe group.
    int previousSubreaper = 0;
    require(::prctl(PR_GET_CHILD_SUBREAPER, &previousSubreaper) == 0, "read fixture subreaper");
    require(::prctl(PR_SET_CHILD_SUBREAPER, 1) == 0, "adopt fixture descendants");
    const std::string grandfile = std::string(argv[2]) + "/grandchild.pid";
    control("inherited-stdout-timeout",
            "child = os.fork()\nif child == 0:\n"
            "    with open(" +
                json_quote(grandfile) +
                ", 'w') as f: f.write(str(os.getpid()))\n"
                "    time.sleep(30)\nelse: os._exit(0)\n",
            .15, "timeout", 0);
    int grandchild = -1;
    std::ifstream(grandfile) >> grandchild;
    require(grandchild > 0, "grandchild pid receipt");
    int grandStatus = 0;
    pid_t done = 0;
    const auto reapDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while ((done = ::waitpid(grandchild, &grandStatus, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < reapDeadline)
        ::usleep(10000);
    require(done == grandchild && WIFSIGNALED(grandStatus) && WTERMSIG(grandStatus) == SIGKILL,
            "inherited stdout descendant killed with the helper group");
    require(::prctl(PR_SET_CHILD_SUBREAPER, previousSubreaper) == 0, "restore fixture subreaper");
#endif
    std::cout << "PASS: CPU protocol/key/ABI/error/deadline controls (no GPU or Triton)\n";
    return 0;
}
catch (const std::exception &e)
{
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
