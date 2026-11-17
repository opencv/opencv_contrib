// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_TRITON_LIFETIME_HPP
#define OPENCV_TRITON_LIFETIME_HPP
#include <atomic>
#include <map>
#include <mutex>
#include <vector>
namespace cv_triton_runtime
{
enum class ContextLife
{
    Alive,
    Gone,
    Unknown
};
enum class Completion
{
    Complete,
    NotReady,
    Error
};
enum class Reclaim
{
    Release,
    Hold,
    Error
};

// Upload cleanup decisions:
// - capture: retain without driver cleanup; destroyed context: drop host ownership;
// - unknown context or an upload without a recorded completion event: retain;
// - event error/not-ready: retain (a device wait alone cannot release host bytes);
// - proven completion, or no upload started: free storage, then destroy the event.
// False keeps the entire owner for later cleanup; true permits owner destruction.
template <class Query, class Wait, class Free, class Destroy>
bool releaseUpload(bool capture, ContextLife life, bool started, bool recorded, Query query,
                   Wait wait, Free free, Destroy destroy)
{
    if (capture)
        return false;
    if (life == ContextLife::Gone)
        return true;
    if (life == ContextLife::Unknown || (started && !recorded))
        return false;
    if (started)
    {
        const Completion completion = query();
        if (completion == Completion::Error)
            return false;
        if (completion == Completion::NotReady)
        {
            // Even a successful device wait does not establish that host staging
            // bytes can be released. Both success and failure preserve ownership.
            wait();
            return false;
        }
    }
    return free() && destroy();
}

// The query is deliberately lazy: capture and dead/unknown contexts never touch an event.
template <class Query> Reclaim completionDecision(bool capturing, ContextLife life, Query query)
{
    if (capturing)
        return Reclaim::Hold;
    if (life == ContextLife::Gone)
        return Reclaim::Release;
    if (life == ContextLife::Unknown)
        return Reclaim::Error;
    const Completion result = query();
    return result == Completion::Complete   ? Reclaim::Release
           : result == Completion::NotReady ? Reclaim::Hold
                                            : Reclaim::Error;
}

// A last-reference cleanup failure retains the ENTIRE owner, including upload host bytes.
// Retry is explicit/noncapturing. Never destroyed at static exit after driver teardown.
template <class T> class Retirement
{
    std::mutex mutex;
    std::vector<T *> held;
    std::atomic<std::size_t> count{0};

  public:
    void hold(T *value)
    {
        std::lock_guard<std::mutex> lock(mutex);
        held.push_back(value);
        count.store(held.size());
    }
    void dispose(T *value)
    {
        if (value->release())
        {
            delete value;
            return;
        }
        hold(value);
    }
    void retry(bool capturing)
    {
        if (capturing || count.load() == 0)
            return;
        std::vector<T *> retry;
        {
            std::lock_guard<std::mutex> lock(mutex);
            retry.swap(held);
            count.store(0);
        }
        for (T *value : retry)
            dispose(value);
    }
    std::size_t size()
    {
        return count.load();
    }
    bool hasFailure()
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (const T *value : held)
            if (value->failed())
                return true;
        return false;
    }
};

// One deliberately process-lifetime retain per device, shared by all host threads.
// Driver calls are injected to test exact ownership/counts without creating GPU faults.
template <class Context> class PrimaryOwners
{
    std::mutex mutex;
    std::map<int, Context> owned;

  public:
    template <class Retain> int get(int device, Context &context, Retain retain)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = owned.find(device);
        if (it != owned.end())
        {
            context = it->second;
            return 0;
        }
        const int result = retain(context, device);
        if (result == 0)
            owned.emplace(device, context);
        return result;
    }
};
} // namespace cv_triton_runtime
#endif
