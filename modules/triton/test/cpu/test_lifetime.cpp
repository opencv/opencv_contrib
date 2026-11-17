// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright 2026, The FlagOS Contributors.
// CPU-only decisions and ownership controls using injected callbacks, not real GPU faults.
// Run through tools/check_source.py; driver-bound code is checked separately by
// check_native_lifetime.py.
#include "lifetime.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
using namespace cv_triton_runtime;
struct Owner
{
    static int deleted;
    Completion done = Completion::Error;
    int calls = 0;
    bool failed() const
    {
        return done == Completion::Error;
    }
    bool release()
    {
        ++calls;
        return done == Completion::Complete;
    }
    ~Owner()
    {
        ++deleted;
    }
};
int Owner::deleted = 0;
int main()
{
    int query = 0;
    for (bool capture : {false, true})
        for (auto life : {ContextLife::Alive, ContextLife::Gone, ContextLife::Unknown})
            for (auto result : {Completion::Complete, Completion::NotReady, Completion::Error})
            {
                int before = query;
                const auto decision = completionDecision(capture, life,
                                                         [&]()
                                                         {
                                                             ++query;
                                                             return result;
                                                         });
                const int expectedQueries = !capture && life == ContextLife::Alive ? 1 : 0;
                const auto expected = capture                     ? Reclaim::Hold
                                      : life == ContextLife::Gone ? Reclaim::Release
                                      : life == ContextLife::Unknown || result == Completion::Error
                                          ? Reclaim::Error
                                      : result == Completion::Complete ? Reclaim::Release
                                                                       : Reclaim::Hold;
                if (query - before != expectedQueries || decision != expected)
                {
                    const char *contextName = life == ContextLife::Alive  ? "Alive"
                                              : life == ContextLife::Gone ? "Gone"
                                                                          : "Unknown";
                    const char *completionName = result == Completion::Complete   ? "Complete"
                                                 : result == Completion::NotReady ? "NotReady"
                                                                                  : "Error";
                    std::cerr << "completionDecision: capture=" << capture
                              << " context=" << contextName << " completion=" << completionName
                              << " queries=" << query - before
                              << " expectedQueries=" << expectedQueries
                              << " decision=" << static_cast<int>(decision)
                              << " expected=" << static_cast<int>(expected) << '\n';
                }
                assert(query - before == expectedQueries);
                assert(decision == expected);
            }
    int waits = 0, frees = 0, destroys = 0;
    bool waitSucceeds = false;
    auto wait = [&]()
    {
        ++waits;
        return waitSucceeds;
    };
    auto free = [&]()
    {
        ++frees;
        return true;
    };
    auto destroy = [&]()
    {
        ++destroys;
        return true;
    };
    for (bool success : {false, true})
    {
        waitSucceeds = success;
        assert(!releaseUpload(
            false, ContextLife::Alive, true, true,
            []()
            {
                return Completion::NotReady;
            },
            wait, free, destroy));
    }
    assert(waits == 2 && frees == 0 && destroys == 0);
    assert(!releaseUpload(
        false, ContextLife::Alive, true, true,
        []()
        {
            return Completion::Error;
        },
        wait, free, destroy));
    assert(!releaseUpload(
        false, ContextLife::Unknown, true, true,
        [&]()
        {
            ++query;
            return Completion::Complete;
        },
        wait, free, destroy));
    const int before = query;
    assert(releaseUpload(
        false, ContextLife::Gone, true, true,
        [&]()
        {
            ++query;
            return Completion::Error;
        },
        wait, free, destroy));
    assert(!releaseUpload(
        true, ContextLife::Alive, true, true,
        [&]()
        {
            ++query;
            return Completion::Complete;
        },
        wait, free, destroy));
    assert(query == before && frees == 0 && destroys == 0);
    assert(!releaseUpload(
        false, ContextLife::Alive, true, false,
        []()
        {
            assert(false);
            return Completion::Complete;
        },
        wait, free, destroy));
    assert(releaseUpload(
        false, ContextLife::Alive, true, true,
        []()
        {
            return Completion::Complete;
        },
        wait, free, destroy));
    assert(frees == 1 && destroys == 1);
    Retirement<Owner> retirement;
    auto *owner = new Owner();
    retirement.dispose(owner);
    assert(retirement.size() == 1 && Owner::deleted == 0 && retirement.hasFailure());
    retirement.retry(true);
    assert(owner->calls == 1); // no query on capture retry
    owner->done = Completion::Complete;
    retirement.retry(false);
    assert(retirement.size() == 0 && Owner::deleted == 1);
    retirement.retry(false);
    assert(Owner::deleted == 1);
    auto *waiting = new Owner();
    waiting->done = Completion::NotReady;
    retirement.dispose(waiting);
    assert(!retirement.hasFailure()); // old errors cannot poison unrelated pending uploads
    waiting->done = Completion::Complete;
    retirement.retry(false);
    PrimaryOwners<int> primaries;
    std::atomic<int> retained{0};
    auto retain = [&](int &context, int device)
    {
        ++retained;
        context = device + 100;
        return 0;
    };
    int context = 0;
    assert(primaries.get(0, context, retain) == 0 && context == 100);
    assert(primaries.get(0, context, retain) == 0 && retained == 1);
    std::vector<std::thread> threads;
    for (int i = 0; i < 64; ++i)
        threads.emplace_back(
            [&]()
            {
                int c = 0;
                assert(primaries.get(0, c, retain) == 0 && c == 100);
            });
    for (auto &thread : threads)
        thread.join();
    assert(retained == 1);
    assert(primaries.get(1, context, retain) == 0 && context == 101 && retained == 2);
    // SetCurrent failure is outside acquisition: ownership survives it, no extra retain.
    int setFailures = 0;
    for (int i = 0; i < 5; ++i)
    {
        assert(primaries.get(0, context, retain) == 0);
        ++setFailures;
    }
    assert(setFailures == 5 && retained == 2);
    PrimaryOwners<int> retry;
    int attempts = 0;
    auto failOnce = [&](int &c, int)
    {
        c = 99;
        return ++attempts == 1 ? 7 : 0;
    };
    assert(retry.get(0, context, failOnce) == 7);
    assert(retry.get(0, context, failOnce) == 0 && attempts == 2);
    std::cout
        << "PASS: event tri-state, capture-no-query, wait failure, whole-owner retention, destroyed-context and primary ownership controls\n";
}
