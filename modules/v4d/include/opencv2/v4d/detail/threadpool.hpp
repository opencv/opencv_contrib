/*
 * Simple header-only thread pool implementation in modern C++.
 *
 * Created:     Aug 9, 2020.
 * Repository:  https://github.com/leiless/threadpool.hpp
 * LICENSE:     BSD-2-Clause
 */

#ifndef THE_THREADPOOL_HPP
#define THE_THREADPOOL_HPP

#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <thread>
#include <future>

#define THE_NAMESPACE_BEGIN(name)       namespace name {
#define THE_NAMESPACE_END()             }

THE_NAMESPACE_BEGIN(concurrent)

class threadpool {
public:
    explicit threadpool(size_t threads) : alive(true) {
        if (threads == 0) {
            throw std::runtime_error("thread pool size cannot be zero");
        }

        for (auto i = 0llu; i < threads; i++) {
            workers.emplace_back([this] { worker_main(); });
        }
    }

    // see: https://stackoverflow.com/a/23771245/13600780
    threadpool(const threadpool &) = delete;
    threadpool & operator=(const threadpool &) = delete;

    ~threadpool() noexcept {
        {
            std::lock_guard<decltype(mtx)> lock(mtx);
            alive = false;
        }

        cv.notify_all();

        for (auto & worker : workers) {
            worker.join();
        }
    }

    template<typename Fn, typename... Args>
    decltype(auto) enqueue(Fn && fn, Args &&... args) {
        return enqueue(false, fn, args...);
    }

    template<typename Fn, typename... Args>
    decltype(auto) enqueue_r(Fn && fn, Args &&... args) {
        return enqueue(true, fn, args...);
    }

private:
    template<typename Fn, typename... Args>
    decltype(auto) enqueue(bool block_on_shutdown, Fn && fn, Args &&... args) {
        using return_type = std::invoke_result_t<Fn, Args...>;
        using pack_task = std::packaged_task<return_type()>;

        auto t = std::make_shared<pack_task>(
                std::bind(std::forward<Fn>(fn), std::forward<Args>(args)...)
        );
        auto future = t->get_future();

        {
            std::lock_guard<decltype(mtx)> lock(mtx);
            if (!alive) {
                throw std::runtime_error("enqueue on stopped thread pool");
            }
            tasks.emplace([t = std::move(t)] { (*t)(); }, block_on_shutdown);
        }

        cv.notify_one();
        return future;
    }

    using task = std::pair<std::function<void()>, bool>;

    [[nodiscard]] inline task poll_task() noexcept {
        task t;

        std::unique_lock<decltype(mtx)> lock(mtx);
        cv.wait(lock, [this] { return !tasks.empty() || !alive; });

        while (!tasks.empty()) {
            if (!alive && !tasks.front().second) {
                tasks.pop();
                continue;
            }

            t = std::move(tasks.front());
            tasks.pop();
            break;
        }

        return t;
   }

    void worker_main() {
        while (true) {
            task t = poll_task();
            // The thread pool is going to shutdown
            if (t.first == nullptr) break;
            t.first();
        }
    }

    bool alive;
    std::mutex mtx;
    std::condition_variable cv;
    std::queue<task> tasks;
    std::vector<std::thread> workers;
};

THE_NAMESPACE_END()

#endif

