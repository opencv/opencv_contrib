//started with https://stackoverflow.com/a/51400041/1884837
#ifndef SRC_COMMON_FUNCTIONPOOL_HPP_
#define SRC_COMMON_FUNCTIONPOOL_HPP_

#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cassert>

namespace kb {
namespace viz2d {
namespace detail {

class FunctionPool {

private:
    std::queue<std::function<void()>> m_function_queue;
    std::mutex m_lock;
    std::condition_variable m_data_condition;
    std::atomic<bool> m_accept_functions;

public:

    FunctionPool();
    ~FunctionPool();
    void push(std::function<void()> func);
    void done();
    void infinite_loop_func();
};

} /* namespace detail */
} /* namespace viz2d */
} /* namespace kb */

#endif /* SRC_COMMON_FUNCTIONPOOL_HPP_ */
