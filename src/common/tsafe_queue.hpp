//started with https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
#ifndef SAFE_QUEUE
#define SAFE_QUEUE

#include <queue>
#include <mutex>
#include <condition_variable>

// A threadsafe-queue.
template<class T>
class SafeQueue {
public:
    SafeQueue(size_t maxSize) :
            q(), m(), cempty(), cfull(), maxSize(maxSize) {
    }

    ~SafeQueue() {
    }

    // Add an element to the queue.
    void enqueue(T t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.size() >= maxSize) {
            cfull.wait(lock);
        }
        q.push(t);
        cempty.notify_one();
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue() {
        cfull.notify_one();
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            // release lock as long as the wait and reaquire it afterwards.
            cempty.wait(lock);
        }
        T val = q.front();
        q.pop();
        return val;
    }
private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable cempty;
    std::condition_variable cfull;
    int maxSize;
};
#endif
