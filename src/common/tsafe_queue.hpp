//https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
#ifndef SAFE_QUEUE
#define SAFE_QUEUE

#include <queue>
#include <mutex>
#include <condition_variable>

// A threadsafe-queue.
template<class T>
class SafeQueue {
public:
    SafeQueue() :
            q(), m(), c() {
    }

    ~SafeQueue() {
    }

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the "front"-element.
    // If the queue is empty, wait till a element is avaiable.
    T dequeue() {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            // release lock as long as the wait and reaquire it afterwards.
            c.wait(lock);
        }

        T val = q.front();
        q.pop();
        return val;
    }
private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
};
#endif
