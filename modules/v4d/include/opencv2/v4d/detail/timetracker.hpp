#ifndef TIME_TRACKER_HPP_
#define TIME_TRACKER_HPP_

#include <chrono>
#include <map>
#include <string>
#include <sstream>
#include <ostream>
#include <limits>
#include <mutex>
#include <opencv2/core/cvdef.h>

using std::ostream;
using std::stringstream;
using std::string;
using std::map;
using std::chrono::microseconds;
using std::mutex;

struct CV_EXPORTS TimeInfo {
    long totalCnt_ = 0;
    long totalTime_ = 0;
    long iterCnt_ = 0;
    long iterTime_ = 0;
    long last_ = 0;

    void add(size_t t) {
        last_ = t;
        totalTime_ += t;
        iterTime_ += t;
        ++totalCnt_;
        ++iterCnt_;

        if (totalCnt_ == std::numeric_limits<long>::max() || totalTime_ == std::numeric_limits<long>::max()) {
            totalCnt_ = 0;
            totalTime_ = 0;
        }

        if (iterCnt_ == std::numeric_limits<long>::max() || iterTime_ == std::numeric_limits<long>::max()) {
            iterCnt_ = 0;
            iterTime_ = 0;
        }
    }

    void newCount() {
        iterCnt_ = 0;
        iterTime_ = 0;
    }

    string str() const {
        stringstream ss;
        ss << (totalTime_ / 1000.0) / totalCnt_ << "ms = (" << totalTime_ / 1000.0 << '\\' << totalCnt_  << ")\t";
        ss << (iterTime_ / 1000.0) / iterCnt_ << "ms = (" << iterTime_  / 1000.0 << '\\' << iterCnt_ << ")\t";
        return ss.str();
    }
};

inline std::ostream& operator<<(ostream &os, TimeInfo &ti) {
    os << (ti.totalTime_ / 1000.0) / ti.totalCnt_ << "ms = (" << ti.totalTime_ / 1000.0 << '\\' << ti.totalCnt_  << ")\t";
    os << (ti.iterTime_ / 1000.0) / ti.iterCnt_ << "ms = (" << ti.iterTime_  / 1000.0 << '\\' << ti.iterCnt_ << ")";
    return os;
}

class CV_EXPORTS TimeTracker {
private:
    static TimeTracker *instance_;
    mutex mapMtx_;
    map<string, TimeInfo> tiMap_;
    bool enabled_;
    TimeTracker();
public:
    virtual ~TimeTracker();

    map<string, TimeInfo>& getMap() {
        return tiMap_;
    }

    template<typename F> void execute(const string &name, F const &func) {
        auto start = std::chrono::system_clock::now();
        func();
        auto duration = std::chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start);
        std::unique_lock lock(mapMtx_);
        tiMap_[name].add(duration.count());
    }

    template<typename F> size_t measure(F const &func) {
        auto start = std::chrono::system_clock::now();
        func();
        auto duration = std::chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start);
        return duration.count();
    }

    bool isEnabled() {
        return enabled_;
    }

    void setEnabled(bool e) {
        enabled_ = e;
    }

    void print(ostream &os) {
        std::unique_lock lock(mapMtx_);
        stringstream ss;
        ss << "Time tracking info: " << std::endl;
        for (auto it : tiMap_) {
            ss << "\t" << it.first << ": " << it.second << std::endl;
        }

        os << ss.str();
    }

    void reset() {
        std::unique_lock lock(mapMtx_);
        tiMap_.clear();
    }

    static TimeTracker* getInstance() {
        if (instance_ == NULL)
            instance_ = new TimeTracker();

        return instance_;
    }

    static void destroy() {
        if (instance_)
            delete instance_;

        instance_ = NULL;
    }

    void newCount() {
        std::unique_lock lock(mapMtx_);
        for (auto& pair : getMap()) {
            pair.second.newCount();
        }
    }
};

#endif /* TIME_TRACKER_HPP_ */
