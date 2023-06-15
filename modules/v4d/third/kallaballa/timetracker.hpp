#ifndef TIME_TRACKER_HPP_
#define TIME_TRACKER_HPP_

#include <chrono>
#include <map>
#include <string>
#include <sstream>
#include <ostream>
#include <limits>
#include <mutex>

using std::ostream;
using std::stringstream;
using std::string;
using std::map;
using std::chrono::microseconds;
using std::mutex;

struct TimeInfo {
    long totalCnt_ = 0;
    long totalTime_ = 0;
    long gameCnt_ = 0;
    long gameTime_ = 0;
    long last_ = 0;
    map<string, TimeInfo> children_;

    void add(size_t t) {
        last_ = t;
        totalTime_ += t;
        gameTime_ += t;
        ++totalCnt_;
        ++gameCnt_;

        if (totalCnt_ == std::numeric_limits<long>::max() || totalTime_ == std::numeric_limits<long>::max()) {
            totalCnt_ = 0;
            totalTime_ = 0;
        }

        if (gameCnt_ == std::numeric_limits<long>::max() || gameTime_ == std::numeric_limits<long>::max()) {
            gameCnt_ = 0;
            gameTime_ = 0;
        }
    }

    void newCount() {
        gameCnt_ = 0;
        gameTime_ = 0;
    }

    string str() const {
        stringstream ss;
        ss << (totalTime_ / 1000.0) / totalCnt_ << "ms = (" << totalTime_ / 1000.0 << '\\' << totalCnt_  << ")\t";
        ss << (gameTime_ / 1000.0) / gameCnt_ << "ms = (" << gameTime_  / 1000.0 << '\\' << gameCnt_ << ")\t";
        return ss.str();
    }
};

inline std::ostream& operator<<(ostream &os, TimeInfo &ti) {
    os << (ti.totalTime_ / 1000.0) / ti.totalCnt_ << "ms = (" << ti.totalTime_ / 1000.0 << '\\' << ti.totalCnt_  << ")\t";
    os << (ti.gameTime_ / 1000.0) / ti.gameCnt_ << "ms = (" << ti.gameTime_  / 1000.0 << '\\' << ti.gameCnt_ << ")";
    return os;
}

class TimeTracker {
private:
    static TimeTracker *instance_;
    mutex mapMtx_;
    map<string, TimeInfo> tiMap_;
    bool enabled_;
    TimeTracker();

    map<string, TimeInfo>& getMap() {
        return tiMap_;
    }
public:
    virtual ~TimeTracker();

    template<typename F> void execute(const string &name, F const &func) {
        auto start = std::chrono::system_clock::now();
        func();
        auto duration = std::chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start);
        std::unique_lock lock(mapMtx_);
        tiMap_[name].add(duration.count());
    }

    template<typename F> void execute(const string &parentName, const string &name, F const &func) {
        auto start = std::chrono::system_clock::now();
        func();
        auto duration = std::chrono::duration_cast<microseconds>(std::chrono::system_clock::now() - start);
        std::unique_lock lock(mapMtx_);
        tiMap_[parentName].children_[name].add(duration.count());
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
            for (auto itc : it.second.children_) {
                ss << "\t\t" << itc.first << ": " << itc.second << std::endl;
            }
        }
        long totalTime = 0;
        long totalGameTime = 0;
        long totalCnt = 0;
        long gameCnt = 0;
        for (auto& pair : getMap()) {
            totalTime += pair.second.totalTime_;
            totalGameTime += pair.second.gameTime_;
            totalCnt = pair.second.totalCnt_;
            gameCnt = pair.second.gameCnt_;
        }

        ss << std::endl << "FPS: " << (float(totalCnt) / float(totalTime / 1000000.0f)) << " / " << (float(gameCnt) / float(totalGameTime / 1000000.0f)) << std::endl;
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
            for (auto& pairc : pair.second.children_) {
                pairc.second.newCount();
            }
        }
    }
};

#endif /* TIME_TRACKER_HPP_ */
