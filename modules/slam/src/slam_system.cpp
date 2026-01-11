// SPDX-License-Identifier: Apache-2.0
#include "opencv2/slam/slam_system.hpp"
#include "opencv2/slam/optimizer.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <iomanip>
#include <opencv2/core/quaternion.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <numeric>
#include <unordered_map>

namespace cv {
namespace vo {

struct TrajEntry { int frameId = -1; double ts = 0.0; Mat R; Mat t; };

static cv::Vec4d rotationMatrixToQuaternion_(const cv::Mat& R_in)
{
    cv::Quatd q = cv::Quatd::createFromRotMat(R_in);
    // OpenCV Quat stores [w, x, y, z]; we export [qx, qy, qz, qw].
    return cv::Vec4d(q.at(1), q.at(2), q.at(3), q.at(0));
}

class SlamSystem::Impl {
public:
    Impl(Ptr<Feature2D> det, Ptr<DescriptorMatcher> matcher)
        : frontend_(std::move(det), std::move(matcher)) {}
    ~Impl(){ stopBackend(); }

    void setFrontendOptions(const VisualOdometryOptions& o){ feOpts_ = o; frontend_.setOptions(o); }
    void setSystemOptions(const SlamSystemOptions& o){
        sysOpts_ = o;
        map_.setMapPointCullingParams(sysOpts_.mapMinObservations, sysOpts_.mapMinFoundRatio,
                                      sysOpts_.mapMaxReprojErrorPx, sysOpts_.mapMaxPointsKeep);
        map_.setRedundantKeyframeCullingParams(sysOpts_.redundantKeyframeMinObs,
                                               sysOpts_.redundantKeyframeMinPointObs,
                                               sysOpts_.redundantKeyframeRatio);
    }
    VisualOdometryOptions frontendOptions() const { return feOpts_; }
    SlamSystemOptions systemOptions() const { return sysOpts_; }

    void setIntrinsics(double fx, double fy, double cx, double cy){
        fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
        frontend_.setCameraIntrinsics(fx, fy, cx, cy);
        map_.setCameraIntrinsics(fx, fy, cx, cy);
    }
    void setScale(double s){ scale_ = s; frontend_.setWorldScale(s); }

    void setMode(int mode){
        mode_ = mode;
        if(mode_ == MODE_LOCALIZATION){
            // Localization mode should be lightweight and map-frozen.
            stopBackend();
        }
    }
    int getMode() const { return mode_; }

    TrackingResult track(InputArray frame, double ts){
        const bool isLocalization = (mode_ == MODE_LOCALIZATION);
        if(!isLocalization) ensureBackend();

        TrackingResult res;
        {
            // Protect map_ against concurrent backend snapshots/writeback.
            std::lock_guard<std::mutex> lk(mapMutex_);
            res = frontend_.track(frame, ts, &map_, !isLocalization);
        }

        if(!isLocalization && sysOpts_.enableBackend && res.keyframeInserted){
            kfSinceBackend_++;
            const int interval = std::max(1, sysOpts_.backendTriggerInterval);
            if(kfSinceBackend_ >= interval){
                kfSinceBackend_ = 0;
                backendRequests_.fetch_add(1);
                backendCv_.notify_one();
            }
        }

        if(!isLocalization && sysOpts_.enableMapMaintenance && res.keyframeInserted){
            kfSinceMaintenance_++;
            const int interval = std::max(1, sysOpts_.maintenanceInterval);
            if(kfSinceMaintenance_ >= interval){
                kfSinceMaintenance_ = 0;
                std::lock_guard<std::mutex> lk(mapMutex_);
                map_.cullRedundantKeyFrames(std::max(1, sysOpts_.maxKeyframeCullsPerMaintenance));
                map_.cullBadMapPoints();
            }
        }
        if(res.ok){
            TrajEntry te; te.frameId = res.frameId; te.ts = ts; te.R = res.R_w; te.t = res.t_w;
            trajectory_.push_back(te);
        }
        return res;
    }

    bool saveTrajectoryTUM(const std::string& path) const {
        std::ofstream out(path, std::ios::out);
        if(!out.is_open()) return false;
        out << "timestamp,tx,ty,tz,qx,qy,qz,qw\n";
        for(const auto &te : trajectory_){
            if(te.R.empty() || te.t.empty()) continue;
            cv::Vec4d q = rotationMatrixToQuaternion_(te.R);
            cv::Mat t; te.t.convertTo(t, CV_64F);
            if(t.rows < 3) continue;
            out << std::fixed << std::setprecision(9)
                << te.ts << "," << t.at<double>(0) << "," << t.at<double>(1) << "," << t.at<double>(2) << ","
                << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << "\n";
        }
        return true;
    }

    bool saveMap(const std::string& path) const { return map_.save(path); }
    bool loadMap(const std::string& path){ bool ok = map_.load(path); return ok; }
    void reset(){ stopBackend(); map_.clear(); frontend_.reset(); trajectory_.clear(); }
    const MapManager& map() const { return map_; }
    MapManager& mapMutable() { return map_; }

private:
    void ensureBackend(){
        if(backendStarted_ || !sysOpts_.enableBackend) return;
        backendStop_.store(false);
        backendThread_ = std::thread([this](){ backendLoop(); });
        backendStarted_ = true;
    }

    void stopBackend(){
        if(!backendStarted_) return;
        backendStop_.store(true);
        backendCv_.notify_one();
        if(backendThread_.joinable()) backendThread_.join();
        backendStarted_ = false;
    }

    void backendLoop(){
        while(!backendStop_.load()){
            std::unique_lock<std::mutex> lk(backendMutex_);
            backendCv_.wait(lk, [&]{ return backendStop_.load() || backendRequests_.load() > 0; });
            if(backendStop_.load()) break;
            backendRequests_.store(0);
            backendBusy_ = true;

            std::vector<KeyFrame> kfs;
            std::vector<MapPoint> mps;
            {
                std::lock_guard<std::mutex> mapLk(mapMutex_);
                // Copy snapshots under lock to avoid data races.
                kfs = map_.keyframes();
                mps = map_.mappoints();
            }
            lk.unlock();

            int K = static_cast<int>(kfs.size());
            if(K > 0){
                int lastIdx = K - 1; int lastId = kfs[lastIdx].id;
                std::unordered_map<int,int> idToIdx; idToIdx.reserve(kfs.size());
                for(int i=0;i<K;++i) idToIdx[kfs[i].id] = i;
                std::unordered_map<int,int> shared;
                for(const auto &mp : mps){
                    bool hasLast = false;
                    for(const auto &o : mp.observations){ if(o.first == lastId){ hasLast = true; break; } }
                    if(!hasLast) continue;
                    for(const auto &o : mp.observations){ if(o.first == lastId) continue; auto it = idToIdx.find(o.first); if(it != idToIdx.end()) shared[it->second]++; }
                }
                std::vector<int> localIdx; localIdx.reserve(sysOpts_.backendWindow);
                auto pushU = [&](int idx){ if(idx<0||idx>=K) return; if(idx==0||idx==1) return; if(std::find(localIdx.begin(), localIdx.end(), idx)!=localIdx.end()) return; localIdx.push_back(idx); };
                pushU(lastIdx);
                std::vector<std::pair<int,int>> scored(shared.begin(), shared.end());
                std::sort(scored.begin(), scored.end(), [](auto a, auto b){ return a.second > b.second; });
                for(const auto &p : scored){ if((int)localIdx.size() >= sysOpts_.backendWindow) break; pushU(p.first); }
                if(localIdx.empty()) localIdx.push_back(lastIdx);
                std::vector<int> fixed{0}; if(K>1) fixed.push_back(1);
        #if defined(HAVE_SFM)
                Optimizer::localBundleAdjustmentSFM(kfs, mps, localIdx, fixed, fx_, fy_, cx_, cy_, sysOpts_.backendIterations);
                {
                    std::lock_guard<std::mutex> mapLk(mapMutex_);
                    for(const auto &kf : kfs) map_.applyOptimizedKeyframePose(kf.id, kf.R_w, kf.t_w);
                    for(const auto &mp : mps) if(mp.id>0) map_.applyOptimizedMapPoint(mp.id, mp.p);
                }
        #endif
            }
            lk.lock();
            backendBusy_ = false;
            backendCv_.notify_all();
        }
    }

    VisualOdometry frontend_;
    MapManager map_;
    VisualOdometryOptions feOpts_;
    SlamSystemOptions sysOpts_;
    int mode_ = MODE_SLAM;
    int kfSinceBackend_ = 0;
    int kfSinceMaintenance_ = 0;
    double fx_ = 0.0, fy_ = 0.0, cx_ = 0.0, cy_ = 0.0;
    double scale_ = 1.0;

    std::vector<TrajEntry> trajectory_;

    std::thread backendThread_;
    std::mutex backendMutex_;
    std::condition_variable backendCv_;
    std::atomic<bool> backendStop_{false};
    std::atomic<int> backendRequests_{0};
    bool backendBusy_ = false;
    bool backendStarted_ = false;

    std::mutex mapMutex_;
};

SlamSystem::SlamSystem() : impl_(std::make_unique<Impl>(ORB::create(), BFMatcher::create(NORM_HAMMING))) {}
SlamSystem::SlamSystem(Ptr<Feature2D> det, Ptr<DescriptorMatcher> matcher)
    : impl_(std::make_unique<Impl>(std::move(det), std::move(matcher))) {}
SlamSystem::~SlamSystem() = default;

void SlamSystem::setFrontendOptions(const VisualOdometryOptions& o){ impl_->setFrontendOptions(o); }
void SlamSystem::setSystemOptions(const SlamSystemOptions& o){ impl_->setSystemOptions(o); }
void SlamSystem::setCameraIntrinsics(double fx, double fy, double cx, double cy){ impl_->setIntrinsics(fx, fy, cx, cy); }
void SlamSystem::setWorldScale(double scale_m){ impl_->setScale(scale_m); }
void SlamSystem::setMode(int mode){ impl_->setMode(mode); }
int SlamSystem::getMode() const { return impl_->getMode(); }
VisualOdometryOptions SlamSystem::getFrontendOptions() const { return impl_->frontendOptions(); }
SlamSystemOptions SlamSystem::getSystemOptions() const { return impl_->systemOptions(); }
void SlamSystem::setEnableBackend(bool enable){ auto opts = impl_->systemOptions(); opts.enableBackend = enable; impl_->setSystemOptions(opts); }
void SlamSystem::setBackendWindow(int w){ auto opts = impl_->systemOptions(); opts.backendWindow = std::max(1, w); impl_->setSystemOptions(opts); }
void SlamSystem::setBackendIterations(int it){ auto opts = impl_->systemOptions(); opts.backendIterations = std::max(1, it); impl_->setSystemOptions(opts); }
TrackingResult SlamSystem::track(InputArray frame, double ts){ return impl_->track(frame, ts); }
bool SlamSystem::saveTrajectoryTUM(const std::string& path) const { return impl_->saveTrajectoryTUM(path); }
bool SlamSystem::saveMap(const std::string& path) const { return impl_->saveMap(path); }
bool SlamSystem::loadMap(const std::string& path){ return impl_->loadMap(path); }
void SlamSystem::reset(){ impl_->reset(); }
const MapManager& SlamSystem::getMap() const { return impl_->map(); }
MapManager& SlamSystem::getMapMutable(){ return impl_->mapMutable(); }

} // namespace vo
} // namespace cv
