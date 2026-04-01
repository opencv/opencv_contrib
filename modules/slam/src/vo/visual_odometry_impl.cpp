#include "vo/visual_odometry_impl.hpp"
#include "config.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <fstream>

namespace cv::vo {

static cv::utils::logging::LogTag g_log_tag("cv_vo", cv::utils::logging::LOG_LEVEL_INFO);

// ============================================================================

// ============================================================================

cv::Ptr<VisualOdometry> VisualOdometry::create(
    const VOConfig& config,
    const cv::Ptr<cv::Feature2D>& feature_detector,
    const cv::Ptr<cv::DescriptorMatcher>& matcher)
{
    auto vo = cv::makePtr<VisualOdometryImpl>(config);
    vo->setFeatureDetector(feature_detector);
    vo->setMatcher(matcher);
    return vo;
}

cv::Ptr<VisualOdometry> VisualOdometry::create(
    const std::string& config_file,
    const std::string& vocab_file)
{
    return cv::makePtr<VisualOdometryImpl>(config_file, vocab_file);
}

// ============================================================================

// ============================================================================

VisualOdometryImpl::VisualOdometryImpl(
    const std::string& config_file, 
    const std::string& vocab_file)
{
    initialize(config_file, vocab_file);
}

VisualOdometryImpl::VisualOdometryImpl(const VOConfig& config)
{
    config_ = config;
    std::string config_file = config.camera_config_file;
    std::string vocab_file = config.vocab_file;  
    
    initialize(config_file, vocab_file);
}

VisualOdometryImpl::~VisualOdometryImpl()
{
    if (!shutdown_) {
        release();
    }
}

// ============================================================================

// ============================================================================

void VisualOdometryImpl::initialize(
    const std::string& config_file, 
    const std::string& vocab_file)
{
    if (initialized_) {
        return;
    }
    
    try {
        vocab_file_ = vocab_file;
        
        
        auto cfg = std::make_shared<cv::slam::config>(config_file);
        
        
        system_ = std::make_shared<cv::slam::system>(cfg, vocab_file);
        
        
        system_->startup();
        
        initialized_ = true;
        state_ = VOState::NotInitialized;
        
        CV_LOG_INFO(&g_log_tag, "VisualOdometry initialized successfully");
    }
    catch (const std::exception& e) {
        CV_LOG_ERROR(&g_log_tag, "Failed to initialize VisualOdometry: " << e.what());
        throw;
    }
}

// ============================================================================

// ============================================================================

std::optional<cv::Matx44d> VisualOdometryImpl::processFrame(
    const cv::Mat& image, 
    double timestamp)
{
    if (!initialized_ || shutdown_) {
        return std::nullopt;
    }
    
    try {
        
        auto pose_ptr = system_->feed_monocular_frame(image, timestamp, cv::Mat());
        
        if (pose_ptr) {
            
            cv::Matx44d pose_cv;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    pose_cv(i, j) = (*pose_ptr)(i, j);
                }
            }
            
            
            {
                std::lock_guard<std::mutex> lock(trajectory_mutex_);
                trajectory_.emplace_back(timestamp, pose_cv);
            }
            
            
            if (system_->tracker_is_paused()) {
                state_ = VOState::Initializing;
            }
            else {
                state_ = VOState::Tracking;
            }
            
            return pose_cv;
        }
        
        return std::nullopt;
    }
    catch (const std::exception& e) {
        CV_LOG_ERROR(&g_log_tag, "Error processing frame: " << e.what());
        return std::nullopt;
    }
}

// ============================================================================

// ============================================================================

VOState VisualOdometryImpl::getState() const
{
    return state_;
}

bool VisualOdometryImpl::isInitialized() const
{
    return state_ != VOState::NotInitialized && state_ != VOState::Initializing;
}

bool VisualOdometryImpl::isEmpty() const
{
    return state_ == VOState::Lost;
}

// ============================================================================

// ============================================================================

std::vector<cv::Point3d> VisualOdometryImpl::getMapPoints() const
{
    
    
    return {};
}

std::vector<cv::KeyPoint> VisualOdometryImpl::getCurrentKeypoints() const
{
    
    return {};
}

cv::Matx44d VisualOdometryImpl::getCurrentPose() const
{
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    if (trajectory_.empty()) {
        return cv::Matx44d::eye();
    }
    return trajectory_.back().second;
}

// ============================================================================

// ============================================================================

std::vector<std::pair<double, cv::Matx44d>> VisualOdometryImpl::getTrajectory() const
{
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    return trajectory_;
}

bool VisualOdometryImpl::saveTrajectory(const std::string& path, const std::string& format)
{
    std::lock_guard<std::mutex> lock(trajectory_mutex_);
    
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        CV_LOG_ERROR(&g_log_tag, "Cannot open file: " << path);
        return false;
    }
    
    if (format == "TUM") {
        
        for (const auto& [timestamp, pose] : trajectory_) {
            
            double tx = pose(0, 3);
            double ty = pose(1, 3);
            double tz = pose(2, 3);
            
            
            cv::Matx33d R;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    R(i, j) = pose(i, j);
                }
            }
            
            
            double w = std::sqrt(1.0 + R(0, 0) + R(1, 1) + R(2, 2)) / 2.0;
            double x = (R(2, 1) - R(1, 2)) / (4.0 * w);
            double y = (R(0, 2) - R(2, 0)) / (4.0 * w);
            double z = (R(1, 0) - R(0, 1)) / (4.0 * w);
            
            ofs << std::fixed << timestamp << " "
                << tx << " " << ty << " " << tz << " "
                << x << " " << y << " " << z << " " << w << std::endl;
        }
    }
    else {
        
        for (const auto& [timestamp, pose] : trajectory_) {
            ofs << std::fixed << timestamp;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    ofs << " " << pose(i, j);
                }
            }
            ofs << std::endl;
        }
    }
    
    CV_LOG_INFO(&g_log_tag, "Trajectory saved to: " << path);
    return true;
}

// ============================================================================

// ============================================================================

void VisualOdometryImpl::reset()
{
    
    
    {
        std::lock_guard<std::mutex> lock(trajectory_mutex_);
        trajectory_.clear();
    }
    
    state_ = VOState::NotInitialized;
}

bool VisualOdometryImpl::saveMap(const std::string& path)
{
    if (!system_) {
        CV_LOG_ERROR(&g_log_tag, "System not initialized");
        return false;
    }
    return system_->save_map_database(path);
}

bool VisualOdometryImpl::loadMap(const std::string& path)
{
    if (!system_) {
        CV_LOG_ERROR(&g_log_tag, "System not initialized");
        return false;
    }
    return system_->load_map_database(path);
}

void VisualOdometryImpl::release()
{
    if (shutdown_) {
        return;
    }
    
    shutdown_ = true;
    
    if (system_) {
        system_->shutdown();
        system_.reset();
    }
    
    CV_LOG_INFO(&g_log_tag, "VisualOdometry released");
}

// ============================================================================

// ============================================================================

void VisualOdometryImpl::setMode(cv::vo::SLAMMode mode)
{
    mode_ = mode;
    
    CV_LOG_INFO(&g_log_tag, "SLAM mode changed to: " << (mode == static_cast<cv::vo::SLAMMode>(0) ? "SLAM" : "LOCALIZATION"));
    
    if (system_) {
        if (mode == static_cast<cv::vo::SLAMMode>(1)) {
            
            system_->disable_mapping_module();
            system_->set_allow_initialization(false);
            CV_LOG_INFO(&g_log_tag, "Mapping module disabled (LOCALIZATION mode)");
        } else {
            
            system_->enable_mapping_module();
            system_->set_allow_initialization(true);
            CV_LOG_INFO(&g_log_tag, "Mapping module enabled (SLAM mode)");
        }
    }
}

cv::vo::SLAMMode VisualOdometryImpl::getMode() const
{
    return mode_;
}

// ============================================================================

// ============================================================================

void VisualOdometryImpl::setBackendEnabled(bool enable, int window_size)
{
    backend_enabled_ = enable;
    ba_window_size_ = window_size;
    
    if (system_) {
        system_->set_enable_backend(enable, window_size);
    }
}

bool VisualOdometryImpl::isBackendEnabled() const
{
    return backend_enabled_;
}

void VisualOdometryImpl::setLoopClosureEnabled(bool enable)
{
    loop_closure_enabled_ = enable;
    
    if (system_) {
        system_->set_enable_loop_closure(enable);
    }
}

bool VisualOdometryImpl::isLoopClosureEnabled() const
{
    return loop_closure_enabled_;
}
void VisualOdometryImpl::setFeatureDetector(const cv::Ptr<cv::Feature2D>& detector)
{
    feature_detector_ = detector;
    if (system_) system_->set_feature_detector(detector);
}

void VisualOdometryImpl::setMatcher(const cv::Ptr<cv::DescriptorMatcher>& m)
{
    matcher_ = m;
}

// ============================================================================

// ============================================================================

void VisualOdometryImpl::enableMappingModule(bool enable)
{
    if (!system_) {
        return;
    }
    
    if (enable) {
        system_->enable_mapping_module();
            system_->set_allow_initialization(true);
    } else {
        system_->disable_mapping_module();
            system_->set_allow_initialization(false);
    }
}

bool VisualOdometryImpl::isMappingModuleEnabled() const
{
    return system_ && system_->mapping_module_is_enabled();
}

void VisualOdometryImpl::enableLoopDetector(bool enable)
{
    if (!system_) {
        return;
    }
    
    if (enable) {
        system_->enable_loop_detector();
    } else {
        system_->disable_loop_detector();
    }
}

bool VisualOdometryImpl::isLoopDetectorEnabled() const
{
    return system_ && system_->loop_detector_is_enabled();
}

} // namespace cv::vo
