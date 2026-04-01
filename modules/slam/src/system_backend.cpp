/**
 * @file system_backend.cpp
 * @brief Backend control implementation (simple flag-based)
 */

#include "system.hpp"
#include "mapping_module.hpp"
#include "global_optimization_module.hpp"

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam_module", cv::utils::logging::LOG_LEVEL_INFO);

void system::set_enable_backend(bool enable, int window_size) {
    CV_LOG_INFO(&g_log_tag, "Setting backend enabled: " << enable << " (window_size=" << window_size << ")");
    backend_enabled_ = enable;
    
    
}

bool system::backend_is_enabled() const {
    return backend_enabled_;
}

bool system::backend_is_running() const {
    return mapper_ && !mapper_->is_paused();
}

void system::set_ba_window_size(int size) {
    CV_LOG_INFO(&g_log_tag, "Setting BA window size to: " << size);
}

void system::set_enable_loop_closure(bool enable) {
    CV_LOG_INFO(&g_log_tag, "Setting loop closure enabled: " << enable);
    loop_closure_enabled_ = enable;
    
    
    if (global_optimizer_) {
        if (enable) {
            CV_LOG_INFO(&g_log_tag, "Enabling loop detector");
            global_optimizer_->enable_loop_detector();
        } else {
            CV_LOG_INFO(&g_log_tag, "Disabling loop detector");
            global_optimizer_->disable_loop_detector();
        }
    }
}

bool system::loop_closure_is_enabled() const {
    return loop_closure_enabled_;
}

bool system::loop_closure_is_running() const {
    return global_optimizer_ && !global_optimizer_->is_paused();
}

bool system::should_skip_backend() const {
    return !backend_enabled_;
}

bool system::should_skip_loop_closure() const {
    return !loop_closure_enabled_;
}

} // namespace cv::slam
