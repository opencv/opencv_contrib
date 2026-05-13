#ifndef SLAM_SYSTEM_H
#define SLAM_SYSTEM_H

#include "type.hpp"
#include "data/bow_vocabulary_fwd.hpp"

#include <string>
#include <thread>
#include <memory>
#include <mutex>
#include <atomic>
#include <memory>

#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace cv::slam {

class config;
class tracking_module;
class mapping_module;
class global_optimization_module;

namespace camera {
class base;
} // namespace camera

namespace data {
class frame;
class camera_database;
class orb_params_database;
class map_database;
class bow_database;
} // namespace data

namespace feature {
class orb_extractor;
struct orb_params;
} // namespace feature

namespace io {
class map_database_io_base;
}

class system {
public:
    //! Constructor
    system(const std::shared_ptr<config>& cfg, const std::string& vocab_file_path);

    //! Destructor
    ~system();

    //-----------------------------------------
    // system startup and shutdown

    //! Print system information
    void print_info();

    //! Startup the SLAM system
    void startup(const bool need_initialize = true);

    //! Shutdown the SLAM system
    void shutdown();

    //-----------------------------------------
    // data I/O

    //! Save the frame trajectory in the specified format
    void save_frame_trajectory(const std::string& path, const std::string& format) const;

    //! Save the keyframe trajectory in the specified format
    void save_keyframe_trajectory(const std::string& path, const std::string& format) const;

    //! Load the map database from file
    bool load_map_database(const std::string& path) const;

    //! Save the map database to file
    bool save_map_database(const std::string& path) const;

    //-----------------------------------------
    // module management

    //! Enable the mapping module
    void enable_mapping_module();

    //! Disable the mapping module
    void disable_mapping_module();

    //! The mapping module is enabled or not
    bool mapping_module_is_enabled() const;

    //! Enable the loop detector
    void enable_loop_detector();

    //! Disable the loop detector
    void disable_loop_detector();

    //! The loop detector is enabled or not
    bool loop_detector_is_enabled() const;

    //! Request loop closure
    bool request_loop_closure(int keyfrm1_id, int keyfrm2_id);

    //! Loop BA is running or not
    bool loop_BA_is_running() const;

    //! Abort the loop BA externally
    void abort_loop_BA();

    //! Enable temporal mapping
    void enable_temporal_mapping();

    //-----------------------------------------
    // data feeding methods

    std::shared_ptr<Mat44_t> feed_frame(const data::frame& frm, const cv::Mat& img, double extraction_time_elapsed_ms);

    //! Feed a monocular frame to SLAM system
    //! (NOTE: distorted images are acceptable if calibrated)
    data::frame create_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask = cv::Mat{});
    std::shared_ptr<Mat44_t> feed_monocular_frame(const cv::Mat& img, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //! Feed a stereo frame to SLAM system
    //! (Note: Left and Right images must be stereo-rectified)
    data::frame create_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask = cv::Mat{});
    std::shared_ptr<Mat44_t> feed_stereo_frame(const cv::Mat& left_img, const cv::Mat& right_img, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //! Feed an RGBD frame to SLAM system
    //! (Note: RGB and Depth images must be aligned)
    data::frame create_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask);
    std::shared_ptr<Mat44_t> feed_RGBD_frame(const cv::Mat& rgb_img, const cv::Mat& depthmap, const double timestamp, const cv::Mat& mask = cv::Mat{});

    //-----------------------------------------
    // pose initializing/updating

    //! Request to update the pose to a given one.
    //! Return failure in case if previous request was not finished.
    bool relocalize_by_pose(const Mat44_t& cam_pose_wc);
    bool relocalize_by_pose_2d(const Mat44_t& cam_pose_wc, const Vec3_t& normal_vector);

    //-----------------------------------------
    // management for pause

    //! Pause the tracking module
    void pause_tracker();

    //! The tracking module is paused or not
    bool tracker_is_paused() const;

    //! Set whether initialization is allowed (false for LOCALIZATION mode)
    void set_allow_initialization(bool allow);

    //! Resume the tracking module
    void resume_tracker();

    //-----------------------------------------
    // management for reset

    //! Request to reset the system
    void request_reset();

    //! Reset of the system is requested or not
    bool reset_is_requested() const;

    //-----------------------------------------
    // management for terminate

    //! Request to terminate the system
    void request_terminate();

    //!! Termination of the system is requested or not
    bool terminate_is_requested() const;

    //-----------------------------------------
    // Phase 2: External feature detector support

    //! Set an external feature detector (e.g., SIFT, AKAZE, BRISK)
    //! If set, this will be used instead of the internal ORB extractor
    //! Call this BEFORE startup() for best results
    void set_feature_detector(const cv::Ptr<cv::Feature2D>& detector);

    //! Get the current external feature detector (nullptr if using internal ORB)
    cv::Ptr<cv::Feature2D> get_feature_detector() const;

    //-----------------------------------------
    // Phase 3: Backend control

    //! Enable/disable local Bundle Adjustment
    void set_enable_backend(bool enable, int window_size = 10);

    //! Check if backend is enabled
    bool backend_is_enabled() const;
    //! Check if backend is actually running
    bool backend_is_running() const;

    //! Set BA window size
    void set_ba_window_size(int size);


    //! Check if loop closure is actually running
    bool loop_closure_is_running() const;


    //! Enable/disable loop closure detection
    void set_enable_loop_closure(bool enable);

    //! Check if loop closure is enabled
    bool loop_closure_is_enabled() const;

    //-----------------------------------------
    // config

    camera::base* get_camera() const;

    //! depthmap factor (pixel_value / depthmap_factor = true_depth)
    double depthmap_factor_ = 1.0;

private:
    //! Check reset request of the system
    void check_reset_request();

    //! Pause the mapping module and the global optimization module
    void pause_other_threads() const;

    //! Resume the mapping module and the global optimization module
    void resume_other_threads() const;

    //! config
    const std::shared_ptr<config> cfg_;
    //! camera model
    camera::base* camera_ = nullptr;

    //! camera database
    data::camera_database* cam_db_ = nullptr;

    //! parameters for orb feature extraction
    feature::orb_params* orb_params_ = nullptr;

    //! orb_params database
    data::orb_params_database* orb_params_db_ = nullptr;

    //! map database
    data::map_database* map_db_ = nullptr;

    //! BoW vocabulary
    data::bow_vocabulary* bow_vocab_ = nullptr;

    //! BoW database
    data::bow_database* bow_db_ = nullptr;

    //! tracker
    tracking_module* tracker_ = nullptr;

    //! mapping module
    mapping_module* mapper_ = nullptr;
    //! mapping thread
    std::unique_ptr<std::thread> mapping_thread_ = nullptr;

    //! next frame ID
    std::atomic<unsigned int> next_frame_id_{0};

    //! global optimization module
    global_optimization_module* global_optimizer_ = nullptr;
    //! global optimization thread
    std::unique_ptr<std::thread> global_optimization_thread_ = nullptr;

    // ORB extractors
    //! ORB extractor for left/monocular image
    feature::orb_extractor* extractor_left_ = nullptr;
    //! ORB extractor for right image
    feature::orb_extractor* extractor_right_ = nullptr;
    //! ORB extractor only when used in initializing
    feature::orb_extractor* ini_extractor_left_ = nullptr;

    //! number of columns of grid to accelerate reprojection matching
    unsigned int num_grid_cols_ = 64;
    //! number of rows of grid to accelerate reprojection matching
    unsigned int num_grid_rows_ = 48;

    //! map I/O
    std::shared_ptr<io::map_database_io_base> map_database_io_ = nullptr;

    //! system running status flag
    std::atomic<bool> system_is_running_{false};

    //! mutex for reset flag
    mutable std::mutex mtx_reset_;
    //! reset flag
    bool reset_is_requested_ = false;

    //! mutex for terminate flag
    mutable std::mutex mtx_terminate_;
    //! terminate flag
    bool terminate_is_requested_ = false;

    //! mutex for flags of enable/disable mapping module
    mutable std::mutex mtx_mapping_;

    //! mutex for flags of enable/disable loop detector
    mutable std::mutex mtx_loop_detector_;

    //! Temporary variables for visualization
    std::vector<cv::KeyPoint> keypts_;

    // Phase 2: External feature detector support
    //! External feature detector (if set, used instead of internal orb_extractor)
    cv::Ptr<cv::Feature2D> external_feature_detector_ = nullptr;

    // Phase 3: Backend control flags
    //! Flag to enable/disable local BA
    bool backend_enabled_ = true;
    //! Flag to enable/disable loop closure
    bool loop_closure_enabled_ = true;
};

} // namespace cv::slam

#endif // SLAM_SYSTEM_H
