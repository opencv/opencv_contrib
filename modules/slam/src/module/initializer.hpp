#ifndef SLAM_MODULE_INITIALIZER_H
#define SLAM_MODULE_INITIALIZER_H

#include "data/frame.hpp"
#include "initialize/base.hpp"
#include "data/bow_vocabulary_fwd.hpp"

#include <memory>

namespace cv::slam {

class config;

namespace data {
class frame;
class map_database;
class bow_database;
} // namespace data

namespace module {

// initializer state
enum class initializer_state_t {
    NotReady,
    Initializing,
    Wrong,
    Succeeded
};

class initializer {
public:
    initializer() = delete;

    //! Constructor
    initializer(data::map_database* map_db,
                const cv::FileNode& yaml_node);

    //! Destructor
    ~initializer();

    //! Reset initializer
    void reset();

    //! Get initialization state
    initializer_state_t get_state() const;

    //! Get the initial frame ID which succeeded in initialization
    unsigned int get_initial_frame_id() const;

    //! Get the initial frame stamp which succeeded in initialization
    double get_initial_frame_timestamp() const;

    //! Get whether to use a fixed seed for RANSAC
    bool get_use_fixed_seed() const;

    //! Initialize with the current frame
    bool initialize(const camera::setup_type_t setup_type,
                    data::bow_vocabulary* bow_vocab, data::frame& curr_frm);

private:
    //! map database
    data::map_database* map_db_ = nullptr;
    //! initializer status
    initializer_state_t state_ = initializer_state_t::NotReady;

    //! ID of frame used for initialization (will be set after succeeded)
    unsigned int init_frm_id_ = 0;
    //! timestamp of frame used for initialization (will be set after succeeded)
    double init_frm_stamp_ = 0.0;

    //-----------------------------------------
    // parameters

    //! max number of iterations of RANSAC (only for monocular initializer)
    const unsigned int num_ransac_iters_;
    //! min number of valid pts (It should be greater than or equal to min_num_triangulated_)
    const unsigned int min_num_valid_pts_;
    //! min number of triangulated pts
    const unsigned int min_num_triangulated_pts_;
    //! min parallax (only for monocular initializer)
    const float parallax_deg_thr_;
    //! reprojection error threshold (only for monocular initializer)
    const float reproj_err_thr_;
    //! max number of iterations of BA (only for monocular initializer)
    const unsigned int num_ba_iters_;
    //! initial scaling factor (only for monocular initializer)
    const float scaling_factor_;
    //! Use fixed random seed for RANSAC if true
    const bool use_fixed_seed_;
    //! Gain threshold (for g2o)
    const float gain_threshold_;
    //! Verbosity (for g2o)
    const bool verbose_;

    //-----------------------------------------
    // for monocular camera model

    //! Create initializer for monocular
    void create_initializer(data::frame& curr_frm);

    //! Try to initialize a map with monocular camera setup
    bool try_initialize_for_monocular(data::frame& curr_frm);

    //! Create an initial map with monocular camera setup
    bool create_map_for_monocular(data::bow_vocabulary* bow_vocab, data::frame& curr_frm);

    //! Scaling up or down a initial map
    void scale_map(const std::shared_ptr<data::keyframe>& init_keyfrm, const std::shared_ptr<data::keyframe>& curr_keyfrm, const double scale);

    //! initializer for monocular
    std::unique_ptr<initialize::base> initializer_ = nullptr;
    //! initial frame
    data::frame init_frm_;
    //! coordinates of previously matched points to perform area-based matching
    std::vector<cv::Point2f> prev_matched_coords_;
    //! initial matching indices (index: idx of initial frame, value: idx of current frame)
    std::vector<int> init_matches_;

    size_t required_keyframes_for_marker_initialization_;

    //-----------------------------------------
    // for stereo or RGBD camera model

    //! Try to initialize a map with stereo or RGBD camera setup
    bool try_initialize_for_stereo(data::frame& curr_frm);

    //! Create an initial map with stereo or RGBD camera setup
    bool create_map_for_stereo(data::bow_vocabulary* bow_vocab, data::frame& curr_frm);
};

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_INITIALIZER_H
