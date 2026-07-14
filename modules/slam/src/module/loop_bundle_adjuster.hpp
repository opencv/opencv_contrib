#ifndef SLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H
#define SLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H

#include <mutex>

namespace cv::slam {

class mapping_module;

namespace data {
class keyframe;
class map_database;
} // namespace data

namespace module {

class loop_bundle_adjuster {
public:
    /**
     * Constructor
     */
    explicit loop_bundle_adjuster(data::map_database* map_db,
                                  const unsigned int num_iter = 10,
                                  const bool use_huber_kernel = false,
                                  const bool verbose = false);

    /**
     * Destructor
     */
    ~loop_bundle_adjuster() = default;

    /**
     * Set the mapping module
     */
    void set_mapping_module(mapping_module* mapper);

    /**
     * Abort loop BA externally
     */
    void abort();

    /**
     * Loop BA is running or not
     */
    bool is_running() const;

    /**
     * Run loop BA
     */
    void optimize(const std::shared_ptr<data::keyframe>& curr_keyfrm);

private:
    //! map database
    data::map_database* map_db_ = nullptr;

    //! mapping module
    mapping_module* mapper_ = nullptr;

    //! number of iteration for optimization
    const unsigned int num_iter_ = 10;
    //! True if using Huber kernel (for g2o)
    const bool use_huber_kernel_ = false;
    //! Verbosity (for g2o)
    const bool verbose_ = false;

    //-----------------------------------------
    // thread management

    //! mutex for access to pause procedure
    mutable std::mutex mtx_thread_;

    //! flag to abort loop BA
    bool abort_loop_BA_ = false;

    //! flag which indicates loop BA is running or not
    bool loop_BA_is_running_ = false;
};

} // namespace module
} // namespace cv::slam

#endif // SLAM_MODULE_LOOP_BUNDLE_ADJUSTER_H
