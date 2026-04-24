// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SLAM_HPP
#define OPENCV_SLAM_HPP

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <memory>
#include <vector>
#include <optional>

namespace cv {

namespace vo {

/** @brief Visual odometry tracking state. */
enum class VOState {
    NotInitialized, //!< System created but no frames processed yet
    Initializing,   //!< Collecting initial features for map bootstrap
    Tracking,       //!< Normal tracking with sufficient feature matches
    Lost            //!< Tracking failed; attempting relocalization
};

/** @brief SLAM operating mode.
 *
 *  Controls whether the system builds a new map or localizes
 *  against a pre-built map.
 */
enum class SLAMMode {
    SLAM,          //!< Full SLAM mode: tracking + mapping + loop closure
    LOCALIZATION   //!< Localization-only mode: uses pre-built map, no new keyframes
};

/** @brief Visual odometry configuration.
 *
 *  Configuration is loaded from YAML files. Camera parameters, feature
 *  settings, backend parameters, etc. are all read from the YAML file
 *  specified by camera_config_file.
 */
struct CV_EXPORTS_W_SIMPLE VOConfig {
    String camera_config_file; //!< Path to camera YAML config (Camera/Feature/Mapping/System sections)
    String vocab_file;         //!< Path to FBoW vocabulary file for loop closure and global relocalization
};

/** @brief Visual Odometry / SLAM system.
 *
 *  Provides a complete visual SLAM pipeline with pluggable feature
 *  extractors and descriptor matchers. Supports monocular input with
 *  ORB, SIFT, AKAZE, or custom feature detectors.
 *
 *  The system has two operating modes controlled by setMode():
 *  - **SLAM mode**: Full pipeline — tracking, local mapping, loop closure.
 *  - **LOCALIZATION mode**: Relocalizes against a pre-built map without
 *    inserting new keyframes. Useful for deployment scenarios where the
 *    map is already built.
 *
 *  Typical usage:
 *  @code
 *  cv::vo::VOConfig config;
 *  config.camera_config_file = "EuRoC.yaml";
 *  config.vocab_file = "orb_vocab.fbow";
 *
 *  auto vo = cv::vo::VisualOdometry::create(config,
 *      cv::ORB::create(), cv::BFMatcher::create(cv::NORM_HAMMING));
 *
 *  cv::Mat frame = cv::imread("frame.png", cv::IMREAD_GRAYSCALE);
 *  auto pose = vo->processFrame(frame, timestamp);
 *  if (pose) {
 *      // pose is a 4x4 camera-to-world SE(3) matrix
 *  }
 *  @endcode
 */
class CV_EXPORTS_W VisualOdometry {
public:
    /** @brief Create a VisualOdometry instance with explicit feature detector and matcher.
     *
     *  Use this overload when you want full control over the feature extraction
     *  and matching pipeline (e.g., ORB+Hamming, SIFT+L2, AKAZE+Hamming).
     *
     *  @param config           VO configuration (camera YAML path, vocab path).
     *  @param feature_detector Feature detector/descriptor (e.g., ORB::create(), SIFT::create()).
     *  @param matcher          Descriptor matcher (e.g., BFMatcher with NORM_HAMMING or NORM_L2).
     *  @return Shared pointer to the created instance.
     */
    CV_WRAP static Ptr<VisualOdometry> create(
        const VOConfig& config,
        const Ptr<Feature2D>& feature_detector,
        const Ptr<DescriptorMatcher>& matcher);

    /** @brief Create a VisualOdometry instance from a YAML config file.
     *
     *  Use this convenience overload when configuration (including feature
     *  type, matcher type, and camera params) is fully specified in the YAML
     *  file. Features and matcher are created automatically from config.
     *
     *  @param config_file Path to YAML configuration file.
     *  @param vocab_file  Path to FBoW vocabulary file. If empty, loop closure
     *                     detection is disabled and the system runs in pure VO
     *                     + local BA mode.
     *  @return Shared pointer to the created instance.
     */
    CV_WRAP static Ptr<VisualOdometry> create(
        const String& config_file,
        const String& vocab_file = "");

    virtual ~VisualOdometry() = default;

    /** @name Frame Processing
     *  Process monocular image frames and estimate camera pose.
     */
    ///@{

    /** @brief Process a single monocular frame.
     *
     *  Extracts features, matches against the local map, and estimates the
     *  camera pose. May insert a new keyframe if sufficient parallax or
     *  tracking quality criteria are met.
     *
     *  @param image     Input grayscale or color image (CV_8UC1 or CV_8UC3).
     *                   Color images are converted to grayscale internally.
     *  @param timestamp Frame timestamp in seconds. Must use absolute time
     *                   values (e.g., Unix epoch or dataset timestamps from
     *                   data.csv). Do NOT use relative offsets like i*0.05.
     *  @return 4x4 camera-to-world SE(3) pose matrix on success, or
     *          std::nullopt if tracking is lost or the system is not
     *          yet initialized.
     *  @note Call release() when done to allow background threads to finish.
     */
    CV_WRAP virtual std::optional<Matx44d> processFrame(
        const Mat& image,
        double timestamp) = 0;
    ///@}

    /** @name State Queries
     *  Query the current tracking state and system properties.
     */
    ///@{

    /** @brief Get the current tracking state.
     *  @return Current VOState (NotInitialized, Initializing, Tracking, or Lost).
     */
    CV_WRAP virtual VOState getState() const = 0;

    /** @brief Check if the system has been initialized (first keyframe inserted).
     *  @return true after successful initialization.
     */
    CV_WRAP virtual bool isInitialized() const = 0;

    /** @brief Check if the map is empty (no keyframes or landmarks).
     *  @return true if the map contains no data.
     */
    CV_WRAP virtual bool isEmpty() const = 0;
    ///@}

    /** @name Map and Trajectory Access
     *  Access map points, keypoints, poses, and trajectory data.
     */
    ///@{

    /** @brief Get all reconstructed 3D map points.
     *  @return Vector of 3D points in world coordinates.
     */
    CV_WRAP virtual std::vector<Point3d> getMapPoints() const = 0;

    /** @brief Get keypoints detected in the current frame.
     *  @return Vector of keypoints from the most recent processFrame() call.
     */
    CV_WRAP virtual std::vector<KeyPoint> getCurrentKeypoints() const = 0;

    /** @brief Get the current camera pose.
     *  @return 4x4 camera-to-world SE(3) pose matrix.
     */
    CV_WRAP virtual Matx44d getCurrentPose() const = 0;

    /** @brief Get the full trajectory as timestamp-pose pairs.
     *  @return Vector of (timestamp, pose) pairs.
     */
    CV_WRAP virtual std::vector<std::pair<double, Matx44d>> getTrajectory() const = 0;

    /** @brief Save trajectory to file.
     *  @param path   Output file path.
     *  @param format File format (e.g., "TUM", "KITTI", "EuRoC").
     *  @return true on success.
     */
    CV_WRAP virtual bool saveTrajectory(const String& path, const String& format) = 0;
    ///@}

    /** @name Map Persistence
     *  Save and load reconstructed maps for later reuse.
     */
    ///@{

    /** @brief Save the map to a file.
     *  @param path Output file path (.msgpack format).
     *  @return true on success.
     */
    CV_WRAP virtual bool saveMap(const String& path) = 0;

    /** @brief Load a map from a file.
     *  @param path Input file path (.msgpack format).
     *  @return true on success.
     */
    CV_WRAP virtual bool loadMap(const String& path) = 0;
    ///@}

    /** @name System Control
     *  Reset, release, and control system lifecycle.
     */
    ///@{

    /** @brief Reset the system to initial state, clearing the map. */
    CV_WRAP virtual void reset() = 0;

    /** @brief Release system resources and stop background threads.
     *  Must be called before destroying the VisualOdometry instance to
     *  ensure clean shutdown of background threads (mapping, loop closure).
     */
    CV_WRAP virtual void release() = 0;
    ///@}

    /** @name Mode Control
     *  Switch between SLAM and Localization modes.
     */
    ///@{

    /** @brief Set the SLAM operating mode.
     *  @param mode SLAM mode (SLAM or LOCALIZATION).
     */
    CV_WRAP virtual void setMode(SLAMMode mode) = 0;

    /** @brief Get the current SLAM operating mode.
     *  @return Current mode.
     */
    CV_WRAP virtual SLAMMode getMode() const = 0;
    ///@}

    /** @name Backend Control
     *  Enable/disable backend optimization components.
     */
    ///@{

    /** @brief Enable or disable local Bundle Adjustment.
     *  @param enable      Whether to enable backend BA.
     *  @param window_size Sliding window size for local BA (default: 10).
     */
    CV_WRAP virtual void setBackendEnabled(bool enable, int window_size = 10) = 0;

    /** @brief Check if backend BA is enabled. */
    CV_WRAP virtual bool isBackendEnabled() const = 0;

    /** @brief Enable or disable loop closure detection.
     *  @param enable Whether to enable loop closure.
     */
    CV_WRAP virtual void setLoopClosureEnabled(bool enable) = 0;

    /** @brief Check if loop closure detection is enabled. */
    CV_WRAP virtual bool isLoopClosureEnabled() const = 0;
    ///@}

    /** @name Component Access
     *  Access or replace feature detector and matcher components.
     */
    ///@{

    /** @brief Replace the feature detector at runtime.
     *  @param detector New feature detector/descriptor extractor.
     */
    CV_WRAP virtual void setFeatureDetector(const Ptr<Feature2D>& detector) = 0;

    /** @brief Get the current feature detector. */
    CV_WRAP virtual Ptr<Feature2D> getFeatureDetector() const = 0;

    /** @brief Replace the descriptor matcher at runtime.
     *  @param matcher New descriptor matcher.
     */
    CV_WRAP virtual void setMatcher(const Ptr<DescriptorMatcher>& matcher) = 0;

    /** @brief Get the current descriptor matcher. */
    CV_WRAP virtual Ptr<DescriptorMatcher> getMatcher() const = 0;
    ///@}
};

} // namespace vo
} // namespace cv

#endif // OPENCV_SLAM_HPP