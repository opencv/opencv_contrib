#ifndef CV_SLAM_VO_VISUAL_ODOMETRY_IMPL_HPP
#define CV_SLAM_VO_VISUAL_ODOMETRY_IMPL_HPP

#include "opencv2/slam.hpp"
#include "system.hpp"

#include <memory>
#include <mutex>

namespace cv::vo {

/**
 * @brief VisualOdometry implementation class
 * 
 * Implemented on top of cv::slam::system and provides a modular interface.
 */
class VisualOdometryImpl : public VisualOdometry {
public:
    /**
     * @brief Constructor - create from config file
     * @param config_file Config file path
     * @param vocab_file Vocabulary file path
     */
    VisualOdometryImpl(const std::string& config_file, 
                       const std::string& vocab_file);
    
    /**
     * @brief Constructor - create from VOConfig
     * @param config VO configuration
     */
    explicit VisualOdometryImpl(const VOConfig& config);
    
    ~VisualOdometryImpl() override;
    
    // ========== Core interface ==========
    
    std::optional<cv::Matx44d> processFrame(
        const cv::Mat& image, 
        double timestamp) override;
    
    // ========== State query ==========
    
    VOState getState() const override;
    bool isInitialized() const override;
    bool isEmpty() const override;
    
    // ========== Map access ==========
    
    std::vector<cv::Point3d> getMapPoints() const override;
    std::vector<cv::KeyPoint> getCurrentKeypoints() const override;
    cv::Matx44d getCurrentPose() const override;
    
    // ========== Trajectory access ==========
    
    std::vector<std::pair<double, cv::Matx44d>> getTrajectory() const override;
    bool saveTrajectory(const std::string& path, const std::string& format) override;
    
    // ========== Map save/load ==========
    
    bool saveMap(const std::string& path) override;
    bool loadMap(const std::string& path) override;
    
    // ========== Control interface ==========
    
    void reset() override;
    void release() override;
    
    // ========== Mode control ==========
    
    void setMode(cv::vo::SLAMMode mode) override;
    cv::vo::SLAMMode getMode() const override;
    
    void setFeatureDetector(const cv::Ptr<cv::Feature2D>& detector);
    void setMatcher(const cv::Ptr<cv::DescriptorMatcher>& m);
    
    // ========== Backend control ==========
    
    void setBackendEnabled(bool enable, int window_size = 10) override;
    bool isBackendEnabled() const override;
    
    void setLoopClosureEnabled(bool enable) override;
    bool isLoopClosureEnabled() const override;
    
    // ========== Extension interface ==========
    
    void enableMappingModule(bool enable);
    bool isMappingModuleEnabled() const;
    
    void enableLoopDetector(bool enable);
    bool isLoopDetectorEnabled() const;
    
    std::shared_ptr<cv::slam::system> getSystem() const { return system_; }
    
private:
    /**
    * @brief Initialize the system
     */
    void initialize(const std::string& config_file, const std::string& vocab_file);
    
    std::shared_ptr<cv::slam::system> system_;
    
    // Trajectory
    std::vector<std::pair<double, cv::Matx44d>> trajectory_;
    mutable std::mutex trajectory_mutex_;
    
    // State
    bool initialized_ = false;
    bool shutdown_ = false;
    VOState state_ = VOState::NotInitialized;
    
    // SLAM mode
    cv::vo::SLAMMode mode_ = static_cast<cv::vo::SLAMMode>(0);
    
    // Vocabulary path
    std::string vocab_file_;
    
    // Backend configuration
    bool backend_enabled_ = false;
    bool loop_closure_enabled_ = false;
    int ba_window_size_ = 10;
    
    // Feature detector and matcher
    cv::Ptr<cv::Feature2D> feature_detector_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

} // namespace cv::vo

#endif // CV_SLAM_VO_VISUAL_ODOMETRY_IMPL_HPP
