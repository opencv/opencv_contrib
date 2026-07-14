#ifndef SLAM_VIEWER_CV_VIEWER_H
#define SLAM_VIEWER_CV_VIEWER_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Core>
#include <vector>
#include <mutex>

namespace cv::slam {
namespace viewer {

struct trajectory_sample {
    double timestamp = 0.0;
    cv::Point2d xy;
    Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
};

struct similarity3d {
    double scale = 1.0;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();

    Eigen::Vector3d apply(const Eigen::Vector3d& p) const {
        return scale * rotation * p + translation;
    }
};

class cv_viewer {
public:
    enum class projection_mode {
        XZ = 0,
        XY = 1,
        YZ = 2
    };

    cv_viewer();
    void update(const cv::Mat& image,
                const std::vector<cv::KeyPoint>& keypts,
                const Eigen::Matrix4d& pose_wc,
                const std::vector<trajectory_sample>& estimated_xy,
                const std::vector<Eigen::Vector3d>& map_points_xyz,
                double fps, int frame_idx);
    void set_ground_truth(const std::vector<trajectory_sample>& ground_truth_xy);
    void render();
    bool should_quit() const { return quit_; }
    void request_quit() { quit_ = true; }

private:
    struct button_layout {
        cv::Rect projection;
        cv::Rect ground_truth;
        cv::Rect map_points;
        cv::Rect pause;
    };

    static void on_mouse(int event, int x, int y, int flags, void* userdata);
    void handle_mouse(int event, int x, int y);

    std::mutex mtx_;
    cv::Mat current_image_;
    std::vector<cv::KeyPoint> current_keypts_;
    std::vector<trajectory_sample> estimated_xy_;
    std::vector<trajectory_sample> ground_truth_xy_;
    std::vector<Eigen::Vector3d> map_points_xyz_;
    Eigen::Matrix4d current_pose_;
    similarity3d gt_to_est_;
    size_t gt_alignment_matched_count_ = 0;
    double gt_alignment_path_len_ = 0.0;
    double viewport_span_y_ = 0.0;
    double fps_ = 0;
    int frame_idx_ = 0;
    button_layout buttons_;
    bool gt_alignment_ready_ = false;
    bool show_ground_truth_ = true;
    bool show_map_points_ = true;
    bool paused_ = false;
    projection_mode projection_mode_ = projection_mode::XZ;
    bool has_new_ = false, quit_ = false;
};

} // namespace viewer
} // namespace cv::slam
#endif
