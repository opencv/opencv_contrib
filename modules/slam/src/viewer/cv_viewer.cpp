#include "viewer/cv_viewer.hpp"
#include <opencv2/imgproc.hpp>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace cv::slam {
namespace viewer {
namespace {

constexpr const char* kWindowName = "cv_slam Viewer";
constexpr int kPanelW = 640;
constexpr int kPanelH = 480;
constexpr int kToolbarH = 48;
constexpr int kStatusH = 32;
constexpr int kWindowW = kPanelW * 2;
constexpr int kWindowH = kToolbarH + kPanelH + kStatusH;
constexpr size_t kMinGtAlignmentMatches = 120;
constexpr double kMinGtAlignmentPathLen = 1.5;

static const char* projection_mode_label(cv_viewer::projection_mode mode) {
    switch (mode) {
        case cv_viewer::projection_mode::XZ: return "VIEW XZ";
        case cv_viewer::projection_mode::XY: return "VIEW XY";
        case cv_viewer::projection_mode::YZ: return "VIEW YZ";
    }
    return "VIEW XZ";
}

static void draw_button(cv::Mat& canvas, const cv::Rect& rect, const char* text, bool active) {
    const cv::Scalar fill = active ? cv::Scalar(46, 92, 70) : cv::Scalar(54, 58, 64);
    const cv::Scalar border = active ? cv::Scalar(92, 190, 132) : cv::Scalar(92, 98, 106);
    cv::rectangle(canvas, rect, fill, cv::FILLED);
    cv::rectangle(canvas, rect, border, 1);

    int baseline = 0;
    const double scale = 0.45;
    const int thickness = 1;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    const cv::Point origin(rect.x + (rect.width - text_size.width) / 2,
                           rect.y + (rect.height + text_size.height) / 2 - 2);
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(235, 235, 235), thickness);
}

static void draw_disabled_button(cv::Mat& canvas, const cv::Rect& rect, const char* text) {
    cv::rectangle(canvas, rect, cv::Scalar(38, 41, 46), cv::FILLED);
    cv::rectangle(canvas, rect, cv::Scalar(62, 67, 74), 1);

    int baseline = 0;
    const double scale = 0.45;
    const int thickness = 1;
    const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    const cv::Point origin(rect.x + (rect.width - text_size.width) / 2,
                           rect.y + (rect.height + text_size.height) / 2 - 2);
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(130, 138, 148), thickness);
}

static int draw_status_pair(cv::Mat& canvas, int x, int y, const char* label, const char* value) {
    const double scale = 0.43;
    const int thickness = 1;
    cv::putText(canvas, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(132, 140, 150), thickness);
    int baseline = 0;
    const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    x += label_size.width + 4;

    cv::putText(canvas, value, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale,
                cv::Scalar(224, 228, 234), thickness);
    const cv::Size value_size = cv::getTextSize(value, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
    return x + value_size.width + 18;
}

static cv::Rect fit_rect_keep_aspect(const cv::Size& src_size, const cv::Size& dst_size) {
    if (src_size.width <= 0 || src_size.height <= 0) {
        return cv::Rect(0, 0, dst_size.width, dst_size.height);
    }

    const double scale = std::min(static_cast<double>(dst_size.width) / src_size.width,
                                  static_cast<double>(dst_size.height) / src_size.height);
    const int width = std::max(1, cvRound(src_size.width * scale));
    const int height = std::max(1, cvRound(src_size.height * scale));
    return cv::Rect((dst_size.width - width) / 2, (dst_size.height - height) / 2, width, height);
}

static Eigen::Vector3d nearest_ground_truth_xyz(const std::vector<trajectory_sample>& gt,
                                                double timestamp,
                                                bool& found) {
    found = false;
    if (gt.empty()) {
        return {};
    }

    auto it = std::lower_bound(gt.begin(), gt.end(), timestamp,
                               [](const trajectory_sample& sample, double ts) {
                                   return sample.timestamp < ts;
                               });

    auto best = gt.end();
    if (it != gt.end()) {
        best = it;
    }
    if (it != gt.begin()) {
        auto prev = std::prev(it);
        if (best == gt.end() || std::abs(prev->timestamp - timestamp) < std::abs(best->timestamp - timestamp)) {
            best = prev;
        }
    }

    if (best != gt.end() && std::abs(best->timestamp - timestamp) <= 0.03) {
        found = true;
        return best->xyz;
    }
    return Eigen::Vector3d::Zero();
}

static cv::Point2d project_plane(const Eigen::Vector3d& p, cv_viewer::projection_mode mode) {
    switch (mode) {
        case cv_viewer::projection_mode::XZ: return {p(0), p(2)};
        case cv_viewer::projection_mode::XY: return {p(0), p(1)};
        case cv_viewer::projection_mode::YZ: return {p(1), p(2)};
    }
    return {p(0), p(2)};
}

static double planar_yaw_deg(const Eigen::Matrix3d& rotation, cv_viewer::projection_mode mode) {
    switch (mode) {
        case cv_viewer::projection_mode::XZ:
            return std::atan2(rotation(2, 0), rotation(0, 0)) * 180.0 / CV_PI;
        case cv_viewer::projection_mode::XY:
            return std::atan2(rotation(1, 0), rotation(0, 0)) * 180.0 / CV_PI;
        case cv_viewer::projection_mode::YZ:
            return std::atan2(rotation(2, 1), rotation(1, 1)) * 180.0 / CV_PI;
    }
    return 0.0;
}

static similarity3d estimate_ground_truth_to_estimated(
        const std::vector<trajectory_sample>& gt,
        const std::vector<trajectory_sample>& estimated,
        size_t* matched_count = nullptr,
        double* estimated_path_len = nullptr) {
    std::vector<Eigen::Vector3d> src;
    std::vector<Eigen::Vector3d> dst;
    src.reserve(estimated.size());
    dst.reserve(estimated.size());

    for (const auto& sample : estimated) {
        bool found = false;
        Eigen::Vector3d gt_xyz = nearest_ground_truth_xyz(gt, sample.timestamp, found);
        if (found) {
            src.push_back(gt_xyz);
            dst.push_back(sample.xyz);
        }
    }

    if (matched_count) {
        *matched_count = src.size();
    }

    similarity3d sim;
    if (src.empty()) {
        return sim;
    }
    if (src.size() == 1) {
        sim.translation = dst.front() - src.front();
        return sim;
    }

    Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d dst_mean = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < src.size(); ++i) {
        src_mean += src[i];
        dst_mean += dst[i];
    }
    src_mean *= 1.0 / static_cast<double>(src.size());
    dst_mean *= 1.0 / static_cast<double>(dst.size());

    double src_variance = 0.0;
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double dst_path_len = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        Eigen::Vector3d p = src[i] - src_mean;
        Eigen::Vector3d q = dst[i] - dst_mean;
        src_variance += p.squaredNorm();
        covariance += p * q.transpose();
        if (i > 0) {
            dst_path_len += (dst[i] - dst[i - 1]).norm();
        }
    }

    if (estimated_path_len) {
        *estimated_path_len = dst_path_len;
    }

    if (src_variance <= 1e-12) {
        sim.translation = dst_mean - src_mean;
        return sim;
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d u = svd.matrixU();
    Eigen::Matrix3d v = svd.matrixV();
    Eigen::Matrix3d d = Eigen::Matrix3d::Identity();
    if ((v * u.transpose()).determinant() < 0.0) {
        d(2, 2) = -1.0;
    }

    sim.rotation = v * d * u.transpose();
    const Eigen::Vector3d singular_values = svd.singularValues();
    sim.scale = (singular_values(0) * d(0, 0)
                 + singular_values(1) * d(1, 1)
                 + singular_values(2) * d(2, 2)) / src_variance;
    sim.translation = dst_mean - sim.scale * sim.rotation * src_mean;
    return sim;
}

} // namespace

cv_viewer::cv_viewer() {
    cv::namedWindow(kWindowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(kWindowName, kWindowW, kWindowH);
    cv::setMouseCallback(kWindowName, &cv_viewer::on_mouse, this);
}

void cv_viewer::on_mouse(int event, int x, int y, int flags, void* userdata) {
    (void)flags;
    auto* viewer = static_cast<cv_viewer*>(userdata);
    if (viewer) {
        viewer->handle_mouse(event, x, y);
    }
}

void cv_viewer::handle_mouse(int event, int x, int y) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    std::lock_guard<std::mutex> lk(mtx_);
    const cv::Point pt(x, y);
    if (buttons_.projection.contains(pt)) {
        projection_mode_ = static_cast<projection_mode>((static_cast<int>(projection_mode_) + 1) % 3);
        viewport_span_y_ = 0.0;
    }
    else if (buttons_.ground_truth.contains(pt)) {
        if (!ground_truth_xy_.empty() && gt_alignment_ready_) {
            show_ground_truth_ = !show_ground_truth_;
        }
    }
    else if (buttons_.map_points.contains(pt)) {
        show_map_points_ = !show_map_points_;
    }
    else if (buttons_.pause.contains(pt)) {
        paused_ = !paused_;
    }
}

void cv_viewer::update(const cv::Mat& img, const std::vector<cv::KeyPoint>& kpts,
                       const Eigen::Matrix4d& pose, const std::vector<trajectory_sample>& estimated_xy,
                       const std::vector<Eigen::Vector3d>& map_points_xyz, double fps, int fidx) {
    std::lock_guard<std::mutex> lk(mtx_);
    current_image_ = img.clone();
    current_keypts_ = kpts;
    current_pose_ = pose;
    estimated_xy_ = estimated_xy;
    map_points_xyz_ = map_points_xyz;
    fps_ = fps;
    frame_idx_ = fidx;
    has_new_ = true;
}

void cv_viewer::set_ground_truth(const std::vector<trajectory_sample>& ground_truth_xy) {
    std::lock_guard<std::mutex> lk(mtx_);
    ground_truth_xy_ = ground_truth_xy;
    gt_to_est_ = similarity3d{};
    gt_alignment_matched_count_ = 0;
    gt_alignment_path_len_ = 0.0;
    gt_alignment_ready_ = false;
}

void cv_viewer::render() {
    cv::Mat img; std::vector<cv::KeyPoint> kpts;
    std::vector<trajectory_sample> estimated;
    std::vector<trajectory_sample> gt;
    std::vector<Eigen::Vector3d> map_points;
    similarity3d locked_gt_to_est;
    size_t gt_alignment_matched_count = 0;
    double gt_alignment_path_len = 0.0;
    bool gt_alignment_ready = false;
    projection_mode active_projection_mode = projection_mode::XZ;
    double fps; int fidx;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!has_new_) { cv::waitKey(2); return; }
        img = current_image_.clone(); kpts = current_keypts_;
        estimated = estimated_xy_;
        gt = ground_truth_xy_;
        map_points = map_points_xyz_;
        locked_gt_to_est = gt_to_est_;
        gt_alignment_matched_count = gt_alignment_matched_count_;
        gt_alignment_path_len = gt_alignment_path_len_;
        gt_alignment_ready = gt_alignment_ready_;
        active_projection_mode = projection_mode_;
        fps = fps_; fidx = frame_idx_;
        has_new_ = false;
    }

    auto draw_display = [&]() {
        bool show_ground_truth = true;
        bool show_map_points = true;
        bool paused = false;
        bool has_ground_truth = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            show_ground_truth = show_ground_truth_;
            show_map_points = show_map_points_;
            paused = paused_;
            has_ground_truth = !ground_truth_xy_.empty();
        }

        cv::Mat disp(kWindowH, kWindowW, CV_8UC3, cv::Scalar(22, 24, 28));
        cv::rectangle(disp, cv::Rect(0, 0, kWindowW, kToolbarH), cv::Scalar(34, 37, 42), cv::FILLED);
        cv::rectangle(disp, cv::Rect(0, kToolbarH + kPanelH, kWindowW, kStatusH), cv::Scalar(28, 31, 35), cv::FILLED);
        cv::line(disp, cv::Point(kPanelW, kToolbarH), cv::Point(kPanelW, kToolbarH + kPanelH), cv::Scalar(54, 58, 64), 1);

        cv::putText(disp, "OpenCV SLAM Viewer", cv::Point(16, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(235, 238, 242), 1);
        cv::putText(disp, "Frame + features", cv::Point(270, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(190, 195, 202), 1);
        cv::putText(disp, "Trajectory projection", cv::Point(kPanelW + 16, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(190, 195, 202), 1);

        cv::Mat left_src;
        if (img.channels() == 1) cv::cvtColor(img, left_src, cv::COLOR_GRAY2BGR);
        else left_src = img.clone();

        cv::Mat left(kPanelH, kPanelW, CV_8UC3, cv::Scalar(18, 20, 24));
        const cv::Rect image_rect = fit_rect_keep_aspect(left_src.size(), cv::Size(kPanelW, kPanelH));
        cv::Mat fitted;
        cv::resize(left_src, fitted, image_rect.size());
        fitted.copyTo(left(image_rect));

        const double image_scale = static_cast<double>(image_rect.width) / left_src.cols;
        for (const auto& k : kpts) {
            cv::Point pt(cvRound(k.pt.x * image_scale + image_rect.x),
                         cvRound(k.pt.y * image_scale + image_rect.y));
            cv::circle(left, pt, 2, cv::Scalar(0, 255, 0), -1);
        }
        left.copyTo(disp(cv::Rect(0, kToolbarH, kPanelW, kPanelH)));

        cv::Mat right(kPanelH, kPanelW, CV_8UC3, cv::Scalar(26, 29, 33));
        char buf[192];
        double status_scale = 1.0;
        double status_yaw_deg = 0.0;
        size_t status_matched_count = gt_alignment_matched_count;
        double status_path_len = gt_alignment_path_len;
        bool status_alignment_ready = gt_alignment_ready;

        if (estimated.size() >= 2 || gt.size() >= 2) {
            similarity3d gt_to_est = locked_gt_to_est;
            if (!gt.empty() && !estimated.empty()) {
                size_t matched_count = 0;
                double estimated_path_len = 0.0;
                similarity3d candidate = estimate_ground_truth_to_estimated(gt, estimated, &matched_count, &estimated_path_len);

                if (matched_count >= kMinGtAlignmentMatches && estimated_path_len >= kMinGtAlignmentPathLen) {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (!gt_alignment_ready_) {
                        gt_to_est_ = candidate;
                        gt_alignment_matched_count_ = matched_count;
                        gt_alignment_path_len_ = estimated_path_len;
                        gt_alignment_ready_ = true;
                    }
                    gt_to_est = gt_to_est_;
                    gt_alignment_matched_count = gt_alignment_matched_count_;
                    gt_alignment_path_len = gt_alignment_path_len_;
                    gt_alignment_ready = gt_alignment_ready_;
                }
                else {
                    gt_to_est = candidate;
                    gt_alignment_matched_count = matched_count;
                    gt_alignment_path_len = estimated_path_len;
                }
            }
            const cv::Point2d center = estimated.empty()
                                       ? project_plane(gt_to_est.apply(gt.front().xyz), active_projection_mode)
                                       : project_plane(estimated.back().xyz, active_projection_mode);
            status_scale = gt_to_est.scale;
            status_yaw_deg = planar_yaw_deg(gt_to_est.rotation, active_projection_mode);
            status_matched_count = gt_alignment_matched_count;
            status_path_len = gt_alignment_path_len;
            status_alignment_ready = gt_alignment_ready;

            double max_abs_x = 0.25;
            double max_abs_y = 0.25;
            for (const auto& sample : estimated) {
                const cv::Point2d projected = project_plane(sample.xyz, active_projection_mode);
                max_abs_x = std::max(max_abs_x, std::abs(projected.x - center.x));
                max_abs_y = std::max(max_abs_y, std::abs(projected.y - center.y));
            }

            const double aspect = static_cast<double>(kPanelW) / kPanelH;
            const double target_occupancy = 0.90;
            const double needed_span_y = std::max(2.0 * max_abs_y / target_occupancy,
                                                 (2.0 * max_abs_x / target_occupancy) / aspect);
            viewport_span_y_ = std::max(viewport_span_y_, std::max(needed_span_y, 1.0));

            const double sc = kPanelH / viewport_span_y_;
            const double ox = kPanelW * 0.5 - center.x * sc;
            const double oy = kPanelH * 0.5 + center.y * sc;

            auto to_screen = [&](double x, double y) {
                return cv::Point(cvRound(x * sc + ox), cvRound(y * (-sc) + oy));
            };

            if (show_map_points) {
                for (const auto& p : map_points) {
                    const cv::Point2d projected = project_plane(p, active_projection_mode);
                    cv::circle(right, to_screen(projected.x, projected.y), 1, cv::Scalar(150, 150, 150), -1);
                }
            }

            if (show_ground_truth && gt_alignment_ready) {
                for (size_t i = 1; i < gt.size(); ++i) {
                    const cv::Point2d p0 = project_plane(gt_to_est.apply(gt[i - 1].xyz), active_projection_mode);
                    const cv::Point2d p1 = project_plane(gt_to_est.apply(gt[i].xyz), active_projection_mode);
                    cv::line(right, to_screen(p0.x, p0.y), to_screen(p1.x, p1.y), cv::Scalar(80, 220, 80), 1);
                }
            }

            for (size_t i = 1; i < estimated.size(); ++i) {
                const cv::Point2d p0 = project_plane(estimated[i - 1].xyz, active_projection_mode);
                const cv::Point2d p1 = project_plane(estimated[i].xyz, active_projection_mode);
                cv::line(right, to_screen(p0.x, p0.y), to_screen(p1.x, p1.y), cv::Scalar(255, 120, 40), 2);
            }

            if (!estimated.empty()) {
                const cv::Point2d current = project_plane(estimated.back().xyz, active_projection_mode);
                cv::circle(right, to_screen(current.x, current.y), 5, cv::Scalar(0, 0, 255), -1);
            }
        } else {
            cv::putText(right, "Waiting for keyframes or ground truth...", cv::Point(18, kPanelH / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.62, cv::Scalar(180, 186, 194), 1);
        }

        right.copyTo(disp(cv::Rect(kPanelW, kToolbarH, kPanelW, kPanelH)));

        const int y = 10;
        const int h = 28;
        button_layout buttons;
        buttons.pause = cv::Rect(kWindowW - 98, y, 82, h);
        buttons.map_points = cv::Rect(buttons.pause.x - 96, y, 88, h);
        buttons.ground_truth = cv::Rect(buttons.map_points.x - 94, y, 86, h);
        buttons.projection = cv::Rect(buttons.ground_truth.x - 110, y, 102, h);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            buttons_ = buttons;
        }
        draw_button(disp, buttons.projection, projection_mode_label(active_projection_mode), true);
        if (!has_ground_truth) {
            draw_disabled_button(disp, buttons.ground_truth, "GT N/A");
        }
        else if (!status_alignment_ready) {
            draw_disabled_button(disp, buttons.ground_truth, "GT Pending");
        }
        else {
            draw_button(disp, buttons.ground_truth, show_ground_truth ? "GT ON" : "GT OFF", show_ground_truth);
        }
        draw_button(disp, buttons.map_points, show_map_points ? "MAP ON" : "MAP OFF", show_map_points);
        draw_button(disp, buttons.pause, paused ? "RESUME" : "PAUSE", paused);

        const int status_y = kToolbarH + kPanelH + 21;
        int x = 16;
        snprintf(buf, sizeof(buf), "%d", fidx);
        x = draw_status_pair(disp, x, status_y, "Frame", buf);
        snprintf(buf, sizeof(buf), "%.0f", fps);
        x = draw_status_pair(disp, x, status_y, "FPS", buf);
        snprintf(buf, sizeof(buf), "%zu", estimated.size());
        x = draw_status_pair(disp, x, status_y, "EST", buf);
        snprintf(buf, sizeof(buf), "%zu", gt.size());
        x = draw_status_pair(disp, x, status_y, "GT", buf);
        snprintf(buf, sizeof(buf), "%zu", map_points.size());
        x = draw_status_pair(disp, x, status_y, "MAP", buf);
        x = draw_status_pair(disp, x, status_y,
                             "View",
                             active_projection_mode == projection_mode::XZ ? "XZ"
                             : active_projection_mode == projection_mode::XY ? "XY" : "YZ");
        if (paused) {
            draw_status_pair(disp, x, status_y, "Mode", "PAUSED");
        }

        cv::circle(disp, cv::Point(kPanelW + 18, status_y - 5), 4, cv::Scalar(255, 120, 40), -1);
        cv::putText(disp, "SLAM", cv::Point(kPanelW + 28, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(174, 181, 190), 1);
        cv::circle(disp, cv::Point(kPanelW + 82, status_y - 5), 4, cv::Scalar(80, 220, 80), -1);
        cv::putText(disp, "GT", cv::Point(kPanelW + 92, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(174, 181, 190), 1);
        cv::circle(disp, cv::Point(kPanelW + 132, status_y - 5), 4, cv::Scalar(150, 150, 150), -1);
        cv::putText(disp, "Map", cv::Point(kPanelW + 142, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(174, 181, 190), 1);
        cv::circle(disp, cv::Point(kPanelW + 196, status_y - 5), 4, cv::Scalar(0, 0, 255), -1);
        cv::putText(disp, "Now", cv::Point(kPanelW + 206, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(174, 181, 190), 1);

        x = kPanelW + 260;
        snprintf(buf, sizeof(buf), "%.2f", status_scale);
        x = draw_status_pair(disp, x, status_y, "S", buf);
        snprintf(buf, sizeof(buf), "%.1f", status_yaw_deg);
        x = draw_status_pair(disp, x, status_y, "Yaw", buf);
        snprintf(buf, sizeof(buf), "%zu", status_matched_count);
        x = draw_status_pair(disp, x, status_y, "N", buf);
        snprintf(buf, sizeof(buf), "%.1f", status_path_len);
        x = draw_status_pair(disp, x, status_y, "Len", buf);
        if (!status_alignment_ready) {
            cv::putText(disp, "pending", cv::Point(x, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.42,
                        cv::Scalar(220, 190, 120), 1);
        }
        return disp;
    };

    while (true) {
        cv::Mat disp = draw_display();
        cv::imshow(kWindowName, disp);
        bool paused_before_wait = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            paused_before_wait = paused_;
        }
        const int key = cv::waitKey(paused_before_wait ? 30 : 1);
        if (key == 27) {
            std::lock_guard<std::mutex> lk(mtx_);
            quit_ = true;
            paused_ = false;
            break;
        }

        bool paused = false;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            paused = paused_;
        }
        if (!paused) {
            break;
        }
    }
}

} // namespace viewer
} // namespace cv::slam
