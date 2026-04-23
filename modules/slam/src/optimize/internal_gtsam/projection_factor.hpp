#ifndef SLAM_OPTIMIZE_INTERNAL_GTSAM_PROJECTION_FACTOR_H
#define SLAM_OPTIMIZE_INTERNAL_GTSAM_PROJECTION_FACTOR_H

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/StereoCamera.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <boost/optional.hpp>

#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

namespace optimize {
namespace internal_gtsam {

template<class POSE, class LANDMARK>
class ProjectionFactorBase : public gtsam::NoiseModelFactor2<POSE, LANDMARK> {
private:
    typedef gtsam::NoiseModelFactor2<POSE, LANDMARK> Base;
    typedef ProjectionFactorBase<POSE, LANDMARK> This;

public:
    ProjectionFactorBase(const gtsam::SharedNoiseModel& model,
                         gtsam::Key key_pose, gtsam::Key key_landmark)
        : Base(model, key_pose, key_landmark) {
    }
    virtual ~ProjectionFactorBase() {}

    virtual gtsam::Vector project(const POSE& pose, const LANDMARK& point) const = 0;
};

template<class POSE, class LANDMARK, class CALIBRATION, class CAMERA>
class ProjectionFactor : public ProjectionFactorBase<POSE, LANDMARK> {
private:
    typedef ProjectionFactorBase<POSE, LANDMARK> Base;
    typedef ProjectionFactor<POSE, LANDMARK, CALIBRATION, CAMERA> This;

protected:
    gtsam::Point2 measurement_;
    boost::shared_ptr<CALIBRATION> calibration_;

public:
    ProjectionFactor(const gtsam::Point2& measurement, const gtsam::SharedNoiseModel& model,
                     gtsam::Key key_pose, gtsam::Key key_landmark, const boost::shared_ptr<CALIBRATION>& calibration)
        : Base(model, key_pose, key_landmark), measurement_(measurement), calibration_(calibration) {}
    virtual ~ProjectionFactor() {}

    gtsam::Vector evaluateError(const POSE& pose, const LANDMARK& point,
                                boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        try {
            CAMERA camera(pose, calibration_);
            return camera.project(point, H1, H2) - measurement_;
        }
        catch (gtsam::CheiralityException& e) {
            // Landmarks is behind the camera
            if (H1)
                *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2)
                *H2 = gtsam::Matrix::Zero(2, 3);
            CV_LOG_DEBUG(&g_log_tag, "{} : Landmarks is behind the camera", e.what());
        }
        // return some large error
        return gtsam::Vector2::Constant(1e+6);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    const gtsam::Point2& measurement() const {
        return measurement_;
    }

    gtsam::Vector project(const POSE& pose, const LANDMARK& point) const override {
        CAMERA camera(pose, calibration_);
        return camera.project(point);
    }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

template<class CALIBRATION>
class PinholeCamera : public gtsam::PinholeCamera<CALIBRATION> {
private:
    typedef gtsam::PinholeCamera<CALIBRATION> Base;

public:
    PinholeCamera(const gtsam::Pose3& pose, const boost::shared_ptr<CALIBRATION>& calibration)
        : Base(pose, *calibration) {}
    virtual ~PinholeCamera() {}
};

class SphericalCameraCalibration {
public:
    SphericalCameraCalibration(unsigned int rows, unsigned int cols)
        : rows_(rows), cols_(cols) {}
    virtual ~SphericalCameraCalibration() {}

    unsigned int rows_;
    unsigned int cols_;
};

template<class CALIBRATION>
class SphericalCamera {
protected:
    gtsam::Pose3 pose_;
    boost::shared_ptr<CALIBRATION> calibration_;

public:
    SphericalCamera(const gtsam::Pose3& pose, const boost::shared_ptr<CALIBRATION>& calibration)
        : pose_(pose), calibration_(calibration) {}
    virtual ~SphericalCamera() {}

    gtsam::Point2 project(const gtsam::Point3& point, gtsam::OptionalJacobian<2, 6> Dpose = boost::none, gtsam::OptionalJacobian<2, 3> Dpoint = boost::none) const {
        const unsigned int cols = calibration_->cols_;
        const unsigned int rows = calibration_->rows_;
        const gtsam::Point3 pos_c = pose_.transformTo(point);
        const double theta = std::atan2(pos_c(0), pos_c(2));
        const double phi = -std::asin(pos_c(1) / pos_c.norm());
        const gtsam::Point2 proj(cols * (0.5 + theta / (2 * M_PI)), rows * (0.5 - phi / M_PI));

        if (Dpose && Dpoint) {
            const Mat33_t rot_cw = pose_.inverse().rotation().matrix();

            const auto pcx = pos_c(0);
            const auto pcy = pos_c(1);
            const auto pcz = pos_c(2);
            const auto L = pos_c.norm();

            // rotation
            const Vec3_t d_pc_d_rx(0, -pcz, pcy);
            const Vec3_t d_pc_d_ry(pcz, 0, -pcx);
            const Vec3_t d_pc_d_rz(-pcy, pcx, 0);
            // translation
            const Vec3_t d_pc_d_tx(1, 0, 0);
            const Vec3_t d_pc_d_ty(0, 1, 0);
            const Vec3_t d_pc_d_tz(0, 0, 1);
            // 3D point
            const Vec3_t d_pc_d_pwx = rot_cw.block<3, 1>(0, 0);
            const Vec3_t d_pc_d_pwy = rot_cw.block<3, 1>(0, 1);
            const Vec3_t d_pc_d_pwz = rot_cw.block<3, 1>(0, 2);

            VecR_t<9> d_pcx_d_x;
            d_pcx_d_x << d_pc_d_rx(0), d_pc_d_ry(0), d_pc_d_rz(0),
                d_pc_d_tx(0), d_pc_d_ty(0), d_pc_d_tz(0),
                d_pc_d_pwx(0), d_pc_d_pwy(0), d_pc_d_pwz(0);
            VecR_t<9> d_pcy_d_x;
            d_pcy_d_x << d_pc_d_rx(1), d_pc_d_ry(1), d_pc_d_rz(1),
                d_pc_d_tx(1), d_pc_d_ty(1), d_pc_d_tz(1),
                d_pc_d_pwx(1), d_pc_d_pwy(1), d_pc_d_pwz(1);
            VecR_t<9> d_pcz_d_x;
            d_pcz_d_x << d_pc_d_rx(2), d_pc_d_ry(2), d_pc_d_rz(2),
                d_pc_d_tx(2), d_pc_d_ty(2), d_pc_d_tz(2),
                d_pc_d_pwx(2), d_pc_d_pwy(2), d_pc_d_pwz(2);

            const VecR_t<9> d_L_d_x = (1.0 / L) * (pcx * d_pcx_d_x + pcy * d_pcy_d_x + pcz * d_pcz_d_x);

            MatRC_t<2, 9> jacobian = MatRC_t<2, 9>::Zero();
            jacobian.block<1, 9>(0, 0) = -(cols / (2 * M_PI)) * (1.0 / (pcx * pcx + pcz * pcz))
                                         * (pcz * d_pcx_d_x - pcx * d_pcz_d_x);
            jacobian.block<1, 9>(1, 0) = -(rows / M_PI) * (1.0 / (L * std::sqrt(pcx * pcx + pcz * pcz)))
                                         * (L * d_pcy_d_x - pcy * d_L_d_x);

            *Dpoint = jacobian.block<2, 3>(0, 6);
            *Dpose = jacobian.block<2, 6>(0, 0);
        }
        else if (Dpose) {
            const auto pcx = pos_c(0);
            const auto pcy = pos_c(1);
            const auto pcz = pos_c(2);
            const auto L = pos_c.norm();

            // rotation
            const Vec3_t d_pc_d_rx(0, -pcz, pcy);
            const Vec3_t d_pc_d_ry(pcz, 0, -pcx);
            const Vec3_t d_pc_d_rz(-pcy, pcx, 0);
            // translation
            const Vec3_t d_pc_d_tx(1, 0, 0);
            const Vec3_t d_pc_d_ty(0, 1, 0);
            const Vec3_t d_pc_d_tz(0, 0, 1);

            VecR_t<6> d_pcx_d_x;
            d_pcx_d_x << d_pc_d_rx(0), d_pc_d_ry(0), d_pc_d_rz(0),
                d_pc_d_tx(0), d_pc_d_ty(0), d_pc_d_tz(0);
            VecR_t<6> d_pcy_d_x;
            d_pcy_d_x << d_pc_d_rx(1), d_pc_d_ry(1), d_pc_d_rz(1),
                d_pc_d_tx(1), d_pc_d_ty(1), d_pc_d_tz(1);
            VecR_t<6> d_pcz_d_x;
            d_pcz_d_x << d_pc_d_rx(2), d_pc_d_ry(2), d_pc_d_rz(2),
                d_pc_d_tx(2), d_pc_d_ty(2), d_pc_d_tz(2);

            const Vec6_t d_L_d_x = (1.0 / L) * (pcx * d_pcx_d_x + pcy * d_pcy_d_x + pcz * d_pcz_d_x);

            MatRC_t<2, 6> jacobian = MatRC_t<2, 6>::Zero();
            jacobian.block<1, 6>(0, 0) = -(cols / (2 * M_PI)) * (1.0 / (pcx * pcx + pcz * pcz))
                                         * (pcz * d_pcx_d_x - pcx * d_pcz_d_x);
            jacobian.block<1, 6>(1, 0) = -(rows / M_PI) * (1.0 / (L * std::sqrt(pcx * pcx + pcz * pcz)))
                                         * (L * d_pcy_d_x - pcy * d_L_d_x);
            *Dpose = jacobian;
        }
        else if (Dpoint) {
            // FIXME: Dpoint (landmark Jacobian) for GTSAM projection factor not yet implemented.
            // Only Dpose (pose Jacobian) is currently computed. Dpoint is required for
            // joint optimization of camera poses and landmarks. Currently, GTSAM-based
            // local BA uses pose-only optimization as a fallback.
            CV_LOG_WARNING(&g_log_tag, "Dpoint Jacobian not implemented for GTSAM projection factor");
        }
        return proj;
    }
};

template<class POSE, class LANDMARK, class CALIBRATION, class STEREO_CAMERA>
class StereoProjectionFactor : public ProjectionFactorBase<POSE, LANDMARK> {
private:
    typedef ProjectionFactorBase<POSE, LANDMARK> Base;
    typedef StereoProjectionFactor<POSE, LANDMARK, CALIBRATION, STEREO_CAMERA> This;

protected:
    gtsam::StereoPoint2 measurement_;
    boost::shared_ptr<CALIBRATION> calibration_;

public:
    StereoProjectionFactor(const gtsam::StereoPoint2& measurement, const gtsam::SharedNoiseModel& model,
                           gtsam::Key key_pose, gtsam::Key key_landmark, const boost::shared_ptr<CALIBRATION>& calibration)
        : Base(model, key_pose, key_landmark), measurement_(measurement), calibration_(calibration) {}
    virtual ~StereoProjectionFactor() {}

    gtsam::Vector evaluateError(const POSE& pose, const LANDMARK& point,
                                boost::optional<gtsam::Matrix&> H1 = boost::none, boost::optional<gtsam::Matrix&> H2 = boost::none) const override {
        try {
            STEREO_CAMERA camera(pose, calibration_);
            return (camera.project(point, H1, H2) - measurement_).vector();
        }
        catch (gtsam::StereoCheiralityException& e) {
            // Landmarks is behind the camera
            if (H1)
                *H1 = gtsam::Matrix::Zero(3, 6);
            if (H2)
                *H2 = gtsam::Matrix::Zero(3, 3);
            CV_LOG_DEBUG(&g_log_tag, "{} : Landmarks is behind the camera", e.what());
        }
        // return some large error
        return gtsam::Vector3::Constant(1e+6);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    const gtsam::StereoPoint2& measurement() const {
        return measurement_;
    }

    gtsam::Vector project(const POSE& pose, const LANDMARK& point) const override {
        STEREO_CAMERA camera(pose, calibration_);
        return camera.project(point).vector();
    }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

template<class POSE, class LANDMARK>
class PoseOptFactorBase : public gtsam::NoiseModelFactor1<POSE> {
private:
    typedef gtsam::NoiseModelFactor1<POSE> Base;
    typedef PoseOptFactorBase<POSE, LANDMARK> This;

protected:
    LANDMARK point_;
    unsigned int idx_;

public:
    PoseOptFactorBase(const LANDMARK& point,
                      unsigned int idx,
                      const gtsam::SharedNoiseModel& model,
                      gtsam::Key key_pose)
        : Base(model, key_pose), point_(point), idx_(idx) {}
    virtual ~PoseOptFactorBase() {}

    unsigned int idx() const {
        return idx_;
    }

    virtual gtsam::Vector project(const gtsam::Pose3& pose) const = 0;
};

template<class POSE, class LANDMARK, class CALIBRATION, class CAMERA>
class PoseOptFactor : public PoseOptFactorBase<POSE, LANDMARK> {
private:
    typedef PoseOptFactorBase<POSE, LANDMARK> Base;
    typedef PoseOptFactor<POSE, LANDMARK, CALIBRATION, CAMERA> This;

protected:
    gtsam::Point2 measurement_;
    boost::shared_ptr<CALIBRATION> calibration_;

public:
    PoseOptFactor(const LANDMARK& point, unsigned int idx, const gtsam::Point2& measurement, const gtsam::SharedNoiseModel& model,
                  gtsam::Key key_pose, const boost::shared_ptr<CALIBRATION>& calibration)
        : Base(point, idx, model, key_pose), measurement_(measurement), calibration_(calibration) {}
    virtual ~PoseOptFactor() {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                                boost::optional<gtsam::Matrix&> H1 = boost::none) const override {
        try {
            CAMERA camera(pose, calibration_);
            return camera.project(this->point_, H1) - measurement_;
        }
        catch (gtsam::CheiralityException& e) {
            // Landmarks is behind the camera
            if (H1)
                *H1 = gtsam::Matrix::Zero(2, 6);
            CV_LOG_DEBUG(&g_log_tag, "{} : Landmarks is behind the camera", e.what());
        }
        // return some large error
        return gtsam::Vector2::Constant(1e+6);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    const gtsam::Point2& measurement() const {
        return measurement_;
    }

    gtsam::Vector project(const gtsam::Pose3& pose) const override {
        CAMERA camera(pose, calibration_);
        return camera.project(this->point_);
    }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

template<class POSE, class LANDMARK, class CALIBRATION, class STEREO_CAMERA>
class StereoPoseOptFactor : public PoseOptFactorBase<POSE, LANDMARK> {
private:
    typedef PoseOptFactorBase<POSE, LANDMARK> Base;
    typedef StereoPoseOptFactor<POSE, LANDMARK, CALIBRATION, STEREO_CAMERA> This;

protected:
    gtsam::StereoPoint2 measurement_;
    boost::shared_ptr<CALIBRATION> calibration_;

public:
    StereoPoseOptFactor(const LANDMARK& point, unsigned int idx, const gtsam::StereoPoint2& measurement, const gtsam::SharedNoiseModel& model,
                        gtsam::Key key_pose, const boost::shared_ptr<CALIBRATION>& calibration)
        : Base(point, idx, model, key_pose), measurement_(measurement), calibration_(calibration) {}
    virtual ~StereoPoseOptFactor() {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                                boost::optional<gtsam::Matrix&> H1 = boost::none) const override {
        try {
            STEREO_CAMERA camera(pose, calibration_);
            return (camera.project(this->point_, H1) - measurement_).vector();
        }
        catch (gtsam::StereoCheiralityException& e) {
            // Landmarks is behind the camera
            if (H1)
                *H1 = gtsam::Matrix::Zero(3, 6);
            CV_LOG_DEBUG(&g_log_tag, "{} : Landmarks is behind the camera", e.what());
        }
        // return some large error
        return gtsam::Vector3::Constant(1e+6);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    const gtsam::StereoPoint2& measurement() const {
        return measurement_;
    }

    gtsam::Vector project(const gtsam::Pose3& pose) const override {
        STEREO_CAMERA camera(pose, calibration_);
        return camera.project(this->point_).vector();
    }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace internal_gtsam
} // namespace optimize
} // namespace cv::slam

#endif // SLAM_OPTIMIZE_INTERNAL_GTSAM_PROJECTION_FACTOR_H
