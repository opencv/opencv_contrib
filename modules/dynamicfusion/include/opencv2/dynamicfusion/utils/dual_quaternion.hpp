#ifndef DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#define DYNAMIC_FUSION_DUAL_QUATERNION_HPP
#include<iostream>
#include<opencv2/dynamicfusion/utils/quaternion.hpp>

//Adapted from https://github.com/Poofjunior/QPose
/**
 * \brief a dual quaternion class for encoding transformations.
 * \details transformations are stored as first a translation; then a
 *          rotation. It is possible to switch the order. See this paper:
 *  https://www.thinkmind.org/download.php?articleid=intsys_v6_n12_2013_5
 */
namespace cv {
    namespace kfusion {
        namespace utils {
            static float epsilon() {
                return 1e-6;
            }

            template<typename T>
            class DualQuaternion {
            public:
                /**
                 * \brief default constructor.
                 */
                DualQuaternion() {};

                /**
                 * \brief constructor that takes cartesian coordinates and Euler angles as
                 *        arguments.
                 */
//            FIXME: always use Rodrigues angles, not Euler
                DualQuaternion(T x, T y, T z, T roll, T pitch, T yaw) {
                    // convert here.
                    rotation_.w_ = cos(roll / 2) * cos(pitch / 2) * cos(yaw / 2) +
                                   sin(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
                    rotation_.x_ = sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2) -
                                   cos(roll / 2) * sin(pitch / 2) * sin(yaw / 2);
                    rotation_.y_ = cos(roll / 2) * sin(pitch / 2) * cos(yaw / 2) +
                                   sin(roll / 2) * cos(pitch / 2) * sin(yaw / 2);
                    rotation_.z_ = cos(roll / 2) * cos(pitch / 2) * sin(yaw / 2) -
                                   sin(roll / 2) * cos(pitch / 2) * cos(yaw / 2);

                    translation_ = 0.5 * Quaternion<T>(0, x, y, z) * rotation_;
                }

                /**
                 * \brief constructor that takes two quaternions as arguments.
                 * \details The rotation
                 *          quaternion has the conventional encoding for a rotation as a
                 *          quaternion. The translation quaternion is a quaternion with
                 *          cartesian coordinates encoded as (0, x, y, z)
                 */
                DualQuaternion(Quaternion <T> translation, Quaternion <T> rotation) {
                    rotation_ = rotation;
                    translation_ = 0.5 * translation * rotation;
                }

                ~DualQuaternion() {};

                /**
                 * \brief store a rotation
                 * \param angle is in radians
                 */
                void encodeRotation(T angle, T x, T y, T z) {
                    rotation_.encodeRotation(angle, x, y, z);
                }

                /**
                 * \brief store a rotation
                 * \param angle is in radians
                 */
                void encodeRotation(T x, T y, T z) {
                    rotation_.encodeRotation(sqrt(x * x + y * y + z * z), x, y, z);
                }

                void encodeTranslation(T x, T y, T z) {
                    translation_ = 0.5 * Quaternion<T>(0, x, y, z) * rotation_;
                }

                /// handle accumulating error.
                void normalize() {
                    T x, y, z;
                    getTranslation(x, y, z);

                    rotation_.normalize();

                    encodeTranslation(x, y, z);
                }

                /**
                 * \brief a reference-based method for acquiring the latest
                 *        translation data.
                 */
                void getTranslation(T &x, T &y, T &z) const {
                    auto rot = rotation_;
                    rot.normalize();
                    Quaternion <T> result = 2 * translation_ * rot.conjugate();
                    /// note: inverse of a quaternion is the same as the conjugate.
                    x = result.x_;
                    y = result.y_;
                    z = result.z_;
                }

                /**
                 * \brief a reference-based method for acquiring the latest
                 *        translation data.
                 */
//FIXME: need to make sure rotation is normalized in all getTranslation functions.
                void getTranslation(Vec3f &vec3f) const {

                    auto rot = rotation_;
                    rot.normalize();
                    Quaternion <T> result = 2 * translation_ * rot.conjugate();
                    vec3f = Vec3f(result.x_, result.y_, result.z_);
                }

                Quaternion <T> getTranslation() const {
                    return 2 * translation_ * rotation_.conjugate();
                }


                /**
                 * \brief a reference-based method for acquiring the latest rotation data.
                 */
                void getEuler(T &roll, T &pitch, T &yaw) {
                    // FIXME: breaks for some value around PI.
                    roll = getRoll();
                    pitch = getPitch();
                    yaw = getYaw();
                }

                Quaternion <T> getRotation() const {
                    return rotation_;
                }


                /**
                 * \brief Extraction everything (in a nice format)
                 */
                void get6DOF(T &x, T &y, T &z, T &roll, T &pitch, T &yaw) {
                    getTranslation(x, y, z);
                    getEuler(roll, pitch, yaw);
                }

                DualQuaternion operator+(const DualQuaternion &other) {
                    DualQuaternion result;
                    result.rotation_ = rotation_ + other.rotation_;
                    result.translation_ = translation_ + other.translation_;
                    return result;
                }

                DualQuaternion operator-(const DualQuaternion &other) {
                    DualQuaternion result;
                    result.rotation_ = rotation_ - other.rotation_;
                    result.translation_ = translation_ - other.translation_;
                    return result;
                }

                DualQuaternion operator*(const DualQuaternion &other) {
                    DualQuaternion<T> result;
                    result.rotation_ = rotation_ * other.rotation_;
                    result.translation_ = (rotation_ * other.translation_) + (translation_ * other.rotation_);
                    return result;
                }

                DualQuaternion operator/(const std::pair <T, T> divisor) {
                    DualQuaternion<T> result;
                    result.rotation_ = 1 / divisor.first * rotation_;
                    result.translation_ = 1 / divisor.second * translation_;
                    return result;
                }

                /// (left) Scalar Multiplication
                /**
                 * \fn template <typename U> friend Quaternion operator*(const U scalar,
                 * \brief implements scalar multiplication for arbitrary scalar types
                 */
                template<typename U>
                friend DualQuaternion operator*(const U scalar, const DualQuaternion &q) {
                    DualQuaternion<T> result;
                    result.rotation_ = scalar * q.rotation_;
                    result.translation_ = scalar * q.translation_;
                    return result;
                }

                DualQuaternion conjugate() {
                    DualQuaternion<T> result;
                    result.rotation_ = rotation_.conjugate();
                    result.translation_ = translation_.conjugate();
                    return result;
                }

                inline DualQuaternion identity() {
                    return DualQuaternion(Quaternion<T>(0, 0, 0, 0), Quaternion<T>(0, 1, 0, 0));
                }

                void transform(Vec3f &point) // TODO: this should be a lot more generic
                {
                    Vec3f translation;
                    getTranslation(translation);
                    rotation_.rotate(point);
                    point += translation;
                }

                void from_twist(const float &r0, const float &r1, const float &r2,
                                const float &x, const float &y, const float &z) {
                    Vec3f r(r0, r1, r2), t(x, y, z);
                    float norm = sqrt(r0 * r0 + r1 * r1 + r2 * r2);
                    Quaternion <T> rotation;
                    if (norm > epsilon()) {
                        float cosNorm = cos(norm);
                        float sign = (cosNorm > 0.f) - (cosNorm < 0.f);
                        cosNorm *= sign;
                        float sinNorm_norm = sign * sin(norm) / norm;
                        rotation = Quaternion<T>(cosNorm, r0 * sinNorm_norm, r1 * sinNorm_norm, r2 * sinNorm_norm);
                    } else
                        rotation = Quaternion<T>();

                    translation_ = Quaternion<T>(0, x, y, z);
                    rotation_ = rotation;
                }

                std::pair <T, T> magnitude() {
                    DualQuaternion result = (*this) * (*this).conjugate();
                    return std::make_pair(result.rotation_.w_, result.translation_.w_);
                }

            private:
                Quaternion <T> rotation_;
                Quaternion <T> translation_;

                T position_[3] = {};    /// default initialize vector to zeros.

                T rotAxis_[3] = {};     /// default initialize vector to zeros.
                T rotAngle_;


                T getRoll() {
                    // TODO: test this!
                    return atan2(2 * ((rotation_.w_ * rotation_.x_) + (rotation_.y_ * rotation_.z_)),
                                 (1 - 2 * ((rotation_.x_ * rotation_.x_) + (rotation_.y_ * rotation_.y_))));
                }

                T getPitch() {
                    // TODO: test this!
                    return asin(2 * (rotation_.w_ * rotation_.y_ - rotation_.z_ * rotation_.x_));
                }

                T getYaw() {
                    // TODO: test this!
                    return atan2(2 * ((rotation_.w_ * rotation_.z_) + (rotation_.x_ * rotation_.y_)),
                                 (1 - 2 * ((rotation_.y_ * rotation_.y_) + (rotation_.z_ * rotation_.z_))));
                }
            };

            template<typename T>
            std::ostream &operator<<(std::ostream &os, const DualQuaternion<T> &q) {
                os << "[" << q.getRotation() << ", " << q.getTranslation() << ", " << "]" << std::endl;
                return os;
            }
        }
    }
}
#endif //DYNAMIC_FUSION_DUAL_QUATERNION_HPP