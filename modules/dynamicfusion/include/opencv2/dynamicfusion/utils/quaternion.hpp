#ifndef DYNAMIC_FUSION_QUATERNION_HPP
#define DYNAMIC_FUSION_QUATERNION_HPP
#pragma once
#include <iostream>
#include <opencv2/dynamicfusion/types.hpp>
//Adapted from https://github.com/Poofjunior/QPose
namespace cv
{
    namespace kfusion{
        namespace utils{


            /**
             * \class Quaternion
             * \brief a templated quaternion class that also enables quick storage and
             *        retrieval of rotations encoded as a vector3 and angle.
             * \details All angles are in radians.
             * \warning This template is intended to be instantiated with a floating point
             *          data type.
             */
            template <typename T> class Quaternion
            {
            public:
                Quaternion() : w_(1), x_(0), y_(0), z_(0)
                {}

                Quaternion(T w, T x, T y, T z) : w_(w), x_(x), y_(y), z_(z)
                {}

                /**
                 * Encodes rotation from a normal
                 * @param normal
                 */
                Quaternion(const Vec3f& normal)
                {
                    Vec3f a(1, 0, 0);
                    Vec3f b(0, 1, 0);

                    Vec3f t0 = normal.cross(a);

                    if (t0.dot(t0) < 0.001f)
                        t0 = normal.cross(b);
                    t0 = cv::normalize(t0);

                    Vec3f t1 = normal.cross(t0);
                    t1 = cv::normalize(t1);

                    cv::Mat3f matrix;
                    matrix.push_back(t0);
                    matrix.push_back(t1);
                    matrix.push_back(normal);
                    w_ = sqrt(1.0 + matrix.at<float>(0,0) + matrix.at<float>(1,1) + matrix.at<float>(2,2)) / 2.0;
//                FIXME: this breaks when w_ = 0;
                    x_ = (matrix.at<float>(2,1) - matrix.at<float>(1,2)) / (w_ * 4);
                    y_ = (matrix.at<float>(0,2) - matrix.at<float>(2,0)) / (w_ * 4);
                    z_ = (matrix.at<float>(1,0) - matrix.at<float>(2,1)) / (w_ * 4);
                }

                ~Quaternion()
                {}


                /**
                 * Quaternion Rotation Properties for straightforward usage of quaternions
                 *  to store rotations.
                 */

                /**
                 * \fn void encodeRotation( T theta, T x, T y, T z)
                 * \brief Store a normalized rotation in the quaternion encoded as a rotation
                 *        of theta about the vector (x,y,z).
                 */
                void encodeRotation(T theta, T x, T y, T z)
                {
                    auto sin_half = sin(theta / 2);
                    w_ = cos(theta / 2);
                    x_ = x * sin_half;
                    y_ = y * sin_half;
                    z_ = z * sin_half;
                    normalize();
                }

                /**
                 * \fn void encodeRotation( T theta, T x, T y, T z)
                 * \brief Store a normalized rotation in the quaternion encoded as a rotation
                 *        of theta about the vector (x,y,z).
                 */
                void getRodrigues(T& x, T& y, T& z)
                {
                    if(w_ == 1)
                    {
                        x = y = z = 0;
                        return;
                    }
                    T half_theta = acos(w_);
                    T k = sin(half_theta) * tan(half_theta);
                    x = x_ / k;
                    y = y_ / k;
                    z = z_ / k;
                }


                /**
                 * \fn void rotate( T& x, T& y, T& z)
                 * \brief rotate a vector3 (x,y,z) by the angle theta about the axis
                 * (U_x, U_y, U_z) stored in the quaternion.
                 */
                void rotate(T& x, T& y, T& z)
                {
                    Quaternion<T> q = (*this);
                    Quaternion<T> qStar = (*this).conjugate();
                    Quaternion<T> rotatedVal = q * Quaternion(0, x, y, z) * qStar;

                    x = rotatedVal.x_;
                    y = rotatedVal.y_;
                    z = rotatedVal.z_;
                }

                /**
                /**
                 * \fn void rotate( T& x, T& y, T& z)
                 * \brief rotate a vector3 (x,y,z) by the angle theta about the axis
                 * (U_x, U_y, U_z) stored in the quaternion.
                 */
                void rotate(Vec3f& v) const
                {
                    auto rot= *this;
                    rot.normalize();
                    Vec3f q_vec(rot.x_, rot.y_, rot.z_);
                    v += (q_vec*2.f).cross( q_vec.cross(v) + v*rot.w_ );
                }

                /**
                 * Quaternion Mathematical Properties
                 * implemented below
                 **/

                Quaternion operator+(const Quaternion& other)
                {
                    return Quaternion(  (w_ + other.w_),
                                        (x_ + other.x_),
                                        (y_ + other.y_),
                                        (z_ + other.z_));
                }

                Quaternion operator-(const Quaternion& other)
                {
                    return Quaternion((w_ - other.w_),
                                      (x_ - other.x_),
                                      (y_ - other.y_),
                                      (z_ - other.z_));
                }

                Quaternion operator-()
                {
                    return Quaternion(-w_, -x_, -y_, -z_);
                }

                bool operator==(const Quaternion& other) const
                {
                    return (w_ == other.w_) && (x_ == other.x_) && (y_ == other.y_) && (z_ == other.z_);
                }

                /**
                 * \fn template <typename U> friend Quaternion operator*(const U scalar,
                 *                                                       const Quaternion& q)
                 * \brief implements scalar multiplication for arbitrary scalar types.
                 */
                template <typename U> friend Quaternion operator*(const U scalar, const Quaternion& other)
                {
                    return Quaternion<T>((scalar * other.w_),
                                         (scalar * other.x_),
                                         (scalar * other.y_),
                                         (scalar * other.z_));
                }

                template <typename U> friend Quaternion operator/(const Quaternion& q, const U scalar)
                {
                    return (1 / scalar) * q;
                }

                /// Quaternion Product
                Quaternion operator*(const Quaternion& other)
                {
                    return Quaternion(
                            ((w_*other.w_) - (x_*other.x_) - (y_*other.y_) - (z_*other.z_)),
                            ((w_*other.x_) + (x_*other.w_) + (y_*other.z_) - (z_*other.y_)),
                            ((w_*other.y_) - (x_*other.z_) + (y_*other.w_) + (z_*other.x_)),
                            ((w_*other.z_) + (x_*other.y_) - (y_*other.x_) + (z_*other.w_))
                    );
                }

                /**
                 * \fn static T dotProduct(Quaternion q1, Quaternion q2)
                 * \brief returns the dot product of two quaternions.
                 */
                T dotProduct(Quaternion other)
                {
                    return 0.5 * ((conjugate() * other) + (*this) * other.conjugate()).w_;
                }

                /// Conjugate
                Quaternion conjugate() const
                {
                    return Quaternion<T>(w_, -x_, -y_, -z_);
                }

                T norm()
                {
                    return sqrt((w_ * w_) + (x_ * x_) + (y_ * y_) + (z_ * z_));
                }

                /**
                 * \fn void normalize()
                 * \brief normalizes the quaternion to magnitude 1
                 */
                void normalize()
                {
                    // should never happen unless the Quaternion<T> wasn't initialized
                    // correctly.
//                CV_Assert( !((w_ == 0) && (x_ == 0) && (y_ == 0) && (z_ == 0)));
                    T theNorm = norm();
//                CV_Assert(theNorm > 0);
                    (*this) = (1.0/theNorm) * (*this);
                }

                /**
                 * \fn template <typename U> friend std::ostream& operator <<
                 *                                  (std::ostream& os, const Quaternion<U>& q);
                 * \brief a templated friend function for printing quaternions.
                 * \details T cannot be used as dummy parameter since it would be shared by
                 *          the class, and this function is not a member function.
                 */
                template <typename U> friend std::ostream& operator << (std::ostream& os, const Quaternion<U>& q)
                {
                    os << "(" << q.w_ << ", " << q.x_ << ", " <<  q.y_ << ", " << q.z_ << ")";
                    return os;
                }
                //TODO: shouldn't have Vec3f but rather Vec3<T>. Not sure how to determine this later

                T w_;
                T x_;
                T y_;
                T z_;
            };
        }
    }
}
#endif // DYNAMIC_FUSION_QUATERNION_HPP