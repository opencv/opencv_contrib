#ifndef __OPENCV_RGBD_DQB_HPP__
#define __OPENCV_RGBD_DQB_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv {
namespace dynafu {

class Quaternion
{
public:
    Quaternion();
    Quaternion(float w, float i, float j, float k);

    // Generate a quaternion from rotation of a Rt matrix.
    Quaternion(const Affine3f& r);

    Affine3f getRotation() const;

    float dot(const Quaternion& q) const {return q.coeff.dot(coeff);}
    float norm() const {return sqrt(dot(*this));}

    float normalize()
    {
        float n = norm();
        coeff /= n;
        return n;
    }

    Quaternion invert() const
    {
        float qn2 = dot(*this);
        return conjugate()/qn2;
    }

    float w() const {return coeff[0];}
    float i() const {return coeff[1];}
    float j() const {return coeff[2];}
    float k() const {return coeff[3];}

    Quaternion conjugate() const
    {
        return Quaternion(w(), -i(), -j(), -k());
    }

    friend Quaternion operator*(float a, const Quaternion& q);
    friend Quaternion operator*(const Quaternion& q, float a);
    friend Quaternion operator/(const Quaternion& q, float a);
    friend Quaternion operator-(const Quaternion& q);
    friend Quaternion operator+(const Quaternion& q1, const Quaternion& q2);
    friend Quaternion operator-(const Quaternion& q1, const Quaternion& q2);
    friend Quaternion operator*(const Quaternion& q1, const Quaternion& q2);
    friend Quaternion& operator+=(Quaternion& q1, const Quaternion& q2);
    friend Quaternion& operator/=(Quaternion& q, float a);

private:
    // w, i, j, k coefficients
    Vec4f coeff;
};

class DualQuaternion
{
public:
    DualQuaternion();
    DualQuaternion(const Affine3f& Rt);
    DualQuaternion(const Quaternion& q0, const Quaternion& qe);

    Quaternion real() const { return q0; }
    Quaternion dual() const { return qe; }

    void normalize();
    DualQuaternion invert() const
    {
        // inv(r+e*t) = inv(r) - e*inv(r)*t*inv(r)
        Quaternion invr = q0.invert();
        return DualQuaternion(invr, -(invr*qe*invr));
    }

    friend DualQuaternion& operator+=(DualQuaternion& q1, const DualQuaternion& q2);
    friend DualQuaternion operator*(float a, const DualQuaternion& q);

    Affine3f getAffine() const;

private:
    Quaternion q0; // rotation quaternion
    Quaternion qe; // translation quaternion
};

DualQuaternion DQB(std::vector<float>& weights, std::vector<DualQuaternion>& quats);

Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms);


} // namespace dynafu
} // namespace cv

#endif
