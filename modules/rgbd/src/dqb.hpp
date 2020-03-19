/*
The code for Dual Quaternion Blending provided here is a modified
version of the sample codes by Arkan.

Copyright (c) 2019 Arkan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef __OPENCV_RGBD_DQB_HPP__
#define __OPENCV_RGBD_DQB_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

namespace cv {
namespace dynafu {

class Quaternion
{
public:
    Quaternion() :
        coeff(Vec4f(1.f, 0.f, 0.f, 0.f))
    { }

    Quaternion(float w, float i, float j, float k) :
        coeff(Vec4f(w, i, j, k))
    { }

    // Generate a quaternion from rotation of a Rt matrix.
    Quaternion(const cv::Affine3f& r);

    Affine3f getRotation() const
    {
        // TODO: assume the norm is 1
        float w = coeff[0], x = -coeff[1], y = -coeff[2], z = -coeff[3];
        float xx = x * x, xy = x * y, xz = x * z, xw = x * w;
        float yy = y * y, yz = y * z, yw = y * w, zz = z * z;
        float zw = z * w;

        // rot = (ww-(ww+xx+yy+zz))*I_3 + 2*(x, y, z)*(x, y, z)^T + 2*w*skewsym(x, y, z)

        cv::Matx33f rot = cv::Matx33f::eye() + 2.f*cv::Matx33f(-yy - zz,  xy + zw,  xz - yw,
                                                                xy - zw, -xx - zz,  yz + xw,
                                                                xz + yw,  yz - xw, -xx - yy);

        cv::Affine3f Rt = cv::Affine3f(rot, cv::Vec3f::all(0));
        return Rt;
    }

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

    Quaternion& operator+=(const Quaternion& q)
    {
        coeff += q.coeff;
        return *this;
    }

    Quaternion operator+(const Quaternion& q2)
    {
        return (Quaternion(*this) += q2);
    }

    Quaternion operator-(const Quaternion& q2)
    {
        return (Quaternion(*this) -= q2);
    }

    Quaternion& operator-=(const Quaternion& q)
    {
        coeff -= q.coeff;
        return *this;
    }

    Quaternion operator-() const
    {
        return Quaternion(-coeff[0], -coeff[1], -coeff[2], -coeff[3]);
    }

    Quaternion& operator*=(float a)
    {
        coeff *= a;
        return *this;
    }

    Quaternion operator*(float a) const
    {
        return (Quaternion(*this) *= a);
    }

    Quaternion& operator/=(float a)
    {
        coeff /= a;
        return *this;
    }

    Quaternion operator/(float a) const
    {
        return (Quaternion(*this) /= a);
    }

    Quaternion operator*=(const Quaternion& b)
    {
        Quaternion a = *this;
        Vec3f av(a.i(), a.j(), a.k()), bv(b.i(), b.j(), b.k());

        // [a0, av]*[b0, bv] = a0*b0 - dot(av, bv) + ijk*(a0*bv + b0*av + cross(av, bv))
        float w = a.w()*b.w() - av.dot(bv);
        Vec3f ijk = a.w()*bv + b.w()*av + av.cross(bv);

        coeff = Vec4f(w, ijk[0], ijk[1], ijk[2]);
        return *this;
    }

    Quaternion operator*(const Quaternion& q2) const
    {
        return (Quaternion(*this) *= q2);
    }

    // w, i, j, k coefficients
    Vec4f coeff;
};

Quaternion operator*(float a, const Quaternion& q);

class DualQuaternion
{
public:
    DualQuaternion() : q0(1, 0, 0, 0), qe(0, 0, 0, 0)
    { }

    DualQuaternion(const Quaternion& _q0, const Quaternion& _qe) : q0(_q0), qe(_qe)
    {}

    DualQuaternion(const Affine3f& rt)
    {
        // (q0 + e*q0) = (r + e*1/2*t*r)
        q0 = Quaternion(rt);
        Vec3f t = rt.translation();
        qe = 0.5f*(Quaternion(0, t[0], t[1], t[2])*q0);
    }

    Quaternion real() const { return q0; }
    Quaternion dual() const { return qe; }

    void normalize()
    {
        // norm(r+e*t) = norm(r) + e*dot(r,t)/norm(r)
        // r_nr = r/norm(r), t_nr = t/norm(r)
        // normalized(r+e*t) = r_nr + e*(t_nr-r_nr*dot(r_nr,t_nr))
        // normalized(r+e*t) = (1+e*Im(t*inv(r)))*r_nr

        float q0norm = q0.norm();
        Quaternion qediv = qe/q0norm, q0div = q0/q0norm;
        q0 = q0div;
        qe = qediv - q0div * (q0div.dot(qediv));
    }

    DualQuaternion invert() const
    {
        // inv(r+e*t) = inv(r) - e*inv(r)*t*inv(r)
        Quaternion invr = q0.invert();
        return DualQuaternion(invr, -(invr*qe*invr));
    }

    Affine3f getAffine() const
    {
        float norm = q0.norm();

        Affine3f Rt = (q0/norm).getRotation();
        // for cases when DualQuaternion's norm is 1:
        // Quaternion t = 2.f*(qe*(q0.conjugate()));
        // common case for any norm:
        Quaternion t = 2.f*(qe*(q0.invert()));

        return Rt.translate(Vec3f(t.i(), t.j(), t.k()));
    }

    DualQuaternion& operator+=(const DualQuaternion& dq)
    {
        q0 += dq.q0;
        qe += dq.qe;
        return *this;
    }

    DualQuaternion operator+(const DualQuaternion& b)
    {
        return (DualQuaternion(*this) += b);
    }

    DualQuaternion& operator*=(float a)
    {
        q0 *= a;
        qe *= a;
        return *this;
    }

    DualQuaternion operator*(float a)
    {
        return (DualQuaternion(*this) *= a);
    }

    DualQuaternion operator*=(const DualQuaternion& dq)
    {
        // (a1 + e*b1)*(a2 + e*b2) = a1*a2 + e*(a1*b2 + b1*a2)
        Quaternion qq0 = q0*dq.q0;
        Quaternion qqe = q0*dq.qe + qe*dq.q0;

        q0 = qq0, qe = qqe;
        return *this;
    }

    DualQuaternion operator*(const DualQuaternion& b)
    {
        return (DualQuaternion(*this) *= b);
    }

    Quaternion q0; // rotation quaternion
    Quaternion qe; // translation quaternion
};

DualQuaternion operator*(float a, const DualQuaternion& dq);

DualQuaternion DQB(std::vector<float>& weights, std::vector<DualQuaternion>& quats);

cv::Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms);


} // namespace dynafu
} // namespace cv

#endif
