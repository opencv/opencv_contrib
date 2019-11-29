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
    Quaternion();
    Quaternion(float w, float i, float j, float k);

    // Generate a quaternion from rotation of a Rt matrix.
    Quaternion(const Affine3f& r);

    float normalize()
    {
        float n = (float)cv::norm(coeff);
        coeff /= n;
        return n;
    }

    Affine3f getRotation() const;

    float w() const {return coeff[0];}
    float i() const {return coeff[1];}
    float j() const {return coeff[2];}
    float k() const {return coeff[3];}

    float norm() const {return (float)cv::norm(coeff);}

    friend Quaternion operator*(float a, const Quaternion& q);
    friend Quaternion operator*(const Quaternion& q, float a);
    friend Quaternion operator/(const Quaternion& q, float a);
    friend Quaternion operator+(const Quaternion& q1, const Quaternion& q2);
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
    DualQuaternion(Quaternion& q0, Quaternion& qe);

    void normalize();

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