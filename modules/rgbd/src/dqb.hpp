/*
The rotation apply formula taken from python quaternion package,
https://github.com/moble/quaternion, with the following license:

The MIT License(MIT)

Copyright(c) 2018 Michael Boyle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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

// Tools for dual quaternions and their optimization

namespace cv {
namespace dynafu {

// Singularity-free primitive functions for quaternion routines

// for quaternion exp
float sinc(float x)
{
    if (abs(x) < 0.01f)
        return 1.f - x * x / 6.f;
    else
        return sin(x) / x;
}

// for quaternion log
float invsinc(float x)
{
    if (abs(x) < 0.01f)
        return 1.f + x * x / 6.f;
    else
        return x / sin(x);
}

// for quaternion jacobian calculation
float csiii(float x)
{
    // even better up to abs(x)=0.1: -1/3 + x**2/30
    if (abs(x) < 0.01f)
        return -1.f / 3.f;
    else
        return (cos(x) - sin(x) / x) / (x * x);
}

// for dual quaternion jacobian calculation
float one15(float x)
{
    if (abs(x) < 1.f)
        return 1.f / 15.f;
    else
    {
        float x2 = x * x, x3 = x2 * x, x4 = x2 * x2, x5 = x3 * x2;
        return (-sin(x) / x3 - 3 * cos(x) / x4 + 3 * sin(x) / x5);
    }
}

// Matrix functions

// concatenate matrices vertically
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m + k, n> concatVert(const Matx<_Tp, m, n>& a, const Matx<_Tp, k, n>& b)
{
    Matx<_Tp, m + k, n> res;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(m + i, j) = b(i, j);
        }
    }
    return res;
}

// concatenate matrices horizontally
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m, n + k> concatHor(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, k>& b)
{
    Matx<_Tp, m, n + k> res;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            res(i, n + j) = b(i, j);
        }
    }
    return res;
}

// TODO URGENT: make them ?static? class methods

const Matx43f Z43 = Matx43f::zeros();
const Matx44f Z4  = Matx44f::zeros();
const Matx33f I3  = Matx33f::eye();
const Matx44f I4  = Matx44f::eye();

Matx33f skew(Vec3f v)
{
    return {    0, -v[2],  v[1],
             v[2],     0, -v[0],
            -v[1],  v[0],     0 };
}

// jacobian for rotation quaternion from known exp arg
Matx43f jExpRogArg(Vec3f er)
{
    float normv = norm(er);
    float sincv = sinc(normv);
    Vec3f up = -er * sincv;
    Matx33f m = I3 * sincv + er * er.t() * csiii(normv);
    return concatVert(up.t(), m);
}

// matrix form of Im(a)
const Matx44f M_Im { 0, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1 };

// matrix form of conjugation
const Matx44f M_Conj { 1,  0,  0,  0,
                       0, -1,  0,  0,
                       0,  0, -1,  0,
                       0,  0,  0, -1 };


Vec4f rotMat2quat(Matx33f m);
Matx33f quat2rotMat(Vec4f q);

class UnitQuaternion;

class Quaternion
{
public:
    // sic! Quaternion() != UnitQuaternion()
    Quaternion() :
        coeff(Vec4f(0.f, 0.f, 0.f, 0.f))
    { }

    Quaternion(float w, float i, float j, float k) :
        coeff(Vec4f(w, i, j, k))
    { }

    Quaternion(const Vec4f& _coeff) :
        coeff(_coeff)
    { }

    Quaternion(float w, const Vec3f& pure) :
        coeff(0, pure[0], pure[1], pure[2])
    { }

    Quaternion(const UnitQuaternion& uq) :
        coeff(uq.w(), uq.i(), uq.j(), uq.k())
    { }

    static Quaternion zero()
    {
        return { 0, 0, 0, 0 };
    }

    float w() const {return coeff[0];}
    float i() const {return coeff[1];}
    float j() const {return coeff[2];}
    float k() const {return coeff[3];}
    Vec3f vec() const { return { i(), j(), k() }; }

    float dot(const Quaternion& q) const { return q.coeff.dot(coeff); }
    float norm() const { return sqrt(dot(*this)); }

    Quaternion normalized() const
    {
        float n = norm();
        Quaternion qn(*this);
        qn.coeff /= n;
        return qn;
    }

    Quaternion inverted() const
    {
        float qn2 = dot(*this);
        return conjugated()/qn2;
    }

    Quaternion exp() const
    {
        return std::exp(w()) * Quaternion(UnitQuaternion::fromAxisAngle(Vec3f(i(), j(), k())*2.f));
    }

    Quaternion log() const
    {
        float n = norm();
        Vec3f unitLog = UnitQuaternion(*this).toAxisAngle() / 2.f;
        return Quaternion(std::log(n), unitLog[0], unitLog[1], unitLog[2]);
    }

    Quaternion conjugated() const
    {
        return Quaternion(w(), -i(), -j(), -k());
    }

    // matrix form of quaternion multiplication from left side
    Matx44f m_left()
    {
        // M_left(a)* V(b) =
        //    = (I_4 * a0 + [ 0 | -av    [    0 | 0_1x3
        //                   av | 0_3] +  0_3x1 | skew(av)]) * V(b)

        float r = w(), x = i(), y = j(), z = k();
        return { r, -x, -y, -z,
                 x,  r, -z,  y,
                 y,  z,  r, -x,
                 z, -y,  x,  r };
    }

    // matrix form of quaternion multiplication from right side
    Matx44f m_right()
    {
        // M_right(b)* V(a) =
        //    = (I_4 * b0 + [ 0 | -bv    [    0 | 0_1x3
        //                   bv | 0_3] +  0_3x1 | skew(-bv)]) * V(a)

        float r = w(), x = i(), y = j(), z = k();
        return { r, -x, -y, -z,
                 x,  r,  z, -y,
                 y, -z,  r,  x,
                 z,  y, -x,  r };
    }

    // matrix form of a*b - b*a
    // a cross product in fact
    Matx44f m_lrdiff()
    {
        // M_lrdiff(a)*V(b) =
        //    = [    0 | 0_1x3
        //       0_3x1 | skew(2 * av)]*V(b)
        return concatVert(Vec4f::zeros().t(), concatHor(Vec3f::zeros(), skew(2.f * vec())));
    }

    Quaternion& operator+=(const Quaternion& q)
    {
        coeff += q.coeff;
        return *this;
    }

    Quaternion operator+(const Quaternion& q2) const
    {
        return (Quaternion(*this) += q2);
    }

    Quaternion operator-(const Quaternion& q2) const
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

Quaternion operator*(float a, const Quaternion& q)
{
    return (Quaternion(q) *= a);
}


// Only norm-keeping operations allowed
class UnitQuaternion
{
public:
    UnitQuaternion() :
        coeff(1.f, 0.f, 0.f, 0.f)
    { }

    UnitQuaternion(const UnitQuaternion& q) :
        coeff(q.coeff)
    { }

    UnitQuaternion(const Quaternion& q) :
        coeff(q.normalized().coeff)
    { }

    // Here the full angle is used, not half as in expForm constructor
    UnitQuaternion(float alpha, Vec3f axis)
    {
        *this = fromAxisAngle(alpha, axis);
    }

    // Exponential form (half angle is used)
    UnitQuaternion(Vec3f expForm)
    {
        *this = fromAxisAngle(expForm * 2.f);
    }

    // Generate a quaternion from rotation of a Rt matrix.
    UnitQuaternion(const Affine3f& r)
    {
        coeff = rotMat2quat(r.rotation());
    }

    float w() const { return coeff[0]; }
    float i() const { return coeff[1]; }
    float j() const { return coeff[2]; }
    float k() const { return coeff[3]; }
    Vec3f vec() const { return { i(), j(), k() }; }
    Vec4f coeffs() const { return coeff;  }

    // axisAngle = axis*angle, using sinc() to avoid singularities
    static UnitQuaternion fromAxisAngle(Vec3f axisAngle)
    {
        // exp(v) = cos(norm(v)) + ijk*sin(norm(v))/norm(v)*v =
        // cos(norm(v)) + ijk*sinc(norm(v))*v
        float halfAngle = cv::norm(axisAngle)/2.f;
        float sn = sinc(halfAngle);
        UnitQuaternion uq;
        uq.coeff[0] = cos(halfAngle);
        uq.coeff[1] = sn * axisAngle[0];
        uq.coeff[2] = sn * axisAngle[1];
        uq.coeff[3] = sn * axisAngle[2];
        return uq;
    }

    static UnitQuaternion fromAxisAngle(float angle, Vec3f axis)
    {
        return fromAxisAngle(angle * axis);
    }

    Vec3f toAxisAngle() const
    {
        float ac = std::acos(w());
        return Vec3f(i(), j(), k()) * invsinc(ac) * 2.f;
    }

    Affine3f getRotation() const
    {
        return Affine3f(quat2rotMat(coeff), Vec3f());
    }

    UnitQuaternion conjugated() const
    {
        UnitQuaternion uq(*this);
        uq.coeff[1] = -uq.coeff[1];
        uq.coeff[2] = -uq.coeff[2];
        uq.coeff[3] = -uq.coeff[3];
        return uq;
    }

    UnitQuaternion inverted() const
    {
        return conjugated();
    }

    Vec3f apply(Vec3f point) const
    {
        // The recipe is taken from python quaternion package, here's the comment:
        // "
        // The most efficient formula I know of for rotating a vector by a quaternion is
        //     v' = v + 2 * r x (s * v + r x v) / m
        // where x represents the cross product, s and r are the scalar and vector parts of the quaternion,
        // respectively, and m is the sum of the squares of the components of the quaternion.
        // This requires 22 multiplications and 14 additions, as opposed to 32 and 24 for naive application
        // of `q*v*q.conj()`. In this function, I will further reduce the operation count to
        // 18 and 12 by skipping the normalization by `m`.
        // "
        Vec3f ijk(vec());
        Vec3f rotated = point + 2 * ijk.cross(w() * point + ijk.cross(point));
        return rotated;
    }

    // jacobian for rotation quaternion from known value
    Matx43f jExpRotVal()
    {
        Vec3f er = this->toAxisAngle() / 2.f;
        // upper part is slightly better optimized than in jExpRotArg() function
        float normv = norm(er);
        Matx33f m = Matx33f::eye() * sinc(normv) + er * er.t() * csiii(normv);
        return concatVert(-this->vec().t(), m);
    }

    // matrix form of quaternion multiplication from left side
    Matx44f m_left()
    {
        return Quaternion(*this).m_left();
    }

    // matrix form of quaternion multiplication from right side
    Matx44f m_right()
    {
        return Quaternion(*this).m_right();
    }

    // matrix form of a*b - b*a
    // a cross product in fact
    Matx44f m_lrdiff()
    {
        return Quaternion(*this).m_lrdiff();
    }

    UnitQuaternion operator-() const
    {
        UnitQuaternion uq = conjugated();
        uq.coeff[0] = -uq.coeff[0];
        return uq;
    }

    UnitQuaternion operator*=(const UnitQuaternion& q2)
    {
        Quaternion res = Quaternion(*this) * Quaternion(q2);
        coeff = res.coeff;
        return *this;
    }

    UnitQuaternion operator*(const UnitQuaternion& q2) const
    {
        return (UnitQuaternion(*this) *= q2);
    }

private:
    Vec4f coeff;
};


// Jacobians

// node's jacobian for exponential representation
//TODO: docs, see derivations in ipynb
Matx<float, 8, 6> j_dq_exp_arg(Vec3f w_real, Vec3f w_dual)
{
    Matx43f jexp_rot = jExpRogArg(w_real);

    float normv = norm(w_real);
    float sincv = sinc(normv);
    float csiiiv = csiii(normv);
    float one15v = one15(normv);

    Matx33f wrwr = w_real * w_real.t(), wdwr = w_dual * w_real.t();
    float dot_dr = w_dual.dot(w_real);

    Matx13f ddr_up = - w_dual.t() * (sincv * I3 + csiiiv * wrwr);
    Matx33f ddr_dn = one15v * (wrwr * wdwr) + csiiiv * (wdwr.t() + dot_dr * I3 + wdwr);
    Matx43f jexp_dual_dr = concatVert(ddr_up, ddr_dn);

    Matx43f jexp_dual_dd = jexp_rot;

    // made for typical se(3) vars order : (dual, real)
    return concatVert(concatHor(         Z43,     jexp_rot),
                      concatHor(jexp_dual_dd, jexp_dual_dr));
}


// when a node is represented as dual quaternion instead of (r, t)
Matx<float, 8, 8> j_centered(Vec3f c)
{
    // center(x) := (1+e*1/2*с)*x*(1-e*1/2*c)
    // d(center(x))/dreal = I_4 + e*1/2*M_lrdiff(c)
    // d(center(x))/ddual = e*I_4
    Matx44f dcdreal = 0.5f * Quaternion(0, c).m_lrdiff();
    return concatVert(concatHor(     I4, Z4),
                      concatHor(dcdreal, I4));
}


// centered node's jacobian
Matx<float, 8, 6> j_pernode(UnitQuaternion r, Vec3f t, Vec3f c)
{
    // getting Lie derivative of dq for current r, t
    // exp(r, t)->dq by r and t
    // exp(r, t) = [exp(r) as rot, t translation then]

    Matx<float, 8, 6> jcombine = j_combined(r);

    // center(x) := (1+e*1/2*с)*x*(1-e*1/2*c)
    // centered = DualQuaternion.from_rt_centered(r, t, c)
    // d(center(x))/dr = I_4 + e*1/2*(M_left(t) + M_lrdiff(c))
    // d(center(x))/dt = e*1/2*M_right(r)

    Matx44f dcdr = 0.5f * (Quaternion(0, t).m_left() + Quaternion(0, c).m_lrdiff());
    Matx44f dcdt = 0.5f * r.m_right();

    Matx<float, 8, 8> jcenter = concatVert(concatHor(  I4, Z4),
                                           concatHor(dcdr, dcdt));
    return jcenter * jcombine;
}


// node's Jacobian for (r, t) representation
Matx<float, 8, 6> j_combined(UnitQuaternion r)
{
    return concatVert(concatHor(r.jExpRotVal(), Z43),
                      concatHor(           Z43, concatVert(Matx13f::zeros(), I3)));
}


// TODO URGENT: add UnitDualQuaternion class
// which is built from DualQuaternion class
// but with read-only fields
class DualQuaternion
{
public:
    DualQuaternion() : qreal(1, 0, 0, 0), qdual(0, 0, 0, 0)
    { }

    DualQuaternion(const Quaternion& _real, const Quaternion& _dual) : qreal(_real), qdual(_dual)
    {}

    DualQuaternion(const Affine3f& rt)
    {
        // (q0 + e*q0) = (r + e*1/2*t*r)
        qreal = Quaternion(rt);
        Vec3f t = rt.translation();
        qdual = 0.5f*(Quaternion(0, t)*qreal);
    }

    //TODO URGENT: review all this file for new constructors
    static DualQuaternion zero()
    {
        return { Quaternion::zero(), Quaternion::zero() };
    }

    static DualQuaternion one()
    {
        return { Quaternion(1, 0, 0, 0), Quaternion::zero() };
    }

    Quaternion real() const { return qreal; }
    Quaternion dual() const { return qdual; }

    float dot(const DualQuaternion& dq) const { return dq.qreal.dot(qreal) + dq.qdual.dot(qdual); }

    Vec2f norm() const
    {
        // norm(a) + e*dot(a, b)/norm(a)
        float r = qreal.norm();
        float d = qreal.dot(qdual)/r;
        return { r, d };
    }

    DualQuaternion normalized() const
    {
        // proven analytically:
        // norm(r+e*t) = norm(r) + e*dot(r,t)/norm(r)
        // r_nr = r/norm(r), t_nr = t/norm(r)
        // normalized(r+e*t) = r_nr + e*(t_nr-r_nr*dot(r_nr,t_nr))
        // normalized(r+e*t) = (1+e*Im(t*inv(r)))*r_nr

        float realNorm = qreal.norm();
        Quaternion dualDiv = qdual / realNorm, realDiv = qreal / realNorm;
        return { realDiv, dualDiv - realDiv * (realDiv.dot(dualDiv)) };
    }

    // see deduction of exp() and log() in documentation
    // not necessary works with non-pure DQs
    DualQuaternion exp() const
    {
        Quaternion er = qreal.exp();
        Vec4f edv = jExpRogArg(qreal.vec()) * qdual.vec();
        Quaternion ed(edv);
        return { er, ed };
    }

    DualQuaternion log() const
    {
        Quaternion lr = qreal.log();
        Matx43f jexp = jExpRogArg(lr.vec());

        // J_exp_quat(w_real) * w_dual = self.dual
        // let's estimate w_dual with least squares
        Vec4f dvec = qdual.coeff;

        //TODO: get rid of it to make it faster
        Vec3f ldvec = jexp.solve(dvec, DECOMP_SVD);
        Quaternion ld(0, ldvec);

        return { lr, ld };
    }

    DualQuaternion centered(Vec3f c) const
    {
        // make a new dq from current that :
        // shifts from c to 0, performs transformation of dq (rotation then translation),
        // then shifts back to c
        // center(x) = (1 + e*1/2*с) * (r + e*d) * (1 - e*1/2*c) = r + e * (d + 1/2 * (c*r - r*c))
        // c*r - r*c == ijk*2*cross(c.vec(), r.vec())
        Vec3f cross = c.cross(qreal.vec());

        return { qreal, qdual + Quaternion(0, cross) };
    }

    // Factor out common rotation
    DualQuaternion factoredOut(DualQuaternion factor, Vec3f c) const
    {
        // generate dq so that this->centered(c) == out.centered(c)*factor
        return (*this) * (factor.invertedUnit().centered(-c));
    }

    //TODO URGENT: put it in UnitDualQuaternion
    //AND rewrite it to UnitQuaternion
    DualQuaternion invertedUnit() const
    {
        return { qreal.conjugated(), qdual.conjugated() };
    }

    //TODO URGENT: to UnitDualQuaternion
    Affine3f getRtUnit() const
    {
        UnitQuaternion qr(qreal);
        Affine3f aff = qr.getRotation();

        Quaternion t = 2.f * (qdual * (qreal.conjugated()));
        aff.translate(t.vec());
        return aff;
    }

    // TODO URGENT: how to remove this thing to normalize()+getRtUnit() ?
    // works even if this dq is not a unit dual quaternion
    Affine3f getRt() const
    {
        Affine3f aff = UnitQuaternion(qreal).getRotation();

        Quaternion t = 2.f * (qdual * (qreal.inverted()));
        aff.translate(t.vec());
        return aff;
    }

    // Get jacobian of dual quaternion
    // Has a sense only for unit dual quaternions, so:
    //TODO URGENT: put it into UnitDualQuaternion
    //TODO URGENT: return type
    auto jRt(Vec3f c, bool atZero, bool disableCentering, bool useExp, bool needR, bool needT)
    {
        //TODO URGENT: should be UnitDualQuaternion
        DualQuaternion dqEffective = atZero ? one() : (*this);

        Vec3f cc = disableCentering ? Vec3f() : c;

        Matx<float, 8, 6> j;
        //TODO URGENT: all the functions below
        if (useExp)
        {
            j = j_centered(cc) * dqEffective.j_dq_exp_val();
        }
        else
        {
            Affine3f rt = dqEffective.getRtUnit();
            j = j_pernode(dqEffective.real(), rt.translation(), cc);
        }

        // limit degrees of freedom
        bool zero1st = false, zero2nd = false;
        zero1st = (useExp && (!needT)) || (!useExp && (!needR));
        zero2nd = (useExp && (!needR)) || (!useExp && (!needT));

        auto z = Matx<float, 8, 3>::zeros();
        auto jleft  = j.get_minor<8, 3>(0, 0);
        auto jright = j.get_minor<8, 3>(0, 3);
        if (zero1st)
            j = concatHor(z, jright);
        if (zero2nd)
            j = concatHor(jleft, z);

        return j;
    }

    //TODO URGENT: make constructors for UnitDualQuaternion
    static DualQuaternion fromRt(Affine3f rt)
    {
        Quaternion r = UnitQuaternion(rt);
        Vec3f tv = rt.translation();
        Quaternion t(0, tv);
        return { r, t * r * 0.5f };
    }

    // make a new dq from existing r, t that :
    // shifts from c to 0, performs transformation of dq(rotation then translation)
    // then shifts back to c
    static DualQuaternion fromRtCentered(Affine3f rt, Vec3f c)
    {
        // r + e*1/2*(t - r*c_i*r^ + c_i)*r
        Vec3f crcr = (Matx33f::eye() - rt.rotation()) * c;
        Vec3f tvec = rt.translation() + crcr;

        Affine3f rtc(rt.rotation(), tvec);
        return fromRt(rtc);
    }

    // generate a rotation around axis by alpha angle and shift by d
    // axis is given in Plucker coordinates: n is for rotation axis, m is for moment
    // n should be unit vector orthogonal to m, if it's not true then use fromScrew()
    static DualQuaternion fromScrewUnit(float alpha, float d, Vec3f n, Vec3f m)
    {
        Quaternion rot = UnitQuaternion(alpha * 0.5f * n);
        Vec3f cross = n.cross(m);
        Vec3f tr = d * n + (1.f - cos(alpha)) * cross + sin(alpha) * m;
        Quaternion mul = 0.5f * Quaternion(0, tr) * rot;
        return { rot, mul };
    }

    // 1. normalize n
    // 2. make m orthogonal to n by removing its collinear-to-n part
    // 3. call unit version
    static DualQuaternion fromScrew(float alpha, float d, Vec3f n, Vec3f m)
    {
        Vec3f nfixed = n/cv::norm(n);
        Vec3f mfixed = m - nfixed * nfixed.dot(m);
        return fromScrewUnit(alpha, d, nfixed, mfixed);
    }

    DualQuaternion inverted() const
    {
        // inv(r+e*t) = inv(r) - e*inv(r)*t*inv(r)
        Quaternion invr = qreal.inverted();
        return { invr, -invr*qdual*invr };
    }
    
    // jacobian of normalization+application to a point
    Matx<float, 3, 8> j_normapply(Vec3f v)
    {
        Quaternion a = real(), b = dual();
        // normalize:
        // d(nr)/da = 1/norm^3(a)*M_right(a)*M_Im*M_right(a^)
        // d(nr)/db = 0
        // d(nt)/da = -2*M_Im*M_right(a^-1)*M_left(b*(a^-1))
        // d(nt)/db =  2*M_Im*M_right(a^-1)
        // apply:
        // d(apply(nr, nt, v))/dnr = 2*M_Im*M_right(v*r^)
        // d(apply(nr, nt, v))/dnt = I
        // normalize and apply can be joined: jnormapply = japply @ jnormalize
        // also all norms are factorized out
        // d(apply(norm()))/da = 2*M_Im*M_right(a^)/norm4(a)*(M_right(a*v)*M_Im*M_right(a^) - M_left(b*a^))
        // d(apply(norm()))/db = 2*M_Im*M_right(a^)/norm2(a)

        Quaternion aconj = a.conjugated();
        float norm2 = a.dot(a);
        Matx44f mul = 2.f / norm2 * aconj.m_right();
        Matx44f danda = ((a * Quaternion(0, v)).m_right() * M_Im * aconj.m_right() - (b * aconj).m_left()) / norm2;
        // concatenate and skip 1st row
        return concatHor(mul * danda, mul).get_minor<3, 8>(1, 0);
    }

    // assume dq is unit
    //TODO URGENT: to UnitDualQuaternion member
    Matx<float, 3, 8> j_apply(Vec3f v)
    {
        // d(apply(...))/dr = 2*(M_Im*M_right(v*r^) + M_left(dual)*M_conj)
        // d(apply(...))/ddual = 2*M_right(r^)
        Matx44f j_apply_dr = 2.f * (M_Im * (Quaternion(0, v) * real().conjugated()).m_right() +
                                    dual().m_left() * M_Conj);
        Matx44f j_apply_dd = 2.f * real().conjugated().m_right();

        // concatenate and skip 1st row
        return concatHor(j_apply_dr, j_apply_dd).get_minor<3, 8>(1, 0);
    }

    // node's exponential Jacobian based on its value
    Matx<float, 8, 6> j_dq_exp_val()
    {
        DualQuaternion w = log();
        return j_dq_exp_arg(w.real().vec(), w.dual().vec());
    }
    
    DualQuaternion& operator+=(const DualQuaternion& dq)
    {
        qreal += dq.qreal;
        qdual += dq.qdual;
        return *this;
    }

    DualQuaternion operator+(const DualQuaternion& b)
    {
        return (DualQuaternion(*this) += b);
    }

    DualQuaternion& operator-=(const DualQuaternion& dq)
    {
        qreal -= dq.qreal;
        qdual -= dq.qdual;
        return *this;
    }

    DualQuaternion operator-(const DualQuaternion& b) const
    {
        return (DualQuaternion(*this) -= b);
    }

    DualQuaternion operator-()
    {
        return (DualQuaternion(-qreal, -qdual));
    }

    DualQuaternion& operator*=(float a)
    {
        qreal *= a;
        qdual *= a;
        return *this;
    }

    DualQuaternion operator*(float a) const
    {
        return (DualQuaternion(*this) *= a);
    }

    DualQuaternion operator*=(const DualQuaternion& dq)
    {
        // (a1 + e*b1)*(a2 + e*b2) = a1*a2 + e*(a1*b2 + b1*a2)
        Quaternion qq0 = qreal*dq.qreal;
        Quaternion qqe = qreal*dq.qdual + qdual*dq.qreal;

        qreal = qq0, qdual = qqe;
        return *this;
    }

    DualQuaternion operator*(const DualQuaternion& b) const
    {
        return (DualQuaternion(*this) *= b);
    }

    Quaternion qreal; // rotation quaternion
    Quaternion qdual; // translation quaternion
};


DualQuaternion operator*(float a, const DualQuaternion& dq)
{
    return (DualQuaternion(dq) *= a);
}


DualQuaternion DQB(std::vector<float>& weights, std::vector<DualQuaternion>& quats)
{
    size_t n = weights.size();
    DualQuaternion blended = DualQuaternion::zero();
    for (size_t i = 0; i < n; i++)
        blended += weights[i] * quats[i];

    return blended.normalized();
}


cv::Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms)
{
    size_t n = transforms.size();
    DualQuaternion blended = DualQuaternion::zero();
    for (size_t i = 0; i < n; i++)
        blended += weights[i] * DualQuaternion(transforms[i]);

    return blended.getAffine();
}

// tries to fix dual quaternions before DQB so that they will form shortest paths
// the algorithm is described in [Kavan and Zara 2005], Kavan'08
// to make it deterministic we choose a reference dq according to relative flag
void signFix(std::vector<DualQuaternion>& dqs, bool relative = false)
{
    if (relative)
    {
        // in relative mode reference is dq with smallest squared norm
        DualQuaternion ref;
        float refnorm = std::numeric_limits<float>::max();
        for (auto dq : dqs)
        {
            float norm = dq.dot(dq);
            if (norm < refnorm)
            {
                refnorm = norm;
                ref = dq;
            }
        }
        for (auto& dq : dqs)
        {
            dq = ref.dot(dq) >= 0 ? dq : -dq;
        }
    }
    else
    {
        // in absolute mode reference is just 1 which simplifies all the procedure
        for (auto& dq : dqs)
        {
            dq = (dq.real().w() >= 0) ? dq : -dq;
        }
    }
}

//TODO URGENT: this to c++
/*

def dist_weight(p, c, csize):
    diff = p-c.vec
    norm2 = np.inner(diff, diff)
    return math.exp(-(norm2)/(2*csize))

# fixes dq set so that they interpolate in a short trajectory
def dqb_sign_fixed(dqs):
    # a way to choose pivot should be value-dependent (but any in fact)
    # let's use the one with smallest L1 norm
    pivot = DualQuaternion()
    pivotNorm = sys.float_info.max # start value should be very big
    for d in dqs:
        assert isinstance(d, DualQuaternion), "dqs should be a list of dual quaternions!"
        dv = np.concatenate((quaternion.as_float_array(d.real),
                             quaternion.as_float_array(d.dual)))
        dnorm = np.linalg.norm(dv, ord=1)
        if dnorm < pivotNorm:
            pivot = d
            pivotNorm = dnorm

    # dot product with pivot should be >= 0
    # r and -r are for the same rotation
    newDqs = []
    for d in dqs:
        dd = d
        if np.dot(quaternion.as_float_array(d.real),
                  quaternion.as_float_array(pivot.real)) < 0:
            dd = -d
        newDqs.append(dd)
    return newDqs

# dqs are effective quaternions, already centered
def dq_bend(pin, dqs, cs, csize, damping, sign_fix, effective):
    dqsum = DualQuaternion()
    wsum = 0
    for c, dq in zip(cs, dqs):
        w = dist_weight(pin, c, csize)
        dqi = dq
        if not effective:
            dqi = dqi.centered(c)
        #DEBUG
        if sign_fix:
            dqsum += w*dqi.sign_fixed()
        else:
            dqsum += w*dqi
        wsum += w

    dqsum = damped_dqsum(dqsum, len(cs), wsum, damping)

    dqn = dqsum.normalized()

    r, t = dqn.get_rt_unit()
    R = quaternion.as_rotation_matrix(r)
    pout = R @ pin.T + t.vec

    return pout



def decorrelate(jtj, nNodes):
    # no need this check anymore
    #assert jtj.shape == (block*nNodes, block*nNodes), ("each dimension should be %d" % block*nNodes)
    for i in range(nNodes):
        for j in range(nNodes):
            if i != j:
                ri = slice(block*i, block*(i+1))
                rj = slice(block*j, block*(j+1))
                jtj[ri, rj] = 0
    return jtj

*/

} // namespace dynafu
} // namespace cv

#endif
