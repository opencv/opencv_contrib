#include "dqb.hpp"

namespace cv {
namespace dynafu {

Quaternion::Quaternion() : coeff(Vec4f(1.f, 0.f, 0.f, 0.f))
{}

Quaternion::Quaternion(float w, float i, float j, float k) : coeff(Vec4f(w, i, j, k))
{}

Quaternion::Quaternion(const Affine3f& r)
{
    // Compute trace of matrix
    float T = trace(r.matrix);

    float S, X, Y, Z, W;

    if ( T > 0.00000001f ) // to avoid large distortions!
    {
        S = sqrt(T) * 2.f;
        X = (r.matrix(1, 2) - r.matrix(2, 1)) / S;
        Y = (r.matrix(2, 0) - r.matrix(0, 2)) / S;
        Z = (r.matrix(0, 1) - r.matrix(1, 0)) / S;
        W = 0.25f * S;
    }
    else
    {
        if (r.matrix(0, 0) > r.matrix(1, 1) && r.matrix(0, 0) > r.matrix(2, 2))
        {
            // Column 0 :
            S  = sqrt(1.0f + r.matrix(0,0) - r.matrix(1,1) - r.matrix(2,2)) * 2.f;
            X = 0.25f * S;
            Y = (r.matrix(1, 0) + r.matrix(0, 1)) / S;
            Z = (r.matrix(0, 2) + r.matrix(2, 0)) / S;
            W = (r.matrix(2, 1) - r.matrix(1, 2)) / S;
        }
        else if (r.matrix(1, 1) > r.matrix(2, 2))
        {
            // Column 1 :
            S  = sqrt(1.0f + r.matrix(1,1) - r.matrix(0,0) - r.matrix(2,2)) * 2.f;
            X = (r.matrix(1, 0) + r.matrix(0, 1)) / S;
            Y = 0.25f * S;
            Z = (r.matrix(2, 1) + r.matrix(1, 2)) / S;
            W = (r.matrix(0, 2) - r.matrix(2, 0)) / S;
        }
        else
        {   // Column 2 :
            S  = sqrt( 1.0f + r.matrix(2, 2) - r.matrix(0, 0) - r.matrix(1, 1)) * 2.f;
            X = (r.matrix(0, 2) + r.matrix(2, 0)) / S;
            Y = (r.matrix(2, 1) + r.matrix(1, 2)) / S;
            Z = 0.25f * S;
            W = (r.matrix(1,0) - r.matrix(0, 1)) / S;
        }
    }

    coeff = Vec4f(W, -X, -Y, -Z);
}

Affine3f Quaternion::getRotation() const
{
    // TODO: assume the norm is 1
    float w = coeff[0], x = -coeff[1], y = -coeff[2], z = -coeff[3];
    float xx = x * x, xy = x * y, xz = x * z, xw = x * w;
    float yy = y * y, yz = y * z, yw = y * w, zz = z * z;
    float zw = z * w;

    // rot = (ww-(ww+xx+yy+zz))*I_3 + 2*(x, y, z)*(x, y, z)^T + 2*w*skewsym(x, y, z)

    Matx33f rot = Matx33f::eye() + 2.f*Matx33f(-yy - zz,  xy + zw,  xz - yw,
                                                xy - zw, -xx - zz,  yz + xw,
                                                xz + yw,  yz - xw, -xx - yy);

    Affine3f Rt = Affine3f(rot, Vec3f::all(0));
    return Rt;
}

Quaternion operator*(float a, const Quaternion& q)
{
    Vec4f newQ = a*q.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
}

Quaternion operator*(const Quaternion& q, float a)
{
    return a*q;
}

Quaternion operator/(const Quaternion& q, float a)
{
    Vec4f newQ = q.coeff/a;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
}

Quaternion operator+(const Quaternion& q1, const Quaternion& q2)
{
    Vec4f newQ = q1.coeff + q2.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
}

Quaternion operator-(const Quaternion& q)
{
    return Quaternion(-q.coeff[0], -q.coeff[1], -q.coeff[2], -q.coeff[3]);
}

Quaternion operator-(const Quaternion& q1, const Quaternion& q2)
{
    Vec4f newQ = q1.coeff - q2.coeff;
    return Quaternion(newQ[0], newQ[1], newQ[2], newQ[3]);
}

Quaternion& operator+=(Quaternion& q1, const Quaternion& q2)
{
    q1.coeff += q2.coeff;
    return q1;
}

Quaternion operator*(const Quaternion& a, const Quaternion& b)
{
    // [a0, av]*[b0, bv] = a0*b0 - dot(av, bv) + ijk*(a0*bv + b0*av + cross(av, bv))
    Vec3f av(a.i(), a.j(), a.k()), bv(b.i(), b.j(), b.k());
    float w = a.w()*b.w() - av.dot(bv);
    Vec3f ijk = a.w()*bv + b.w()*av + av.cross(bv);

    return Quaternion(w, ijk[0], ijk[1], ijk[2]);
}

Quaternion& operator/=(Quaternion& q, float a)
{
    q.coeff /= a;
    return q;
}

///////////////////////////////////////////////
/// Dual Quaternions
///////////////////////////////////////////////

DualQuaternion::DualQuaternion() : q0(1, 0, 0, 0), qe(0, 0, 0, 0)
{}

DualQuaternion::DualQuaternion(const Affine3f& rt)
{
    // (q0 + e*q0) = (r + e*1/2*t*r)
    q0 = Quaternion(rt);
    Vec3f t = rt.translation();
    qe = 0.5f*(Quaternion(0, t[0], t[1], t[2])*q0);
}

DualQuaternion::DualQuaternion(const Quaternion& _q0, const Quaternion& _qe) : q0(_q0), qe(_qe)
{}

void DualQuaternion::normalize()
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

DualQuaternion& operator+=(DualQuaternion& q1, const DualQuaternion& q2)
{
    q1.q0 += q2.q0;
    q1.qe += q2.qe;
    return q1;
}

DualQuaternion operator*(float a, const DualQuaternion& q)
{
    Quaternion newQ0 = a*q.q0;
    Quaternion newQe = a*q.qe;
    return DualQuaternion(newQ0, newQe);
}

Affine3f DualQuaternion::getAffine() const
{
    float norm = q0.norm();

    Affine3f Rt = (q0/norm).getRotation();
    // for cases when DualQuaternion's norm is 1:
    // Quaternion t = 2.f*(qe*(q0.conjugate()));
    // common case for any norm:
    Quaternion t = 2.f*(qe*(q0.invert()));

    return Rt.translate(Vec3f(t.i(), t.j(), t.k()));
}

DualQuaternion DQB(std::vector<float>& weights, std::vector<DualQuaternion>& quats)
{
    size_t n = weights.size();
    DualQuaternion blended(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0));
    for(size_t i = 0; i < n; i++)
        blended += weights[i] * quats[i];

    blended.normalize();
    return blended;
}

Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms)
{
    size_t n = transforms.size();
    DualQuaternion blended(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0));
    for(size_t i = 0; i < n; i++)
        blended += weights[i] * DualQuaternion(transforms[i]);

    return blended.getAffine();
}


} // namespace dynafu
} // namespace cv
