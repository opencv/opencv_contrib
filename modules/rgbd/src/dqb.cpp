#include "dqb.hpp"

namespace cv {
namespace dynafu {

Quaternion::Quaternion() : coeff(Vec4f(0.f, 0.f, 0.f, 0.f))
{}

Quaternion::Quaternion(float w, float i, float j, float k) : coeff(Vec4f(w, i, j, k))
{}

Quaternion::Quaternion(const Affine3f& r)
{
    // Compute trace of matrix
    float T = (float)trace(r.matrix);

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
    float W = coeff[0], X = -coeff[1], Y = -coeff[2], Z = -coeff[3];
    float xx = X * X, xy = X * Y, xz = X * Z, xw = X * W;
    float yy = Y * Y, yz = Y * Z, yw = Y * W, zz = Z * Z;
    float zw = Z * W;

    Matx33f rot(1.f - 2.f * (yy + zz),  2.f * (xy + zw),        2.f * (xz - yw),
                2.f * (xy - zw),        1.f - 2.f * (xx + zz),  2.f * (yz + xw),
                2.f * (xz + yw),        2.f * (yz - xw),        1.f - 2.f * (xx + yy));

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

Quaternion& operator+=(Quaternion& q1, const Quaternion& q2)
{
    q1.coeff += q2.coeff;
    return q1;
}

Quaternion& operator/=(Quaternion& q, float a)
{
    q.coeff /= a;
    return q;
}



DualQuaternion::DualQuaternion() : q0(), qe()
{}

DualQuaternion::DualQuaternion(const Affine3f& rt)
{
    q0 = Quaternion(rt);
    Vec3f t = rt.translation();
    float w = -0.5f*( t[0] * q0.i() + t[1] * q0.j() + t[2] * q0.k());
    float i =  0.5f*( t[0] * q0.w() + t[1] * q0.k() - t[2] * q0.j());
    float j =  0.5f*(-t[0] * q0.k() + t[1] * q0.w() + t[2] * q0.i());
    float k =  0.5f*( t[0] * q0.j() - t[1] * q0.i() + t[2] * q0.w());
    qe = Quaternion(w, i, j, k);
}

DualQuaternion::DualQuaternion(Quaternion& _q0, Quaternion& _qe) : q0(_q0), qe(_qe)
{}

void DualQuaternion::normalize()
{
    float n = q0.normalize();
    qe /= n;
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
    Vec3f t(0.f, 0.f, 0.f);
    t[0] = 2.f*(-qe.w()*q0.i() + qe.i()*q0.w() - qe.j()*q0.k() + qe.k()*q0.j()) / norm;
    t[1] = 2.f*(-qe.w()*q0.j() + qe.i()*q0.k() + qe.j()*q0.w() - qe.k()*q0.i()) / norm;
    t[2] = 2.f*(-qe.w()*q0.k() - qe.i()*q0.j() + qe.j()*q0.i() + qe.k()*q0.w()) / norm;

    return Rt.translate(t);
}

DualQuaternion DQB(std::vector<float>& weights, std::vector<DualQuaternion>& quats)
{
    size_t n = weights.size();
    DualQuaternion blended;
    for(size_t i = 0; i < n; i++)
        blended += weights[i] * quats[i];

    blended.normalize();
    return blended;
}

Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms)
{
    size_t n = transforms.size();
    std::vector<DualQuaternion> quats(n);

    std::transform(transforms.begin(), transforms.end(),
                   quats.begin(), [](const Affine3f& rt){return DualQuaternion(rt);});

    DualQuaternion blended = DQB(weights, quats);
    return blended.getAffine();
}


} // namespace dynafu
} // namespace cv