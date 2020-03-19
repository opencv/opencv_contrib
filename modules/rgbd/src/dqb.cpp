#include "dqb.hpp"

namespace cv {
namespace dynafu {

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

Quaternion operator*(float a, const Quaternion& q)
{
    return (Quaternion(q) *= a);
}

DualQuaternion operator*(float a, const DualQuaternion& dq)
{
    return (DualQuaternion(dq) *= a);
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

cv::Affine3f DQB(std::vector<float>& weights, std::vector<Affine3f>& transforms)
{
    size_t n = transforms.size();
    DualQuaternion blended(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0));
    for(size_t i = 0; i < n; i++)
        blended += weights[i] * DualQuaternion(transforms[i]);

    return blended.getAffine();
}


} // namespace dynafu
} // namespace cv
