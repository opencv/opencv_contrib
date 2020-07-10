#include "dqb.hpp"

namespace cv {
namespace dynafu {

Vec4f rotMat2quat(Matx33f m)
{
    // Compute trace of matrix
    float tr = trace(m);

    float S, X, Y, Z, W;

    const float TRACE_THRESHOLD = 0.00000001f;
    if (tr > TRACE_THRESHOLD) // to avoid large distortions!
    {
        S = sqrt(tr) * 2.f;
        X = (m(1, 2) - m(2, 1)) / S;
        Y = (m(2, 0) - m(0, 2)) / S;
        Z = (m(0, 1) - m(1, 0)) / S;
        W = 0.25f * S;
    }
    else
    {
        if (m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2))
        {
            // Column 0 :
            S = sqrt(1.0f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.f;
            X = 0.25f * S;
            Y = (m(1, 0) + m(0, 1)) / S;
            Z = (m(0, 2) + m(2, 0)) / S;
            W = (m(2, 1) - m(1, 2)) / S;
        }
        else if (m(1, 1) > m(2, 2))
        {
            // Column 1 :
            S = sqrt(1.0f + m(1, 1) - m(0, 0) - m(2, 2)) * 2.f;
            X = (m(1, 0) + m(0, 1)) / S;
            Y = 0.25f * S;
            Z = (m(2, 1) + m(1, 2)) / S;
            W = (m(0, 2) - m(2, 0)) / S;
        }
        else
        {   // Column 2 :
            S = sqrt(1.0f + m(2, 2) - m(0, 0) - m(1, 1)) * 2.f;
            X = (m(0, 2) + m(2, 0)) / S;
            Y = (m(2, 1) + m(1, 2)) / S;
            Z = 0.25f * S;
            W = (m(1, 0) - m(0, 1)) / S;
        }
    }

    return Vec4f(W, -X, -Y, -Z);
}


// convert unit quaternion to rotation matrix
Matx33f quat2rotMat(Vec4f q)
{
    float w = q[0], x = -q[1], y = -q[2], z = -q[3];
    float xx = x * x, xy = x * y, xz = x * z, xw = x * w;
    float yy = y * y, yz = y * z, yw = y * w, zz = z * z;
    float zw = z * w;

    // rot = (ww-(ww+xx+yy+zz))*I_3 + 2*(x, y, z)*(x, y, z)^T + 2*w*skewsym(x, y, z)

    cv::Matx33f rot = cv::Matx33f::eye() + 2.f * cv::Matx33f(-yy - zz,  xy + zw,  xz - yw,
                                                              xy - zw, -xx - zz,  yz + xw,
                                                              xz + yw,  yz - xw, -xx - yy);

    return rot;
}

} // namespace dynafu
} // namespace cv
