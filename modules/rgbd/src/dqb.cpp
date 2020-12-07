#include "dqb.hpp"

namespace cv {
namespace dynafu {

Vec4f rotMat2quat(Matx33f m)
{
    float tr = trace(m);
    float w, x, y, z;
    const float TRACE_THRESHOLD = 0.00000001f;
    if (tr > TRACE_THRESHOLD)
    {
        float s = sqrt(tr + 1.0f) * 2.f;
        w = 0.25f * s;
        x = (m(2, 1) - m(1, 2)) / s;
        y = (m(0, 2) - m(2, 0)) / s;
        z = (m(1, 0) - m(0, 1)) / s;
    }
    else if ((m(0, 0) > m(1, 1)) && (m(0, 0) > m(2, 2)))
    {
        float s = sqrt(1.0f + m(0, 0) - m(1, 1) - m(2, 2)) * 2.f;
        w = (m(2, 1) - m(1, 2)) / s;
        x = 0.25f * s;
        y = (m(0, 1) + m(1, 0)) / s;
        z = (m(0, 2) + m(2, 0)) / s;
    }
    else if (m(1, 1) > m(2, 2))
    {
        float s = sqrt(1.0f + m(1, 1) - m(0, 0) - m(2, 2)) * 2.f;
        w = (m(0, 2) - m(2, 0)) / s;
        x = (m(0, 1) + m(1, 0)) / s;
        y = 0.25f * s;
        z = (m(1, 2) + m(2, 1)) / s;
    }
    else
    {
        float s = sqrt(1.0f + m(2, 2) - m(0, 0) - m(1, 1)) * 2.f;
        w = (m(1, 0) - m(0, 1)) / s;
        x = (m(0, 2) + m(2, 0)) / s;
        y = (m(1, 2) + m(2, 1)) / s;
        z = 0.25f * s;
    }

    return { w, x, y, z };
}


// convert unit quaternion to rotation matrix
Matx33f quat2rotMat(Vec4f q)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    Vec3f rv(x, y, z);
    cv::Matx33f rot = 2.f * ((w * w - 0.5f) * I3 + rv * rv.t() + w * skew(rv));

    return rot;
}

} // namespace dynafu
} // namespace cv
