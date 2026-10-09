// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "rpc_utils.hpp"

namespace cv { namespace sfm {

RPCLocalFrame::RPCLocalFrame(const Point3d& origin_, double orientation_)
    : origin(origin_), orientation(orientation_)
{
    Matx33d linear;
    Vec3d offset;
    rpc_detail::localTransform(*this, linear, offset);
}

namespace rpc_detail {

void localTransform(const RPCLocalFrame& frame, Matx33d& linear, Vec3d& offset,
                    Matx33d* inverse)
{
    offset = Vec3d(frame.origin.x, frame.origin.y, frame.origin.z);
    CV_CheckEQ(finite(offset), true, "RPC local-frame origin must be finite");
    CV_CheckEQ(std::isfinite(frame.orientation), true, "RPC local-frame orientation must be finite");
    CV_CheckGE(frame.origin.x, -180.0, "RPC origin longitude must be in [-180,180]");
    CV_CheckLE(frame.origin.x, 180.0, "RPC origin longitude must be in [-180,180]");
    CV_CheckGT(frame.origin.y, -90.0, "RPC origin latitude must be strictly between -90 and 90");
    CV_CheckLT(frame.origin.y, 90.0, "RPC origin latitude must be strictly between -90 and 90");

    // WGS84 radii of curvature. The local approximation is linear in latitude
    // and longitude; it is intentionally not the legacy VXL LVCS rotation.
    const double majorAxis = 6378137.0;
    const double flattening = 1.0 / 298.257223563;
    const double eccentricitySquared = flattening * (2.0 - flattening);
    const double latitude = frame.origin.y * (CV_PI / 180.0);
    const double sinLatitude = std::sin(latitude);
    const double denominator = 1.0 - eccentricitySquared * sinLatitude * sinLatitude;
    const double primeVertical = majorAxis / std::sqrt(denominator);
    const double meridional = primeVertical * (1.0 - eccentricitySquared) / denominator;
    const double northRadius = meridional + frame.origin.z;
    const double eastRadius = primeVertical + frame.origin.z;
    CV_CheckGT(northRadius, 0.0, "RPC height must leave a positive meridional radius");
    CV_CheckGT(eastRadius, 0.0, "RPC height must leave a positive prime-vertical radius");
    const double latitudeScale = (180.0 / CV_PI) / northRadius;
    const double longitudeScale = (180.0 / CV_PI) / (eastRadius * std::cos(latitude));
    const double cosine = std::cos(frame.orientation);
    const double sine = std::sin(frame.orientation);
    linear = Matx33d(longitudeScale * cosine, -longitudeScale * sine, 0,
                    latitudeScale * sine, latitudeScale * cosine, 0,
                    0, 0, 1);
    for (int i = 0; i < 9; ++i)
        CV_CheckEQ(std::isfinite(linear.val[i]), true, "RPC local-frame transform must be finite");
    if (inverse)
    {
        // Invert the independent angular scales and rotation analytically;
        // a general matrix inverse can underflow its determinant at large height.
        *inverse = Matx33d(cosine / longitudeScale, sine / latitudeScale, 0,
                           -sine / longitudeScale, cosine / latitudeScale, 0,
                           0, 0, 1);
        for (int i = 0; i < 9; ++i)
            CV_CheckEQ(std::isfinite(inverse->val[i]), true, "RPC inverse local-frame transform must be finite");
    }
}

} // namespace rpc_detail

void localToGeographicRPC(InputArray localPoints, const RPCLocalFrame& frame,
                         OutputArray geographicPoints)
{
    Matx33d linear;
    Vec3d offset;
    rpc_detail::localTransform(frame, linear, offset);
    Mat points = rpc_detail::pointRows(localPoints, 3);
    if (points.empty())
    {
        geographicPoints.release();
        return;
    }
    Mat output(points.rows, 1, CV_64FC3);
    for (int i = 0; i < points.rows; ++i)
    {
        const double* row = points.ptr<double>(i);
        const Vec3d point(row[0], row[1], row[2]);
        CV_CheckEQ(rpc_detail::finite(point), true, "RPC local coordinates must be finite");
        const Vec3d transformed = linear * point + offset;
        CV_CheckEQ(rpc_detail::finite(transformed), true, "RPC geographic coordinates overflowed");
        output.at<Vec3d>(i, 0) = transformed;
    }
    output.copyTo(geographicPoints);
}

void geographicToLocalRPC(InputArray geographicPoints, const RPCLocalFrame& frame,
                         OutputArray localPoints)
{
    Matx33d linear, inverse;
    Vec3d offset;
    rpc_detail::localTransform(frame, linear, offset, &inverse);
    Mat points = rpc_detail::pointRows(geographicPoints, 3);
    if (points.empty())
    {
        localPoints.release();
        return;
    }

    Mat output(points.rows, 1, CV_64FC3);
    for (int i = 0; i < points.rows; ++i)
    {
        const double* row = points.ptr<double>(i);
        const Vec3d point(row[0], row[1], row[2]);
        CV_CheckEQ(rpc_detail::finite(point), true, "RPC geographic coordinates must be finite");
        const Vec3d transformed = inverse * (point - offset);
        CV_CheckEQ(rpc_detail::finite(transformed), true, "RPC local coordinates overflowed");
        output.at<Vec3d>(i, 0) = transformed;
    }
    output.copyTo(localPoints);
}

}} // namespace cv::sfm
