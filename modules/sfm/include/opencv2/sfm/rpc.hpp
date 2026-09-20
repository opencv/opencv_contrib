// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_SFM_RPC_HPP
#define OPENCV_SFM_RPC_HPP

#include <opencv2/core.hpp>

namespace cv { namespace sfm {

/** @addtogroup projection
 * @{
 */

/** @brief Ordering of the 20 coefficients of a cubic RPC polynomial.
 * Variables are normalized longitude x, latitude y and ellipsoidal height z.
 * RPC00A and RPC00B follow the corresponding NITF definitions.
 */
enum RPCCoefficientOrder
{
    RPC_ORDER_RPC00B = 0, //!< 1,x,y,z,xy,xz,yz,x^2,y^2,z^2,xyz,x^3,xy^2,xz^2,x^2y,y^3,yz^2,x^2z,y^2z,z^3
    RPC_ORDER_RPC00A = 1, //!< 1,x,y,z,xy,xz,yz,xyz,x^2,y^2,z^2,x^3,x^2y,x^2z,xy^2,y^3,y^2z,xz^2,yz^2,z^3
    RPC_ORDER_VXL = 2    //!< x^3,x^2y,x^2z,x^2,xy^2,xyz,xy,xz^2,xz,x,y^3,y^2z,y^2,yz^2,yz,y,z^3,z^2,z,1
};

/** @brief Rational polynomial camera, as used in satellite imagery.
 *
 * Each image coordinate is a ratio of two cubic polynomials. The four rows of
 * coefficients are sample numerator, sample denominator, line numerator, line
 * denominator. Image points are (sample, line), with no half-pixel adjustment.
 * World points are (longitude in degrees, latitude in degrees, WGS84 ellipsoidal
 * height in metres). The same equations can also model other Cartesian coordinates.
 *
 * Normalize world coordinates as (point - worldOffset) / worldScale; evaluate the
 * polynomial ratios; then denormalize as ratio * imageScale + imageOffset.
 * All scales must be strictly positive and all parameters finite. Coefficients
 * describe a calibrated model; fitting coefficients and reading metadata files
 * are outside this API. The default model projects (x,y,z) to (x,y).
 *
 * This implements the projection model of VXL's vpgl_rational_camera without a
 * dependency on VXL. It is distinct from CALIB_RATIONAL_MODEL lens distortion.
 */
struct CV_EXPORTS RPCModel
{
    RPCModel();
    Matx<double, 4, 20> coefficients;
    Vec3d worldOffset, worldScale;
    Vec2d imageOffset, imageScale;
    RPCCoefficientOrder order;
};

/** @brief Local linear approximation to WGS84 geographic coordinates.
 *
 * origin is (longitude degrees, latitude degrees, ellipsoidal height metres).
 * Local coordinates are metres. At orientation = 0 the axes are east, north, up.
 * Positive orientation, in radians, rotates local +x towards north. Angular
 * scales use WGS84 meridional and prime-vertical radii at origin, including its
 * height. This is a fixed linear approximation for a small neighbourhood, not
 * a terrain model, map projection, or ECEF/ENU transform. Heights remain
 * ellipsoidal. Geographic longitudes are not wrapped at the antimeridian.
 *
 * The origin must have longitude in [-180,180], latitude strictly between
 * -90 and 90 degrees, and strictly positive curvature radii after adding height.
 */
struct CV_EXPORTS RPCLocalFrame
{
    explicit RPCLocalFrame(const Point3d& origin = Point3d(), double orientation = 0);
    Point3d origin;
    double orientation;
};

/** @brief Converts local metre coordinates to geographic longitude, latitude and height.
 * @param localPoints N-by-3 CV_64F matrix or vector of Point3d (including a strided matrix view).
 * @param frame Local frame.
 * @param geographicPoints N-by-1 CV_64FC3 output, in input order.
 * Empty input produces empty output. Inputs must be finite. Invalid arguments
 * raise cv::Exception. The conversion is the inverse of geographicToLocalRPC.
 */
CV_EXPORTS void localToGeographicRPC(InputArray localPoints, const RPCLocalFrame& frame,
                                    OutputArray geographicPoints);

/** @brief Inverse of localToGeographicRPC with the same frame.
 * @param geographicPoints N-by-3 CV_64F matrix or vector of Point3d.
 * @param frame Local frame.
 * @param localPoints N-by-1 CV_64FC3 output, in metres.
 */
CV_EXPORTS void geographicToLocalRPC(InputArray geographicPoints, const RPCLocalFrame& frame,
                                    OutputArray localPoints);

/** @brief Projects points through a rational polynomial camera.
 * @param objectPoints N-by-3 CV_64F matrix or vector of Point3d.
 * @param model RPC camera.
 * @param imagePoints N-by-1 CV_64FC2 output (sample, line).
 * @param localFrame Optional frame: objectPoints are local metres when supplied;
 * otherwise they are world coordinates in the model's units.
 *
 * Strided inputs and empty batches are supported. Nonfinite inputs, invalid
 * models, zero polynomial denominators or nonfinite projections raise cv::Exception.
 * No range clipping or half-pixel offset is applied.
 */
CV_EXPORTS void projectPointsRPC(InputArray objectPoints, const RPCModel& model,
                                OutputArray imagePoints, const RPCLocalFrame* localFrame = 0);

/** @brief Backprojects image points onto a plane using a local numerical search.
 * @param imagePoints N-by-2 CV_64F matrix or vector of Point2d.
 * @param model RPC camera.
 * @param plane Plane (a,b,c,d), a*x+b*y+c*z+d=0, with a nonzero normal.
 * @param initialPoints N initial estimates, with the same format as objectPoints in projectPointsRPC.
 * @param objectPoints N-by-1 CV_64FC3 solutions. Failed entries contain NaN.
 * @param success N-by-1 CV_8U: 1 for a finite solution within errorTolerance pixels,
 * 0 for failure to converge or an undefined projection.
 * @param localFrame Optional frame. The plane, initial estimates and solutions
 * are all in local metre coordinates when this is supplied.
 * @param errorTolerance Strictly positive, finite Euclidean reprojection tolerance, in pixels.
 * @param maxIterations Positive search budget passed to cv::DownhillSolver (function evaluations).
 *
 * Optimization uses normalized world coordinates and eliminates the coordinate
 * with the largest absolute plane coefficient, including ties. The plane is
 * enforced throughout the search. There may be several valid roots; the initial
 * estimate selects a local search, not a unique solution. Failure does not prove
 * that no root exists. A small image residual does not guarantee accurate 3D
 * coordinates for an ill-conditioned camera. Invalid arguments raise cv::Exception.
 */
CV_EXPORTS void backProjectPointsRPC(InputArray imagePoints, const RPCModel& model,
                                    const Vec4d& plane, InputArray initialPoints,
                                    OutputArray objectPoints, OutputArray success,
                                    const RPCLocalFrame* localFrame = 0,
                                    double errorTolerance = 0.05, int maxIterations = 5000);

//! @}
}} // namespace cv::sfm
#endif
