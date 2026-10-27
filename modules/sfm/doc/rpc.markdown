Rational polynomial cameras {#tutorial_sfm_rpc}
===========================

Satellite image products often describe the camera with rational polynomial
coefficients (RPCs), rather than a pinhole matrix. The SFM RPC API projects those
models using OpenCV point arrays and can find a point on a specified plane from
an image observation. It needs no VXL or GDAL runtime. Include
`<opencv2/sfm/rpc.hpp>` and link `opencv_sfm`.

Model and coordinate conventions
-------------------------------

`cv::sfm::RPCModel` stores four rows of twenty coefficients: sample numerator,
sample denominator, line numerator and line denominator. Set `order` to
`RPC_ORDER_RPC00B`, `RPC_ORDER_RPC00A`, or `RPC_ORDER_VXL` to match the metadata.
Sample is image **x** (column); line is image **y** (row). World coordinates are
**longitude, latitude, ellipsoidal height**, in degrees, degrees and metres.
Height above a geoid or mean sea level must first be converted to ellipsoidal
height. No half-pixel shift is added: match the pixel convention of your input.
The RPC model is unrelated to the rational radial lens-distortion model.

The mapping is:

1. Normalize world coordinates by subtracting `worldOffset` and dividing by
   `worldScale` componentwise.
2. Evaluate each cubic polynomial in the specified coefficient order.
3. Divide sample numerator by sample denominator, and line numerator by line
   denominator.
4. Multiply by `imageScale` and add `imageOffset` componentwise.

All arithmetic uses doubles. Supply a vector of `cv::Point3d`, or an N-by-3
single-channel `CV_64F` matrix, to `projectPointsRPC`. Strided matrix views are
accepted. Results use a vector of `cv::Point2d` or an N-by-1 `CV_64FC2` matrix.
All normalization scales must be strictly positive. Invalid models, nonfinite
input, and undefined forward projections raise `cv::Exception`.

Inverse projection
------------------

One image observation alone does not determine a 3D point. Call
`backProjectPointsRPC` with a plane `(a,b,c,d)` representing
`a*x + b*y + c*z + d = 0`, and one initial estimate per image point. For example,
`(0,0,1,-height)` restricts the solution to a constant ellipsoidal height.

The search uses OpenCV's `DownhillSolver` in normalized coordinates while
enforcing the plane. The solver's search budget is exposed as `maxIterations`.
The final reprojection must be finite and within `errorTolerance` pixels; solver
termination alone is not success. Always inspect the returned byte mask before
using a solution. Failed entries contain NaNs. Several roots can be valid, and
a failed local search does not prove that the requested point is unreachable.
A small image error can still correspond to a large ground error when the
camera/plane intersection is ill-conditioned.

Local coordinates
-----------------

`RPCLocalFrame` provides a small-neighbourhood **linear approximation** to WGS84
coordinates. Its origin is longitude, latitude and ellipsoidal height. Local
axes default to east, north and up, in metres. Positive orientation rotates
local +x towards north. Angular scales are derived from WGS84 radii of curvature
at the origin, including height. Geographic longitudes are not wrapped.

Use `localToGeographicRPC` and `geographicToLocalRPC` for explicit conversion,
or pass the frame to the projection functions. In the latter case, the inverse
plane, initial estimates and outputs are all expressed in local coordinates.
This frame is not an ECEF/ENU transformation, map projection, or terrain model.
For a large area, use an appropriate geodetic library and the global RPC API.

Example
-------

@include samples/rpc_camera.cpp

Build with `BUILD_EXAMPLES=ON`, then build and run `example_sfm_rpc_camera`.
The example uses synthetic coefficients and needs no image downloads, Ceres,
or visualization support. SFM's existing Eigen, Glog and Gflags dependencies
are still required to build the module.

Provenance and boundaries
-------------------------

The projection convention and coefficient permutations follow
[VXL v3.5.0 vpgl_rational_camera](https://github.com/vxl/vxl/tree/ecacd8cb08773d1dc100605e6d634706d34c9fc5/core/vpgl).
The tests reuse its rational-camera and backprojection fixtures, with exact
attribution in the test code. The VXL BSD notice is retained and installed with
the library. Inverse tests accept any geometrically valid root, rather than
requiring the root chosen by VXL's optimizer.

This is a native OpenCV API, not an exact compatibility wrapper for all VXL
behaviour. Zero normalization scales are rejected. Plane elimination handles
tied normal components. Local frame rotations are true inverses; angular scales
use analytic WGS84 curvature rather than VXL's finite-displacement approximation.
No VXL camera serialization, coefficient fitting, DEM intersection, datum
conversion, or new language bindings are supplied.
