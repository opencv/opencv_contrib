// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Polynomial ordering and projection adapted from VXL v3.5.0:
// https://github.com/vxl/vxl/tree/ecacd8cb08773d1dc100605e6d634706d34c9fc5/core/vpgl
// vpgl_rational_camera.h/.hxx, originally by Joseph Mundy.
// Original notice (core/vxl_copyright.h):
// Copyright 2000-2013 VXL Contributors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of their
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#include "precomp.hpp"
#include "rpc_utils.hpp"
#include <opencv2/core/optim.hpp>
#include <algorithm>
#include <limits>

namespace cv { namespace sfm {

RPCModel::RPCModel()
    : coefficients(Matx<double, 4, 20>::zeros()), worldOffset(0, 0, 0), worldScale(1, 1, 1),
      imageOffset(0, 0), imageScale(1, 1), order(RPC_ORDER_RPC00B)
{
    coefficients(0, 1) = coefficients(2, 2) = 1;
    coefficients(1, 0) = coefficients(3, 0) = 1;
}

namespace {

using rpc_detail::finite;
using rpc_detail::pointRows;

struct PreparedRPC
{
    Matx<double, 4, 20> coefficients;
    Vec3d worldOffset, worldScale, frameOffset;
    Vec2d imageOffset, imageScale;
    Matx33d frameLinear, inputFromNormalized;
    Vec3d inputOffset;

    PreparedRPC(const RPCModel& model, const RPCLocalFrame* frame)
        : worldOffset(model.worldOffset), worldScale(model.worldScale),
          imageOffset(model.imageOffset), imageScale(model.imageScale)
    {
        CV_CheckGE(static_cast<int>(model.order), static_cast<int>(RPC_ORDER_RPC00B), "Invalid RPC coefficient order");
        CV_CheckLE(static_cast<int>(model.order), static_cast<int>(RPC_ORDER_VXL), "Invalid RPC coefficient order");
        for (int i = 0; i < 3; ++i)
            CV_CheckEQ(std::isfinite(worldOffset[i]) && std::isfinite(worldScale[i]) && worldScale[i] > 0, true,
                     "RPC world offsets must be finite and scales strictly positive");
        for (int i = 0; i < 2; ++i)
            CV_CheckEQ(std::isfinite(imageOffset[i]) && std::isfinite(imageScale[i]) && imageScale[i] > 0, true,
                     "RPC image offsets must be finite and scales strictly positive");

        // Index in the supplied order for each monomial in VXL order.
        static const int orders[3][20] = {
            {11, 14, 17, 7, 12, 10, 4, 13, 5, 1, 15, 18, 8, 16, 6, 2, 19, 9, 3, 0},
            {11, 12, 13, 8, 14, 7, 4, 17, 5, 1, 15, 16, 9, 18, 6, 2, 19, 10, 3, 0},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
        };
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 20; ++col)
            {
                double value = model.coefficients(row, orders[model.order][col]);
                CV_CheckEQ(std::isfinite(value), true, "RPC coefficients must be finite");
                coefficients(row, col) = value;
            }

        frameLinear = Matx33d::eye();
        frameOffset = Vec3d(0, 0, 0);
        Matx33d inverse = Matx33d::eye();
        if (frame)
            rpc_detail::localTransform(*frame, frameLinear, frameOffset, &inverse);
        inputFromNormalized = inverse * Matx33d::diag(worldScale);
        inputOffset = inverse * (worldOffset - frameOffset);
    }

    Vec3d normalize(const Vec3d& input) const
    {
        Vec3d world = frameLinear * input + frameOffset - worldOffset;
        return Vec3d(world[0] / worldScale[0], world[1] / worldScale[1], world[2] / worldScale[2]);
    }

    bool projectNormalized(const Vec3d& point, Vec2d& image) const
    {
        if (!finite(point))
            return false;
        double x = point[0], y = point[1], z = point[2];
        double xx = x*x, yy = y*y, zz = z*z;
        double terms[20] = {x*xx, xx*y, xx*z, xx, x*yy, x*y*z, x*y, x*zz, x*z, x,
                            y*yy, yy*z, yy, y*zz, y*z, y, z*zz, zz, z, 1};
        double polynomials[4] = {0, 0, 0, 0};
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 20; ++col)
                polynomials[row] += coefficients(row, col) * terms[col];
        if (polynomials[1] == 0 || polynomials[3] == 0)
            return false;
        image = Vec2d(polynomials[0] / polynomials[1] * imageScale[0] + imageOffset[0],
                      polynomials[2] / polynomials[3] * imageScale[1] + imageOffset[1]);
        return std::isfinite(image[0]) && std::isfinite(image[1]);
    }
};

class PlaneProjection : public MinProblemSolver::Function
{
public:
    PlaneProjection(const PreparedRPC& camera_, const Vec4d& inputPlane)
        : camera(camera_), eliminated(0)
    {
        double scale = 0;
        for (int i = 0; i < 4; ++i)
        {
            CV_CheckEQ(std::isfinite(inputPlane[i]), true, "RPC plane coefficients must be finite");
            if (i < 3)
                scale = std::max(scale, std::abs(inputPlane[i]));
        }
        CV_CheckGT(scale, 0., "RPC plane normal must be nonzero");
        Vec3d normal(inputPlane[0] / scale, inputPlane[1] / scale, inputPlane[2] / scale);
        Vec3d transformed = camera.inputFromNormalized.t() * normal;
        double offset = inputPlane[3] / scale + normal.dot(camera.inputOffset);
        CV_CheckEQ(finite(transformed) && std::isfinite(offset), true, "RPC normalized plane must be finite");
        for (int i = 1; i < 3; ++i)
            if (std::abs(transformed[i]) > std::abs(transformed[eliminated]))
                eliminated = i;
        CV_CheckGT(std::abs(transformed[eliminated]), 0., "RPC normalized plane must be nondegenerate");
        for (int i = 0, j = 0; i < 3; ++i)
            if (i != eliminated)
                axes[j++] = i;
        plane = Vec4d(transformed[0], transformed[1], transformed[2], offset) / transformed[eliminated];
    }

    int getDims() const CV_OVERRIDE { return 2; }

    Vec3d point(const double* parameters) const
    {
        Vec3d result;
        result[axes[0]] = parameters[0];
        result[axes[1]] = parameters[1];
        result[eliminated] = -plane[3] - plane[axes[0]] * parameters[0] - plane[axes[1]] * parameters[1];
        return result;
    }

    double calc(const double* parameters) const CV_OVERRIDE
    {
        Vec2d image;
        if (camera.projectNormalized(point(parameters), image))
        {
            double residual = std::hypot(image[0] - desired[0], image[1] - desired[1]);
            if (std::isfinite(residual))
                return residual;
        }
        // Keep optimizer arithmetic finite even when a vertex lies on an RPC pole.
        return std::numeric_limits<double>::max() / 16;
    }

    const PreparedRPC& camera;
    Vec2d desired;
    Vec4d plane;
    int eliminated, axes[2];
};

} // namespace

void projectPointsRPC(InputArray objectPoints, const RPCModel& model,
                      OutputArray imagePoints, const RPCLocalFrame* localFrame)
{
    PreparedRPC camera(model, localFrame);
    Mat points = pointRows(objectPoints, 3);
    Mat result(points.rows, 1, CV_64FC2);
    for (int i = 0; i < points.rows; ++i)
    {
        const double* row = points.ptr<double>(i);
        Vec3d point(row[0], row[1], row[2]);
        CV_CheckEQ(finite(point), true, "RPC object points must be finite");
        Vec2d image;
        if (!camera.projectNormalized(camera.normalize(point), image))
            CV_Error(Error::StsOutOfRange, "RPC projection is undefined or nonfinite");
        result.at<Vec2d>(i) = image;
    }
    result.copyTo(imagePoints);
}

void backProjectPointsRPC(InputArray imagePoints, const RPCModel& model,
                          const Vec4d& plane, InputArray initialPoints,
                          OutputArray objectPoints, OutputArray success,
                          const RPCLocalFrame* localFrame, double errorTolerance, int maxIterations)
{
    CV_CheckEQ(std::isfinite(errorTolerance) && errorTolerance > 0, true, "RPC error tolerance must be positive and finite");
    CV_CheckGT(maxIterations, 0, "RPC search budget must be positive");
    PreparedRPC camera(model, localFrame);
    Ptr<PlaneProjection> objective = makePtr<PlaneProjection>(camera, plane);
    Mat images = pointRows(imagePoints, 2), initial = pointRows(initialPoints, 3);
    CV_CheckEQ(images.rows, initial.rows, "RPC image points and initial estimates must have equal lengths");
    Mat result(images.rows, 1, CV_64FC3, Scalar::all(std::numeric_limits<double>::quiet_NaN()));
    Mat status = Mat::zeros(images.rows, 1, CV_8U);
    // DownhillSolver uses one epsilon for both function spread (pixels) and
    // simplex diameter (normalized world units). Account for image scaling so
    // its diameter test does not stop before a subpixel tolerance is reached.
    const double epsilon = std::max(std::numeric_limits<double>::min(),
        std::min(1e-12, (errorTolerance / (1 + std::max(model.imageScale[0], model.imageScale[1]))) * 1e-3));
    Ptr<DownhillSolver> solver = DownhillSolver::create(objective, Vec2d(0.1, 0.1),
        TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxIterations, epsilon));
    for (int i = 0; i < images.rows; ++i)
    {
        const double* image = images.ptr<double>(i);
        const double* guess = initial.ptr<double>(i);
        Vec3d initialPoint(guess[0], guess[1], guess[2]);
        CV_CheckEQ(finite(initialPoint) && std::isfinite(image[0]) && std::isfinite(image[1]), true,
                 "RPC image points and initial estimates must be finite");
        Vec3d normalized = camera.normalize(initialPoint);
        if (!finite(normalized))
            continue;
        objective->desired = Vec2d(image[0], image[1]);
        Vec2d parameters(normalized[objective->axes[0]], normalized[objective->axes[1]]);
        if (objective->calc(parameters.val) > errorTolerance)
            solver->minimize(parameters);
        Vec3d candidate = camera.inputFromNormalized * objective->point(parameters.val) + camera.inputOffset;
        // Reimpose the plane after coordinate conversion to remove rounding from
        // geographic offsets much larger than the local metre coordinates.
        int eliminated = 0;
        for (int axis = 1; axis < 3; ++axis)
            if (std::abs(plane[axis]) > std::abs(plane[eliminated]))
                eliminated = axis;
        candidate[eliminated] = -plane[3] / plane[eliminated];
        for (int axis = 0; axis < 3; ++axis)
            if (axis != eliminated)
                candidate[eliminated] -= (plane[axis] / plane[eliminated]) * candidate[axis];
        Vec2d projected;
        if (!finite(candidate) || !camera.projectNormalized(camera.normalize(candidate), projected))
            continue;
        if (std::hypot(projected[0] - image[0], projected[1] - image[1]) > errorTolerance)
            continue;
        // The algebraic plane constraint should also survive denormalization and rounding.
        double scale = std::max(std::abs(plane[0]), std::max(std::abs(plane[1]), std::abs(plane[2])));
        Vec3d normal(plane[0] / scale, plane[1] / scale, plane[2] / scale);
        double planeResidual = std::abs(normal.dot(candidate) + plane[3] / scale);
        double planeTolerance = 64 * std::numeric_limits<double>::epsilon() *
            (1 + std::abs(normal[0] * candidate[0]) + std::abs(normal[1] * candidate[1]) +
             std::abs(normal[2] * candidate[2]) + std::abs(plane[3] / scale));
        if (!std::isfinite(planeResidual) || planeResidual > planeTolerance)
            continue;
        result.at<Vec3d>(i) = candidate;
        status.at<uchar>(i) = 1;
    }
    result.copyTo(objectPoints);
    status.copyTo(success);
}

}} // namespace cv::sfm
