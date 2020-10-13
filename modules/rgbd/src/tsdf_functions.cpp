// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "tsdf_functions.hpp"

namespace cv {

namespace kinfu {

v_float32x4 tsdfToFloat_INTR(const v_int32x4& num)
{
    v_float32x4 num128 = v_setall_f32(-1.f / 128.f);
    return v_cvt_f32(num) * num128;
}

TsdfType floatToTsdf(float num)
{
    //CV_Assert(-1 < num <= 1);
    int8_t res = int8_t(num * (-128.f));
    res = res ? res : (num < 0 ? 1 : -1);
    return res;
}

float tsdfToFloat(TsdfType num)
{
    return float(num) * (-1.f / 128.f);
}


cv::Mat preCalculationPixNorm(Depth depth, const Intr& intrinsics)
{
    int height = depth.rows;
    int widht = depth.cols;
    Point2f fl(intrinsics.fx, intrinsics.fy);
    Point2f pp(intrinsics.cx, intrinsics.cy);
    Mat pixNorm(height, widht, CV_32F);
    std::vector<float> x(widht);
    std::vector<float> y(height);
    for (int i = 0; i < widht; i++)
        x[i] = (i - pp.x) / fl.x;
    for (int i = 0; i < height; i++)
        y[i] = (i - pp.y) / fl.y;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < widht; j++)
        {
            pixNorm.at<float>(i, j) = sqrtf(x[j] * x[j] + y[i] * y[i] + 1.0f);
        }
    }
    return pixNorm;
}

const bool fixMissingData = false;
depthType bilinearDepth(const Depth& m, cv::Point2f pt)
{
    const depthType defaultValue = qnan;
    if (pt.x < 0 || pt.x >= m.cols - 1 ||
        pt.y < 0 || pt.y >= m.rows - 1)
        return defaultValue;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

    const depthType* row0 = m[yi + 0];
    const depthType* row1 = m[yi + 1];

    depthType v00 = row0[xi + 0];
    depthType v01 = row0[xi + 1];
    depthType v10 = row1[xi + 0];
    depthType v11 = row1[xi + 1];

    // assume correct depth is positive
    bool b00 = v00 > 0;
    bool b01 = v01 > 0;
    bool b10 = v10 > 0;
    bool b11 = v11 > 0;

    if (!fixMissingData)
    {
        if (!(b00 && b01 && b10 && b11))
            return defaultValue;
        else
        {
            float tx = pt.x - xi, ty = pt.y - yi;
            depthType v0 = v00 + tx * (v01 - v00);
            depthType v1 = v10 + tx * (v11 - v10);
            return v0 + ty * (v1 - v0);
        }
    }
    else
    {
        int nz = b00 + b01 + b10 + b11;
        if (nz == 0)
        {
            return defaultValue;
        }
        if (nz == 1)
        {
            if (b00) return v00;
            if (b01) return v01;
            if (b10) return v10;
            if (b11) return v11;
        }
        if (nz == 2)
        {
            if (b00 && b10) v01 = v00, v11 = v10;
            if (b01 && b11) v00 = v01, v10 = v11;
            if (b00 && b01) v10 = v00, v11 = v01;
            if (b10 && b11) v00 = v10, v01 = v11;
            if (b00 && b11) v01 = v10 = (v00 + v11) * 0.5f;
            if (b01 && b10) v00 = v11 = (v01 + v10) * 0.5f;
        }
        if (nz == 3)
        {
            if (!b00) v00 = v10 + v01 - v11;
            if (!b01) v01 = v00 + v11 - v10;
            if (!b10) v10 = v00 + v11 - v01;
            if (!b11) v11 = v01 + v10 - v00;
        }

        float tx = pt.x - xi, ty = pt.y - yi;
        depthType v0 = v00 + tx * (v01 - v00);
        depthType v1 = v10 + tx * (v11 - v10);
        return v0 + ty * (v1 - v0);
    }
}

void integrateVolumeUnit(
    float truncDist, float voxelSize, int maxWeight,
    cv::Matx44f _pose, Point3i volResolution, Vec4i volStrides,
    InputArray _depth, float depthFactor, const cv::Matx44f& cameraPose,
    const cv::kinfu::Intr& intrinsics, InputArray _pixNorms, InputArray _volume)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    cv::Affine3f vpose(_pose);
    Depth depth = _depth.getMat();

    Range integrateRange(0, volResolution.x);

    Mat volume = _volume.getMat();
    Mat pixNorms = _pixNorms.getMat();
    const Intr::Projector proj(intrinsics.makeProjector());
    const cv::Affine3f vol2cam(Affine3f(cameraPose.inv()) * vpose);
    const float truncDistInv(1.f / truncDist);
    const float dfac(1.f / depthFactor);
    TsdfVoxel* volDataStart = volume.ptr<TsdfVoxel>();;

    auto IntegrateInvoker = [&](const Range& range)
    {
        for (int x = range.start; x < range.end; x++)
        {
            TsdfVoxel* volDataX = volDataStart + x * volStrides[0];
            for (int y = 0; y < volResolution.y; y++)
            {
                TsdfVoxel* volDataY = volDataX + y * volStrides[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam * (Point3f(float(x), float(y), 0.0f) * voxelSize);
                Point3f camSpacePt = basePt;
                // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                // zStep == vol2cam*[Point3f(x, y, 1) - Point3f(x, y, 0)]*voxelSize
                Point3f zStep = Point3f(vol2cam.matrix(0, 2),
                    vol2cam.matrix(1, 2),
                    vol2cam.matrix(2, 2)) * voxelSize;
                int startZ, endZ;
                if (abs(zStep.z) > 1e-5)
                {
                    int baseZ = int(-basePt.z / zStep.z);
                    if (zStep.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        startZ = 0;
                        endZ = baseZ;
                    }
                }
                else
                {
                    if (basePt.z > 0)
                    {
                        startZ = 0;
                        endZ = volResolution.z;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(volResolution.z, endZ);

                for (int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*volume.voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;

                    camSpacePt += zStep;
                    if (camSpacePt.z <= 0)
                        continue;

                    Point3f camPixVec;
                    Point2f projected = proj(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if (v == 0) {
                        continue;
                    }

                    int _u = projected.x;
                    int _v = projected.y;
                    if (!(_u >= 0 && _u < depth.cols && _v >= 0 && _v < depth.rows))
                        continue;
                    float pixNorm = pixNorms.at<float>(_v, _u);

                    // difference between distances of point and of surface to camera
                    float sdf = pixNorm * (v * dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);
                    if (sdf >= -truncDist)
                    {
                        TsdfType tsdf = floatToTsdf(fmin(1.f, sdf * truncDistInv));

                        TsdfVoxel& voxel = volDataY[z * volStrides[2]];
                        WeightType& weight = voxel.weight;
                        TsdfType& value = voxel.tsdf;

                        // update TSDF
                        value = floatToTsdf((tsdfToFloat(value) * weight + tsdfToFloat(tsdf)) / (weight + 1));
                        weight = min(int(weight + 1), int(maxWeight));
                    }
                }
            }
        }
    };
    parallel_for_(integrateRange, IntegrateInvoker);
}

} // namespace kinfu
} // namespace cv
