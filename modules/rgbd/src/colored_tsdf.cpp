// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "colored_tsdf.hpp"
#include "tsdf_functions.hpp"
#include "opencl_kernels_rgbd.hpp"

#define USE_INTERPOLATION_IN_GETNORMAL 1

namespace cv {

namespace kinfu {

ColoredTSDFVolume::ColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor, float _truncDist,
                       int _maxWeight, Point3i _resolution, bool zFirstMemOrder)
    : Volume(_voxelSize, _pose, _raycastStepFactor),
      volResolution(_resolution),
      maxWeight( WeightType(_maxWeight) )
{
    CV_Assert(_maxWeight < 255);
    // Unlike original code, this should work with any volume size
    // Not only when (x,y,z % 32) == 0
    volSize   = Point3f(volResolution) * voxelSize;
    truncDist = std::max(_truncDist, 2.1f * voxelSize);

    // (xRes*yRes*zRes) array
    // Depending on zFirstMemOrder arg:
    // &elem(x, y, z) = data + x*zRes*yRes + y*zRes + z;
    // &elem(x, y, z) = data + x + y*xRes + z*xRes*yRes;
    int xdim, ydim, zdim;
    if(zFirstMemOrder)
    {
        xdim = volResolution.z * volResolution.y;
        ydim = volResolution.z;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volResolution.x;
        zdim = volResolution.x * volResolution.y;
    }

    volDims = Vec4i(xdim, ydim, zdim);
    neighbourCoords = Vec8i(
        volDims.dot(Vec4i(0, 0, 0)),
        volDims.dot(Vec4i(0, 0, 1)),
        volDims.dot(Vec4i(0, 1, 0)),
        volDims.dot(Vec4i(0, 1, 1)),
        volDims.dot(Vec4i(1, 0, 0)),
        volDims.dot(Vec4i(1, 0, 1)),
        volDims.dot(Vec4i(1, 1, 0)),
        volDims.dot(Vec4i(1, 1, 1))
    );
}

class ColoredTSDFVolumeCPU : public ColoredTSDFVolume
{
public:
    // dimension in voxels, size in meters
    ColoredTSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor, float _truncDist,
        int _maxWeight, Vec3i _resolution, bool zFirstMemOrder = true);
    virtual void integrate(InputArray, float, const Matx44f&, const kinfu::Intr&, const int) override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };
    virtual void integrate(InputArray _depth, InputArray _rgb, float depthFactor, const Matx44f& cameraPose,
        const kinfu::Intr& depth_intrinsics, const Intr& rgb_intrinsics, const int frameId = 0) override;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& depth_intrinsics, const Size& frameSize,
        OutputArray points, OutputArray normals, OutputArray colors) const override;
    virtual void raycast(const Matx44f&, const kinfu::Intr&, const Size&, OutputArray, OutputArray) const override
        { CV_Error(Error::StsNotImplemented, "Not implemented"); };

    virtual void fetchNormals(InputArray points, OutputArray _normals) const override;
    void fetchPointsNormalsColors(OutputArray points, OutputArray normals, OutputArray colors) const override;

    void fetchPointsNormals(OutputArray points, OutputArray normals) const override
    {
        fetchPointsNormalsColors(points, normals, noArray());
    }

    virtual void reset() override;
    virtual RGBTsdfVoxel at(const Vec3i& volumeIdx) const;

    float interpolateVoxel(const cv::Point3f& p) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;
    float interpolateColor(float tx, float ty, float tz, float vx[8]) const;
    Point3f getColorVoxel(const cv::Point3f& p) const;

#if USE_INTRINSICS
    float interpolateVoxel(const v_float32x4& p) const;
    v_float32x4 getNormalVoxel(const v_float32x4& p) const;
    v_float32x4 getColorVoxel(const v_float32x4& p) const;
#endif

    Vec4i volStrides;
    Vec6f frameParams;
    Mat pixNorms;
    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Consist of Voxel elements
    Mat volume;
};

// dimension in voxels, size in meters
ColoredTSDFVolumeCPU::ColoredTSDFVolumeCPU(float _voxelSize, cv::Matx44f _pose, float _raycastStepFactor,
                             float _truncDist, int _maxWeight, Vec3i _resolution,
                             bool zFirstMemOrder)
    : ColoredTSDFVolume(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _resolution,
                 zFirstMemOrder)
{
    int xdim, ydim, zdim;
    if (zFirstMemOrder)
    {
        xdim = volResolution.z * volResolution.y;
        ydim = volResolution.z;
        zdim = 1;
    }
    else
    {
        xdim = 1;
        ydim = volResolution.x;
        zdim = volResolution.x * volResolution.y;
    }
    volStrides = Vec4i(xdim, ydim, zdim);

    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<RGBTsdfVoxel>());

    reset();
}

// zero volume, leave rest params the same
void ColoredTSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecRGBTsdfVoxel>([](VecRGBTsdfVoxel& vv, const int* /* position */)
    {
        RGBTsdfVoxel& v = reinterpret_cast<RGBTsdfVoxel&>(vv);
        v.tsdf = floatToTsdf(0.0f); v.weight = 0;
    });
}

RGBTsdfVoxel ColoredTSDFVolumeCPU::at(const Vec3i& volumeIdx) const
{
    //! Out of bounds
    if ((volumeIdx[0] >= volResolution.x || volumeIdx[0] < 0) ||
        (volumeIdx[1] >= volResolution.y || volumeIdx[1] < 0) ||
        (volumeIdx[2] >= volResolution.z || volumeIdx[2] < 0))
    {
        return RGBTsdfVoxel(floatToTsdf(1.f), 0, 160, 160, 160);
    }

    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();
    int coordBase =
        volumeIdx[0] * volDims[0] + volumeIdx[1] * volDims[1] + volumeIdx[2] * volDims[2];
    return volData[coordBase];
}

// use depth instead of distance (optimization)
void ColoredTSDFVolumeCPU::integrate(InputArray _depth, InputArray _rgb, float depthFactor, const Matx44f& cameraPose,
                              const Intr& depth_intrinsics, const Intr& rgb_intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();
    CV_UNUSED(frameId);
    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    Depth depth = _depth.getMat();
    Colors rgb = _rgb.getMat();
    Vec6f newParams((float)depth.rows, (float)depth.cols,
        depth_intrinsics.fx, depth_intrinsics.fy,
        depth_intrinsics.cx, depth_intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, depth_intrinsics);
    }

    integrateRGBVolumeUnit(truncDist, voxelSize, maxWeight, (this->pose).matrix, volResolution, volStrides, depth, rgb,
        depthFactor, cameraPose, depth_intrinsics, rgb_intrinsics, pixNorms, volume);
}

#if USE_INTRINSICS
// all coordinate checks should be done in inclosing cycle
inline float ColoredTSDFVolumeCPU::interpolateVoxel(const Point3f& _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0);
    return interpolateVoxel(p);
}

inline float ColoredTSDFVolumeCPU::interpolateVoxel(const v_float32x4& p) const
{
    // tx, ty, tz = floor(p)
    v_int32x4 ip = v_floor(p);
    v_float32x4 t = v_sub(p, v_cvt_f32(ip));
    float tx = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = v_get0(t);

    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip);
    ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip);
    ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;

    TsdfType vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = volData[neighbourCoords[i] + coordBase].tsdf;

    v_float32x4 v0246 = tsdfToFloat_INTR(v_int32x4(vx[0], vx[2], vx[4], vx[6]));
    v_float32x4 v1357 = tsdfToFloat_INTR(v_int32x4(vx[1], vx[3], vx[5], vx[7]));
    v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
    float v0 = v_get0(v0_1);
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v_get0(v0_1);

    return v0 + tx * (v1 - v0);
}
#else
inline float ColoredTSDFVolumeCPU::interpolateVoxel(const Point3f& p) const
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz*zdim;
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    float vx[8];
    for (int i = 0; i < 8; i++)
        vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase].tsdf);

    float v00 = vx[0] + tz*(vx[1] - vx[0]);
    float v01 = vx[2] + tz*(vx[3] - vx[2]);
    float v10 = vx[4] + tz*(vx[5] - vx[4]);
    float v11 = vx[6] + tz*(vx[7] - vx[6]);

    float v0 = v00 + ty*(v01 - v00);
    float v1 = v10 + ty*(v11 - v10);

    return v0 + tx*(v1 - v0);
}
#endif


#if USE_INTRINSICS
//gradientDeltaFactor is fixed at 1.0 of voxel size
inline Point3f ColoredTSDFVolumeCPU::getNormalVoxel(const Point3f& _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getNormalVoxel(p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}

inline v_float32x4 ColoredTSDFVolumeCPU::getNormalVoxel(const v_float32x4& p) const
{
    if (v_check_any(v_lt(p, v_float32x4(1.f, 1.f, 1.f, 0.f))) ||
        v_check_any(v_ge(p, v_float32x4((float)(volResolution.x - 2),
            (float)(volResolution.y - 2),
            (float)(volResolution.z - 2), 1.f)))
        )
        return nanv;

    v_int32x4 ip = v_floor(p);
    v_float32x4 t = v_sub(p, v_cvt_f32(ip));
    float tx = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = v_get0(t);
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = v_get0(t);

    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;

    float CV_DECL_ALIGNED(16) an[4];
    an[0] = an[1] = an[2] = an[3] = 0.f;
    for (int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv = an[c];

        float vx[8];
        for (int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1 * dim].tsdf) -
                    tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1 * dim].tsdf);

        v_float32x4 v0246(vx[0], vx[2], vx[4], vx[6]);
        v_float32x4 v1357(vx[1], vx[3], vx[5], vx[7]);
        v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

        v_float32x4 v00_10 = vxx;
        v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

        v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
        float v0 = v_get0(v0_1);
        v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
        float v1 = v_get0(v0_1);

        nv = v0 + tx * (v1 - v0);
    }

    v_float32x4 n = v_load_aligned(an);
    v_float32x4 Norm = v_sqrt(v_setall_f32(v_reduce_sum(v_mul(n, n))));

    return v_get0(Norm) < 0.0001f ? nanv : v_div(n, Norm);
}
#else
inline Point3f ColoredTSDFVolumeCPU::getNormalVoxel(const Point3f& p) const
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    if(p.x < 1 || p.x >= volResolution.x - 2 ||
       p.y < 1 || p.y >= volResolution.y - 2 ||
       p.z < 1 || p.z >= volResolution.z - 2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz*zdim;

    Vec3f an;
    for(int c = 0; c < 3; c++)
    {
        const int dim = volDims[c];
        float& nv = an[c];

        float vx[8];
        for(int i = 0; i < 8; i++)
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1 * dim].tsdf) -
                    tsdfToFloat(volData[neighbourCoords[i] + coordBase - 1 * dim].tsdf);

        float v00 = vx[0] + tz*(vx[1] - vx[0]);
        float v01 = vx[2] + tz*(vx[3] - vx[2]);
        float v10 = vx[4] + tz*(vx[5] - vx[4]);
        float v11 = vx[6] + tz*(vx[7] - vx[6]);

        float v0 = v00 + ty*(v01 - v00);
        float v1 = v10 + ty*(v11 - v10);

        nv = v0 + tx*(v1 - v0);
    }

    float nv = sqrt(an[0] * an[0] +
                    an[1] * an[1] +
                    an[2] * an[2]);
    return nv < 0.0001f ? nan3 : an / nv;
}
#endif

#if USE_INTRINSICS
inline float ColoredTSDFVolumeCPU::interpolateColor(float tx, float ty, float tz, float vx[8]) const
{
    v_float32x4 v0246, v1357;
    v_load_deinterleave(vx, v0246, v1357);

    v_float32x4 vxx = v_add(v0246, v_mul(v_setall_f32(tz), v_sub(v1357, v0246)));

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v_add(v00_10, v_mul(v_setall_f32(ty), v_sub(v01_11, v00_10)));
    float v0 = v_get0(v0_1);
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v_get0(v0_1);

    return v0 + tx * (v1 - v0);
}
#else
inline float ColoredTSDFVolumeCPU::interpolateColor(float tx, float ty, float tz, float vx[8]) const
{
    float v00 = vx[0] + tz * (vx[1] - vx[0]);
    float v01 = vx[2] + tz * (vx[3] - vx[2]);
    float v10 = vx[4] + tz * (vx[5] - vx[4]);
    float v11 = vx[6] + tz * (vx[7] - vx[6]);

    float v0 = v00 + ty * (v01 - v00);
    float v1 = v10 + ty * (v11 - v10);

    return v0 + tx * (v1 - v0);
}
#endif

#if USE_INTRINSICS
//gradientDeltaFactor is fixed at 1.0 of voxel size
inline Point3f ColoredTSDFVolumeCPU::getColorVoxel(const Point3f& _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getColorVoxel(p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}
inline v_float32x4 ColoredTSDFVolumeCPU::getColorVoxel(const v_float32x4& p) const
{
    if (v_check_any(v_lt(p, v_float32x4(1.f, 1.f, 1.f, 0.f))) ||
        v_check_any(v_ge(p, v_float32x4((float)(volResolution.x - 2),
            (float)(volResolution.y - 2),
            (float)(volResolution.z - 2), 1.f)))
        )
        return nanv;

    v_int32x4 ip = v_floor(p);

    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();

    int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
    int iz = v_get0(ip);

    int coordBase = ix * xdim + iy * ydim + iz * zdim;
    float CV_DECL_ALIGNED(16) rgb[4];

#if USE_INTERPOLATION_IN_GETNORMAL
    float r[8], g[8], b[8];
    for (int i = 0; i < 8; i++)
    {
        r[i] = (float)volData[neighbourCoords[i] + coordBase].r;
        g[i] = (float)volData[neighbourCoords[i] + coordBase].g;
        b[i] = (float)volData[neighbourCoords[i] + coordBase].b;
    }

    v_float32x4 vsi(voxelSizeInv, voxelSizeInv, voxelSizeInv, voxelSizeInv);
    v_float32x4 ptVox = v_mul(p, vsi);
    v_int32x4 iptVox = v_floor(ptVox);
    v_float32x4 t = v_sub(ptVox, v_cvt_f32(iptVox));
    float tx = v_get0(t); t = v_rotate_right<1>(t);
    float ty = v_get0(t); t = v_rotate_right<1>(t);
    float tz = v_get0(t);
    rgb[0] = interpolateColor(tx, ty, tz, r);
    rgb[1] = interpolateColor(tx, ty, tz, g);
    rgb[2] = interpolateColor(tx, ty, tz, b);
    rgb[3] = 0.f;
#else
    rgb[0] = volData[coordBase].r;
    rgb[1] = volData[coordBase].g;
    rgb[2] = volData[coordBase].b;
    rgb[3] = 0.f;
#endif
    v_float32x4 res = v_load_aligned(rgb);
    return res;
}
#else
inline Point3f ColoredTSDFVolumeCPU::getColorVoxel(const Point3f& p) const
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();



    if(p.x < 1 || p.x >= volResolution.x - 2 ||
       p.y < 1 || p.y >= volResolution.y - 2 ||
       p.z < 1 || p.z >= volResolution.z - 2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    int coordBase = ix*xdim + iy*ydim + iz*zdim;
    Point3f res;

#if USE_INTERPOLATION_IN_GETNORMAL
    // TODO: create better interpolation or remove this simple version
    float r[8], g[8], b[8];
    for (int i = 0; i < 8; i++)
    {
        r[i] = (float) volData[neighbourCoords[i] + coordBase].r;
        g[i] = (float) volData[neighbourCoords[i] + coordBase].g;
        b[i] = (float) volData[neighbourCoords[i] + coordBase].b;
    }

    Point3f ptVox = p * voxelSizeInv;
    Vec3i iptVox(cvFloor(ptVox.x), cvFloor(ptVox.y), cvFloor(ptVox.z));
    float tx = ptVox.x - iptVox[0];
    float ty = ptVox.y - iptVox[1];
    float tz = ptVox.z - iptVox[2];

    res=Point3f(interpolateColor(tx, ty, tz, r),
                interpolateColor(tx, ty, tz, g),
                interpolateColor(tx, ty, tz, b));
#else
    res=Point3f(volData[coordBase].r, volData[coordBase].g, volData[coordBase].b);
#endif
    colorFix(res);
    return res;
}
#endif

struct ColorRaycastInvoker : ParallelLoopBody
{
    ColorRaycastInvoker(Points& _points, Normals& _normals, Colors& _colors, const Matx44f& cameraPose,
                  const Intr& depth_intrinsics, const ColoredTSDFVolumeCPU& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        colors(_colors),
        volume(_volume),
        tstep(volume.truncDist * volume.raycastStepFactor),
        // We do subtract voxel size to minimize checks after
        // Note: origin of volume coordinate is placed
        // in the center of voxel (0,0,0), not in the corner of the voxel!
        boxMax(volume.volSize - Point3f(volume.voxelSize,
                                        volume.voxelSize,
                                        volume.voxelSize)),
        boxMin(),
        cam2vol(volume.pose.inv() * Affine3f(cameraPose)),
        vol2cam(Affine3f(cameraPose.inv()) * volume.pose),
        reprojDepth(depth_intrinsics.makeReprojector())
    {  }
#if USE_INTRINSICS
    virtual void operator() (const Range& range) const override
    {
        const v_float32x4 vfxy(reprojDepth.fxinv, reprojDepth.fyinv, 0, 0);
        const v_float32x4 vcxy(reprojDepth.cx, reprojDepth.cy, 0, 0);

        const float(&cm)[16] = cam2vol.matrix.val;
        const v_float32x4 camRot0(cm[0], cm[4], cm[8], 0);
        const v_float32x4 camRot1(cm[1], cm[5], cm[9], 0);
        const v_float32x4 camRot2(cm[2], cm[6], cm[10], 0);
        const v_float32x4 camTrans(cm[3], cm[7], cm[11], 0);

        const v_float32x4 boxDown(boxMin.x, boxMin.y, boxMin.z, 0.f);
        const v_float32x4 boxUp(boxMax.x, boxMax.y, boxMax.z, 0.f);

        const v_float32x4 invVoxelSize = v_float32x4(volume.voxelSizeInv,
            volume.voxelSizeInv,
            volume.voxelSizeInv, 1.f);

        const float(&vm)[16] = vol2cam.matrix.val;
        const v_float32x4 volRot0(vm[0], vm[4], vm[8], 0);
        const v_float32x4 volRot1(vm[1], vm[5], vm[9], 0);
        const v_float32x4 volRot2(vm[2], vm[6], vm[10], 0);
        const v_float32x4 volTrans(vm[3], vm[7], vm[11], 0);

        for (int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];
            ptype* clrRow = colors[y];

            for (int x = 0; x < points.cols; x++)
            {
                v_float32x4 point = nanv, normal = nanv, color = nanv;

                v_float32x4 orig = camTrans;

                // get direction through pixel in volume space:

                // 1. reproject (x, y) on projecting plane where z = 1.f
                v_float32x4 planed = v_mul(v_sub(v_float32x4((float)x, (float)y, 0.F, 0.F), vcxy), vfxy);
                planed = v_combine_low(planed, v_float32x4(1.f, 0.f, 0.f, 0.f));

                // 2. rotate to volume space
                planed = v_matmuladd(planed, camRot0, camRot1, camRot2, v_setzero_f32());

                // 3. normalize
                v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(v_mul(planed, planed))));
                v_float32x4 dir = v_mul(planed, invNorm);

                // compute intersection of ray with all six bbox planes
                v_float32x4 rayinv = v_div(v_setall_f32(1.F), dir);
                // div by zero should be eliminated by these products
                v_float32x4 tbottom = v_mul(rayinv, v_sub(boxDown, orig));
                v_float32x4 ttop = v_mul(rayinv, v_sub(boxUp, orig));

                // re-order intersections to find smallest and largest on each axis
                v_float32x4 minAx = v_min(ttop, tbottom);
                v_float32x4 maxAx = v_max(ttop, tbottom);

                // near clipping plane
                const float clip = 0.f;
                float _minAx[4], _maxAx[4];
                v_store(_minAx, minAx);
                v_store(_maxAx, maxAx);
                float tmin = max({ _minAx[0], _minAx[1], _minAx[2], clip });
                float tmax = min({ _maxAx[0], _maxAx[1], _maxAx[2] });

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if (tmin < tmax)
                {
                    // interpolation optimized a little
                    orig = v_mul(orig, invVoxelSize);
                    dir = v_mul(dir, invVoxelSize);

                    int xdim = volume.volDims[0];
                    int ydim = volume.volDims[1];
                    int zdim = volume.volDims[2];
                    v_float32x4 rayStep = v_mul(dir, v_setall_f32(this->tstep));
                    v_float32x4 next = (v_add(orig, v_mul(dir, v_setall_f32(tmin))));
                    float f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = cvFloor((tmax - tmin) / tstep);
                    for (; steps < nSteps; steps++)
                    {
                        next = v_add(next, rayStep);
                        v_int32x4 ip = v_round(next);
                        int ix = v_get0(ip); ip = v_rotate_right<1>(ip);
                        int iy = v_get0(ip); ip = v_rotate_right<1>(ip);
                        int iz = v_get0(ip);
                        int coord = ix * xdim + iy * ydim + iz * zdim;

                        fnext = tsdfToFloat(volume.volume.at<RGBTsdfVoxel>(coord).tsdf);
                        if (fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);

                            // when ray crosses a surface
                            if (std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }

                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if (f > 0.f && fnext < 0.f)
                    {
                        v_float32x4 tp = v_sub(next, rayStep);
                        float ft = volume.interpolateVoxel(tp);
                        float ftdt = volume.interpolateVoxel(next);
                        float ts = tmin + tstep * (steps - ft / (ftdt - ft));

                        // avoid division by zero
                        if (!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            v_float32x4 pv = (v_add(orig, v_mul(dir, v_setall_f32(ts))));
                            v_float32x4 nv = volume.getNormalVoxel(pv);
                            v_float32x4 cv = volume.getColorVoxel(pv);

                            if (!isNaN(nv))
                            {
                                color = cv;
                                //convert pv and nv to camera space
                                normal = v_matmuladd(nv, volRot0, volRot1, volRot2, v_setzero_f32());
                                // interpolation optimized a little
                                point = v_matmuladd(v_mul(pv, v_float32x4(this->volume.voxelSize, this->volume.voxelSize, this->volume.voxelSize, 1.F)),
                                    volRot0, volRot1, volRot2, volTrans);
                            }
                        }
                    }
                }

                v_store((float*)(&ptsRow[x]), point);
                v_store((float*)(&nrmRow[x]), normal);
                v_store((float*)(&clrRow[x]), color);
            }
        }
    }
#else
    virtual void operator() (const Range& range) const override
    {
        const Point3f camTrans = cam2vol.translation();
        const Matx33f  camRot  = cam2vol.rotation();
        const Matx33f  volRot  = vol2cam.rotation();

        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];
            ptype* clrRow = colors[y];

            for(int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3, color = nan3;

                Point3f orig = camTrans;
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(camRot * reprojDepth(Point3f(float(x), float(y), 1.f))));

                // compute intersection of ray with all six bbox planes
                Vec3f rayinv(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);
                Point3f tbottom = rayinv.mul(boxMin - orig);
                Point3f ttop    = rayinv.mul(boxMax - orig);

                // re-order intersections to find smallest and largest on each axis
                Point3f minAx(min(ttop.x, tbottom.x), min(ttop.y, tbottom.y), min(ttop.z, tbottom.z));
                Point3f maxAx(max(ttop.x, tbottom.x), max(ttop.y, tbottom.y), max(ttop.z, tbottom.z));

                // near clipping plane
                const float clip = 0.f;
                //float tmin = max(max(max(minAx.x, minAx.y), max(minAx.x, minAx.z)), clip);
                //float tmax =     min(min(maxAx.x, maxAx.y), min(maxAx.x, maxAx.z));
                float tmin = max({ minAx.x, minAx.y, minAx.z, clip });
                float tmax = min({ maxAx.x, maxAx.y, maxAx.z });

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if(tmin < tmax)
                {
                    // interpolation optimized a little
                    orig = orig*volume.voxelSizeInv;
                    dir  =  dir*volume.voxelSizeInv;

                    Point3f rayStep = dir * tstep;
                    Point3f next = (orig + dir * tmin);
                    float f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = int(floor((tmax - tmin)/tstep));
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volume.volDims[0];
                        int ydim = volume.volDims[1];
                        int zdim = volume.volDims[2];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = tsdfToFloat(volume.volume.at<RGBTsdfVoxel>(ix*xdim + iy*ydim + iz*zdim).tsdf);
                        if(fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);
                            // when ray crosses a surface
                            if(std::signbit(f) != std::signbit(fnext))
                                break;

                            f = fnext;
                        }
                    }
                    // if ray penetrates a surface from outside
                    // linearly interpolate t between two f values
                    if(f > 0.f && fnext < 0.f)
                    {
                        Point3f tp    = next - rayStep;
                        float ft   = volume.interpolateVoxel(tp);
                        float ftdt = volume.interpolateVoxel(next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep*(steps - ft/(ftdt - ft));

                        // avoid division by zero
                        if(!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir*ts);
                            Point3f nv = volume.getNormalVoxel(pv);
                            Point3f cv = volume.getColorVoxel(pv);
                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                color = cv;
                                // interpolation optimized a little
                                point = vol2cam * (pv*volume.voxelSize);
                            }
                        }
                    }
                }
                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
                clrRow[x] = toPtype(color);
            }
        }
    }
#endif

    Points& points;
    Normals& normals;
    Colors& colors;
    const ColoredTSDFVolumeCPU& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reprojDepth;
};


void ColoredTSDFVolumeCPU::raycast(const Matx44f& cameraPose, const Intr& depth_intrinsics, const Size& frameSize,
                            OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    CV_TRACE_FUNCTION();

    CV_Assert(frameSize.area() > 0);

    _points.create (frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);
    _colors.create(frameSize, POINT_TYPE);

    Points points   =  _points.getMat();
    Normals normals = _normals.getMat();
    Colors colors = _colors.getMat();
    ColorRaycastInvoker ri(points, normals, colors, cameraPose, depth_intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}


struct ColorFetchPointsNormalsInvoker : ParallelLoopBody
{
    ColorFetchPointsNormalsInvoker(const ColoredTSDFVolumeCPU& _volume,
                              std::vector<std::vector<ptype>>& _pVecs,
                              std::vector<std::vector<ptype>>& _nVecs,
                              std::vector<std::vector<ptype>>& _cVecs,
                              bool _needNormals, bool _needColors) :
        ParallelLoopBody(),
        vol(_volume),
        pVecs(_pVecs),
        nVecs(_nVecs),
        cVecs(_cVecs),
        needNormals(_needNormals),
        needColors(_needColors)
    {
        volDataStart = vol.volume.ptr<RGBTsdfVoxel>();
    }

    inline void coord(std::vector<ptype>& points, std::vector<ptype>& normals, std::vector<ptype>& colors,
                      int x, int y, int z, Point3f V, float v0, int axis) const
    {
        // 0 for x, 1 for y, 2 for z
        bool limits = false;
        Point3i shift;
        float Vc = 0.f;
        if(axis == 0)
        {
            shift  = Point3i(1, 0, 0);
            limits = (x + 1 < vol.volResolution.x);
            Vc     = V.x;
        }
        if(axis == 1)
        {
            shift  = Point3i(0, 1, 0);
            limits = (y + 1 < vol.volResolution.y);
            Vc     = V.y;
        }
        if(axis == 2)
        {
            shift  = Point3i(0, 0, 1);
            limits = (z + 1 < vol.volResolution.z);
            Vc     = V.z;
        }

        if(limits)
        {
            const RGBTsdfVoxel& voxeld = volDataStart[(x+shift.x)*vol.volDims[0] +
                                                   (y+shift.y)*vol.volDims[1] +
                                                   (z+shift.z)*vol.volDims[2]];
            float vd = tsdfToFloat(voxeld.tsdf);

            if(voxeld.weight != 0 && vd != 1.f)
            {
                if((v0 > 0 && vd < 0) || (v0 < 0 && vd > 0))
                {
                    //linearly interpolate coordinate
                    float Vn    = Vc + vol.voxelSize;
                    float dinv  = 1.f/(abs(v0)+abs(vd));
                    float inter = (Vc*abs(vd) + Vn*abs(v0))*dinv;

                    Point3f p(shift.x ? inter : V.x,
                              shift.y ? inter : V.y,
                              shift.z ? inter : V.z);
                    {
                        points.push_back(toPtype(vol.pose * p));
                        if(needNormals)
                            normals.push_back(toPtype(vol.pose.rotation() *
                                                      vol.getNormalVoxel(p*vol.voxelSizeInv)));
                        if(needColors)
                            colors.push_back(toPtype(vol.getColorVoxel(p*vol.voxelSizeInv)));
                    }
                }
            }
        }
    }

    virtual void operator() (const Range& range) const override
    {
        std::vector<ptype> points, normals, colors;
        for(int x = range.start; x < range.end; x++)
        {
            const RGBTsdfVoxel* volDataX = volDataStart + x*vol.volDims[0];
            for(int y = 0; y < vol.volResolution.y; y++)
            {
                const RGBTsdfVoxel* volDataY = volDataX + y*vol.volDims[1];
                for(int z = 0; z < vol.volResolution.z; z++)
                {
                    const RGBTsdfVoxel& voxel0 = volDataY[z*vol.volDims[2]];
                    float v0             = tsdfToFloat(voxel0.tsdf);
                    if(voxel0.weight != 0 && v0 != 1.f)
                    {
                        Point3f V(Point3f((float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f)*vol.voxelSize);

                        coord(points, normals, colors, x, y, z, V, v0, 0);
                        coord(points, normals, colors, x, y, z, V, v0, 1);
                        coord(points, normals, colors, x, y, z, V, v0, 2);

                    } // if voxel is not empty
                }
            }
        }

        AutoLock al(mutex);
        pVecs.push_back(points);
        nVecs.push_back(normals);
        cVecs.push_back(colors);
    }

    const ColoredTSDFVolumeCPU& vol;
    std::vector<std::vector<ptype>>& pVecs;
    std::vector<std::vector<ptype>>& nVecs;
    std::vector<std::vector<ptype>>& cVecs;
    const RGBTsdfVoxel* volDataStart;
    bool needNormals;
    bool needColors;
    mutable Mutex mutex;
};

void ColoredTSDFVolumeCPU::fetchPointsNormalsColors(OutputArray _points, OutputArray _normals, OutputArray _colors) const
{
    CV_TRACE_FUNCTION();

    if(_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs, cVecs;
        ColorFetchPointsNormalsInvoker fi(*this, pVecs, nVecs, cVecs, _normals.needed(), _colors.needed());
        Range range(0, volResolution.x);
        const int nstripes = -1;
        parallel_for_(range, fi, nstripes);
        std::vector<ptype> points, normals, colors;
        for(size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
            colors.insert(colors.end(), cVecs[i].begin(), cVecs[i].end());
        }

        _points.create((int)points.size(), 1, POINT_TYPE);
        if(!points.empty())
            Mat((int)points.size(), 1, POINT_TYPE, &points[0]).copyTo(_points.getMat());

        if(_normals.needed())
        {
            _normals.create((int)normals.size(), 1, POINT_TYPE);
            if(!normals.empty())
                Mat((int)normals.size(), 1, POINT_TYPE, &normals[0]).copyTo(_normals.getMat());
        }

        if(_colors.needed())
        {
            _colors.create((int)colors.size(), 1, COLOR_TYPE);
            if(!colors.empty())
                Mat((int)colors.size(), 1, COLOR_TYPE, &colors[0]).copyTo(_colors.getMat());
        }
    }
}

void ColoredTSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();
    CV_Assert(!_points.empty());
    if(_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Normals normals = _normals.getMat();

        const ColoredTSDFVolumeCPU& _vol = *this;
        auto PushNormals = [&](const ptype& pp, const int* position)
        {
            const ColoredTSDFVolumeCPU& vol(_vol);
            Affine3f invPose(vol.pose.inv());
            Point3f p = fromPtype(pp);
            Point3f n = nan3;
            if (!isNaN(p))
            {
                Point3f voxPt = (invPose * p);
                voxPt = voxPt * vol.voxelSizeInv;
                n = vol.pose.rotation() * vol.getNormalVoxel(voxPt);
            }
            normals(position[0], position[1]) = toPtype(n);
        };
        points.forEach(PushNormals);
    }
}

Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(float _voxelSize, Matx44f _pose, float _raycastStepFactor,
                                   float _truncDist, int _maxWeight, Point3i _resolution)
{
    return makePtr<ColoredTSDFVolumeCPU>(_voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _resolution);
}

Ptr<ColoredTSDFVolume> makeColoredTSDFVolume(const VolumeParams& _params)
{
    return makePtr<ColoredTSDFVolumeCPU>(_params.voxelSize, _params.pose.matrix, _params.raycastStepFactor,
                                  _params.tsdfTruncDist, _params.maxWeight, _params.resolution);
}

} // namespace kinfu
} // namespace cv
