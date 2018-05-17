// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "tsdf.hpp"

namespace cv {

namespace kinfu {

typedef float volumeType; // can be float16
struct Voxel
{
    volumeType v;
    int weight;
};

}

template<> class DataType<kinfu::Voxel>
{
public:
    typedef kinfu::Voxel       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_64F,
           channels     = 1,
           fmt          = (int)'v',
           type         = CV_MAKETYPE(depth, channels)
         };
};

namespace kinfu {


class TSDFVolumeCPU : public TSDFVolume
{
    typedef cv::Mat_<Voxel> Volume;

public:
    // dimension in voxels, size in meters
    TSDFVolumeCPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                  float _raycastStepFactor);

    virtual void integrate(cv::Ptr<Frame> depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) override;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, int pyramidLevels,
                         cv::Ptr<FrameGenerator> frameGenerator, cv::Ptr<Frame> frame) const override;

    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;

    virtual void reset() override;

    volumeType interpolateVoxel(cv::Point3f p) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

#if CV_SIMD128
    volumeType interpolateVoxel(const v_float32x4& p) const;
    v_float32x4 getNormalVoxel(const v_float32x4& p) const;
#endif

    // edgeResolution^3 array
    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    Volume volume;

    int neighbourCoords[8];
    int dims[4];
    float voxelSize;
    float voxelSizeInv;
    float truncDist;
    float raycastStepFactor;
};


TSDFVolume::TSDFVolume(int _res, float _size, Affine3f _pose, float /*_truncDist*/, int _maxWeight,
                       float /*_raycastStepFactor*/) :
    edgeSize(_size),
    edgeResolution(_res),
    maxWeight(_maxWeight),
    pose(_pose)
{ }

// dimension in voxels, size in meters
TSDFVolumeCPU::TSDFVolumeCPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                             float _raycastStepFactor) :
    TSDFVolume(_res, _size, _pose, _truncDist, _maxWeight, _raycastStepFactor)
{
    CV_Assert(_res % 32 == 0);

    int xdim = edgeResolution*edgeResolution;
    int ydim = edgeResolution;
    int steps[4] = { xdim, ydim, 1, 0 };
    for(int i = 0; i < 4; i++)
        dims[i] = steps[i];
    int coords[8] = {
        xdim*0 + ydim*0 + 1*0,
        xdim*0 + ydim*0 + 1*1,
        xdim*0 + ydim*1 + 1*0,
        xdim*0 + ydim*1 + 1*1,
        xdim*1 + ydim*0 + 1*0,
        xdim*1 + ydim*0 + 1*1,
        xdim*1 + ydim*1 + 1*0,
        xdim*1 + ydim*1 + 1*1
    };
    for(int i = 0; i < 8; i++)
        neighbourCoords[i] = coords[i];

    voxelSize = edgeSize/edgeResolution;
    voxelSizeInv = edgeResolution/edgeSize;
    volume = Volume(1, _res * _res * _res);

    truncDist = std::max (_truncDist, 2.1f * voxelSize);
    raycastStepFactor = _raycastStepFactor;

    reset();
}

struct FillZero
{
    void operator ()(Voxel &v, const int* /*position*/) const
    {
        v.v = 0; v.weight = 0;
    }
};

// zero volume, leave rest params the same
void TSDFVolumeCPU::reset()
{
    ScopeTime st("tsdf: reset");

    volume.forEach(FillZero());
}

// SIMD version of that code is manually inlined
#if !CV_SIMD128
static const bool fixMissingData = false;

static inline depthType bilinearDepth(const Depth& m, cv::Point2f pt)
{
    const depthType defaultValue = qnan;
    if(pt.x < 0 || pt.x >= m.cols-1 ||
       pt.y < 0 || pt.y >= m.rows-1)
        return defaultValue;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);

    const depthType* row0 = m[yi+0];
    const depthType* row1 = m[yi+1];

    depthType v00 = row0[xi+0];
    depthType v01 = row0[xi+1];
    depthType v10 = row1[xi+0];
    depthType v11 = row1[xi+1];

    // assume correct depth is positive
    bool b00 = v00 > 0;
    bool b01 = v01 > 0;
    bool b10 = v10 > 0;
    bool b11 = v11 > 0;

    if(!fixMissingData)
    {
        if(!(b00 && b01 && b10 && b11))
            return defaultValue;
        else
        {
            float tx = pt.x - xi, ty = pt.y - yi;
            depthType v0 = v00 + tx*(v01 - v00);
            depthType v1 = v10 + tx*(v11 - v10);
            return v0 + ty*(v1 - v0);
        }
    }
    else
    {
        int nz = b00 + b01 + b10 + b11;
        if(nz == 0)
        {
            return defaultValue;
        }
        if(nz == 1)
        {
            if(b00) return v00;
            if(b01) return v01;
            if(b10) return v10;
            if(b11) return v11;
        }
        else if(nz == 2)
        {
            if(b00 && b10) v01 = v00, v11 = v10;
            if(b01 && b11) v00 = v01, v10 = v11;
            if(b00 && b01) v10 = v00, v11 = v01;
            if(b10 && b11) v00 = v10, v01 = v11;
            if(b00 && b11) v01 = v10 = (v00 + v11)*0.5f;
            if(b01 && b10) v00 = v11 = (v01 + v10)*0.5f;
        }
        else if(nz == 3)
        {
            if(!b00) v00 = v10 + v01 - v11;
            if(!b01) v01 = v00 + v11 - v10;
            if(!b10) v10 = v00 + v11 - v01;
            if(!b11) v11 = v01 + v10 - v00;
        }

        float tx = pt.x - xi, ty = pt.y - yi;
        depthType v0 = v00 + tx*(v01 - v00);
        depthType v1 = v10 + tx*(v11 - v10);
        return v0 + ty*(v1 - v0);
    }
}
#endif

struct IntegrateInvoker : ParallelLoopBody
{
    IntegrateInvoker(TSDFVolumeCPU& _volume, const Depth& _depth, Intr intrinsics, cv::Affine3f cameraPose,
                     float depthFactor) :
        ParallelLoopBody(),
        volume(_volume),
        depth(_depth),
        proj(intrinsics.makeProjector()),
        vol2cam(cameraPose.inv() * volume.pose),
        truncDistInv(1.f/volume.truncDist),
        dfac(1.f/depthFactor)
    {
        volDataStart = volume.volume[0];
    }

#if CV_SIMD128
    virtual void operator() (const Range& range) const override
    {
        // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
        Point3f zStepPt = Point3f(vol2cam.matrix(0, 2), vol2cam.matrix(1, 2), vol2cam.matrix(2, 2))*volume.voxelSize;
        v_float32x4 zStep(zStepPt.x, zStepPt.y, zStepPt.z, 0);
        v_float32x4 vfxy(proj.fx, proj.fy, 0.f, 0.f), vcxy(proj.cx, proj.cy, 0.f, 0.f);
        const v_float32x4 upLimits = v_cvt_f32(v_int32x4(depth.cols-1, depth.rows-1, 0, 0));

        // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
        for(int x = range.start; x < range.end; x++)
        {
            Voxel* volDataX = volDataStart + x*volume.dims[0];
            for(int y = 0; y < volume.edgeResolution; y++)
            {
                Voxel* volDataY = volDataX+y*volume.dims[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam*Point3f(x*volume.voxelSize, y*volume.voxelSize, 0);
                v_float32x4 camSpacePt(basePt.x, basePt.y, basePt.z, 0);

                int startZ, endZ;
                if(abs(zStepPt.z) > 1e-5)
                {
                    int baseZ = (int)(-basePt.z / zStepPt.z);
                    if(zStepPt.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volume.edgeResolution;
                    }
                    else
                    {
                        startZ = 0;
                        endZ = baseZ;
                    }
                }
                else
                {
                    if(basePt.z > 0)
                    {
                        startZ = 0; endZ = volume.edgeResolution;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(volume.edgeResolution, endZ);
                for(int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt += zStep;

                    float zCamSpace = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(camSpacePt))).get0();

                    if(zCamSpace <= 0.f)
                        continue;

                    v_float32x4 camPixVec = camSpacePt/v_setall_f32(zCamSpace);
                    v_float32x4 projected = v_muladd(camPixVec, vfxy, vcxy);
                    // leave only first 2 lanes
                    projected = v_reinterpret_as_f32(v_reinterpret_as_u32(projected) &
                                                     v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0, 0));

                    depthType v;
                    // bilinearly interpolate depth at projected
                    {
                        const v_float32x4& pt = projected;
                        // check coords >= 0 and < imgSize
                        v_uint32x4 limits = v_reinterpret_as_u32(pt < v_setzero_f32()) |
                                            v_reinterpret_as_u32(pt >= upLimits);
                        limits = limits | v_rotate_right<1>(limits);
                        if(limits.get0())
                            continue;

                        // xi, yi = floor(pt)
                        v_int32x4 ip = v_floor(pt);
                        v_int32x4 ipshift = ip;
                        int xi = ipshift.get0();
                        ipshift = v_rotate_right<1>(ipshift);
                        int yi = ipshift.get0();

                        const depthType* row0 = depth[yi+0];
                        const depthType* row1 = depth[yi+1];

                        // v001 = [v(xi + 0, yi + 0), v(xi + 1, yi + 0)]
                        v_float32x4 v001 = v_load_low(row0 + xi);
                        // v101 = [v(xi + 0, yi + 1), v(xi + 1, yi + 1)]
                        v_float32x4 v101 = v_load_low(row1 + xi);

                        v_float32x4 vall = v_combine_low(v001, v101);

                        // assume correct depth is positive
                        // don't fix missing data
                        if(v_check_all(vall > v_setzero_f32()))
                        {
                            v_float32x4 t = pt - v_cvt_f32(ip);
                            float tx = t.get0();
                            t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
                            v_float32x4 ty = v_setall_f32(t.get0());
                            // vx is y-interpolated between rows 0 and 1
                            v_float32x4 vx = v001 + ty*(v101 - v001);
                            float v0 = vx.get0();
                            vx = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vx)));
                            float v1 = vx.get0();
                            v = v0 + tx*(v1 - v0);
                        }
                        else
                            continue;
                    }

                    // norm(camPixVec) produces double which is too slow
                    float pixNorm = sqrt(v_reduce_sum(camPixVec*camPixVec));
                    // difference between distances of point and of surface to camera
                    volumeType sdf = pixNorm*(v*dfac - zCamSpace);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);

                    if(sdf >= -volume.truncDist)
                    {
                        volumeType tsdf = fmin(1.f, sdf * truncDistInv);

                        Voxel& voxel = volDataY[z];
                        int& weight = voxel.weight;
                        volumeType& value = voxel.v;

                        // update TSDF
                        value = (value*weight+tsdf) / (weight + 1);
                        weight = min(weight + 1, volume.maxWeight);
                    }
                }
            }
        }
    }
#else
    virtual void operator() (const Range& range) const
    {
        // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
        for(int x = range.start; x < range.end; x++)
        {
            Voxel* volDataX = volDataStart + x*volume.dims[0];
            for(int y = 0; y < volume.edgeResolution; y++)
            {
                Voxel* volDataY = volDataX+y*volume.dims[1];
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam*Point3f(x*volume.voxelSize, y*volume.voxelSize, 0);
                Point3f camSpacePt = basePt;
                // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                Point3f zStep = Point3f(vol2cam.matrix(0, 2), vol2cam.matrix(1, 2), vol2cam.matrix(2, 2))*volume.voxelSize;

                int startZ, endZ;
                if(abs(zStep.z) > 1e-5)
                {
                    int baseZ = -basePt.z / zStep.z;
                    if(zStep.z > 0)
                    {
                        startZ = baseZ;
                        endZ = volume.edgeResolution;
                    }
                    else
                    {
                        startZ = 0;
                        endZ = baseZ;
                    }
                }
                else
                {
                    if(basePt.z > 0)
                    {
                        startZ = 0; endZ = volume.edgeResolution;
                    }
                    else
                    {
                        // z loop shouldn't be performed
                        startZ = endZ = 0;
                    }
                }
                startZ = max(0, startZ);
                endZ = min(volume.edgeResolution, endZ);
                for(int z = startZ; z < endZ; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*volume.voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt += zStep;

                    if(camSpacePt.z <= 0)
                        continue;

                    Point3f camPixVec;
                    Point2f projected = proj(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if(v == 0)
                        continue;

                    // norm(camPixVec) produces double which is too slow
                    float pixNorm = sqrt(camPixVec.dot(camPixVec));
                    // difference between distances of point and of surface to camera
                    volumeType sdf = pixNorm*(v*dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);

                    if(sdf >= -volume.truncDist)
                    {
                        volumeType tsdf = fmin(1.f, sdf * truncDistInv);

                        Voxel& voxel = volDataY[z];
                        int& weight = voxel.weight;
                        volumeType& value = voxel.v;

                        // update TSDF
                        value = (value*weight+tsdf) / (weight + 1);
                        weight = min(weight + 1, volume.maxWeight);
                    }
                }
            }
        }
    }
#endif

    TSDFVolumeCPU& volume;
    const Depth& depth;
    const Intr::Projector proj;
    const cv::Affine3f vol2cam;
    const float truncDistInv;
    const float dfac;
    Voxel* volDataStart;
};

// use depth instead of distance (optimization)
void TSDFVolumeCPU::integrate(cv::Ptr<Frame> _depth, float depthFactor, cv::Affine3f cameraPose, Intr intrinsics)
{
    ScopeTime st("tsdf: integrate");

    Depth depth;
    _depth->getDepth(depth);

    IntegrateInvoker ii(*this, depth, intrinsics, cameraPose, depthFactor);
    Range range(0, edgeResolution);
    parallel_for_(range, ii);
}

#if CV_SIMD128
// all coordinate checks should be done in inclosing cycle
inline volumeType TSDFVolumeCPU::interpolateVoxel(Point3f _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0);
    return interpolateVoxel(p);
}

inline volumeType TSDFVolumeCPU::interpolateVoxel(const v_float32x4& p) const
{
    // tx, ty, tz = floor(p)
    v_int32x4 ip = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    float tx = t.get0();
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = t.get0();
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = t.get0();

    int xdim = dims[0], ydim = dims[1];
    const Voxel* volData = volume[0];

    int ix = ip.get0(); ip = v_rotate_right<1>(ip);
    int iy = ip.get0(); ip = v_rotate_right<1>(ip);
    int iz = ip.get0();

    int coordBase = ix*xdim + iy*ydim + iz;

    volumeType vx[8];
    for(int i = 0; i < 8; i++)
        vx[i] = volData[neighbourCoords[i] + coordBase].v;

    v_float32x4 v0246(vx[0], vx[2], vx[4], vx[6]), v1357(vx[1], vx[3], vx[5], vx[7]);
    v_float32x4 vxx = v0246 + v_setall_f32(tz)*(v1357 - v0246);

    v_float32x4 v00_10 = vxx;
    v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

    v_float32x4 v0_1 = v00_10 + v_setall_f32(ty)*(v01_11 - v00_10);
    float v0 = v0_1.get0();
    v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
    float v1 = v0_1.get0();

    return v0 + tx*(v1 - v0);
}
#else
inline volumeType TSDFVolumeCPU::interpolateVoxel(Point3f p) const
{
    int xdim = dims[0], ydim = dims[1];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz;
    const Voxel* volData = volume[0];

    volumeType vx[8];
    for(int i = 0; i < 8; i++)
        vx[i] = volData[neighbourCoords[i] + coordBase].v;

    volumeType v00 = vx[0] + tz*(vx[1] - vx[0]);
    volumeType v01 = vx[2] + tz*(vx[3] - vx[2]);
    volumeType v10 = vx[4] + tz*(vx[5] - vx[4]);
    volumeType v11 = vx[6] + tz*(vx[7] - vx[6]);

    volumeType v0 = v00 + ty*(v01 - v00);
    volumeType v1 = v10 + ty*(v11 - v10);

    return v0 + tx*(v1 - v0);
}
#endif


#if CV_SIMD128
//gradientDeltaFactor is fixed at 1.0 of voxel size
inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f _p) const
{
    v_float32x4 p(_p.x, _p.y, _p.z, 0.f);
    v_float32x4 result = getNormalVoxel(p);
    float CV_DECL_ALIGNED(16) ares[4];
    v_store_aligned(ares, result);
    return Point3f(ares[0], ares[1], ares[2]);
}

inline v_float32x4 TSDFVolumeCPU::getNormalVoxel(const v_float32x4& p) const
{
    if(v_check_any((p < v_float32x4(1.f, 1.f, 1.f, 0.f)) +
                   (p >= v_setall_f32((float)(edgeResolution-2)))))
        return nanv;

    v_int32x4 ip = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    float tx = t.get0();
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float ty = t.get0();
    t = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(t)));
    float tz = t.get0();

    int xdim = dims[0], ydim = dims[1];
    const Voxel* volData = volume[0];

    int ix = ip.get0(); ip = v_rotate_right<1>(ip);
    int iy = ip.get0(); ip = v_rotate_right<1>(ip);
    int iz = ip.get0();

    int coordBase = ix*xdim + iy*ydim + iz;

    float CV_DECL_ALIGNED(16) an[4];
    an[0] = an[1] = an[2] = an[3] = 0.f;
    for(int c = 0; c < 3; c++)
    {
        const int dim = dims[c];
        float& nv = an[c];

        volumeType vx[8];
        for(int i = 0; i < 8; i++)
            vx[i] = volData[neighbourCoords[i] + coordBase + 1*dim].v -
                    volData[neighbourCoords[i] + coordBase - 1*dim].v;

        v_float32x4 v0246(vx[0], vx[2], vx[4], vx[6]), v1357(vx[1], vx[3], vx[5], vx[7]);
        v_float32x4 vxx = v0246 + v_setall_f32(tz)*(v1357 - v0246);

        v_float32x4 v00_10 = vxx;
        v_float32x4 v01_11 = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(vxx)));

        v_float32x4 v0_1 = v00_10 + v_setall_f32(ty)*(v01_11 - v00_10);
        float v0 = v0_1.get0();
        v0_1 = v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(v0_1)));
        float v1 = v0_1.get0();

        nv = v0 + tx*(v1 - v0);
    }

    v_float32x4 n = v_load_aligned(an);
    v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(n*n)));
    return n*invNorm;
}
#else
inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f p) const
{
    const int xdim = dims[0], ydim = dims[1];
    const Voxel* volData = volume[0];

    if(p.x < 1 || p.x >= edgeResolution -2 ||
       p.y < 1 || p.y >= edgeResolution -2 ||
       p.z < 1 || p.z >= edgeResolution -2)
        return nan3;

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz;

    Vec3f an;
    for(int c = 0; c < 3; c++)
    {
        const int dim = dims[c];
        float& nv = an[c];

        volumeType vx[8];
        for(int i = 0; i < 8; i++)
            vx[i] = volData[neighbourCoords[i] + coordBase + 1*dim].v -
                    volData[neighbourCoords[i] + coordBase - 1*dim].v;

        volumeType v00 = vx[0] + tz*(vx[1] - vx[0]);
        volumeType v01 = vx[2] + tz*(vx[3] - vx[2]);
        volumeType v10 = vx[4] + tz*(vx[5] - vx[4]);
        volumeType v11 = vx[6] + tz*(vx[7] - vx[6]);

        volumeType v0 = v00 + ty*(v01 - v00);
        volumeType v1 = v10 + ty*(v11 - v10);

        nv = v0 + tx*(v1 - v0);
    }

    return normalize(an);
}
#endif


struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, Affine3f cameraPose,
                   Intr intrinsics, const TSDFVolumeCPU& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        volume(_volume),
        tstep(volume.truncDist * volume.raycastStepFactor),
        // We do subtract voxel size to minimize checks after
        // Note: origin of volume coordinate is placed
        // in the center of voxel (0,0,0), not in the corner of the voxel!
        boxMax(volume.edgeSize - volume.voxelSize,
               volume.edgeSize - volume.voxelSize,
               volume.edgeSize - volume.voxelSize),
        boxMin(),
        cam2vol(volume.pose.inv() * cameraPose),
        vol2cam(cameraPose.inv() * volume.pose),
        reproj(intrinsics.makeReprojector())
    {  }

#if CV_SIMD128
    virtual void operator() (const Range& range) const override
    {
        const v_float32x4 vfxy(reproj.fxinv, reproj.fyinv, 0, 0);
        const v_float32x4 vcxy(reproj.cx, reproj.cy, 0, 0);

        const float (&cm)[16] = cam2vol.matrix.val;
        const v_float32x4 camRot0(cm[0], cm[4], cm[ 8], 0);
        const v_float32x4 camRot1(cm[1], cm[5], cm[ 9], 0);
        const v_float32x4 camRot2(cm[2], cm[6], cm[10], 0);
        const v_float32x4 camTrans(cm[3], cm[7], cm[11], 0);

        const v_float32x4 boxDown(boxMin.x, boxMin.y, boxMin.z, 0.f);
        const v_float32x4 boxUp(boxMax.x, boxMax.y, boxMax.z, 0.f);

        const v_float32x4 invVoxelSize = v_setall_f32(volume.voxelSizeInv);

        const float (&vm)[16] = vol2cam.matrix.val;
        const v_float32x4 volRot0(vm[0], vm[4], vm[ 8], 0);
        const v_float32x4 volRot1(vm[1], vm[5], vm[ 9], 0);
        const v_float32x4 volRot2(vm[2], vm[6], vm[10], 0);
        const v_float32x4 volTrans(vm[3], vm[7], vm[11], 0);

        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                v_float32x4 point = nanv, normal = nanv;

                v_float32x4 orig = camTrans;

                // get direction through pixel in volume space:

                // 1. reproject (x, y) on projecting plane where z = 1.f
                v_float32x4 planed = (v_float32x4((float)x, (float)y, 0.f, 0.f) - vcxy)*vfxy;
                planed = v_combine_low(planed, v_float32x4(1.f, 0.f, 0.f, 0.f));

                // 2. rotate to volume space
                planed = v_matmuladd(planed, camRot0, camRot1, camRot2, v_setzero_f32());

                // 3. normalize
                v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(planed*planed)));
                v_float32x4 dir = planed*invNorm;

                // compute intersection of ray with all six bbox planes
                v_float32x4 rayinv = v_setall_f32(1.f)/dir;
                // div by zero should be eliminated by these products
                v_float32x4 tbottom = rayinv*(boxDown - orig);
                v_float32x4 ttop    = rayinv*(boxUp   - orig);

                // re-order intersections to find smallest and largest on each axis
                v_float32x4 minAx = v_min(ttop, tbottom);
                v_float32x4 maxAx = v_max(ttop, tbottom);

                // near clipping plane
                const float clip = 0.f;
                float tmin = max(v_reduce_max(minAx), clip);
                float tmax =     v_reduce_min(maxAx);

                // precautions against getting coordinates out of bounds
                tmin = tmin + tstep;
                tmax = tmax - tstep;

                if(tmin < tmax)
                {
                    // interpolation optimized a little
                    orig *= invVoxelSize;
                    dir  *= invVoxelSize;

                    int xdim = volume.dims[0], ydim = volume.dims[1];
                    v_float32x4 rayStep = dir * v_setall_f32(tstep);
                    v_float32x4 next = (orig + dir * v_setall_f32(tmin));
                    volumeType f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = cvFloor((tmax - tmin)/tstep);
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        v_int32x4 ip = v_round(next);
                        int ix = ip.get0(); ip = v_rotate_right<1>(ip);
                        int iy = ip.get0(); ip = v_rotate_right<1>(ip);
                        int iz = ip.get0();
                        int coord = ix*xdim + iy*ydim + iz;

                        fnext = volume.volume(coord).v;
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
                        v_float32x4 tp = next - rayStep;
                        volumeType ft   = volume.interpolateVoxel(tp);
                        volumeType ftdt = volume.interpolateVoxel(next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep*(steps - ft/(ftdt - ft));

                        // avoid division by zero
                        if(!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            v_float32x4 pv = (orig + dir*v_setall_f32(ts));
                            v_float32x4 nv = volume.getNormalVoxel(pv);

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = v_matmuladd(nv, volRot0, volRot1, volRot2, v_setzero_f32());
                                // interpolation optimized a little
                                point = v_matmuladd(pv*v_setall_f32(volume.voxelSize), volRot0, volRot1, volRot2, volTrans);
                            }
                        }
                    }
                }

                v_store((float*)(&ptsRow[x]), point);
                v_store((float*)(&nrmRow[x]), normal);
            }
        }
    }
#else
    virtual void operator() (const Range& range) const
    {
        const Point3f camTrans = cam2vol.translation();
        const Matx33f  camRot  = cam2vol.rotation();
        const Matx33f  volRot  = vol2cam.rotation();

        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3;

                Point3f orig = camTrans;
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(camRot * reproj(Point3f(x, y, 1.f))));

                // compute intersection of ray with all six bbox planes
                Vec3f rayinv(1.f/dir.x, 1.f/dir.y, 1.f/dir.z);
                Point3f tbottom = rayinv.mul(boxMin - orig);
                Point3f ttop    = rayinv.mul(boxMax - orig);

                // re-order intersections to find smallest and largest on each axis
                Point3f minAx(min(ttop.x, tbottom.x), min(ttop.y, tbottom.y), min(ttop.z, tbottom.z));
                Point3f maxAx(max(ttop.x, tbottom.x), max(ttop.y, tbottom.y), max(ttop.z, tbottom.z));

                // near clipping plane
                const float clip = 0.f;
                float tmin = max(max(max(minAx.x, minAx.y), max(minAx.x, minAx.z)), clip);
                float tmax =     min(min(maxAx.x, maxAx.y), min(maxAx.x, maxAx.z));

                if(tmin < tmax)
                {
                    // precautions against getting coordinates out of bounds
                    tmin = tmin + tstep;
                    tmax = tmax - tstep - tstep;

                    // interpolation optimized a little
                    orig *= volume.voxelSizeInv;
                    dir *= volume.voxelSizeInv;

                    Point3f rayStep = dir * tstep;
                    Point3f next = (orig + dir * tmin);
                    volumeType f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = floor((tmax - tmin)/tstep);
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volume.dims[0], ydim = volume.dims[1];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = volume.volume(ix*xdim + iy*ydim + iz).v;
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
                        Point3f tp = next - rayStep;
                        volumeType ft   = volume.interpolateVoxel(tp);
                        volumeType ftdt = volume.interpolateVoxel(next);
                        // float t = tmin + steps*tstep;
                        // float ts = t - tstep*ft/(ftdt - ft);
                        float ts = tmin + tstep*(steps - ft/(ftdt - ft));

                        // avoid division by zero
                        if(!cvIsNaN(ts) && !cvIsInf(ts))
                        {
                            Point3f pv = (orig + dir*ts);
                            Point3f nv = volume.getNormalVoxel(pv);

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                // interpolation optimized a little
                                point = vol2cam * (pv * volume.voxelSize);
                            }
                        }
                    }
                }

                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    }
#endif

    Points& points;
    Normals& normals;
    const TSDFVolumeCPU& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};


void TSDFVolumeCPU::raycast(cv::Affine3f cameraPose, Intr intrinsics, Size frameSize, int pyramidLevels,
                            cv::Ptr<FrameGenerator> frameGenerator, cv::Ptr<Frame> frame) const
{
    ScopeTime st("tsdf: raycast");

    CV_Assert(frameSize.area() > 0);

    Points points(frameSize);
    Normals normals(frameSize);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), RaycastInvoker(points, normals, cameraPose, intrinsics, *this), nstripes);

    // build a pyramid of points and normals
    (*frameGenerator)(frame, points, normals, pyramidLevels);
}


struct FetchPointsNormalsInvoker : ParallelLoopBody
{
    FetchPointsNormalsInvoker(const TSDFVolumeCPU& _volume,
                              std::vector< std::vector<ptype> >& _pVecs,
                              std::vector< std::vector<ptype> >& _nVecs,
                              bool _needNormals) :
        ParallelLoopBody(),
        vol(_volume),
        pVecs(_pVecs),
        nVecs(_nVecs),
        needNormals(_needNormals)
    {
        volDataStart = vol.volume[0];
    }

    inline void coord(std::vector<ptype>& points, std::vector<ptype>& normals,
                      int x, int y, int z, Point3f V, float v0, int axis) const
    {
        // 0 for x, 1 for y, 2 for z
        const int edgeResolution = vol.edgeResolution;
        bool limits = false;
        Point3i shift;
        float Vc = 0.f;
        if(axis == 0)
        {
            shift = Point3i(1, 0, 0);
            limits = (x + 1 < edgeResolution);
            Vc = V.x;
        }
        if(axis == 1)
        {
            shift = Point3i(0, 1, 0);
            limits = (y + 1 < edgeResolution);
            Vc = V.y;
        }
        if(axis == 2)
        {
            shift = Point3i(0, 0, 1);
            limits = (z + 1 < edgeResolution);
            Vc = V.z;
        }

        if(limits)
        {
            const Voxel& voxeld = volDataStart[(x+shift.x)*vol.dims[0] +
                                               (y+shift.y)*vol.dims[1] +
                                               (z+shift.z)];
            volumeType vd = voxeld.v;

            if(voxeld.weight != 0 && vd != 1.f)
            {
                if((v0 > 0 && vd < 0) || (v0 < 0 && vd > 0))
                {
                    //linearly interpolate coordinate
                    float Vn = Vc + vol.voxelSize;
                    float dinv = 1.f/(abs(v0)+abs(vd));
                    float inter = (Vc*abs(vd) + Vn*abs(v0))*dinv;

                    Point3f p(shift.x ? inter : V.x,
                              shift.y ? inter : V.y,
                              shift.z ? inter : V.z);
                    {
                        points.push_back(toPtype(vol.pose * p));
                        if(needNormals)
                            normals.push_back(toPtype(vol.pose.rotation() * vol.getNormalVoxel(p * vol.voxelSizeInv)));
                    }
                }
            }
        }
    }

    virtual void operator() (const Range& range) const override
    {
        // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
        std::vector<ptype> points, normals;
        for(int x = range.start; x < range.end; x++)
        {
            const Voxel* volDataX = volDataStart + x*vol.dims[0];
            for(int y = 0; y < vol.edgeResolution; y++)
            {
                const Voxel* volDataY = volDataX + y*vol.dims[1];
                for(int z = 0; z < vol.edgeResolution; z++)
                {
                    const Voxel& voxel0 = volDataY[z];
                    volumeType v0 = voxel0.v;
                    if(voxel0.weight != 0 && v0 != 1.f)
                    {
                        Point3f V = (Point3f((float)x, (float)y, (float)z) +
                                     Point3f(0.5f, 0.5f, 0.5f))*vol.voxelSize;

                        coord(points, normals, x, y, z, V, v0, 0);
                        coord(points, normals, x, y, z, V, v0, 1);
                        coord(points, normals, x, y, z, V, v0, 2);

                    } // if voxel is not empty
                }
            }
        }

        AutoLock al(mutex);
        pVecs.push_back(points);
        nVecs.push_back(normals);
    }

    const TSDFVolumeCPU& vol;
    std::vector< std::vector<ptype> >& pVecs;
    std::vector< std::vector<ptype> >& nVecs;
    const Voxel* volDataStart;
    bool needNormals;
    mutable Mutex mutex;
};

void TSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    ScopeTime st("tsdf: fetch points+normals");

    if(_points.needed())
    {
        std::vector< std::vector<ptype> > pVecs, nVecs;
        FetchPointsNormalsInvoker fi(*this, pVecs, nVecs, _normals.needed());
        Range range(0, edgeResolution);
        const int nstripes = -1;
        parallel_for_(range, fi, nstripes);
        std::vector<ptype> points, normals;
        for(size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
        }

        _points.create((int)points.size(), 1, DataType<ptype>::type);
        Mat((int)points.size(), 1, DataType<ptype>::type, &points[0]).copyTo(_points.getMat());

        if(_normals.needed())
        {
            _normals.create((int)normals.size(), 1, DataType<ptype>::type);
            Mat((int)normals.size(), 1, DataType<ptype>::type, &normals[0]).copyTo(_normals.getMat());
        }
    }
}


struct PushNormals
{
    PushNormals(const TSDFVolumeCPU& _vol, Mat_<ptype>& _nrm) :
        vol(_vol), normals(_nrm), invPose(vol.pose.inv())
    { }
    void operator ()(const ptype &pp, const int * position) const
    {
        Point3f p = fromPtype(pp);
        Point3f n = nan3;
        if(!isNaN(p))
        {
            n = vol.pose.rotation() * vol.getNormalVoxel(invPose * p * vol.voxelSizeInv);
        }
        normals(position[0], position[1]) = toPtype(n);
    }
    const TSDFVolumeCPU& vol;
    Mat_<ptype>& normals;

    Affine3f invPose;
};


void TSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    ScopeTime st("tsdf: fetch normals");

    if(_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == DataType<ptype>::type);

        _normals.createSameSize(_points, _points.type());
        Mat_<ptype> normals = _normals.getMat();

        points.forEach(PushNormals(*this, normals));
    }
}

///////// GPU implementation /////////

class TSDFVolumeGPU : public TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolumeGPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                  float _raycastStepFactor);

    virtual void integrate(cv::Ptr<Frame> depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics) override;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, int pyramidLevels,
                         cv::Ptr<FrameGenerator> frameGenerator, cv::Ptr<Frame> frame) const override;

    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals) const override;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override;

    virtual void reset() override;
};


TSDFVolumeGPU::TSDFVolumeGPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
              float _raycastStepFactor) :
    TSDFVolume(_res, _size, _pose, _truncDist, _maxWeight, _raycastStepFactor)
{ }


// zero volume, leave rest params the same
void TSDFVolumeGPU::reset()
{
    throw std::runtime_error("Not implemented");
}


// use depth instead of distance (optimization)
void TSDFVolumeGPU::integrate(cv::Ptr<Frame> /*depth*/, float /*depthFactor*/, cv::Affine3f /*cameraPose*/, Intr /*intrinsics*/)
{
    throw std::runtime_error("Not implemented");
}


void TSDFVolumeGPU::raycast(cv::Affine3f /*cameraPose*/, Intr /*intrinsics*/, Size /*frameSize*/, int /*pyramidLevels*/,
                            Ptr<FrameGenerator> /* frameGenerator */, Ptr<Frame> /* frame */) const
{
    throw std::runtime_error("Not implemented");
}


void TSDFVolumeGPU::fetchNormals(InputArray /*_points*/, OutputArray /*_normals*/) const
{
    throw std::runtime_error("Not implemented");
}

void TSDFVolumeGPU::fetchPointsNormals(OutputArray /*points*/, OutputArray /*normals*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<TSDFVolume> makeTSDFVolume(cv::kinfu::KinFu::Params::PlatformType t,
                                   int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor)
{
    switch (t)
    {
    case cv::kinfu::KinFu::Params::PlatformType::PLATFORM_CPU:
        return cv::makePtr<TSDFVolumeCPU>(_res, _size, _pose, _truncDist, _maxWeight,
                                          _raycastStepFactor);
    case cv::kinfu::KinFu::Params::PlatformType::PLATFORM_GPU:
        return cv::makePtr<TSDFVolumeGPU>(_res, _size, _pose, _truncDist, _maxWeight,
                                          _raycastStepFactor);
    default:
        return cv::Ptr<TSDFVolume>();
    }
}

} // namespace kinfu
} // namespace cv
