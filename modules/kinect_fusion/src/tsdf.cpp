//TODO: add license

#include "precomp.hpp"
#include "tsdf.hpp"

using namespace cv;
using namespace cv::kinfu;


typedef float volumeType; // can be float16
struct Voxel
{
    volumeType v;
    int weight;
};

namespace cv
{

template<> class DataType<Voxel>
{
public:
    typedef Voxel       value_type;
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

}

class TSDFVolumeCPU : public TSDFVolume
{
    typedef cv::Mat_<Voxel> Volume;

public:
    // dimension in voxels, size in meters
    TSDFVolumeCPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                  float _raycastStepFactor, float _gradientDeltaFactor);

    virtual void integrate(cv::Ptr<Frame> depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics);
    virtual cv::Ptr<Frame> raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, int pyramidLevels,
                                   cv::Ptr<FrameGenerator> frameGenerator) const;

    virtual void fetchPoints(cv::OutputArray points) const;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const;

    virtual void reset();

    volumeType interpolateVoxel(cv::Point3f p) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

#if CV_SIMD128
    volumeType interpolateVoxel(const v_float32x4& p) const;
    v_float32x4 getNormalVoxel(const v_float32x4& p) const;
#endif

    // edgeResolution^3 array
    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    Volume volume;
    float edgeSize;
    int edgeResolution;
    int neighbourCoords[8];
    int dimStep[4];
    float voxelSize;
    float voxelSizeInv;
    float truncDist;
    float raycastStepFactor;
    float gradientDeltaFactor;
    int maxWeight;
    cv::Affine3f pose;
};


TSDFVolume::TSDFVolume(int /*_res*/, float /*_size*/, Affine3f /*_pose*/, float /*_truncDist*/, int /*_maxWeight*/,
                       float /*_raycastStepFactor*/, float /*_gradientDeltaFactor*/)
{ }

// dimension in voxels, size in meters
TSDFVolumeCPU::TSDFVolumeCPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                             float _raycastStepFactor, float _gradientDeltaFactor) :
    TSDFVolume(_res, _size, _pose, _truncDist, _maxWeight, _raycastStepFactor, _gradientDeltaFactor)
{
    CV_Assert(_res % 32 == 0);
    edgeResolution = _res;
    int xdim = edgeResolution*edgeResolution;
    int ydim = edgeResolution;
    int steps[4] = { xdim, ydim, 1, 0 };
    for(int i = 0; i < 4; i++)
        dimStep[i] = steps[i];
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
    edgeSize = _size;
    voxelSize = edgeSize/edgeResolution;
    voxelSizeInv = edgeResolution/edgeSize;
    volume = Volume(1, _res * _res * _res);
    pose = _pose;
    truncDist = std::max (_truncDist, 2.1f * voxelSize);
    raycastStepFactor = _raycastStepFactor;
    gradientDeltaFactor = _gradientDeltaFactor;
    maxWeight = _maxWeight;
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
            float tx1 = 1.f-tx, ty1 = 1.f-ty;
            return v00*tx1*ty1 + v01*tx*ty1 + v10*tx1*ty + v11*tx*ty;
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

        float tx1 = 1.f-tx, ty1 = 1.f-ty;
        return v00*tx1*ty1 + v01*tx*ty1 + v10*tx1*ty + v11*tx*ty;

        // speed is the same
    //    float txty = tx*ty;
    //    depthType d001 = v00 - v01;
    //    return v00 + tx*d001 + ty*(v10-v00) + txty*(d001 - v10 + v11);
    }
}

struct IntegrateInvoker : ParallelLoopBody
{
    IntegrateInvoker(TSDFVolumeCPU& _volume, const Depth& _depth, Intr intrinsics, cv::Affine3f cameraPose,
                     float depthFactor) :
        ParallelLoopBody(),
        volume(_volume),
        depth(_depth),
        proj(intrinsics.makeProjector()),
        vol2cam(cameraPose.inv() * volume.pose),
        truncDistInv(1./volume.truncDist),
        dfac(1.f/depthFactor)
    { }

    virtual void operator() (const Range& range) const
    {
        // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
        for(int x = range.start; x < range.end; x++)
        {
            for(int y = 0; y < volume.edgeResolution; y++)
            {
                // optimization of camSpace transformation (vector addition instead of matmul at each z)
                Point3f basePt = vol2cam*Point3f(x*volume.voxelSize, y*volume.voxelSize, 0);
                Point3f camSpacePt = basePt;
                // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
                Point3f zStep = Point3f(vol2cam.matrix(0, 2), vol2cam.matrix(1, 2), vol2cam.matrix(2, 2))*volume.voxelSize;
                int baseZ = -basePt.z / zStep.z;
                baseZ = max(0, min(volume.edgeResolution, baseZ));
                for(int z = baseZ; z < volume.edgeResolution; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt += zStep;

                    if(camSpacePt.z <= 0)
                        continue;

                    Point3f camPixVec;
                    Point2f projected = proj(camSpacePt, camPixVec);

                    depthType v = bilinearDepth(depth, projected);
                    if(v == 0)
                        continue;

                    // difference between distances of point and of surface to camera
                    volumeType sdf = norm(camPixVec)*(v*dfac - camSpacePt.z);
                    // possible alternative is:
                    // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);

                    if(sdf >= -volume.truncDist)
                    {
                        volumeType tsdf = fmin(1.f, sdf * truncDistInv);

                        Voxel& voxel = volume.volume(x*volume.dimStep[0] + y*volume.dimStep[1] + z);
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

    TSDFVolumeCPU& volume;
    const Depth& depth;
    const Intr::Projector proj;
    const cv::Affine3f vol2cam;
    const float truncDistInv;
    const float dfac;
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
    const v_int32x4 mulDim = v_load(dimStep);

    v_int32x4 ip = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    v_float32x4 t1 = v_setall_f32(1.f) - t;

    float CV_DECL_ALIGNED(16) ttmp[8];
    v_store_aligned(ttmp + 0, t);
    v_store_aligned(ttmp + 4, t1);
    float tx = ttmp[0], ty = ttmp[1], tz = ttmp[2];
    float tx1 = ttmp[4], ty1 = ttmp[5], tz1 = ttmp[6];

    v_float32x4 tmul = v_float32x4(ty1, ty1, ty, ty)*v_float32x4(tz1, tz, tz1, tz);
    v_float32x4 tv0 = v_setall_f32(tx1)*tmul;
    v_float32x4 tv1 = v_setall_f32(tx )*tmul;

    int coordBase = v_reduce_sum(ip*mulDim);

    const Voxel* volData = volume[0];
    volumeType CV_DECL_ALIGNED(16) av[8];
    for(int i = 0; i < 8; i++)
        av[i] = volData[neighbourCoords[i] + coordBase].v;

    v_float32x4 v0 = v_load_aligned(av);
    v_float32x4 v1 = v_load_aligned(av + 4);

    v_float32x4 mulN = tv0 * v0 + tv1 * v1;

    volumeType sum = v_reduce_sum(mulN);

    return sum;
}
#else
inline volumeType TSDFVolumeCPU::interpolateVoxel(Point3f p) const
{
    int xdim = dimStep[0], ydim = dimStep[1];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;
    float tx1 = 1.f - tx;
    float ty1 = 1.f - ty;
    float tz1 = 1.f - tz;
    float tv[8] = { tx1 * ty1 * tz1,
                    tx1 * ty1 * tz,
                    tx1 * ty  * tz1,
                    tx1 * ty  * tz,
                    tx  * ty1 * tz1,
                    tx  * ty1 * tz,
                    tx  * ty  * tz1,
                    tx  * ty  * tz  };

    int coordBase = ix*xdim + iy*ydim + iz;

    volumeType v[8];
    for(int i = 0; i < 8; i++)
        v[i] = volume.at<Voxel>(neighbourCoords[i] + coordBase).v;

    volumeType mulN[8];
    for(int i = 0; i < 8; i++)
        mulN[i] = tv[i]*v[i];

    volumeType sum = 0;
    for(int i = 0; i < 8; i++)
        sum += mulN[i];

    return sum;
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
    const v_int32x4 mulDim = v_load(dimStep);

    if(v_check_any((p < v_float32x4(1.f, 1.f, 1.f, 0.f)) +
                   (p >= v_setall_f32(edgeResolution-2))))
        return nanv;

    v_int32x4 ip = v_floor(p);
    v_float32x4 t = p - v_cvt_f32(ip);
    v_float32x4 t1 = v_setall_f32(1.f) - t;

    float CV_DECL_ALIGNED(16) ttmp[8];
    v_store_aligned(ttmp + 0, t);
    v_store_aligned(ttmp + 4, t1);
    float tx = ttmp[0], ty = ttmp[1], tz = ttmp[2];
    float tx1 = ttmp[4], ty1 = ttmp[5], tz1 = ttmp[6];

    v_float32x4 tmul = v_float32x4(ty1, ty1, ty, ty)*v_float32x4(tz1, tz, tz1, tz);
    v_float32x4 tv0 = v_setall_f32(tx1)*tmul;
    v_float32x4 tv1 = v_setall_f32(tx )*tmul;

    const Voxel* volData = volume[0];
    int coordBase = v_reduce_sum(ip*mulDim);

    float CV_DECL_ALIGNED(16) an[4];
    an[0] = an[1] = an[2] = an[3] = 0.f;
    for(int c = 0; c < 3; c++)
    {
        const int dim = dimStep[c];
        float& nv = an[c];

        volumeType CV_DECL_ALIGNED(16) avp[8], avn[8];
        for(int i = 0; i < 8; i++)
        {
            avp[i] = volData[neighbourCoords[i] + 1*dim + coordBase].v;
            avn[i] = volData[neighbourCoords[i] - 1*dim + coordBase].v;
        }

        v_float32x4 vp0 = v_load_aligned(avp);
        v_float32x4 vp1 = v_load_aligned(avp + 4);
        v_float32x4 vn0 = v_load_aligned(avn);
        v_float32x4 vn1 = v_load_aligned(avn + 4);

        v_float32x4 v0 = vp0 - vn0;
        v_float32x4 v1 = vp1 - vn1;

        v_float32x4 mulN = tv0*v0 + tv1*v1;

        nv = v_reduce_sum(mulN);
    }

    v_float32x4 n = v_load_aligned(an);
    v_float32x4 invNorm = v_invsqrt(v_setall_f32(v_reduce_sum(n*n)));
    return n*invNorm;
}
#else
inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f p) const
{
    const int xdim = dimStep[0], ydim = dimStep[1];

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
    float tx1 = 1.f - tx;
    float ty1 = 1.f - ty;
    float tz1 = 1.f - tz;
    float tv[8] = { tx1 * ty1 * tz1,
                    tx1 * ty1 * tz,
                    tx1 * ty  * tz1,
                    tx1 * ty  * tz,
                    tx  * ty1 * tz1,
                    tx  * ty1 * tz,
                    tx  * ty  * tz1,
                    tx  * ty  * tz  };

    int coordBase = ix*xdim + iy*ydim + iz;

    Vec3f an;
    for(int c = 0; c < 3; c++)
    {
        const int dim = dimStep[c];
        float& nv = an[c];

        volumeType vp[8], vn[8], v[8], mulN[8];
        for(int i = 0; i < 8; i++)
        {
            vp[i] = volume.at<Voxel>(neighbourCoords[i] + 1*dim + coordBase).v;
            vn[i] = volume.at<Voxel>(neighbourCoords[i] - 1*dim + coordBase).v;
        }

        for(int i = 0; i < 8; i++)
            v[i] = (vp[i] - vn[i]);

        for(int i = 0; i < 8; i++)
            mulN[i] = tv[i]*v[i];

        nv = 0;
        for(int i = 0; i < 8; i++)
            nv += mulN[i];
    }

    return normalize(an);
}
#endif


struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points _points, Normals _normals, Affine3f cameraPose,
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
    virtual void operator() (const Range& range) const
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
                v_float32x4 planed = (v_float32x4(x, y, 0.f, 0.f) - vcxy)*vfxy;
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

                if(tmin < tmax)
                {
                    // precautions against getting coordinates out of bounds
                    tmin = tmin + tstep;
                    tmax = tmax - tstep - tstep;

                    // interpolation optimized a little
                    orig *= invVoxelSize;
                    dir  *= invVoxelSize;

                    int xdim = volume.dimStep[0], ydim = volume.dimStep[1];
                    v_float32x4 rayStep = dir * v_setall_f32(tstep);
                    v_float32x4 next = (orig + dir * v_setall_f32(tmin));
                    volumeType f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = floor((tmax - tmin)/tstep);
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        v_int32x4 ip = v_round(next);

                        // it's a bit faster than v_reduce_sum
                        // int coord = v_reduce_sum(ip*mulDim);
                        int CV_DECL_ALIGNED(16) aip[4];
                        v_store_aligned(aip, ip);
                        int coord = aip[0]*xdim + aip[1]*ydim + aip[2];

                        fnext = volume.volume(coord).v;
                        if(fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);

                            // when ray crosses a surface
                            if(f * fnext < 0.f)
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
                        fnext = volume.fetchVoxel(next);
                        int xdim = dimStep[0], ydim = dimStep[1];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = volume.volume(ix*xdim + iy*ydim + iz).v;
                        if(fnext != f)
                        {
                            fnext = volume.interpolateVoxel(next);

                            // when ray crosses a surface
                            if(f * fnext < 0.f)
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


cv::Ptr<Frame> TSDFVolumeCPU::raycast(cv::Affine3f cameraPose, Intr intrinsics, Size frameSize,
                                      int pyramidLevels, cv::Ptr<FrameGenerator> frameGenerator) const
{
    ScopeTime st("tsdf: raycast");

    CV_Assert(frameSize.area() > 0);

    Points points(frameSize);
    Normals normals(frameSize);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), RaycastInvoker(points, normals, cameraPose, intrinsics, *this), nstripes);

    // build a pyramid of points and normals
    return (*frameGenerator)(points, normals, pyramidLevels);
}


struct PushPoints
{
    PushPoints(const TSDFVolumeCPU& _vol, Mat_<ptype>& _pts, Mutex& _mtx) :
        vol(_vol), points(_pts), mtx(_mtx) { }

    inline void coord(int x, int y, int z, Point3f V, float v0, int axis) const
    {
        // 0 for x, 1 for y, 2 for z
        const int edgeResolution = vol.edgeResolution;
        bool limits;
        Point3i shift;
        float Vc;
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
            const Voxel& voxeld = vol.volume((x+shift.x)*edgeResolution*edgeResolution +
                                             (y+shift.y)*edgeResolution +
                                             (z+shift.z));
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
                        AutoLock al(mtx);
                        points.push_back(toPtype(vol.pose * p));
                    }
                }
            }
        }
    }

    void operator ()(const Voxel &voxel0, const int position[2]) const
    {
        volumeType v0 = voxel0.v;
        if(voxel0.weight != 0 && v0 != 1.f)
        {
            int pi = position[1];

            // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
            int x, y, z;
            z = pi % vol.edgeResolution;
            pi = (pi - z)/vol.edgeResolution;
            y = pi % vol.edgeResolution;
            pi = (pi - y)/vol.edgeResolution;
            x = pi % vol.edgeResolution;

            Point3f V = (Point3f(x, y, z) + Point3f(0.5f, 0.5f, 0.5f))*vol.voxelSize;

            coord(x, y, z, V, v0, 0);
            coord(x, y, z, V, v0, 1);
            coord(x, y, z, V, v0, 2);

        } // if voxel is not empty
    }

    const TSDFVolumeCPU& vol;
    Mat_<ptype>& points;
    Mutex& mtx;
};

void TSDFVolumeCPU::fetchPoints(OutputArray _points) const
{
    ScopeTime st("tsdf: fetch points");

    if(_points.needed())
    {
        Mat_<ptype> points;

        Mutex mutex;
        volume.forEach(PushPoints(*this, points, mutex));

        //TODO: try to use pre-allocated memory if possible
        points.copyTo(_points);
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
                  float _raycastStepFactor, float _gradientDeltaFactor);

    virtual void integrate(cv::Ptr<Frame> depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics);
    virtual cv::Ptr<Frame> raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize, int pyramidLevels,
                                   cv::Ptr<FrameGenerator> frameGenerator) const;

    virtual void fetchPoints(cv::OutputArray points) const;
    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const;

    virtual void reset();
};


TSDFVolumeGPU::TSDFVolumeGPU(int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
              float _raycastStepFactor, float _gradientDeltaFactor) :
    TSDFVolume(_res, _size, _pose, _truncDist, _maxWeight, _raycastStepFactor, _gradientDeltaFactor)
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


cv::Ptr<Frame> TSDFVolumeGPU::raycast(cv::Affine3f /*cameraPose*/, Intr /*intrinsics*/, Size /*frameSize*/, int /*pyramidLevels*/,
                                      Ptr<FrameGenerator> /* frameGenerator */) const
{
    throw std::runtime_error("Not implemented");
}


void TSDFVolumeGPU::fetchPoints(OutputArray /*_points*/) const
{
    throw std::runtime_error("Not implemented");
}


void TSDFVolumeGPU::fetchNormals(InputArray /*_points*/, OutputArray /*_normals*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<TSDFVolume> makeTSDFVolume(cv::kinfu::KinFu::KinFuParams::PlatformType t,
                                   int _res, float _size, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor, float _gradientDeltaFactor)
{
    switch (t)
    {
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_CPU:
        return cv::makePtr<TSDFVolumeCPU>(_res, _size, _pose, _truncDist, _maxWeight,
                                          _raycastStepFactor, _gradientDeltaFactor);
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_GPU:
        return cv::makePtr<TSDFVolumeGPU>(_res, _size, _pose, _truncDist, _maxWeight,
                                          _raycastStepFactor, _gradientDeltaFactor);
    default:
        return cv::Ptr<TSDFVolume>();
    }
}

