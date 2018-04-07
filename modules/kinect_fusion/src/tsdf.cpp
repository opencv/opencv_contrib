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

    volumeType fetchVoxel(cv::Point3f p) const;
    volumeType interpolateVoxel(cv::Point3f p) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

    // edgeResolution^3 array
    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    Volume volume;
    float edgeSize;
    int edgeResolution;
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

    bool b00 = v00 > 0;
    bool b01 = v01 > 0;
    bool b10 = v10 > 0;
    bool b11 = v11 > 0;

    //fix missing data, assume correct depth is positive
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
                for(int z = 0; z < volume.edgeResolution; z++)
                {
                    // optimization of the following:
                    //Point3f volPt = Point3f(x, y, z)*voxelSize;
                    //Point3f camSpacePt = vol2cam * volPt;
                    camSpacePt += zStep;

                    // can be optimized later
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

                        Voxel& voxel = volume.volume(x*volume.edgeResolution*volume.edgeResolution + y*volume.edgeResolution + z);
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


inline volumeType TSDFVolumeCPU::fetchVoxel(Point3f p) const
{
    // all coordinate checks should be done in inclosing cycle

    int xdim = edgeResolution*edgeResolution, ydim = edgeResolution;

    int xi, yi, zi;

    volumeType v = 0.f;
    xi = cvRound(p.x), yi = cvRound(p.y), zi = cvRound(p.z);
    v = volume(xi*xdim + yi*ydim + zi).v;

    return v;
}

inline volumeType TSDFVolumeCPU::interpolateVoxel(Point3f p) const
{
    // all coordinate checks should be done in inclosing cycle

    int xdim = edgeResolution*edgeResolution, ydim = edgeResolution;

    int xi, yi, zi;

    volumeType v = 0.f;

    xi = cvFloor(p.x), yi = cvFloor(p.y), zi = cvFloor(p.z);
    float tx = p.x - xi, ty = p.y - yi, tz = p.z - zi;
    float tx1 = 1.f - tx, ty1 = 1.f - ty, tz1 = 1.f - tz;

    const Voxel* vol00 = &volume.at<Voxel>((xi+0)*xdim + (yi+0)*ydim + zi);
    const Voxel* vol01 = &volume.at<Voxel>((xi+0)*xdim + (yi+1)*ydim + zi);
    const Voxel* vol10 = &volume.at<Voxel>((xi+1)*xdim + (yi+0)*ydim + zi);
    const Voxel* vol11 = &volume.at<Voxel>((xi+1)*xdim + (yi+1)*ydim + zi);

    v += vol00[0].v*tx1*ty1*tz1;
    v += vol00[1].v*tx1*ty1*tz ;
    v += vol01[0].v*tx1*ty *tz1;
    v += vol01[1].v*tx1*ty *tz ;
    v += vol10[0].v*tx *ty1*tz1;
    v += vol10[1].v*tx *ty1*tz ;
    v += vol11[0].v*tx *ty *tz1;
    v += vol11[1].v*tx *ty *tz ;

    return v;
}

inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f p) const
{
    Point3f n;
    volumeType fx1 = interpolateVoxel(Point3f(p.x + gradientDeltaFactor, p.y, p.z));
    volumeType fx0 = interpolateVoxel(Point3f(p.x - gradientDeltaFactor, p.y, p.z));
    // no need to divide, will be normalized after
    // n.x = (fx1-fx0)/gradientDeltaFactor;
    n.x = fx1 - fx0;

    volumeType fy1 = interpolateVoxel(Point3f(p.x, p.y + gradientDeltaFactor, p.z));
    volumeType fy0 = interpolateVoxel(Point3f(p.x, p.y - gradientDeltaFactor, p.z));
    n.y = fy1 - fy0;

    volumeType fz1 = interpolateVoxel(Point3f(p.x, p.y, p.z + gradientDeltaFactor));
    volumeType fz0 = interpolateVoxel(Point3f(p.x, p.y, p.z - gradientDeltaFactor));
    n.z = fz1 - fz0;

    return normalize(Vec3f(n));
}


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

    virtual void operator() (const Range& range) const
    {
        for(int y = range.start; y < range.end; y++)
        {
            ptype* ptsRow = points[y];
            ptype* nrmRow = normals[y];

            for(int x = 0; x < points.cols; x++)
            {
                Point3f point = nan3, normal = nan3;

                Point3f orig = cam2vol.translation();
                // direction through pixel in volume space
                Point3f dir = normalize(Vec3f(cam2vol.rotation() * reproj(Point3f(x, y, 1.f))));

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
                    volumeType fnext = volume.interpolateVoxel(next);

                    //raymarch
                    for(float t = tmin; t < tmax; t += tstep)
                    {
                        volumeType f = fnext;
                        Point3f tp = next;
                        next += rayStep;
                        // trying to optimize
                        fnext = volume.fetchVoxel(next);
                        if(fnext != f)
                            fnext = volume.interpolateVoxel(next);

                        // when ray comes from inside of a surface
                        if(f < 0.f && fnext > 0.f)
                            break;

                        // if ray penetrates a surface from outside
                        // linear interpolate t between two f values
                        if(f > 0.f && fnext < 0.f)
                        {
                            volumeType ft   = volume.interpolateVoxel(tp);
                            volumeType ftdt = volume.interpolateVoxel(next);
                            float ts = t - tstep*ft/(ftdt - ft);

                            // avoid division by zero
                            if(!cvIsNaN(ts) && !cvIsInf(ts))
                            {
                                Point3f pv = (orig + dir*ts);
                                Point3f nv = volume.getNormalVoxel(pv);

                                if(!isNaN(nv))
                                {
                                    //convert pv and nv to camera space
                                    normal = vol2cam.rotation() * nv;
                                    // interpolation optimized a little
                                    point = vol2cam * (pv * volume.voxelSize);
                                }
                            }
                            break;
                        }
                    }
                }

                ptsRow[x] = toPtype(point);
                nrmRow[x] = toPtype(normal);
            }
        }
    }

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

    parallel_for_(Range(0, points.rows), RaycastInvoker(points, normals, cameraPose, intrinsics, *this));

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

