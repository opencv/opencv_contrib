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
    volumeType fetchi(cv::Point3i p) const;
    volumeType interpolate(cv::Point3f p) const;
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
    gradientDeltaFactor = _gradientDeltaFactor*voxelSize;
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

                    depthType v = bilinear<depthType, Depth>(depth, projected);
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

    //TODO: find out why it's slower than w/o parallel code
    //parallel_for_(Range(0, edgeResolution), IntegrateInvoker(*this, depth, intrinsics, cameraPose, depthFactor));

    Intr::Projector proj = intrinsics.makeProjector();

    cv::Affine3f vol2cam = cameraPose.inv() * pose;
    float truncDistInv = 1./truncDist;
    float dfac = 1.f/depthFactor;

    // &elem(x, y, z) = data + x*edgeRes^2 + y*edgeRes + z;
    for(int x = 0; x < edgeResolution; x++)
    {
        for(int y = 0; y < edgeResolution; y++)
        {
            // optimization of camSpace transformation (vector addition instead of matmul at each z)
            Point3f basePt = vol2cam*Point3f(x*voxelSize, y*voxelSize, 0);
            Point3f camSpacePt = basePt;
            // zStep == vol2cam*(Point3f(x, y, 1)*voxelSize) - basePt;
            Point3f zStep = Point3f(vol2cam.matrix(0, 2), vol2cam.matrix(1, 2), vol2cam.matrix(2, 2))*voxelSize;
            for(int z = 0; z < edgeResolution; z++)
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

                depthType v = bilinear<depthType, Depth>(depth, projected);
                if(v == 0)
                    continue;

                // difference between distances of point and of surface to camera
                volumeType sdf = norm(camPixVec)*(v*dfac - camSpacePt.z);
                // possible alternative is:
                // kftype sdf = norm(camSpacePt)*(v*dfac/camSpacePt.z - 1);

                if(sdf >= -truncDist)
                {
                    volumeType tsdf = fmin(1.f, sdf * truncDistInv);

                    Voxel& voxel = volume(x*edgeResolution*edgeResolution + y*edgeResolution + z);
                    int& weight = voxel.weight;
                    volumeType& value = voxel.v;

                    // update TSDF
                    value = (value*weight+tsdf) / (weight + 1);
                    weight = min(weight + 1, maxWeight);
                }
            }
        }
    }
}

inline volumeType TSDFVolumeCPU::fetchVoxel(Point3f p) const
{
    p *= voxelSizeInv;
    return volume(cvRound(p.x)*edgeResolution*edgeResolution +
                  cvRound(p.y)*edgeResolution +
                  cvRound(p.z)).v;
}

inline volumeType TSDFVolumeCPU::fetchi(Point3i p) const
{
    return volume(p.x*edgeResolution*edgeResolution +
                  p.y*edgeResolution +
                  p.z).v;
}

inline volumeType TSDFVolumeCPU::interpolate(Point3f p) const
{
    p *= voxelSizeInv;

    if(isNaN(p)||
       p.x < 0 || p.x >= edgeResolution-1 ||
       p.y < 0 || p.y >= edgeResolution-1 ||
       p.z < 0 || p.z >= edgeResolution-1)
        return qnan;

    int xi = cvFloor(p.x), yi = cvFloor(p.y), zi = cvFloor(p.z);
    float tx = p.x - xi, ty = p.y - yi, tz = p.z - zi;

    volumeType v = 0.f;

    v += fetchi(Point3i(xi+0, yi+0, zi+0))*(1.f-tx)*(1.f-ty)*(1.f-tz);
    v += fetchi(Point3i(xi+0, yi+0, zi+1))*(1.f-tx)*(1.f-ty)*(    tz);
    v += fetchi(Point3i(xi+0, yi+1, zi+0))*(1.f-tx)*(    ty)*(1.f-tz);
    v += fetchi(Point3i(xi+0, yi+1, zi+1))*(1.f-tx)*(    ty)*(    tz);
    v += fetchi(Point3i(xi+1, yi+0, zi+0))*(    tx)*(1.f-ty)*(1.f-tz);
    v += fetchi(Point3i(xi+1, yi+0, zi+1))*(    tx)*(1.f-ty)*(    tz);
    v += fetchi(Point3i(xi+1, yi+1, zi+0))*(    tx)*(    ty)*(1.f-tz);
    v += fetchi(Point3i(xi+1, yi+1, zi+1))*(    tx)*(    ty)*(    tz);

    return v;
}

inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f p) const
{
    Point3f n;
    volumeType fx1 = interpolate(Point3f(p.x + gradientDeltaFactor, p.y, p.z));
    volumeType fx0 = interpolate(Point3f(p.x - gradientDeltaFactor, p.y, p.z));
    // no need to divide, will be normalized after
    // n.x = (fx1-fx0)/gradientDeltaFactor;
    n.x = fx1 - fx0;

    volumeType fy1 = interpolate(Point3f(p.x, p.y + gradientDeltaFactor, p.z));
    volumeType fy0 = interpolate(Point3f(p.x, p.y - gradientDeltaFactor, p.z));
    n.y = fy1 - fy0;

    volumeType fz1 = interpolate(Point3f(p.x, p.y, p.z + gradientDeltaFactor));
    volumeType fz0 = interpolate(Point3f(p.x, p.y, p.z - gradientDeltaFactor));
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
            Point3f* ptsRow = points[y];
            Point3f* nrmRow = normals[y];

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
                    tmax -= tstep;
                    Point3f rayStep = dir * tstep;
                    Point3f next = orig + dir * tmin;
                    volumeType fnext = volume.interpolate(next);

                    //raymarch
                    for(float t = tmin; t < tmax; t += tstep)
                    {
                        volumeType f = fnext;
                        Point3f tp = next;
                        next += rayStep;
                        //trying to optimize
                        fnext = volume.fetchVoxel(next);
                        if(fnext != f)
                            fnext = volume.interpolate(next);

                        // when ray comes from inside of a surface
                        if(f < 0.f && fnext > 0.f)
                            break;

                        // if ray penetrates a surface from outside
                        // linear interpolate t between two f values
                        if(f > 0.f && fnext < 0.f)
                        {
                            volumeType ft   = volume.interpolate(tp);
                            volumeType ftdt = volume.interpolate(next);
                            float ts = t - tstep*ft/(ftdt - ft);

                            Point3f pv = orig + dir*ts;
                            Point3f nv = volume.getNormalVoxel(pv);

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = vol2cam.rotation() * nv;
                                point = vol2cam * pv;
                            }
                            break;
                        }
                    }
                }

                ptsRow[x] = point;
                nrmRow[x] = normal;
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
    PushPoints(const TSDFVolumeCPU& _vol, Mat_<Point3f>& _pts, Mutex& _mtx) :
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
                        points.push_back(vol.pose * p);
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
    Mat_<Point3f>& points;
    Mutex& mtx;
};

void TSDFVolumeCPU::fetchPoints(OutputArray _points) const
{
    ScopeTime st("tsdf: fetch points");

    if(_points.needed())
    {
        Mat_<Point3f> points;

        //TODO: use parallel_for instead
        Mutex mutex;
        volume.forEach(PushPoints(*this, points, mutex));

        //TODO: try to use pre-allocated memory if possible
        points.copyTo(_points);
    }
}


struct PushNormals
{
    PushNormals(const TSDFVolumeCPU& _vol, Mat_<Point3f>& _nrm, Mutex& _mtx) :
        vol(_vol), normals(_nrm), mtx(_mtx) { }
    void operator ()(const Point3f &p, const int * /*position*/) const
    {
        AutoLock al(mtx);
        normals.push_back(vol.pose.rotation() * vol.getNormalVoxel(p));
    }
    const TSDFVolumeCPU& vol;
    Mat_<Point3f>& normals;
    Mutex& mtx;
};

void TSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    ScopeTime st("tsdf: fetch normals");

    if(_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == CV_32FC3);

        //TODO: try to use pre-allocated memory if possible
        Mat_<Point3f> normals;
        normals.reserve(points.total());

        Mutex mutex;
        points.forEach(PushNormals(*this, normals, mutex));

        _normals.assign(normals);
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

