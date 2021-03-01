// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "colored_tsdf.hpp"
#include "tsdf_functions.hpp"
#include "opencl_kernels_rgbd.hpp"

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
    virtual void integrate(InputArray _depth, float depthFactor, const Matx44f& cameraPose,
        const kinfu::Intr& intrinsics, const int frameId = 0) override {};
    virtual void integrate(InputArray _depth, InputArray _rgb, float depthFactor, const Matx44f& cameraPose,
        const kinfu::Intr& intrinsics, const int frameId = 0) override;
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
        OutputArray points, OutputArray normals, OutputArray colors) const override {};
    virtual void raycast(const Matx44f& cameraPose, const kinfu::Intr& intrinsics, const Size& frameSize,
        OutputArray points, OutputArray normals) const override;

    virtual void fetchNormals(InputArray points, OutputArray _normals) const override;
    virtual void fetchPointsNormals(OutputArray points, OutputArray normals) const override;

    virtual void reset() override;
    virtual RGBTsdfVoxel at(const Vec3i& volumeIdx) const;

    float interpolateVoxel(const cv::Point3f& p) const;
    Point3f getNormalVoxel(const cv::Point3f& p) const;

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

    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<TsdfVoxel>());

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
        return RGBTsdfVoxel(floatToTsdf(1.f), 0);
    }

    const RGBTsdfVoxel* volData = volume.ptr<RGBTsdfVoxel>();
    int coordBase =
        volumeIdx[0] * volDims[0] + volumeIdx[1] * volDims[1] + volumeIdx[2] * volDims[2];
    return volData[coordBase];
}

// use depth instead of distance (optimization)
void ColoredTSDFVolumeCPU::integrate(InputArray _depth, InputArray _rgb, float depthFactor, const Matx44f& cameraPose,
                              const Intr& intrinsics, const int frameId)
{
    CV_TRACE_FUNCTION();
    CV_UNUSED(frameId);
    CV_Assert(_depth.type() == DEPTH_TYPE);
    CV_Assert(!_depth.empty());
    Depth depth = _depth.getMat();

    Vec6f newParams((float)depth.rows, (float)depth.cols,
        intrinsics.fx, intrinsics.fy,
        intrinsics.cx, intrinsics.cy);
    if (!(frameParams == newParams))
    {
        frameParams = newParams;
        pixNorms = preCalculationPixNorm(depth, intrinsics);
    }

    integrateVolumeUnit(truncDist, voxelSize, maxWeight, (this->pose).matrix, volResolution, volStrides, depth,
        depthFactor, cameraPose, intrinsics, pixNorms, volume);
}


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
            vx[i] = tsdfToFloat(volData[neighbourCoords[i] + coordBase + 1*dim].tsdf -
                                volData[neighbourCoords[i] + coordBase - 1*dim].tsdf);

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


struct RaycastInvoker : ParallelLoopBody
{
    RaycastInvoker(Points& _points, Normals& _normals, const Matx44f& cameraPose,
                  const Intr& intrinsics, const ColoredTSDFVolumeCPU& _volume) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
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
        reproj(intrinsics.makeReprojector())
    {  }

    virtual void operator() (const Range& range) const override
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
                Point3f dir = normalize(Vec3f(camRot * reproj(Point3f(float(x), float(y), 1.f))));

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

                            if(!isNaN(nv))
                            {
                                //convert pv and nv to camera space
                                normal = volRot * nv;
                                // interpolation optimized a little
                                point = vol2cam * (pv*volume.voxelSize);
                            }
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
    const ColoredTSDFVolumeCPU& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};


void ColoredTSDFVolumeCPU::raycast(const Matx44f& cameraPose, const Intr& intrinsics, const Size& frameSize,
                            OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    CV_Assert(frameSize.area() > 0);

    _points.create (frameSize, POINT_TYPE);
    _normals.create(frameSize, POINT_TYPE);

    Points points   =  _points.getMat();
    Normals normals = _normals.getMat();

    RaycastInvoker ri(points, normals, cameraPose, intrinsics, *this);

    const int nstripes = -1;
    parallel_for_(Range(0, points.rows), ri, nstripes);
}


struct FetchPointsNormalsInvoker : ParallelLoopBody
{
    FetchPointsNormalsInvoker(const ColoredTSDFVolumeCPU& _volume,
                              std::vector<std::vector<ptype>>& _pVecs,
                              std::vector<std::vector<ptype>>& _nVecs,
                              bool _needNormals) :
        ParallelLoopBody(),
        vol(_volume),
        pVecs(_pVecs),
        nVecs(_nVecs),
        needNormals(_needNormals)
    {
        volDataStart = vol.volume.ptr<RGBTsdfVoxel>();
    }

    inline void coord(std::vector<ptype>& points, std::vector<ptype>& normals,
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
                    }
                }
            }
        }
    }

    virtual void operator() (const Range& range) const override
    {
        std::vector<ptype> points, normals;
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

    const ColoredTSDFVolumeCPU& vol;
    std::vector<std::vector<ptype>>& pVecs;
    std::vector<std::vector<ptype>>& nVecs;
    const RGBTsdfVoxel* volDataStart;
    bool needNormals;
    mutable Mutex mutex;
};

void ColoredTSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if(_points.needed())
    {
        std::vector<std::vector<ptype>> pVecs, nVecs;
        FetchPointsNormalsInvoker fi(*this, pVecs, nVecs, _normals.needed());
        Range range(0, volResolution.x);
        const int nstripes = -1;
        parallel_for_(range, fi, nstripes);
        std::vector<ptype> points, normals;
        for(size_t i = 0; i < pVecs.size(); i++)
        {
            points.insert(points.end(), pVecs[i].begin(), pVecs[i].end());
            normals.insert(normals.end(), nVecs[i].begin(), nVecs[i].end());
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
