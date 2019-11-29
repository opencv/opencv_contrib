// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "dynafu_tsdf.hpp"
#include "marchingcubes.hpp"

namespace cv {

namespace dynafu {

using namespace kinfu;

// TODO: Optimization possible:
// * volumeType can be FP16
// * weight can be int16
typedef float volumeType;
struct Voxel
{
    volumeType v;
    float weight;
    nodeNeighboursType neighbours;
    float neighbourDists[DYNAFU_MAX_NEIGHBOURS];
    int n;
};
typedef Vec<uchar, sizeof(Voxel)> VecT;


class TSDFVolumeCPU : public TSDFVolume
{
public:
    // dimension in voxels, size in meters
    TSDFVolumeCPU(Point3i _res, float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                  float _raycastStepFactor, bool zFirstMemOrder = true);

    virtual void integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, Ptr<WarpField> wf) override;
    virtual void raycast(cv::Affine3f cameraPose, cv::kinfu::Intr intrinsics, cv::Size frameSize,
                         cv::OutputArray points, cv::OutputArray normals) const override;

    virtual void fetchNormals(cv::InputArray points, cv::OutputArray _normals) const override;
    virtual void fetchPointsNormals(cv::OutputArray points, cv::OutputArray normals, bool fetchVoxels) const override;

    virtual void marchCubes(OutputArray _vertices, OutputArray _edges) const override;

    virtual void reset() override;

    volumeType interpolateVoxel(cv::Point3f p) const;
    Point3f getNormalVoxel(cv::Point3f p) const;

    nodeNeighboursType const& getVoxelNeighbours(Point3i coords, int& n) const override;

    // See zFirstMemOrder arg of parent class constructor
    // for the array layout info
    // Consist of Voxel elements
    Mat volume;

private:
    Point3f interpolate(Point3f p1, Point3f p2, float v1, float v2) const;
};


TSDFVolume::TSDFVolume(Point3i _res, float _voxelSize, Affine3f _pose, float _truncDist, int _maxWeight,
                       float _raycastStepFactor, bool zFirstMemOrder) :
    voxelSize(_voxelSize),
    voxelSizeInv(1.f/_voxelSize),
    volResolution(_res),
    maxWeight((float)_maxWeight),
    pose(_pose),
    raycastStepFactor(_raycastStepFactor)
{
    // Unlike original code, this should work with any volume size
    // Not only when (x,y,z % 32) == 0

    volSize = Point3f(volResolution) * voxelSize;

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

// dimension in voxels, size in meters
TSDFVolumeCPU::TSDFVolumeCPU(Point3i _res, float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                             float _raycastStepFactor, bool zFirstMemOrder) :
    TSDFVolume(_res, _voxelSize, _pose, _truncDist, _maxWeight, _raycastStepFactor, zFirstMemOrder)
{
    volume = Mat(1, volResolution.x * volResolution.y * volResolution.z, rawType<Voxel>());

    reset();
}

// zero volume, leave rest params the same
void TSDFVolumeCPU::reset()
{
    CV_TRACE_FUNCTION();

    volume.forEach<VecT>([](VecT& vv, const int* /* position */)
    {
        Voxel& v = reinterpret_cast<Voxel&>(vv);
        v.v = 0; v.weight = 0;
    });
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

struct IntegrateInvoker : ParallelLoopBody
{
    IntegrateInvoker(TSDFVolumeCPU& _volume, const Depth& _depth, Intr intrinsics, cv::Affine3f cameraPose,
                     float depthFactor, Ptr<WarpField> wf) :
        ParallelLoopBody(),
        volume(_volume),
        depth(_depth),
        proj(intrinsics.makeProjector()),
        vol2cam(cameraPose.inv() * _volume.pose),
        truncDistInv(1.f/_volume.truncDist),
        dfac(1.f/depthFactor),
        warpfield(wf)
    {
        volDataStart = volume.volume.ptr<Voxel>();
    }

    virtual void operator() (const Range& range) const override
    {
        CV_TRACE_FUNCTION();

        for(int x = range.start; x < range.end; x++)
        {
            Voxel* volDataX = volDataStart + x*volume.volDims[0];
            for(int y = 0; y < volume.volResolution.y; y++)
            {
                Voxel* volDataY = volDataX+y*volume.volDims[1];

                for(int z = 0; z < volume.volResolution.z; z++)
                {
                    Voxel& voxel = volDataY[z*volume.volDims[2]];

                    Point3f volPt = Point3f((float)x, (float)y, (float)z)*volume.voxelSize;

                    if(warpfield->getNodeIndex())
                    {
                        std::vector<int> indices(warpfield->k);
                        std::vector<float> dists(warpfield->k);
                        warpfield->findNeighbours(volPt, indices, dists);

                        voxel.n = 0;
                        for(size_t i = 0; i < indices.size(); i++)
                        {
                            if(std::isnan(dists[i])) continue;

                            voxel.neighbourDists[voxel.n] = dists[i];
                            voxel.neighbours[voxel.n++] = indices[i];
                        }
                    }

                    Point3f camSpacePt =
                    vol2cam * warpfield->applyWarp(volPt, voxel.neighbours, voxel.n);

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

                        float& weight = voxel.weight;
                        volumeType& value = voxel.v;

                        // update TSDF
                        float newWeight = 0;

                        if(warpfield->getNodesLen() >= (size_t)warpfield->k)
                        {
                            for(int i = 0; i < voxel.n; i++)
                                newWeight += sqrt(voxel.neighbourDists[i]);

                            if(voxel.n > 0) newWeight /= voxel.n;

                        } else newWeight = 1.f;

                        if((weight + newWeight) != 0)
                        {
                            value = (value*weight+tsdf*newWeight) / (weight+newWeight);
                            weight = min(weight+newWeight, volume.maxWeight);
                        }
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
    Voxel* volDataStart;
    Ptr<WarpField> warpfield;
};

// use depth instead of distance (optimization)
void TSDFVolumeCPU::integrate(InputArray _depth, float depthFactor, cv::Affine3f cameraPose, Intr intrinsics, Ptr<WarpField> wf)
{
    CV_TRACE_FUNCTION();

    CV_Assert(_depth.type() == DEPTH_TYPE);
    Depth depth = _depth.getMat();

    IntegrateInvoker ii(*this, depth, intrinsics, cameraPose, depthFactor, wf);
    Range range(0, volResolution.x);
    parallel_for_(range, ii);
}

inline volumeType TSDFVolumeCPU::interpolateVoxel(Point3f p) const
{
    int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];

    int ix = cvFloor(p.x);
    int iy = cvFloor(p.y);
    int iz = cvFloor(p.z);

    float tx = p.x - ix;
    float ty = p.y - iy;
    float tz = p.z - iz;

    int coordBase = ix*xdim + iy*ydim + iz*zdim;
    const Voxel* volData = volume.ptr<Voxel>();

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

inline Point3f TSDFVolumeCPU::getNormalVoxel(Point3f p) const
{
    const int xdim = volDims[0], ydim = volDims[1], zdim = volDims[2];
    const Voxel* volData = volume.ptr<Voxel>();

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
        boxMax(volume.volSize - Point3f(volume.voxelSize,
                                        volume.voxelSize,
                                        volume.voxelSize)),
        boxMin(),
        cam2vol(volume.pose.inv() * cameraPose),
        vol2cam(cameraPose.inv() * volume.pose),
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
                Point3f dir = normalize(Vec3f(camRot * reproj(Point3f((float)x, (float)y, 1.f))));

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
                    volumeType f = volume.interpolateVoxel(next), fnext = f;

                    //raymarch
                    int steps = 0;
                    int nSteps = (int)floor((tmax - tmin)/tstep);
                    for(; steps < nSteps; steps++)
                    {
                        next += rayStep;
                        int xdim = volume.volDims[0];
                        int ydim = volume.volDims[1];
                        int zdim = volume.volDims[2];
                        int ix = cvRound(next.x);
                        int iy = cvRound(next.y);
                        int iz = cvRound(next.z);
                        fnext = volume.volume.at<Voxel>(ix*xdim + iy*ydim + iz*zdim).v;
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
    const TSDFVolumeCPU& volume;

    const float tstep;

    const Point3f boxMax;
    const Point3f boxMin;

    const Affine3f cam2vol;
    const Affine3f vol2cam;
    const Intr::Reprojector reproj;
};


void TSDFVolumeCPU::raycast(cv::Affine3f cameraPose, Intr intrinsics, Size frameSize,
                            cv::OutputArray _points, cv::OutputArray _normals) const
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
    FetchPointsNormalsInvoker(const TSDFVolumeCPU& _volume,
                              std::vector< std::vector<ptype> >& _pVecs,
                              std::vector< std::vector<ptype> >& _nVecs,
                              bool _needNormals, bool _fetchVoxels) :
        ParallelLoopBody(),
        vol(_volume),
        pVecs(_pVecs),
        nVecs(_nVecs),
        needNormals(_needNormals),
        fetchVoxels(_fetchVoxels)
    {
        volDataStart = vol.volume.ptr<Voxel>();
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
            shift = Point3i(1, 0, 0);
            limits = (x + 1 < vol.volResolution.x);
            Vc = V.x;
        }
        if(axis == 1)
        {
            shift = Point3i(0, 1, 0);
            limits = (y + 1 < vol.volResolution.y);
            Vc = V.y;
        }
        if(axis == 2)
        {
            shift = Point3i(0, 0, 1);
            limits = (z + 1 < vol.volResolution.z);
            Vc = V.z;
        }

        if(limits)
        {
            const Voxel& voxeld = volDataStart[(x+shift.x)*vol.volDims[0] +
                                               (y+shift.y)*vol.volDims[1] +
                                               (z+shift.z)*vol.volDims[2]];
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
                        if(fetchVoxels)
                        {
                            points.push_back(toPtype(p));
                            if(needNormals)
                                normals.push_back(toPtype(vol.getNormalVoxel(p*vol.voxelSizeInv)));
                        } else {

                            points.push_back(toPtype(vol.pose * p));
                            if(needNormals)
                                normals.push_back(toPtype(vol.pose.rotation() *
                                                          vol.getNormalVoxel(p*vol.voxelSizeInv)));
                        }
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
            const Voxel* volDataX = volDataStart + x*vol.volDims[0];
            for(int y = 0; y < vol.volResolution.y; y++)
            {
                const Voxel* volDataY = volDataX + y*vol.volDims[1];
                for(int z = 0; z < vol.volResolution.z; z++)
                {
                    const Voxel& voxel0 = volDataY[z*vol.volDims[2]];
                    volumeType v0 = voxel0.v;
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

    const TSDFVolumeCPU& vol;
    std::vector< std::vector<ptype> >& pVecs;
    std::vector< std::vector<ptype> >& nVecs;
    const Voxel* volDataStart;
    bool needNormals;
    bool fetchVoxels;
    mutable Mutex mutex;
};

void TSDFVolumeCPU::fetchPointsNormals(OutputArray _points, OutputArray _normals, bool fetchVoxels) const
{
    CV_TRACE_FUNCTION();

    if(_points.needed())
    {
        std::vector< std::vector<ptype> > pVecs, nVecs;
        FetchPointsNormalsInvoker fi(*this, pVecs, nVecs, _normals.needed(), fetchVoxels);
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
            Point3f voxPt = (invPose * p);
            voxPt = voxPt * vol.voxelSizeInv;
            n = vol.pose.rotation() * vol.getNormalVoxel(voxPt);
        }
        normals(position[0], position[1]) = toPtype(n);
    }
    const TSDFVolumeCPU& vol;
    Mat_<ptype>& normals;

    Affine3f invPose;
};


void TSDFVolumeCPU::fetchNormals(InputArray _points, OutputArray _normals) const
{
    CV_TRACE_FUNCTION();

    if(_normals.needed())
    {
        Points points = _points.getMat();
        CV_Assert(points.type() == POINT_TYPE);

        _normals.createSameSize(_points, _points.type());
        Mat_<ptype> normals = _normals.getMat();

        points.forEach(PushNormals(*this, normals));
    }
}

struct MarchCubesInvoker : ParallelLoopBody
{
    MarchCubesInvoker(const TSDFVolumeCPU& _volume,
                      std::vector<Vec4f>& _meshPoints) :
        volume(_volume),
        meshPoints(_meshPoints),
        mcNeighbourPts{
            Point3f(0.f, 0.f, 0.f),
            Point3f(0.f, 0.f, 1.f),
            Point3f(0.f, 1.f, 1.f),
            Point3f(0.f, 1.f, 0.f),
            Point3f(1.f, 0.f, 0.f),
            Point3f(1.f, 0.f, 1.f),
            Point3f(1.f, 1.f, 1.f),
            Point3f(1.f, 1.f, 0.f)},
        mcNeighbourCoords(
            Vec8i(
            volume.volDims.dot(Vec4i(0, 0, 0)),
            volume.volDims.dot(Vec4i(0, 0, 1)),
            volume.volDims.dot(Vec4i(0, 1, 1)),
            volume.volDims.dot(Vec4i(0, 1, 0)),
            volume.volDims.dot(Vec4i(1, 0, 0)),
            volume.volDims.dot(Vec4i(1, 0, 1)),
            volume.volDims.dot(Vec4i(1, 1, 1)),
            volume.volDims.dot(Vec4i(1, 1, 0))
            ))
    {
        volData = volume.volume.ptr<Voxel>();
    }

    Point3f interpolate(Point3f p1, Point3f p2, float v1, float v2) const
    {
        float dV = 0.5f;
        if (abs(v1 - v2) > 0.0001f)
            dV = v1 / (v1 - v2);

        Point3f p = p1 + dV * (p2 - p1);
        return p;
    }

    virtual void operator()(const Range &range) const override
    {
        std::vector<Vec4f> points;
        for (int x = range.start; x < range.end; x++)
        {
            int coordBaseX = x * volume.volDims[0];
            for (int y = 0; y < volume.volResolution.y - 1; y++)
            {
                int coordBaseY = coordBaseX + y * volume.volDims[1];
                for (int z = 0; z < volume.volResolution.z - 1; z++)
                {
                    int coordBase = coordBaseY + z * volume.volDims[2];

                    if (volData[coordBase].weight == 0)
                        continue;

                    uint8_t cubeIndex = 0;
                    float tsdfValues[8] = {0};
                    for (int i = 0; i < 8; i++)
                    {
                        if (volData[mcNeighbourCoords[i] + coordBase].weight == 0)
                            continue;

                        tsdfValues[i] = volData[mcNeighbourCoords[i] + coordBase].v;
                        if (tsdfValues[i] <= 0)
                            cubeIndex |= (1 << i);
                    }

                    if (edgeTable[cubeIndex] == 0)
                        continue;

                    Point3f vertices[12];
                    Point3f basePt((float)x, (float)y, (float)z);

                    if (edgeTable[cubeIndex] & 1)
                        vertices[0] = basePt + interpolate(mcNeighbourPts[0], mcNeighbourPts[1],
                                                           tsdfValues[0], tsdfValues[1]);
                    if (edgeTable[cubeIndex] & 2)
                        vertices[1] = basePt + interpolate(mcNeighbourPts[1], mcNeighbourPts[2],
                                                           tsdfValues[1], tsdfValues[2]);
                    if (edgeTable[cubeIndex] & 4)
                        vertices[2] = basePt + interpolate(mcNeighbourPts[2], mcNeighbourPts[3],
                                                           tsdfValues[2], tsdfValues[3]);
                    if (edgeTable[cubeIndex] & 8)
                        vertices[3] = basePt + interpolate(mcNeighbourPts[3], mcNeighbourPts[0],
                                                           tsdfValues[3], tsdfValues[0]);
                    if (edgeTable[cubeIndex] & 16)
                        vertices[4] = basePt + interpolate(mcNeighbourPts[4], mcNeighbourPts[5],
                                                           tsdfValues[4], tsdfValues[5]);
                    if (edgeTable[cubeIndex] & 32)
                        vertices[5] = basePt + interpolate(mcNeighbourPts[5], mcNeighbourPts[6],
                                                           tsdfValues[5], tsdfValues[6]);
                    if (edgeTable[cubeIndex] & 64)
                        vertices[6] = basePt + interpolate(mcNeighbourPts[6], mcNeighbourPts[7],
                                                           tsdfValues[6], tsdfValues[7]);
                    if (edgeTable[cubeIndex] & 128)
                        vertices[7] = basePt + interpolate(mcNeighbourPts[7], mcNeighbourPts[4],
                                                           tsdfValues[7], tsdfValues[4]);
                    if (edgeTable[cubeIndex] & 256)
                        vertices[8] = basePt + interpolate(mcNeighbourPts[0], mcNeighbourPts[4],
                                                           tsdfValues[0], tsdfValues[4]);
                    if (edgeTable[cubeIndex] & 512)
                        vertices[9] = basePt + interpolate(mcNeighbourPts[1], mcNeighbourPts[5],
                                                           tsdfValues[1], tsdfValues[5]);
                    if (edgeTable[cubeIndex] & 1024)
                        vertices[10] = basePt + interpolate(mcNeighbourPts[2], mcNeighbourPts[6],
                                                            tsdfValues[2], tsdfValues[6]);
                    if (edgeTable[cubeIndex] & 2048)
                        vertices[11] = basePt + interpolate(mcNeighbourPts[3], mcNeighbourPts[7],
                                                            tsdfValues[3], tsdfValues[7]);

                    for (int i = 0; triTable[cubeIndex][i] != -1; i += 3)
                    {
                        Point3f p = volume.pose * (vertices[triTable[cubeIndex][i]] * volume.voxelSize);
                        points.push_back(Vec4f(p.x, p.y, p.z, 1.f));

                        p = volume.pose * (vertices[triTable[cubeIndex][i + 1]] * volume.voxelSize);
                        points.push_back(Vec4f(p.x, p.y, p.z, 1.f));

                        p = volume.pose * (vertices[triTable[cubeIndex][i + 2]] * volume.voxelSize);
                        points.push_back(Vec4f(p.x, p.y, p.z, 1.f));
                    }
                }
            }
        }

        if(points.size() > 0)
        {
            AutoLock al(m);
            meshPoints.insert(meshPoints.end(), points.begin(), points.end());
        }
    }

    const TSDFVolumeCPU& volume;
    std::vector<Vec4f>& meshPoints;
    const Point3f mcNeighbourPts[8];
    const Vec8i mcNeighbourCoords;
    const Voxel* volData;
    mutable Mutex m;
};

void TSDFVolumeCPU::marchCubes(OutputArray _vertices, OutputArray _edges) const
{
    std::vector<Vec4f> meshPoints;
    std::vector<int> meshEdges;
    MarchCubesInvoker mci(*this, meshPoints);
    Range range(0, volResolution.x - 1);
    parallel_for_(range, mci);

    for(int i = 0; i < (int)meshPoints.size(); i+= 3)
    {
        meshEdges.push_back(i);
        meshEdges.push_back(i+1);

        meshEdges.push_back(i+1);
        meshEdges.push_back(i+2);

        meshEdges.push_back(i+2);
        meshEdges.push_back(i);
    }

    if (_vertices.needed())
        Mat((int)meshPoints.size(), 1, CV_32FC4, &meshPoints[0]).copyTo(_vertices);

    if (_edges.needed())
        Mat((int)meshPoints.size(), 2, CV_32S, &meshEdges[0]).copyTo(_edges);
}

nodeNeighboursType const& TSDFVolumeCPU::getVoxelNeighbours(Point3i v, int& n) const
{
    int baseX = v.x * volDims[0];
    int baseY = baseX + v.y * volDims[1];
    int base = baseY + v.z * volDims[2];
    const Voxel *vox = volume.ptr<Voxel>()+base;

    n = vox->n;
    return vox->neighbours;
}

cv::Ptr<TSDFVolume> makeTSDFVolume(Point3i _res,  float _voxelSize, cv::Affine3f _pose, float _truncDist, int _maxWeight,
                                   float _raycastStepFactor)
{
    return cv::makePtr<TSDFVolumeCPU>(_res, _voxelSize, _pose, _truncDist, _maxWeight, _raycastStepFactor);
}

} // namespace dynafu
} // namespace cv
