// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "fast_icp.hpp"

using namespace std;

namespace cv {
namespace kinfu {


ICP::ICP(const Intr _intrinsics, const std::vector<int>& _iterations, float _angleThreshold, float _distanceThreshold) :
    iterations(_iterations), angleThreshold(_angleThreshold), distanceThreshold(_distanceThreshold),
    intrinsics(_intrinsics)
{ }

///////// CPU implementation /////////

class ICPCPU : public ICP
{
public:
    ICPCPU(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const override;

    virtual ~ICPCPU() { }

private:
    void getAb(const Points &oldPts, const Normals &oldNrm, const Points &newPts, const Normals &newNrm,
               cv::Affine3f pose, int level, cv::Matx66f& A, cv::Vec6f& b) const;
};


ICPCPU::ICPCPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }

bool ICPCPU::estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> _oldFrame, cv::Ptr<Frame> _newFrame) const
{
    ScopeTime st("icp");

    cv::Ptr<FrameCPU> oldFrame = _oldFrame.dynamicCast<FrameCPU>();
    cv::Ptr<FrameCPU> newFrame = _newFrame.dynamicCast<FrameCPU>();

    const std::vector<Points>& oldPoints   = oldFrame->points;
    const std::vector<Normals>& oldNormals = oldFrame->normals;
    const std::vector<Points>& newPoints  = newFrame->points;
    const std::vector<Normals>& newNormals = newFrame->normals;

    transform = Affine3f::Identity();
    for(size_t l = 0; l < iterations.size(); l++)
    {
        size_t level = iterations.size() - 1 - l;

        Points  oldPts = oldPoints [level], newPts = newPoints [level];
        Normals oldNrm = oldNormals[level], newNrm = newNormals[level];

        for(int iter = 0; iter < iterations[level]; iter++)
        {
            Matx66f A;
            Vec6f b;

            getAb(oldPts, oldNrm, newPts, newNrm, transform, (int)level, A, b);

            double det = cv::determinant(A);

            if (abs (det) < 1e-15 || cvIsNaN(det))
                return false;

            Vec6f x;
            // theoretically, any method of solving is applicable
            // since there are usual least square matrices
            solve(A, b, x, DECOMP_SVD);
            Affine3f tinc(Vec3f(x.val), Vec3f(x.val+3));
            transform = tinc * transform;
        }
    }

    return true;
}

// 1 any coord to check is enough since we know the generation


#if CV_SIMD128
static inline bool fastCheck(const v_float32x4& p0, const v_float32x4& p1)
{
    float check = (p0.get0() + p1.get0());
    return !cvIsNaN(check);
}

static inline void getCrossPerm(const v_float32x4& a, v_float32x4& yzx, v_float32x4& zxy)
{
    v_uint32x4 aa = v_reinterpret_as_u32(a);
    v_uint32x4 yz00 = v_extract<1>(aa, v_setzero_u32());
    v_uint32x4 x0y0, tmp;
    v_zip(aa, v_setzero_u32(), x0y0, tmp);
    v_uint32x4 yzx0 = v_combine_low(yz00, x0y0);
    v_uint32x4 y000 = v_extract<2>(x0y0, v_setzero_u32());
    v_uint32x4 zx00 = v_extract<1>(yzx0, v_setzero_u32());
    zxy = v_reinterpret_as_f32(v_combine_low(zx00, y000));
    yzx = v_reinterpret_as_f32(yzx0);
}

static inline v_float32x4 crossProduct(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 ayzx, azxy, byzx, bzxy;
    getCrossPerm(a, ayzx, azxy);
    getCrossPerm(b, byzx, bzxy);
    return ayzx*bzxy - azxy*byzx;
}
#else
static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}

#endif

typedef Matx<float, 6, 7> ABtype;

struct GetAbInvoker : ParallelLoopBody
{
    enum
    {
        UTSIZE = 27
    };

    GetAbInvoker(ABtype& _globalAb, Mutex& _mtx,
                 const Points& _oldPts, const Normals& _oldNrm, const Points& _newPts, const Normals& _newNrm,
                 Affine3f _pose, Intr::Projector _proj, float _sqDistanceThresh, float _minCos) :
        ParallelLoopBody(),
        globalSumAb(_globalAb), mtx(_mtx),
        oldPts(_oldPts), oldNrm(_oldNrm), newPts(_newPts), newNrm(_newNrm), pose(_pose),
        proj(_proj), sqDistanceThresh(_sqDistanceThresh), minCos(_minCos)
    { }

    virtual void operator ()(const Range& range) const override
    {
#if CV_SIMD128
        CV_Assert(ptype::channels == 4);

        const size_t utBufferSize = 9;
        float CV_DECL_ALIGNED(16) upperTriangle[utBufferSize*4];
        for(size_t i = 0; i < utBufferSize*4; i++)
            upperTriangle[i] = 0;
        // how values are kept in upperTriangle
        const int NA = 0;
        const size_t utPos[] =
        {
           0,  1,  2,  4,  5,  6,  3,
          NA,  9, 10, 12, 13, 14, 11,
          NA, NA, 18, 20, 21, 22, 19,
          NA, NA, NA, 24, 28, 30, 32,
          NA, NA, NA, NA, 25, 29, 33,
          NA, NA, NA, NA, NA, 26, 34
        };

        const float (&pm)[16] = pose.matrix.val;
        v_float32x4 poseRot0(pm[0], pm[4], pm[ 8], 0);
        v_float32x4 poseRot1(pm[1], pm[5], pm[ 9], 0);
        v_float32x4 poseRot2(pm[2], pm[6], pm[10], 0);
        v_float32x4 poseTrans(pm[3], pm[7], pm[11], 0);

        v_float32x4 vfxy(proj.fx, proj.fy, 0, 0), vcxy(proj.cx, proj.cy, 0, 0);
        v_float32x4 vframe((float)(oldPts.cols - 1), (float)(oldPts.rows - 1), 1.f, 1.f);

        float sqThresh = sqDistanceThresh;
        float cosThresh = minCos;

        for(int y = range.start; y < range.end; y++)
        {
            const CV_DECL_ALIGNED(16) float* newPtsRow = (const float*)newPts[y];
            const CV_DECL_ALIGNED(16) float* newNrmRow = (const float*)newNrm[y];

            for(int x = 0; x < newPts.cols; x++)
            {
                v_float32x4 newP = v_load_aligned(newPtsRow + x*4);
                v_float32x4 newN = v_load_aligned(newNrmRow + x*4);

                if(!fastCheck(newP, newN))
                    continue;

                //transform to old coord system
                newP = v_matmuladd(newP, poseRot0, poseRot1, poseRot2, poseTrans);
                newN = v_matmuladd(newN, poseRot0, poseRot1, poseRot2, v_setzero_f32());

                //find correspondence by projecting the point
                v_float32x4 oldCoords;
                float pz = (v_reinterpret_as_f32(v_rotate_right<2>(v_reinterpret_as_u32(newP))).get0());
                // x, y, 0, 0
                oldCoords = v_muladd(newP/v_setall_f32(pz), vfxy, vcxy);

                if(!v_check_all((oldCoords >= v_setzero_f32()) & (oldCoords < vframe)))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                v_float32x4 oldP;
                v_float32x4 oldN;
                {
                    v_int32x4 ixy = v_floor(oldCoords);
                    v_float32x4 txy = oldCoords - v_cvt_f32(ixy);
                    int xi = ixy.get0();
                    int yi = v_rotate_right<1>(ixy).get0();
                    v_float32x4 tx = v_setall_f32(txy.get0());
                    txy = v_reinterpret_as_f32(v_rotate_right<1>(v_reinterpret_as_u32(txy)));
                    v_float32x4 ty = v_setall_f32(txy.get0());

                    const float* prow0 = (const float*)oldPts[yi+0];
                    const float* prow1 = (const float*)oldPts[yi+1];

                    v_float32x4 p00 = v_load(prow0 + (xi+0)*4);
                    v_float32x4 p01 = v_load(prow0 + (xi+1)*4);
                    v_float32x4 p10 = v_load(prow1 + (xi+0)*4);
                    v_float32x4 p11 = v_load(prow1 + (xi+1)*4);

                    // do not fix missing data
                    // NaN check is done later

                    const float* nrow0 = (const float*)oldNrm[yi+0];
                    const float* nrow1 = (const float*)oldNrm[yi+1];

                    v_float32x4 n00 = v_load(nrow0 + (xi+0)*4);
                    v_float32x4 n01 = v_load(nrow0 + (xi+1)*4);
                    v_float32x4 n10 = v_load(nrow1 + (xi+0)*4);
                    v_float32x4 n11 = v_load(nrow1 + (xi+1)*4);

                    // NaN check is done later

                    v_float32x4 p0 = p00 + tx*(p01 - p00);
                    v_float32x4 p1 = p10 + tx*(p11 - p10);
                    oldP = p0 + ty*(p1 - p0);

                    v_float32x4 n0 = n00 + tx*(n01 - n00);
                    v_float32x4 n1 = n10 + tx*(n11 - n10);
                    oldN = n0 + ty*(n1 - n0);
                }

                bool oldPNcheck = fastCheck(oldP, oldN);

                //filter by distance
                v_float32x4 diff = newP - oldP;
                bool distCheck = !(v_reduce_sum(diff*diff) > sqThresh);

                //filter by angle
                bool angleCheck = !(abs(v_reduce_sum(newN*oldN)) < cosThresh);

                if(!(oldPNcheck && distCheck && angleCheck))
                    continue;

                // build point-wise vector ab = [ A | b ]

                v_float32x4 VxNv = crossProduct(newP, oldN);
                Point3f VxN;
                VxN.x = VxNv.get0();
                VxN.y = v_reinterpret_as_f32(v_extract<1>(v_reinterpret_as_u32(VxNv), v_setzero_u32())).get0();
                VxN.z = v_reinterpret_as_f32(v_extract<2>(v_reinterpret_as_u32(VxNv), v_setzero_u32())).get0();

                float dotp = -v_reduce_sum(oldN*diff);

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                // and gather sum

                v_float32x4 vd = VxNv | v_float32x4(0, 0, 0, dotp);
                v_float32x4 n = oldN;
                v_float32x4 nyzx;
                {
                    v_uint32x4 aa = v_reinterpret_as_u32(n);
                    v_uint32x4 yz00 = v_extract<1>(aa, v_setzero_u32());
                    v_uint32x4 x0y0, tmp;
                    v_zip(aa, v_setzero_u32(), x0y0, tmp);
                    nyzx = v_reinterpret_as_f32(v_combine_low(yz00, x0y0));
                }

                v_float32x4 vutg[utBufferSize];
                for(size_t i = 0; i < utBufferSize; i++)
                    vutg[i] = v_load_aligned(upperTriangle + i*4);

                int p = 0;
                v_float32x4 v;
                // vx * vd, vx * n
                v = v_setall_f32(VxN.x);
                v_store_aligned(upperTriangle + p*4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p*4, v_muladd(v,  n, vutg[p])); p++;
                // vy * vd, vy * n
                v = v_setall_f32(VxN.y);
                v_store_aligned(upperTriangle + p*4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p*4, v_muladd(v,  n, vutg[p])); p++;
                // vz * vd, vz * n
                v = v_setall_f32(VxN.z);
                v_store_aligned(upperTriangle + p*4, v_muladd(v, vd, vutg[p])); p++;
                v_store_aligned(upperTriangle + p*4, v_muladd(v,  n, vutg[p])); p++;
                // nx^2, ny^2, nz^2
                v_store_aligned(upperTriangle + p*4, v_muladd(n, n, vutg[p])); p++;
                // nx*ny, ny*nz, nx*nz
                v_store_aligned(upperTriangle + p*4, v_muladd(n, nyzx, vutg[p])); p++;
                // nx*d, ny*d, nz*d
                v = v_setall_f32(dotp);
                v_store_aligned(upperTriangle + p*4, v_muladd(n, v, vutg[p])); p++;
            }
        }

        ABtype sumAB = ABtype::zeros();
        for(int i = 0; i < 6; i++)
        {
            for(int j = i; j < 7; j++)
            {
                size_t p = utPos[i*7+j];
                sumAB(i, j) = upperTriangle[p];
            }
        }

#else

        float upperTriangle[UTSIZE];
        for(int i = 0; i < UTSIZE; i++)
            upperTriangle[i] = 0;

        for(int y = range.start; y < range.end; y++)
        {
            const ptype* newPtsRow = newPts[y];
            const ptype* newNrmRow = newNrm[y];

            for(int x = 0; x < newPts.cols; x++)
            {
                Point3f newP = fromPtype(newPtsRow[x]);
                Point3f newN = fromPtype(newNrmRow[x]);

                Point3f oldP(nan3), oldN(nan3);

                if(!(fastCheck(newP) && fastCheck(newN)))
                    continue;

                //transform to old coord system
                newP = pose * newP;
                newN = pose.rotation() * newN;

                //find correspondence by projecting the point
                Point2f oldCoords = proj(newP);
                if(!(oldCoords.x >= 0 && oldCoords.x < oldPts.cols - 1 &&
                     oldCoords.y >= 0 && oldCoords.y < oldPts.rows - 1))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                int xi = cvFloor(oldCoords.x), yi = cvFloor(oldCoords.y);
                float tx  = oldCoords.x - xi, ty = oldCoords.y - yi;

                const ptype* prow0 = oldPts[yi+0];
                const ptype* prow1 = oldPts[yi+1];

                Point3f p00 = fromPtype(prow0[xi+0]);
                Point3f p01 = fromPtype(prow0[xi+1]);
                Point3f p10 = fromPtype(prow1[xi+0]);
                Point3f p11 = fromPtype(prow1[xi+1]);

                //do not fix missing data
                if(!(fastCheck(p00) && fastCheck(p01) &&
                     fastCheck(p10) && fastCheck(p11)))
                    continue;

                const ptype* nrow0 = oldNrm[yi+0];
                const ptype* nrow1 = oldNrm[yi+1];

                Point3f n00 = fromPtype(nrow0[xi+0]);
                Point3f n01 = fromPtype(nrow0[xi+1]);
                Point3f n10 = fromPtype(nrow1[xi+0]);
                Point3f n11 = fromPtype(nrow1[xi+1]);

                if(!(fastCheck(n00) && fastCheck(n01) &&
                     fastCheck(n10) && fastCheck(n11)))
                    continue;

                Point3f p0 = p00 + tx*(p01 - p00);
                Point3f p1 = p10 + tx*(p11 - p10);
                oldP = p0 + ty*(p1 - p0);

                Point3f n0 = n00 + tx*(n01 - n00);
                Point3f n1 = n10 + tx*(n11 - n10);
                oldN = n0 + ty*(n1 - n0);

                if(!(fastCheck(oldP) && fastCheck(oldN)))
                    continue;

                //filter by distance
                Point3f diff = newP - oldP;
                if(diff.dot(diff) > sqDistanceThresh)
                {
                    continue;
                }

                //filter by angle
                if(abs(newN.dot(oldN)) < minCos)
                {
                    continue;
                }

                // build point-wise vector ab = [ A | b ]

                //try to optimize
                Point3f VxN = newP.cross(oldN);
                float ab[7] = {VxN.x, VxN.y, VxN.z, oldN.x, oldN.y, oldN.z, oldN.dot(-diff)};

                // build point-wise upper-triangle matrix [ab^T * ab] w/o last row
                // which is [A^T*A | A^T*b]
                // and gather sum
                int pos = 0;
                for(int i = 0; i < 6; i++)
                {
                    for(int j = i; j < 7; j++)
                    {
                        upperTriangle[pos++] += ab[i]*ab[j];
                    }
                }
            }
        }

        ABtype sumAB = ABtype::zeros();
        int pos = 0;
        for(int i = 0; i < 6; i++)
        {
            for(int j = i; j < 7; j++)
            {
                sumAB(i, j) = upperTriangle[pos++];
            }
        }
#endif

        AutoLock al(mtx);
        globalSumAb += sumAB;
    }

    ABtype& globalSumAb;
    Mutex& mtx;
    const Points& oldPts;
    const Normals& oldNrm;
    const Points& newPts;
    const Normals& newNrm;
    Affine3f pose;
    const Intr::Projector proj;
    float sqDistanceThresh;
    float minCos;
};


void ICPCPU::getAb(const Points& oldPts, const Normals& oldNrm, const Points& newPts, const Normals& newNrm,
                   Affine3f pose, int level, Matx66f &A, Vec6f &b) const
{
    ScopeTime st("icp: get ab", false);

    CV_Assert(oldPts.size() == oldNrm.size());
    CV_Assert(newPts.size() == newNrm.size());

    ABtype sumAB = ABtype::zeros();

    Mutex mutex;
    GetAbInvoker invoker(sumAB, mutex, oldPts, oldNrm, newPts, newNrm, pose,
                         intrinsics.scale(level).makeProjector(),
                         distanceThreshold*distanceThreshold, cos(angleThreshold));
    Range range(0, newPts.rows);
    const int nstripes = -1;
    parallel_for_(range, invoker, nstripes);

    // splitting AB matrix to A and b
    for(int i = 0; i < 6; i++)
    {
        // augment lower triangle of A by symmetry
        for(int j = i; j < 6; j++)
        {
            A(i, j) = A(j, i) = sumAB(i, j);
        }

        b(i) = sumAB(i, 6);
    }
}

///////// GPU implementation /////////

class ICPGPU : public ICP
{
public:
    ICPGPU(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const override;

    virtual ~ICPGPU() { }
};

ICPGPU::ICPGPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }


bool ICPGPU::estimateTransform(cv::Affine3f& /*transform*/, cv::Ptr<Frame> /*_oldFrame*/, cv::Ptr<Frame> /*newFrame*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<ICP> makeICP(cv::kinfu::Params::PlatformType t,
                     const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold)
{
    switch (t)
    {
    case cv::kinfu::Params::PlatformType::PLATFORM_CPU:
        return cv::makePtr<ICPCPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    case cv::kinfu::Params::PlatformType::PLATFORM_GPU:
        return cv::makePtr<ICPGPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    default:
        return cv::Ptr<ICP>();
    }
}

} // namespace kinfu
} // namespace cv
