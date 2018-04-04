//TODO: add license

//TODO: organize includes properly
#include "precomp.hpp"
#include "icp.hpp"
#include "opencv2/core/hal/intrin.hpp"

using namespace cv;
using namespace cv::kinfu;
using namespace std;

ICP::ICP(const Intr _intrinsics, const std::vector<int>& _iterations, float _angleThreshold, float _distanceThreshold) :
    iterations(_iterations), angleThreshold(_angleThreshold), distanceThreshold(_distanceThreshold),
    intrinsics(_intrinsics)
{ }

///////// CPU implementation /////////

class ICPCPU : public ICP
{
public:
    ICPCPU(const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold);

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const;

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
    for(int level = iterations.size() - 1; level >= 0; level--)
    {
        Points  oldPts = oldPoints [level], newPts = newPoints [level];
        Normals oldNrm = oldNormals[level], newNrm = newNormals[level];

        for(int iter = 0; iter < iterations[level]; iter++)
        {
            Matx66f A;
            Vec6f b;

            getAb(oldPts, oldNrm, newPts, newNrm, transform, level, A, b);

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
static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}

static inline bool fastCheck(const v_float32x4& p0, const v_float32x4& p1, const v_float32x4& p2, const v_float32x4& p3)
{
    v_float32x4 a0 = v_combine_low(p0, p1);
    v_float32x4 a1 = v_combine_low(p2, p3);
    return !(v_check_any(a0 != a0) || v_check_any(a1 != a1));
}

static inline bool fastCheck(const v_float32x4& p0, const v_float32x4& p1)
{
    v_float32x4 a = v_combine_low(p0, p1);
    return !v_check_any(a != a);
}

static inline bool isNaN(const v_float32x4& p)
{
    return v_check_any(p != p);
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

    virtual void operator ()(const Range& range) const
    {
        CV_Assert(ptype::channels == 4);

        const size_t utBufferSize = 9;
        float CV_DECL_ALIGNED(16) upperTriangle[utBufferSize*4];
        for(size_t i = 0; i < utBufferSize*4; i++)
            upperTriangle[i] = 0;
        const size_t utPos[] =
        {
           0,  1,  2,  4,  5,  6,  3,
          -1,  9, 10, 12, 13, 14, 11,
          -1, -1, 18, 20, 21, 22, 19,
          -1, -1, -1, 24, 28, 30, 32,
          -1, -1, -1, -1, 25, 29, 33,
          -1, -1, -1, -1, -1, 26, 34
        };

        const float (&pm)[16] = pose.matrix.val;
        v_float32x4 poseRot0(pm[0], pm[4], pm[ 8], 0);
        v_float32x4 poseRot1(pm[1], pm[5], pm[ 9], 0);
        v_float32x4 poseRot2(pm[2], pm[6], pm[10], 0);
        v_float32x4 poseTrans(pm[3], pm[7], pm[11], 0);

        v_float32x4 vfxy(proj.fx, proj.fy, 0, 0), vcxy(proj.cx, proj.cy, 0, 0);
        v_float32x4 vframe(oldPts.cols - 1, oldPts.rows - 1, 1, 1);

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
                v_float32x4 oldCoordsv;
                {
                    float z = 1.f/(v_reinterpret_as_f32(v_extract<2>(v_reinterpret_as_u32(newP), v_setzero_u32())).get0());
                    // x, y, 0, 0
                    oldCoordsv = vfxy*newP*v_setall_f32(z) + vcxy;
                }

                if(!(v_check_all(oldCoordsv >= v_setzero_f32()) &&
                     v_check_all(oldCoordsv < vframe)))
                    continue;

                // bilinearly interpolate oldPts and oldNrm under oldCoords point
                v_float32x4 oldP;
                v_float32x4 oldN;
                {
                    Point2f oldCoords;
                    oldCoords.x = oldCoordsv.get0();
                    oldCoords.y = v_reinterpret_as_f32(v_extract<1>(v_reinterpret_as_u32(oldCoordsv), v_setzero_u32())).get0();

                    v_int32x4 ixy = v_floor(oldCoordsv);
                    v_float32x4 txy = oldCoordsv - v_cvt_f32(ixy);
                    int xi = ixy.get0();
                    int yi = v_rotate_right<1>(ixy).get0();
                    // tx, ty, tx, ty
                    txy = v_combine_low(txy, txy);
                    v_float32x4 txy1 = v_setall_f32(1.f) - txy;
                    v_float32x4 mask = v_reinterpret_as_f32(v_uint32x4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0));
                    // tx1, ty1, tx1, ty
                    v_float32x4 tmul0 = v_select(mask, txy1, txy);
                    // ty1,  tx,  ty, tx
                    v_float32x4 tmul1 = v_reinterpret_as_f32(v_extract<3>(v_reinterpret_as_u32(txy1),
                                                                          v_reinterpret_as_u32(txy)));
                    v_float32x4 tv = tmul0*tmul1;

                    const float* prow0 = (const float*)oldPts[yi+0];
                    const float* prow1 = (const float*)oldPts[yi+1];

                    v_float32x4 p00 = v_load_aligned(prow0 + (xi+0)*4);
                    v_float32x4 p01 = v_load_aligned(prow0 + (xi+1)*4);
                    v_float32x4 p10 = v_load_aligned(prow1 + (xi+0)*4);
                    v_float32x4 p11 = v_load_aligned(prow1 + (xi+1)*4);

                    //do not fix missing data
                    if(!fastCheck(p00, p01, p10, p11))
                        continue;

                    const float* nrow0 = (const float*)oldNrm[yi+0];
                    const float* nrow1 = (const float*)oldNrm[yi+1];

                    v_float32x4 n00 = v_load_aligned(nrow0 + (xi+0)*4);
                    v_float32x4 n01 = v_load_aligned(nrow0 + (xi+1)*4);
                    v_float32x4 n10 = v_load_aligned(nrow1 + (xi+0)*4);
                    v_float32x4 n11 = v_load_aligned(nrow1 + (xi+1)*4);

                    if(!fastCheck(n00, n01, n10, n11))
                        continue;

                    oldP = v_matmul(tv, p00, p01, p10, p11);
                    oldN = v_matmul(tv, n00, n01, n10, n11);
                }

                if(!fastCheck(oldP, oldN))
                    continue;

                //filter by distance
                v_float32x4 diff = newP - oldP;
                if(v_reduce_sum(diff*diff) > sqThresh)
                    continue;

                //filter by angle
                if(abs(v_reduce_sum(newN*oldN)) < cosThresh)
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
                int p = utPos[i*7+j];
                sumAB(i, j) = upperTriangle[p];
            }
        }

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
    parallel_for_(range, invoker);

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

    virtual bool estimateTransform(cv::Affine3f& transform, cv::Ptr<Frame> oldFrame, cv::Ptr<Frame> newFrame) const;

    virtual ~ICPGPU() { }
};

ICPGPU::ICPGPU(const Intr _intrinsics, const std::vector<int> &_iterations, float _angleThreshold, float _distanceThreshold) :
    ICP(_intrinsics, _iterations, _angleThreshold, _distanceThreshold)
{ }


bool ICPGPU::estimateTransform(cv::Affine3f& /*transform*/, cv::Ptr<Frame> /*_oldFrame*/, cv::Ptr<Frame> /*newFrame*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<ICP> makeICP(cv::kinfu::KinFu::KinFuParams::PlatformType t,
                     const cv::kinfu::Intr _intrinsics, const std::vector<int> &_iterations,
                     float _angleThreshold, float _distanceThreshold)
{
    switch (t)
    {
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_CPU:
        return cv::makePtr<ICPCPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_GPU:
        return cv::makePtr<ICPGPU>(_intrinsics, _iterations, _angleThreshold, _distanceThreshold);
    default:
        return cv::Ptr<ICP>();
    }
}
