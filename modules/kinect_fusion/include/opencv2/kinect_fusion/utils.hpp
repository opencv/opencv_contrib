//TODO: add license

#ifndef __OPENCV_KINFU_UTILS_H__
#define __OPENCV_KINFU_UTILS_H__

#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"

//TODO: put it into namespace kinfu since it's exposed outside

typedef float kftype;

namespace cv {

template<> class DataType<cv::Point3f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 3,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

}

typedef cv::Mat_< kftype > Depth;
typedef cv::Mat_< kftype > Distance;
typedef cv::Mat_< cv::Point3_<kftype> > Points;
typedef Points Normals;

struct Voxel
{
    kftype v;
    int weight;
};

typedef cv::AutoBuffer<Voxel> Volume;

//TODO: make it better
inline Depth toDepth(cv::InputArray a)
{
    return Depth(a.getMat());
}

// Camera intrinsics
struct Intr
{
    struct Reprojector
    {
        Reprojector() {}
        inline Reprojector(Intr intr)
        {
            fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;
            cx = intr.cx, cy = intr.cy;
        }
        template<typename T>
        inline cv::Point3_<T> operator()(cv::Point3_<T> p) const
        {
            T x = p.z * (p.x - cx) * fxinv;
            T y = p.z * (p.y - cy) * fyinv;
            return cv::Point3_<T>(x, y, p.z);
        }

        float fxinv, fyinv, cx, cy;
    };
    struct Projector
    {
        inline Projector(const Intr& intr) : i(intr) { }
        template<typename T>
        inline cv::Point_<T> operator()(cv::Point3_<T> p) const
        {
            T x = i.fx*p.x/p.z + i.cx;
            T y = i.fy*p.y/p.z + i.cy;
            return cv::Point_<T>(x, y);
        }
        const Intr& i;
    };
    Intr() : fx(), fy(), cx(), cy() { }
    Intr(float _fx, float _fy, float _cx, float _cy) : fx(_fx), fy(_fy), cx(_cx), cy(_cy) { }
    // scale intrinsics to pyramid level
    inline Intr scale(int pyr) const
    {
        float factor = (1.f /(1 << pyr));
        return Intr(fx*factor, fy*factor, cx*factor, cy*factor);
    }
    inline Reprojector makeReprojector() const { return Reprojector(*this); }
    inline Projector   makeProjector()   const { return Projector(*this);   }

    float fx, fy, cx, cy;
};

#endif

