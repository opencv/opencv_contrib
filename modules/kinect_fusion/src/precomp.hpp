//TODO: add license

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/imgproc.hpp"
#include "opencv2/kinect_fusion/kinfu.hpp"

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

inline bool isNaN(cv::Point3_<kftype> p)
{
    return (cvIsNaN(p.x) || cvIsNaN(p.y) || cvIsNaN(p.z));
}

inline bool isNaN(kftype p)
{
    return cvIsNaN(p);
}

typedef cv::Mat_< cv::Point3_<kftype> > Points;
typedef Points Normals;

typedef cv::Mat_< kftype > Depth;

//TODO: make it better
inline Depth toDepth(cv::InputArray a)
{
    return Depth(a.getMat());
}

template<typename Tv>
inline Tv getNaN()
{
    return std::numeric_limits<Tv>::quiet_NaN();
}

template<>
inline cv::Point3f getNaN<cv::Point3f>()
{
    float v = std::numeric_limits<float>::quiet_NaN();
    return cv::Point3f(v, v, v);
}

template<typename Tv, typename Tm>
inline Tv bilinear(Tm m, cv::Point2f pt)
{
    const Tv defaultValue = getNaN<Tv>();
    if(pt.x < 0 || pt.x >= m.cols-1 ||
       pt.y < 0 || pt.y >= m.rows-1)
        return defaultValue;

    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);
    float tx = pt.x - xi, ty = pt.y - yi;
    Tv v00 = m(cv::Point(xi+0, yi+0));
    Tv v01 = m(cv::Point(xi+1, yi+0));
    Tv v10 = m(cv::Point(xi+0, yi+1));
    Tv v11 = m(cv::Point(xi+1, yi+1));

    bool b00 = !isNaN(v00) && v00 != Tv();
    bool b01 = !isNaN(v01) && v01 != Tv();
    bool b10 = !isNaN(v10) && v10 != Tv();
    bool b11 = !isNaN(v11) && v11 != Tv();

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
    return v00*(1.f-tx)*(1.f-ty) + v01*tx*(1.f-ty) + v10*(1.f-tx)*ty + v11*tx*ty;
}

#endif
