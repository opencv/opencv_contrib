//TODO: add license

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include "precomp.hpp"

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

typedef cv::Mat_< cv::Point3f > Points;
typedef Points Normals;

typedef cv::Mat_< depthType > Depth;

struct Frame
{
public:
    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const = 0;
    virtual void getDepth(cv::OutputArray depth) const = 0;
    virtual ~Frame() { }
};

struct FrameCPU : Frame
{
public:
    FrameCPU() : points(), normals() { }
    virtual ~FrameCPU() { }

    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const;
    virtual void getDepth(cv::OutputArray depth) const;

    std::vector<Points> points;
    std::vector<Normals> normals;
    Depth depthData;
};

struct FrameGPU : Frame
{
public:
    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const;
    virtual void getDepth(cv::OutputArray depth) const;
    virtual ~FrameGPU() { }
};

//TODO: change interface to avoid extra memory allocations

struct FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator() (const cv::InputArray depth, const cv::kinfu::Intr, int levels, float depthFactor,
                                       float sigmaDepth, float sigmaSpatial, int kernelSize) const = 0;
    virtual cv::Ptr<Frame> operator() (const cv::InputArray points, const cv::InputArray normals, int levels) const = 0;
    virtual ~FrameGenerator() {}
};

cv::Ptr<FrameGenerator> makeFrameGenerator(cv::kinfu::KinFu::KinFuParams::PlatformType t);

#endif


