// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include "precomp.hpp"
#include "utils.hpp"

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

template<> class DataType<cv::Vec3f>
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

template<> class DataType<cv::Vec4f>
{
public:
    typedef float       value_type;
    typedef value_type  work_type;
    typedef value_type  channel_type;
    typedef value_type  vec_type;
    enum { generic_type = 0,
           depth        = CV_32F,
           channels     = 4,
           fmt          = (int)'f',
           type         = CV_MAKETYPE(depth, channels)
         };
};

namespace kinfu {

typedef cv::Vec4f ptype;
inline cv::Vec3f fromPtype(const ptype& x)
{
    return cv::Vec3f(x[0], x[1], x[2]);
}

inline ptype toPtype(const cv::Vec3f& x)
{
    return ptype(x[0], x[1], x[2], 0);
}

typedef cv::Mat_< ptype > Points;
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

    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const override;
    virtual void getDepth(cv::OutputArray depth) const override;

    std::vector<Points> points;
    std::vector<Normals> normals;
    Depth depthData;
};

struct FrameGPU : Frame
{
public:
    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const override;
    virtual void getDepth(cv::OutputArray depth) const override;
    virtual ~FrameGPU() { }
};

struct FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator ()() const = 0;
    virtual void operator() (cv::Ptr<Frame> frame, cv::InputArray depth, const cv::kinfu::Intr, int levels, float depthFactor,
                                       float sigmaDepth, float sigmaSpatial, int kernelSize) const = 0;
    virtual void operator() (cv::Ptr<Frame> frame, cv::InputArray points, cv::InputArray normals, int levels) const = 0;
    virtual ~FrameGenerator() {}
};

cv::Ptr<FrameGenerator> makeFrameGenerator(cv::kinfu::KinFu::Params::PlatformType t);

} // namespace kinfu
} // namespace cv
#endif
