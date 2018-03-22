//TODO: add license

#ifndef __OPENCV_KINFU_FRAME_H__
#define __OPENCV_KINFU_FRAME_H__

#include "precomp.hpp"

struct Frame
{
public:
    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const = 0;
    virtual ~Frame() { }
};

struct FrameCPU : Frame
{
public:
    FrameCPU() : points(), normals() { }
    virtual ~FrameCPU() { }

    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const;

    std::vector<Points> points;
    std::vector<Normals> normals;
};

struct FrameGPU : Frame
{
public:
    virtual void render(cv::OutputArray image, int level, cv::Affine3f lightPose) const;
    virtual ~FrameGPU() { }
};

//TODO: replace Depth, Points and Normals by InputArrays (getMat/getUMat inside)
//TODO: add conversion to Depth (from CV_16S to CV_32F)

struct FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator() (const Depth, const cv::kinfu::Intr, int levels, float depthFactor,
                                       float sigmaDepth, float sigmaSpatial, int kernelSize) const = 0;
    virtual cv::Ptr<Frame> operator() (const Points, const Normals, int levels) const = 0;
    virtual ~FrameGenerator() {}
};

cv::Ptr<FrameGenerator> makeFrameGenerator(cv::kinfu::KinFu::KinFuParams::PlatformType t);

#endif


