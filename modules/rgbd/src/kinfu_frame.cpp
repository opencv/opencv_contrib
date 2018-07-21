// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "kinfu_frame.hpp"

namespace cv {
namespace kinfu {

struct FrameGeneratorCPU : FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator ()() const override;
    virtual void operator() (Ptr<Frame> _frame, InputArray depth, const kinfu::Intr, int levels, float depthFactor,
                             float sigmaDepth, float sigmaSpatial, int kernelSize) const override;
    virtual void operator() (Ptr<Frame> _frame, InputArray points, InputArray normals, int levels) const override;

    virtual ~FrameGeneratorCPU() {}
};

void computePointsNormals(const cv::kinfu::Intr, float depthFactor, const Depth, Points, Normals );
Depth pyrDownBilateral(const Depth depth, float sigma);
void pyrDownPointsNormals(const Points p, const Normals n, Points& pdown, Normals& ndown);

cv::Ptr<Frame> FrameGeneratorCPU::operator ()() const
{
    return makePtr<FrameCPU>();
}

void FrameGeneratorCPU::operator ()(Ptr<Frame> _frame, InputArray depth, const Intr intr, int levels, float depthFactor,
                                    float sigmaDepth, float sigmaSpatial, int kernelSize) const
{
    ScopeTime st("frameGenerator cpu: from depth");

    Ptr<FrameCPU> frame = _frame.dynamicCast<FrameCPU>();

    CV_Assert(frame);

    //CV_Assert(depth.type() == CV_16S);
    // this should convert CV_16S to CV_32F
    frame->depthData = Depth(depth.getMat());

    // looks like OpenCV's bilateral filter works the same as KinFu's
    Depth smooth;
    bilateralFilter(frame->depthData, smooth, kernelSize, sigmaDepth*depthFactor, sigmaSpatial);

    // depth truncation is not used by default
    //if (p.icp_truncate_depth_dist > 0) kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    // we don't need depth pyramid outside this method
    // if we do, the code is to be refactored

    Depth scaled = smooth;
    Size sz = smooth.size();
    frame->points.resize(levels);
    frame->normals.resize(levels);
    for(int i = 0; i < levels; i++)
    {
        Points&  p = frame->points[i];
        Normals& n = frame->normals[i];
        p.create(sz); n.create(sz);

        computePointsNormals(intr.scale(i), depthFactor, scaled, p, n);

        if(i < levels - 1)
        {
            sz.width /= 2; sz.height /= 2;
            scaled = pyrDownBilateral(scaled, sigmaDepth*depthFactor);
        }
    }
}

void FrameGeneratorCPU::operator ()(Ptr<Frame> _frame, InputArray _points, InputArray _normals, int levels) const
{
    ScopeTime st("frameGenerator cpu: pyrDown p, n");

    CV_Assert( _points.type() == DataType<Points::value_type>::type);
    CV_Assert(_normals.type() == DataType<Points::value_type>::type);

    Ptr<FrameCPU> frame = _frame.dynamicCast<FrameCPU>();

    CV_Assert(frame);

    Mat p0 = _points.getMat(), n0 = _normals.getMat();
    if(_points.kind()  != _InputArray::MAT)
        p0 = p0.clone();
    if(_normals.kind() != _InputArray::MAT)
        n0 = n0.clone();

    frame->depthData = Depth();
    frame->points.resize(levels);
    frame->normals.resize(levels);
    frame->points[0]  = p0;
    frame->normals[0] = n0;
    Size sz = _points.size();
    for(int i = 1; i < levels; i++)
    {
        sz.width /= 2; sz.height /= 2;
        frame->points[i].create(sz);
        frame->normals[i].create(sz);
        pyrDownPointsNormals(frame->points[i-1], frame->normals[i-1],
                             frame->points[i  ], frame->normals[i  ]);
    }
}

template<int p>
inline float specPow(float x)
{
    if(p % 2 == 0)
    {
        float v = specPow<p/2>(x);
        return v*v;
    }
    else
    {
        float v = specPow<(p-1)/2>(x);
        return v*v*x;
    }
}

template<>
inline float specPow<0>(float /*x*/)
{
    return 1.f;
}

template<>
inline float specPow<1>(float x)
{
    return x;
}

struct RenderInvoker : ParallelLoopBody
{
    RenderInvoker(const Points& _points, const Normals& _normals, Mat_<Vec3b>& _img, Affine3f _lightPose, Size _sz) :
        ParallelLoopBody(),
        points(_points),
        normals(_normals),
        img(_img),
        lightPose(_lightPose),
        sz(_sz)
    { }

    virtual void operator ()(const Range& range) const override
    {
        for(int y = range.start; y < range.end; y++)
        {
            Vec3b* imgRow = img[y];
            const ptype* ptsRow = points[y];
            const ptype* nrmRow = normals[y];

            for(int x = 0; x < sz.width; x++)
            {
                Point3f p = fromPtype(ptsRow[x]);
                Point3f n = fromPtype(nrmRow[x]);

                Vec3b color;

                if(isNaN(p))
                {
                    color = Vec3b(0, 32, 0);
                }
                else
                {
                    const float Ka = 0.3f;  //ambient coeff
                    const float Kd = 0.5f;  //diffuse coeff
                    const float Ks = 0.2f;  //specular coeff
                    const int   sp = 20;  //specular power

                    const float Ax = 1.f;   //ambient color,  can be RGB
                    const float Dx = 1.f;   //diffuse color,  can be RGB
                    const float Sx = 1.f;   //specular color, can be RGB
                    const float Lx = 1.f;   //light color

                    Point3f l = normalize(lightPose.translation() - Vec3f(p));
                    Point3f v = normalize(-Vec3f(p));
                    Point3f r = normalize(Vec3f(2.f*n*n.dot(l) - l));

                    uchar ix = (uchar)((Ax*Ka*Dx + Lx*Kd*Dx*max(0.f, n.dot(l)) +
                                        Lx*Ks*Sx*specPow<sp>(max(0.f, r.dot(v))))*255.f);
                    color = Vec3b(ix, ix, ix);
                }

                imgRow[x] = color;
            }
        }
    }

    const Points& points;
    const Normals& normals;
    Mat_<Vec3b>& img;
    Affine3f lightPose;
    Size sz;
};

void FrameCPU::render(OutputArray image, int level, Affine3f lightPose) const
{
    ScopeTime st("frame cpu render");

    CV_Assert(level < (int)points.size());
    CV_Assert(level < (int)normals.size());

    Size sz = points[level].size();
    image.create(sz, CV_8UC3);
    Mat_<Vec3b> img = image.getMat();

    RenderInvoker ri(points[level], normals[level], img, lightPose, sz);
    Range range(0, sz.height);
    const int nstripes = -1;
    parallel_for_(range, ri, nstripes);
}


void FrameCPU::getDepth(OutputArray _depth) const
{
    CV_Assert(!depthData.empty());
    _depth.createSameSize(depthData, depthData.type());
    _depth.assign(depthData);
}


void FrameCPU::getPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_Assert(!points.empty());
    CV_Assert(!normals.empty());
    std::vector<Mat> pts(points.size());
    std::vector<Mat> nrm(normals.size());
    for(size_t i = 0; i < pts.size(); i++)
    {
        pts[i] = points [i];
        nrm[i] = normals[i];
    }
    _points.create(1, points.size(), DataType<Points::value_type>::type);
    _points.assign(pts);
    _normals.create(1, normals.size(), DataType<Points::value_type>::type);
    _normals.assign(nrm);
}


void pyrDownPointsNormals(const Points p, const Normals n, Points &pdown, Normals &ndown)
{
    for(int y = 0; y < pdown.rows; y++)
    {
        ptype* ptsRow = pdown[y];
        ptype* nrmRow = ndown[y];
        const ptype* pUpRow0 = p[2*y];
        const ptype* pUpRow1 = p[2*y+1];
        const ptype* nUpRow0 = n[2*y];
        const ptype* nUpRow1 = n[2*y+1];
        for(int x = 0; x < pdown.cols; x++)
        {
            Point3f point = nan3, normal = nan3;

            Point3f d00 = fromPtype(pUpRow0[2*x]);
            Point3f d01 = fromPtype(pUpRow0[2*x+1]);
            Point3f d10 = fromPtype(pUpRow1[2*x]);
            Point3f d11 = fromPtype(pUpRow1[2*x+1]);

            if(!(isNaN(d00) || isNaN(d01) || isNaN(d10) || isNaN(d11)))
            {
                point = (d00 + d01 + d10 + d11)*0.25f;

                Point3f n00 = fromPtype(nUpRow0[2*x]);
                Point3f n01 = fromPtype(nUpRow0[2*x+1]);
                Point3f n10 = fromPtype(nUpRow1[2*x]);
                Point3f n11 = fromPtype(nUpRow1[2*x+1]);

                normal = (n00 + n01 + n10 + n11)*0.25f;
            }

            ptsRow[x] = toPtype(point);
            nrmRow[x] = toPtype(normal);
        }
    }
}

struct PyrDownBilateralInvoker : ParallelLoopBody
{
    PyrDownBilateralInvoker(const Depth& _depth, Depth& _depthDown, float _sigma) :
        ParallelLoopBody(),
        depth(_depth),
        depthDown(_depthDown),
        sigma(_sigma)
    { }

    virtual void operator ()(const Range& range) const override
    {
        float sigma3 = sigma*3;
        const int D = 5;

        for(int y = range.start; y < range.end; y++)
        {
            depthType* downRow = depthDown[y];
            const depthType* srcCenterRow = depth[2*y];

            for(int x = 0; x < depthDown.cols; x++)
            {
                depthType center = srcCenterRow[2*x];

                int sx = max(0, 2*x - D/2), ex = min(2*x - D/2 + D, depth.cols-1);
                int sy = max(0, 2*y - D/2), ey = min(2*y - D/2 + D, depth.rows-1);

                depthType sum = 0;
                int count = 0;

                for(int iy = sy; iy < ey; iy++)
                {
                    const depthType* srcRow = depth[iy];
                    for(int ix = sx; ix < ex; ix++)
                    {
                        depthType val = srcRow[ix];
                        if(abs(val - center) < sigma3)
                        {
                            sum += val; count ++;
                        }
                    }
                }

                downRow[x] = (count == 0) ? 0 : sum / count;
            }
        }
    }

    const Depth& depth;
    Depth& depthDown;
    float sigma;
};


Depth pyrDownBilateral(const Depth depth, float sigma)
{
    Depth depthDown(depth.rows/2, depth.cols/2);

    PyrDownBilateralInvoker pdi(depth, depthDown, sigma);
    Range range(0, depthDown.rows);
    const int nstripes = -1;
    parallel_for_(range, pdi, nstripes);

    return depthDown;
}

struct ComputePointsNormalsInvoker : ParallelLoopBody
{
    ComputePointsNormalsInvoker(const Depth& _depth, Points& _points, Normals& _normals,
                                const Intr::Reprojector& _reproj, float _dfac) :
        ParallelLoopBody(),
        depth(_depth),
        points(_points),
        normals(_normals),
        reproj(_reproj),
        dfac(_dfac)
    { }

    virtual void operator ()(const Range& range) const override
    {
        for(int y = range.start; y < range.end; y++)
        {
            const depthType* depthRow0 = depth[y];
            const depthType* depthRow1 = (y < depth.rows - 1) ? depth[y + 1] : 0;
            ptype    *ptsRow = points[y];
            ptype   *normRow = normals[y];

            for(int x = 0; x < depth.cols; x++)
            {
                depthType d00 = depthRow0[x];
                depthType z00 = d00*dfac;
                Point3f v00 = reproj(Point3f((float)x, (float)y, z00));

                Point3f p = nan3, n = nan3;

                if(x < depth.cols - 1 && y < depth.rows - 1)
                {
                    depthType d01 = depthRow0[x+1];
                    depthType d10 = depthRow1[x];

                    depthType z01 = d01*dfac;
                    depthType z10 = d10*dfac;

                    // before it was
                    //if(z00*z01*z10 != 0)
                    if(z00 != 0 && z01 != 0 && z10 != 0)
                    {
                        Point3f v01 = reproj(Point3f((float)(x+1), (float)(y+0), z01));
                        Point3f v10 = reproj(Point3f((float)(x+0), (float)(y+1), z10));

                        cv::Vec3f vec = (v01-v00).cross(v10-v00);
                        n = -normalize(vec);
                        p = v00;
                    }
                }

                ptsRow[x] = toPtype(p);
                normRow[x] = toPtype(n);
            }
        }
    }

    const Depth& depth;
    Points& points;
    Normals& normals;
    const Intr::Reprojector& reproj;
    float dfac;
};

void computePointsNormals(const Intr intr, float depthFactor, const Depth depth,
                          Points points, Normals normals)
{
    CV_Assert(!points.empty() && !normals.empty());
    CV_Assert(depth.size() == points.size());
    CV_Assert(depth.size() == normals.size());

    // conversion to meters
    // before it was:
    //float dfac = 0.001f/depthFactor;
    float dfac = 1.f/depthFactor;

    Intr::Reprojector reproj = intr.makeReprojector();

    ComputePointsNormalsInvoker ci(depth, points, normals, reproj, dfac);
    Range range(0, depth.rows);
    const int nstripes = -1;
    parallel_for_(range, ci, nstripes);
}

///////// GPU implementation /////////

static const int GPU_DEPTH_TYPE  = CV_32F;
static const int GPU_POINTS_TYPE = CV_32FC4;

void computePointsNormalsGpu(const Intr intr, float depthFactor, const UMat depth, UMat points, UMat normals);
UMat pyrDownBilateralGpu(const UMat depth, float sigma);
void customBilateralFilterGpu(UMat src, UMat dst, int kernelSize, float sigmaDepth, float sigmaSpatial);
void pyrDownPointsNormalsGpu(const UMat p, const UMat n, UMat &pdown, UMat &ndown);


struct FrameGeneratorGPU : FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator ()() const override;
    virtual void operator() (Ptr<Frame> frame, InputArray depth, const kinfu::Intr, int levels, float depthFactor,
                             float sigmaDepth, float sigmaSpatial, int kernelSize) const override;
    virtual void operator() (Ptr<Frame> frame, InputArray points, InputArray normals, int levels) const override;

    virtual ~FrameGeneratorGPU() {}
};

cv::Ptr<Frame> FrameGeneratorGPU::operator ()() const
{
    return makePtr<FrameGPU>();
}

void FrameGeneratorGPU::operator ()(Ptr<Frame> _frame, InputArray depth, const Intr intr, int levels, float depthFactor,
                                    float sigmaDepth, float sigmaSpatial, int kernelSize) const
{
    ScopeTime st("frameGenerator gpu: from depth");

    Ptr<FrameGPU> frame = _frame.dynamicCast<FrameGPU>();

    CV_Assert(frame);

    UMat udepth = depth.getUMat();
    if(depth.type() != GPU_DEPTH_TYPE)
        udepth.convertTo(udepth, GPU_DEPTH_TYPE);

    frame->depthData = udepth;

    // looks like OpenCV's bilateral filter works the same as KinFu's
    UMat smooth;
    //TODO: fix that
    // until 32f isn't implemented in OpenCV, we should use our workarounds
    //bilateralFilter(udepth, smooth, kernelSize, sigmaDepth*depthFactor, sigmaSpatial);
    customBilateralFilterGpu(udepth, smooth, kernelSize, sigmaDepth*depthFactor, sigmaSpatial);

    //TODO: remove that
    // 5 meters should fit all
//    float scaleFactor = 255.f/(depthFactor*5.f);
//    UMat depth8, smooth8;
//    udepth.convertTo(depth8, CV_8U, scaleFactor);
//    float sigDepth = sigmaDepth*255.f/depthFactor;
//    bilateralFilter(depth8, smooth8, kernelSize, sigDepth, sigmaSpatial);
//    smooth8.convertTo(smooth, CV_32F, 1.f/scaleFactor);

    // depth truncation is not used by default
    //if (p.icp_truncate_depth_dist > 0) kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    UMat scaled = smooth;
    Size sz = smooth.size();
    frame->points.resize(levels);
    frame->normals.resize(levels);
    for(int i = 0; i < levels; i++)
    {
        UMat& p = frame->points[i];
        UMat& n = frame->normals[i];
        p.create(sz, GPU_POINTS_TYPE); n.create(sz, GPU_POINTS_TYPE);

        computePointsNormalsGpu(intr.scale(i), depthFactor, scaled, p, n);

        if(i < levels - 1)
        {
            sz.width /= 2, sz.height /= 2;
            scaled = pyrDownBilateralGpu(scaled, sigmaDepth*depthFactor);
        }
    }
}

void computePointsNormalsGpu(const Intr intr, float depthFactor, const UMat depth,
                             UMat points, UMat normals)
{
    CV_Assert(!points.empty() && !normals.empty());
    CV_Assert(depth.size() == points.size());
    CV_Assert(depth.size() == normals.size());
    CV_Assert(depth.type() == GPU_DEPTH_TYPE);
    CV_Assert(points.type() == GPU_POINTS_TYPE);
    CV_Assert(normals.type() == GPU_POINTS_TYPE);

    // conversion to meters
    float dfac = 1.f/depthFactor;

    Intr::Reprojector reproj = intr.makeReprojector();

    cv::String errorStr;
    cv::String name = "computePointsNormals";
    ocl::ProgramSource source = ocl::rgbd::kinfu_frame_oclsrc;
    cv::String options;
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    k.args(ocl::KernelArg::WriteOnly(points),
           ocl::KernelArg::WriteOnly(normals),
           ocl::KernelArg::ReadOnly(depth),
           reproj.fxinv, reproj.fyinv,
           reproj.cx, reproj.cy,
           dfac);

    size_t globalSize[2];
    globalSize[0] = (size_t)depth.cols;
    globalSize[1] = (size_t)depth.rows;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

UMat pyrDownBilateralGpu(const UMat depth, float sigma)
{
    UMat depthDown(depth.rows/2, depth.cols/2, GPU_DEPTH_TYPE);

    cv::String errorStr;
    cv::String name = "pyrDownBilateral";
    ocl::ProgramSource source = ocl::rgbd::kinfu_frame_oclsrc;
    cv::String options;
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    k.args(ocl::KernelArg::ReadOnly(depth),
           ocl::KernelArg::WriteOnly(depthDown),
           sigma);

    size_t globalSize[2];
    globalSize[0] = (size_t)depthDown.cols;
    globalSize[1] = (size_t)depthDown.rows;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");

    return depthDown;
}

//TODO: remove it when OpenCV's bilateral processes 32f on GPU
void customBilateralFilterGpu(UMat src /* udepth */, UMat dst /* smooth */,
                              int kernelSize, float sigmaDepth, float sigmaSpatial)
{
    CV_Assert(src.size().area() > 0);
    CV_Assert(src.type() == GPU_DEPTH_TYPE);

    dst.create(src.size(), GPU_DEPTH_TYPE);

    cv::String errorStr;
    cv::String name = "customBilateral";
    ocl::ProgramSource source = ocl::rgbd::kinfu_frame_oclsrc;
    cv::String options;
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    k.args(ocl::KernelArg::ReadOnly(src),
           ocl::KernelArg::WriteOnly(dst),
           kernelSize,
           0.5f / (sigmaSpatial * sigmaSpatial),
           0.5f / (sigmaDepth * sigmaDepth));

    size_t globalSize[2];
    globalSize[0] = (size_t)src.cols;
    globalSize[1] = (size_t)src.rows;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

void FrameGeneratorGPU::operator ()(Ptr<Frame> _frame, InputArray _points, InputArray _normals, int levels) const
{
    ScopeTime st("frameGenerator gpu: pyrDown p, n");

    CV_Assert( _points.type() == GPU_POINTS_TYPE);
    CV_Assert(_normals.type() == GPU_POINTS_TYPE);

    Ptr<FrameGPU> frame = _frame.dynamicCast<FrameGPU>();

    CV_Assert(frame);

    UMat p0 = _points.getUMat(), n0 = _normals.getUMat();
    frame->depthData = UMat();
    frame->points .resize(levels);
    frame->normals.resize(levels);
    frame->points [0] = p0;
    frame->normals[0] = n0;
    Size sz = _points.size();
    for(int i = 1; i < levels; i++)
    {
        sz.width /= 2; sz.width /= 2;
        frame->points [i].create(sz, GPU_POINTS_TYPE);
        frame->normals[i].create(sz, GPU_POINTS_TYPE);
        pyrDownPointsNormalsGpu(frame->points[i-1], frame->normals[i-1],
                                frame->points[i  ], frame->normals[i  ]);
    }
}

void pyrDownPointsNormalsGpu(const UMat p, const UMat n, UMat &pdown, UMat &ndown)
{
    cv::String errorStr;
    cv::String name = "pyrDownPointsNormals";
    ocl::ProgramSource source = ocl::rgbd::kinfu_frame_oclsrc;
    cv::String options;
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    k.args(ocl::KernelArg::ReadOnly(p),
           ocl::KernelArg::ReadOnly(n),
           ocl::KernelArg::WriteOnly(pdown),
           ocl::KernelArg::WriteOnly(ndown));

    size_t globalSize[2];
    globalSize[0] = (size_t)pdown.cols;
    globalSize[1] = (size_t)pdown.rows;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

void FrameGPU::render(OutputArray image, int level, Affine3f lightPose) const
{
    ScopeTime st("frame gpu render");

    CV_Assert(level < (int) points.size());
    CV_Assert(level < (int)normals.size());

    Size sz = points[level].size();
    image.create(sz, CV_8UC3);
    UMat img = image.getUMat();

    cv::String errorStr;
    cv::String name = "render";
    ocl::ProgramSource source = ocl::rgbd::kinfu_frame_oclsrc;
    cv::String options;
    ocl::Kernel k;
    k.create(name.c_str(), source, options, &errorStr);

    if(k.empty())
        throw std::runtime_error("Failed to create kernel: " + errorStr);

    Vec4f lightPt(lightPose.translation()[0],
                  lightPose.translation()[1],
                  lightPose.translation()[2]);

    k.args(ocl::KernelArg::ReadOnly(points [level]),
           ocl::KernelArg::ReadOnly(normals[level]),
           ocl::KernelArg::WriteOnly(img),
           lightPt.val);

    size_t globalSize[2];
    globalSize[0] = (size_t)sz.width;
    globalSize[1] = (size_t)sz.height;

    if(!k.run(2, globalSize, NULL, true))
        throw std::runtime_error("Failed to run kernel");
}

void FrameGPU::getDepth(OutputArray _depth) const
{
    CV_Assert(!depthData.empty());
    _depth.createSameSize(depthData, depthData.type());
    _depth.assign(depthData);
}

void FrameGPU::getPointsNormals(OutputArray _points, OutputArray _normals) const
{
    CV_Assert(!points.empty());
    CV_Assert(!normals.empty());
    _points.create(1, points.size(), GPU_POINTS_TYPE);
    _points.assign(points);
    _normals.create(1, normals.size(), GPU_POINTS_TYPE);
    _normals.assign(normals);
}

cv::Ptr<FrameGenerator> makeFrameGenerator(cv::kinfu::Params::PlatformType t)
{
    switch (t)
    {
    case cv::kinfu::Params::PlatformType::PLATFORM_CPU:
        return cv::makePtr<FrameGeneratorCPU>();
    case cv::kinfu::Params::PlatformType::PLATFORM_GPU:
        return cv::makePtr<FrameGeneratorGPU>();
    default:
        return cv::Ptr<FrameGenerator>();
    }
}

} // namespace kinfu
} // namespace cv
