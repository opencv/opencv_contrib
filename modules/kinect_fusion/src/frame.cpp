//TODO: add license

#include "precomp.hpp"
#include "frame.hpp"

using namespace cv;
using namespace cv::kinfu;

struct FrameGeneratorCPU : FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator() (const Depth, const cv::kinfu::Intr, int levels, float depthFactor,
                                       float sigmaDepth, float sigmaSpatial, int kernelSize) const;
    virtual cv::Ptr<Frame> operator() (const Points, const Normals, int levels) const;
    virtual ~FrameGeneratorCPU() {}
};

struct FrameGeneratorGPU : FrameGenerator
{
public:
    virtual cv::Ptr<Frame> operator() (const Depth, const cv::kinfu::Intr, int levels, float depthFactor,
                                       float sigmaDepth, float sigmaSpatial, int kernelSize) const;
    virtual cv::Ptr<Frame> operator() (const Points, const Normals, int levels) const;
    virtual ~FrameGeneratorGPU() {}
};

cv::Ptr<FrameGenerator> makeFrameGenerator(cv::kinfu::KinFu::KinFuParams::PlatformType t)
{
    switch (t)
    {
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_CPU:
        return cv::makePtr<FrameGeneratorCPU>();
    case cv::kinfu::KinFu::KinFuParams::PlatformType::PLATFORM_GPU:
        return cv::makePtr<FrameGeneratorGPU>();
    default:
        return cv::Ptr<FrameGenerator>();
    }
}

void computePointsNormals(const cv::kinfu::Intr, float depthFactor, const Depth, Points, Normals );
Depth pyrDownBilateral(const Depth depth, float sigma);
void pyrDownPointsNormals(const Points p, const Normals n, Points& pdown, Normals& ndown);

cv::Ptr<Frame> FrameGeneratorCPU::operator ()(const Depth depth, const Intr intr, int levels, float depthFactor,
                                              float sigmaDepth, float sigmaSpatial, int kernelSize) const
{
    cv::Ptr<FrameCPU> frame = makePtr<FrameCPU>();

    // looks like OpenCV's bilateral filter works the same as KinFu's
    Depth smooth;
    bilateralFilter(depth, smooth, kernelSize, sigmaDepth*depthFactor, sigmaSpatial);

    // depth truncation is not used by default
    //if (p.icp_truncate_depth_dist > 0) kfusion::cuda::depthTruncation(curr_.depth_pyr[0], p.icp_truncate_depth_dist);

    // we don't need depth pyramid outside this method
    // if we do, the code is to be refactored

    Depth scaled = smooth;
    Size sz = smooth.size();
    for(int i = 0; i < levels; i++)
    {
        Points p(sz); Normals n(sz);
        computePointsNormals(intr.scale(i), depthFactor, scaled, p, n);

        frame->points.push_back(p);
        frame->normals.push_back(n);
        if(i < levels - 1)
        {
            sz.width /= 2; sz.height /= 2;
            scaled = pyrDownBilateral(scaled, sigmaDepth*depthFactor);
        }
    }

    return frame;
}

cv::Ptr<Frame> FrameGeneratorCPU::operator ()(const Points _points, const Normals _normals, int levels) const
{
    cv::Ptr<FrameCPU> frame = makePtr<FrameCPU>();

    CV_Assert(_points.type() == CV_32FC3);
    CV_Assert(_normals.type() == CV_32FC3);

    std::vector<Points>  points  = std::vector<Points>(levels);
    std::vector<Normals> normals = std::vector<Normals>(levels);
    points[0]  = _points;
    normals[0] = _normals;
    Size sz = _points.size();
    for(int i = 1; i < levels; i++)
    {
        sz.width /= 2; sz.height /= 2;
        points[i]  = Points(sz);
        normals[i] = Normals(sz);
        pyrDownPointsNormals(points[i-1], normals[i-1], points[i], normals[i]);
    }

    frame->points = points;
    frame->normals = normals;

    return frame;
}

cv::Ptr<Frame> FrameGeneratorGPU::operator ()(const Depth /*depth*/, const Intr /*intr*/, int /*levels*/, float /*depthFactor*/,
                                              float /*sigmaDepth*/, float /*sigmaSpatial*/, int /*kernelSize*/) const
{
    throw std::runtime_error("Not implemented");
}

cv::Ptr<Frame> FrameGeneratorGPU::operator ()(const Points /*_points*/, const Normals /*_normals*/, int /*levels*/) const
{
    throw std::runtime_error("Not implemented");
}

void FrameCPU::render(OutputArray image, int level, Affine3f lightPose) const
{
    typedef Points::value_type p3type;

    CV_Assert(level < (int)points.size());
    CV_Assert(level < (int)normals.size());

    Size sz = points[level].size();
    image.create(sz, CV_8UC3);
    Mat_<Vec3b> img = image.getMat();

    for(int y = 0; y < sz.height; y++)
    {
        Vec3b* imgRow = img[y];
        const p3type* ptsRow = points[level][y];
        const p3type* nrmRow = normals[level][y];

        for(int x = 0; x < sz.width; x++)
        {
            p3type p = ptsRow[x];
            p3type n = nrmRow[x];

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
                const float sp = 20.f;  //specular power

                const float Ax = 1.f;   //ambient color,  can be RGB
                const float Dx = 1.f;   //diffuse color,  can be RGB
                const float Sx = 1.f;   //specular color, can be RGB
                const float Lx = 1.f;   //light color

                Point3f l = normalize(lightPose.translation() - Vec<kftype, 3>(p));
                Point3f v = normalize(-Vec<kftype, 3>(p));
                Point3f r = normalize(Vec<kftype, 3>(2.f*n*n.dot(l) - l));

                uchar ix = (Ax*Ka*Dx + Lx*Kd*Dx*max(0.f, n.dot(l)) + Lx*Ks*Sx*pow(max(0.f, r.dot(v)), sp))*255;
                color = Vec3b(ix, ix, ix);
            }

            imgRow[x] = color;
        }
    }
}

void FrameGPU::render(OutputArray /* image */, int /*level*/, Affine3f /*lightPose*/) const
{
    throw std::runtime_error("Not implemented");
}


void pyrDownPointsNormals(const Points p, const Normals n, Points &pdown, Normals &ndown)
{
    typedef Points::value_type p3type;

    const kftype qnan = std::numeric_limits<kftype>::quiet_NaN();
    p3type nan3(qnan, qnan, qnan);

    for(int y = 0; y < pdown.rows; y++)
    {
        p3type* ptsRow = pdown[y];
        p3type* nrmRow = ndown[y];
        const p3type* pUpRow0 = p[2*y];
        const p3type* pUpRow1 = p[2*y+1];
        const p3type* nUpRow0 = n[2*y];
        const p3type* nUpRow1 = n[2*y+1];
        for(int x = 0; x < pdown.cols; x++)
        {
            p3type point = nan3, normal = nan3;

            p3type d00 = pUpRow0[2*x];
            p3type d01 = pUpRow0[2*x+1];
            p3type d10 = pUpRow1[2*x];
            p3type d11 = pUpRow1[2*x+1];

            if(!(isNaN(d00) || isNaN(d01) || isNaN(d10) || isNaN(d11)))
            {
                point = (d00 + d01 + d10 + d11)*0.25f;

                p3type n00 = nUpRow0[2*x];
                p3type n01 = nUpRow0[2*x+1];
                p3type n10 = nUpRow1[2*x];
                p3type n11 = nUpRow1[2*x+1];

                normal = (n00 + n01 + n10 + n11)*0.25f;
            }

            ptsRow[x] = point;
            nrmRow[x] = normal;
        }
    }
}


Depth pyrDownBilateral(const Depth depth, float sigma)
{
    Depth depthDown(depth.rows/2, depth.cols/2);

    float sigma3 = sigma*3;
    const int D = 5;

    for(int y = 0; y < depthDown.rows; y++)
    {
        kftype* downRow = depthDown[y];
        const kftype* srcCenterRow = depth[2*y];

        for(int x = 0; x < depthDown.cols; x++)
        {
            kftype center = srcCenterRow[2*x];

            int sx = max(0, 2*x - D/2), ex = min(2*x - D/2 + D, depth.cols-1);
            int sy = max(0, 2*y - D/2), ey = min(2*y - D/2 + D, depth.rows-1);

            kftype sum = 0;
            int count = 0;

            for(int iy = sy; iy < ey; iy++)
            {
                const kftype* srcRow = depth[iy];
                for(int ix = sx; ix < ex; ix++)
                {
                    kftype val = srcRow[ix];
                    if(abs(val - center) < sigma3)
                    {
                        sum += val; count ++;
                    }
                }
            }

            downRow[x] = (count == 0) ? 0 : sum / count;
        }
    }
    return depthDown;
}

void computePointsNormals(const Intr intr, float depthFactor, const Depth depth,
                          Points points, Normals normals)
{
    CV_Assert(!points.empty() && !normals.empty());;
    CV_Assert(depth.size() == points.size());
    CV_Assert(depth.size() == normals.size());

    typedef Points::value_type p3type;
    typedef Depth::value_type depthType;

    const kftype qnan = std::numeric_limits<kftype>::quiet_NaN ();
    p3type nan3(qnan, qnan, qnan);

    // conversion to meters
    // before it was:
    //float dfac = 0.001f/depthFactor;
    float dfac = 1.f/depthFactor;

    Intr::Reprojector reproj = intr.makeReprojector();

    for(int y = 0; y < depth.rows; y++)
    {
        const depthType* depthRow0 = depth[y];
        const depthType* depthRow1 = (y < depth.rows - 1) ? depth[y + 1] : 0;
        p3type    *ptsRow = points[y];
        p3type   *normRow = normals[y];

        for(int x = 0; x < depth.cols; x++)
        {
            depthType d00 = depthRow0[x];
            depthType z00 = d00*dfac;
            p3type v00 = reproj(p3type(x, y, z00));

            p3type p = nan3, n = nan3;

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
                    p3type v01 = reproj(p3type(x+1, y, z01));
                    p3type v10 = reproj(p3type(x, y+1, z10));

                    cv::Vec<kftype, 3> vec = (v01-v00).cross(v10-v00);
                    n = -normalize(vec);
                    p = v00;
                }
            }

            ptsRow[x] = p;
            normRow[x] = n;
        }
    }
}

