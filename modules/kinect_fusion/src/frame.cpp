//TODO: add license

#include "precomp.hpp"

using namespace cv;
using namespace cv::kinfu;

Frame::Frame(int levels, cv::Size frameSize) : pose(cv::Affine3f::Identity())
{
    points.reserve(levels);
    normals.reserve(levels);

    cv::Size sz = frameSize;
    for(int i = 0; i < levels; i++)
    {
        Points p(sz);
        Normals n(sz);

        points.push_back(p);
        normals.push_back(n);
        sz.width /= 2; sz.height /= 2;
    }
}

void Frame::computePointsNormals(const std::vector<Depth>& pyramid, Intr intrinsics, float depthFactor)
{
    for (int i = 0; i < params.pyramidLevels; ++i)
    {
        // scale intrinsics to pyramid level
        Intr intr = intrinsics.scale(i);

        computePointsNormals(intr, pyramid[i], points[i], normals[i], depthFactor);
    }
}


Distance depthToDistance(Depth depth, Intr intr, float depthFactor)
{
    float cx = intr.cx, cy = intr.cy;
    float fxinv = 1.f/intr.fx, fyinv = 1.f/intr.fy;

    //TODO: find out why 0.001f for meters
    //TODO: is this procedure necessary? why not just do d/depthFactor ?
    float dfac = 0.001f / depthFactor;

    Distance distance(depth.size());
    for(int y = 0; y < depth.rows; y++)
    {
        const Depth::value_type* depthRow = depth.ptr(y);
        Distance::value_type* distRow = distance.ptr(y);
        for(int x = 0; x < depth.cols; x++)
        {
            Depth::value_type d = depthRow[x];

            float xl = (x - cx) * fxinv;
            float yl = (y - cy) * fyinv;
            float lambda = sqrt (xl * xl + yl * yl + 1);

            distRow[x] = d * lambda * dfac;
        }
    }
    return distance;
}


void computePointsNormals(Intr intr, const Depth& depth, Points& points, Normals& normals, float depthFactor)
{
    CV_Assert(depth.size() == points.size && depth.size() == normals.size());

    typedef Points::value_type p3type;
    typedef DataType<p3type>::value_type ptype;
    typedef Depth::value_type dtype;

    const ptype qnan = numeric_limits<ptype>::quiet_NaN ();
    p3type nan3(qnan, qnan, qnan);

    // conversion to meters
    // TODO: do we need it?
    float dfac = 0.001f/depthFactor;

    Intr::Reprojector reproj = intr.makeReprojector();
    for(int y = 0; y < depth.rows - 1; y++)
    {
        const dtype* depthRow0 = depth.ptr(y);
        const dtype* depthRow1 = depth.ptr(y + 1);
        p3type * ptsRow = points.ptr(y);
        p3type *normRow = normals.ptr(y);
        for(int x = 0; x < depth.cols - 1; x++)
        {
            p3type p = nan3, n = nan3;

            dtype z00 = depthRow0[x  ]*dfac;
            dtype z01 = depthRow0[x+1]*dfac;
            dtype z10 = depthRow1[x  ]*dfac;

            //if(z00*z01*z10 != 0)
            if(z00 != 0 && z01 != 0 && z10 != 0)
            {
                p3type v00 = reproj(p3type(x,   y, z00));
                p3type v01 = reproj(p3type(x+1, y, z01));
                p3type v10 = reproj(p3type(x, y+1, z10));

                n = -normalize((v01-v00).cross(v10-v00));
                p = v00;
            }

            ptsRow[x] = p;
            normRow[x] = n;
        }
        // fill last column by NaNs
        ptsRow [depth.cols - 1] = nan3;
        normRow[depth.cols - 1] = nan3;
    }
    // fill last row by NaNs
    p3type *ptsRow =  points.ptr(depth.rows - 1);
    p3type *normRow = normals.ptr(depth.rows - 1);
    for(int x = 0; x < depth.cols; x++)
    {
        ptsRow[x]  = nan3;
        normRow[x] = nan3;
    }
}

