#pragma once

#include <kfusion/cuda/device_array.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <iosfwd>

struct CUevent_st;

namespace kfusion
{
    typedef cv::Matx33f Mat3f;
    typedef cv::Matx44f Mat4f;
    typedef cv::Vec3f Vec3f;
    typedef cv::Vec4f Vec4f;
    typedef cv::Vec3i Vec3i;
    typedef cv::Affine3f Affine3f;

    struct KF_EXPORTS Intr
    {
        float fx, fy, cx, cy;

        Intr ();
        Intr (float fx, float fy, float cx, float cy);
        Intr operator()(int level_index) const;
    };

    KF_EXPORTS std::ostream& operator << (std::ostream& os, const Intr& intr);

    struct Point
    {
        union
        {
            float data[4];
            struct { float x, y, z; };
        };
    };

    typedef Point Normal;

    struct RGB
    {
        union
        {
            struct { unsigned char b, g, r; };
            int bgra;
        };
    };

    struct PixelRGB
    {
        unsigned char r, g, b;
    };

    namespace cuda
    {
        typedef cuda::DeviceMemory CudaData;
        typedef cuda::DeviceArray2D<unsigned short> Depth;
        typedef cuda::DeviceArray2D<unsigned short> Dists;
        typedef cuda::DeviceArray2D<RGB> Image;
        typedef cuda::DeviceArray2D<Normal> Normals;
        typedef cuda::DeviceArray2D<Point> Cloud;

        struct Frame
        {
            bool use_points;

            std::vector<Depth> depth_pyr;
            std::vector<Cloud> points_pyr;
            std::vector<Normals> normals_pyr;
        };
    }

    inline float deg2rad (float alpha) { return alpha * 0.017453293f; }

    struct KF_EXPORTS ScopeTime
    {
        const char* name;
        double start;
        ScopeTime(const char *name);
        ~ScopeTime();
    };

    struct KF_EXPORTS SampledScopeTime
    {
    public:
        enum { EACH = 33 };
        SampledScopeTime(double& time_ms);
        ~SampledScopeTime();
    private:
        double getTime();
        SampledScopeTime(const SampledScopeTime&);
        SampledScopeTime& operator=(const SampledScopeTime&);

        double& time_ms_;
        double start;
    };

}
