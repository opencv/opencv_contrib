#pragma once

#include <opencv2/dynamicfusion/types.hpp>

namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {
             void depthBilateralFilter(const Depth& in, Depth& out, int ksz, float sigma_spatial, float sigma_depth);

             void depthTruncation(Depth& depth, float threshold);

             void depthBuildPyramid(const Depth& depth, Depth& pyramid, float sigma_depth);

             void computeNormalsAndMaskDepth(const Intr& intr, Depth& depth, Normals& normals);

             void computePointNormals(const Intr& intr, const Depth& depth, Cloud& points, Normals& normals);

             void computeDists(const Depth& depth, Dists& dists, const Intr& intr);

             void cloudToDepth(const Cloud& cloud, Depth& depth);

             void resizeDepthNormals(const Depth& depth, const Normals& normals, Depth& depth_out, Normals& normals_out);

             void resizePointsNormals(const Cloud& points, const Normals& normals, Cloud& points_out, Normals& normals_out);

             void waitAllDefaultStream();

             void renderTangentColors(const Normals& normals, Image& image);

             void renderImage(const Depth& depth, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image);

             void renderImage(const Cloud& points, const Normals& normals, const Intr& intr, const Vec3f& light_pose, Image& image);
        }
    }
}
