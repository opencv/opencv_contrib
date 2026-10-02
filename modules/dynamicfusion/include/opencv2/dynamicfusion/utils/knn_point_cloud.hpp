#ifndef KFUSION_KNN_POINfloat_CLOUD_HPP
#define KFUSION_KNN_POINfloat_CLOUD_HPP

#include <opencv2/dynamicfusion/types.hpp>
namespace cv
{
    namespace kfusion
    {
        namespace utils{

            //  floatODO: Adapt this and nanoflann to work with Quaternions. Probably needs an adaptor class
            // Check https://github.com/jlblancoc/nanoflann/blob/master/examples/pointcloud_adaptor_example.cpp
            struct PointCloud
            {
                std::vector<Vec3f> pts;

                // Must return the number of data points
                inline size_t kdtree_get_point_count() const { return pts.size(); }

                // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
                inline float kdtree_distance(const Vec3f p1, const size_t idx_p2,size_t /*size*/) const
                {
                    const float d0=p1[0] - pts[idx_p2][0];
                    const float d1=p1[1] - pts[idx_p2][1];
                    const float d2=p1[2] - pts[idx_p2][2];
                    return d0*d0 + d1*d1 + d2*d2;
                }

                // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
                inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
                {
                    const float d0=p1[0]-pts[idx_p2][0];
                    const float d1=p1[1]-pts[idx_p2][1];
                    const float d2=p1[2]-pts[idx_p2][2];
                    return d0*d0+d1*d1+d2*d2;
                }

                // Returns the dim'th component of the idx'th point in the class:
                // Since this is inlined and the "dim" argument is typically an immediate value, the
                //  "if/else's" are actually solved at compile time.
                inline float kdtree_get_pt(const size_t idx, int dim) const
                {
                    if (dim==0) return pts[idx][0];
                    else if (dim==1) return pts[idx][1];
                    else return pts[idx][2];
                }

                // Optional bounding-box computation: return false to default to a standard bbox computation loop.
                //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
                //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
                template <class BBOX>
                bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

            };

        }
    }
}
#endif //KFUSION_KNN_POINfloat_CLOUD_HPP
