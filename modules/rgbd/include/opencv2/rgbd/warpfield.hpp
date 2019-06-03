#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/affine.hpp"
#include <flann/flann.hpp>

namespace cv {
namespace dynafu {

typedef float RadiusType;
struct CV_EXPORTS_W WarpNode {
    Point3f pos;
    float radius;
};

class CV_EXPORTS_W WarpField {
public:
    WarpField();
    void updateNodesFromPoints(InputArray _points, float resolution);

private:
    std::vector<Ptr<WarpNode> > nodes;
    void removeSupported(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& supInd);
    void subsampleIndex(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& supInd, float res);
    void appendNodeFromCluster(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<int> indices, float res);
};

} // namepsace dynafu
} // namespace cv

#endif