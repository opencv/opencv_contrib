#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/affine.hpp"
#include <flann/flann.hpp>

namespace cv {
namespace dynafu {

struct CV_EXPORTS_W WarpNode {
    Point3f pos;
    float radius;
};

class CV_EXPORTS_W WarpField {
public:
    WarpField();
    void updateNodesFromPoints(InputArray _points, float resolution);

    std::vector<Ptr<WarpNode> > getNodes();

private:
    std::vector<Ptr<WarpNode> > nodes;
    void removeSupported(::flann::KDTreeSingleIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& supInd);
    void subsampleIndex(::flann::KDTreeSingleIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& supInd, float res);
    void appendNodeFromCluster(float res, Point3f p);
};

} // namepsace dynafu
} // namespace cv

#endif
