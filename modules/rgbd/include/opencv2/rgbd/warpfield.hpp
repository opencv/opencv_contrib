#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/affine.hpp"
#include "opencv2/flann.hpp"

namespace cv {
namespace dynafu {

struct CV_EXPORTS_W WarpNode
{
    Point3f pos;
    float radius;
    std::vector<Ptr<WarpNode> > children;
};

typedef std::vector<Ptr<WarpNode> > NodesLevelType;
typedef std::vector<std::vector<int> > EdgesLevelType;

class CV_EXPORTS_W WarpField
{
public:
    WarpField(int _maxNeighbours=1000000, int K=4, int levels=4, float baseResolution=0.025, float resolutionGrowth=2);
    void updateNodesFromPoints(InputArray _points);

    std::vector<Ptr<WarpNode> > getNodes();
    std::vector<NodesLevelType> getGraphNodes();

private:
    void removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind, std::vector<bool>& supInd);
    std::vector<Ptr<WarpNode> > subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind, std::vector<bool>& supInd, 
        float res, Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex = nullptr);
    void constructRegGraph();

    std::vector<Ptr<WarpNode> > nodes; //heirarchy level 0
    int maxNeighbours;

    int k; //k-nearest neighbours will be used
    int n_levels; // number of levels in the heirarchy
    float baseRes;
    float resGrowthRate;
    
    std::vector<NodesLevelType> regGraphNodes; // heirarchy levels 1 to L
};

} // namepsace dynafu
} // namespace cv

#endif
