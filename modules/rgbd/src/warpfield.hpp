#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/flann.hpp"

#define DYNAFU_MAX_NEIGHBOURS 10

namespace cv {
namespace dynafu {

struct WarpNode
{
    Point3f pos;
    float radius;
    Affine3f transform;

    std::vector<Ptr<WarpNode> > children;

    float weight(Point3f x)
    {
        Point3f diff = pos - x;
        float L2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
        return expf(-L2/(2.0*radius*radius));
    }
};

typedef std::vector<Ptr<WarpNode> > NodeVectorType;

class WarpField
{
public:
    WarpField(int _maxNeighbours=1000000, int K=4, int levels=4, float baseResolution=.010f, 
              float resolutionGrowth=4);
              
    void updateNodesFromPoints(InputArray _points);

    NodeVectorType getNodes() const;
    std::vector<NodeVectorType> getGraphNodes() const;
    
    size_t getNodesLen() const 
    {
        return nodes.size();
    }

    Point3f applyWarp(Point3f p, int neighbours[4], int n) const;

    void setAllRT(Affine3f warpRT, Affine3f invCamPose);

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > getNodeIndex() const;
    int k; //k-nearest neighbours will be used

private:
    void removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind,
                         std::vector<bool>& supInd);
                         
    NodeVectorType subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind, 
                                  std::vector<bool>& supInd, float res, 
                                  Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex = nullptr);
    void constructRegGraph();

    void initTransforms(NodeVectorType nv);

    NodeVectorType nodes; //heirarchy level 0
    int maxNeighbours;

    int n_levels; // number of levels in the heirarchy
    float baseRes;
    float resGrowthRate;
    
    std::vector<NodeVectorType> regGraphNodes; // heirarchy levels 1 to L

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > nodeIndex;

    Affine3f cameraPoseInv;

};

bool PtCmp(cv::Point3f a, cv::Point3f b);
Mat getNodesPos(NodeVectorType nv);

} // namepsace dynafu
} // namespace cv

#endif
