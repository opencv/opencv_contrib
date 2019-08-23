#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/flann.hpp"
#include "dqb.hpp"

#define DYNAFU_MAX_NEIGHBOURS 10
typedef std::array<int, DYNAFU_MAX_NEIGHBOURS> nodeNeighboursType;
typedef std::vector<std::vector<nodeNeighboursType> > heirarchyType;

namespace cv {
namespace dynafu {

struct WarpNode
{
    Point3f pos;
    float radius;
    Affine3f transform;

    float weight(Point3f x)
    {
        Point3f diff = pos - x;
        float L2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
        return expf(-L2/(2.f*radius));
    }
};

typedef std::vector<Ptr<WarpNode> > NodeVectorType;

class WarpField
{
public:
    WarpField(int _maxNeighbours=1000000, int K=4, int levels=4, float baseResolution=.10f,
              float resolutionGrowth=4);

    void updateNodesFromPoints(InputArray _points);

    const NodeVectorType& getNodes() const;
    const heirarchyType& getRegGraph() const;
    const std::vector<NodeVectorType>& getGraphNodes() const;

    size_t getNodesLen() const
    {
        return nodes.size();
    }

    Point3f applyWarp(Point3f p, const nodeNeighboursType neighbours, int n, bool normal=false) const;

    void setAllRT(Affine3f warpRT);

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > getNodeIndex() const;

    inline void findNeighbours(Point3f queryPt, std::vector<int>& indices, std::vector<float>& dists)
    {
        std::vector<float> query = {queryPt.x, queryPt.y, queryPt.z};
        nodeIndex->knnSearch(query, indices, dists, k, cvflann::SearchParams());
    }

    int k; //k-nearest neighbours will be used
    int n_levels; // number of levels in the heirarchy

private:
    void removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind, AutoBuffer<bool>& supInd);

    NodeVectorType subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind,
                                  AutoBuffer<bool>& supInd, float res,
                                  Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex = nullptr);
    void constructRegGraph();

    void initTransforms(NodeVectorType nv);

    NodeVectorType nodes; //heirarchy level 0
    int maxNeighbours;

    float baseRes;
    float resGrowthRate;

    std::vector<NodeVectorType> regGraphNodes; // heirarchy levels 1 to L
    heirarchyType heirarchy;

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > nodeIndex;

    Mat nodesPos;

};

bool PtCmp(cv::Point3f a, cv::Point3f b);
Mat getNodesPos(NodeVectorType nv);

} // namepsace dynafu
} // namespace cv

#endif
