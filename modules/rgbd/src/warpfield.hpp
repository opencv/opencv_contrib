#ifndef __OPENCV_RGBD_WARPFIELD_HPP__
#define __OPENCV_RGBD_WARPFIELD_HPP__

#include "opencv2/core.hpp"
#include "opencv2/flann.hpp"
#include "dqb.hpp"

constexpr size_t DYNAFU_MAX_NEIGHBOURS = 10;
typedef std::array<size_t, DYNAFU_MAX_NEIGHBOURS> NodeNeighboursType;

namespace cv {
namespace dynafu {

struct WarpNode
{
    // node's center
    Point3f pos;
    float radius;
    DualQuaternion transform;
    // where it is in params vector
    int place;
    // cached jacobian
    cv::Matx<float, 8, 6> cachedJac;

    WarpNode():
        pos(), radius(), transform(), place(-1), cachedJac()
    {}

    float weight(Point3f x) const
    {
        Point3f diff = pos - x;
        float L2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
        return expf(-L2/(2.f*radius));
    }

    //TODO URGENT: make dq.centered() instead
    // Returns transform applied to node's center
    Affine3f rt_centered() const
    {
        Affine3f r = Affine3f().translate(-pos)
                               .rotate(transform.rotation())
                               .translate(pos)
                               .translate(transform.translation());
        return r;
    }
};


class WarpField
{
public:
    WarpField(int _maxNeighbours=1000000, int K=4, size_t levels=4, float baseResolution=.10f,
              float resolutionGrowth=4) :
        k(K), n_levels(levels),
        nodes(), maxNeighbours(_maxNeighbours), // good amount for dense kinfu pointclouds
        baseRes(baseResolution),
        resGrowthRate(resolutionGrowth),
        regGraphNodes(n_levels - 1),
        hierarchy(n_levels - 1),
        nodeIndex(nullptr)
    {
        CV_Assert(k <= DYNAFU_MAX_NEIGHBOURS);
    }

    void updateNodesFromPoints(InputArray _points);

    Point3f applyWarp(Point3f p, const NodeNeighboursType neighbours, int n, bool normal = false) const;

    void findNeighbours(Point3f queryPt, std::vector<int>& indices, std::vector<float>& dists) const
    {
        //TODO URGENT: preprocessing to get arrays with -1's
        std::vector<float> query = { queryPt.x, queryPt.y, queryPt.z };
        nodeIndex->knnSearch(query, indices, dists, k, cvflann::SearchParams());
    }

    const std::vector<Ptr<WarpNode> >& getNodes() const
    {
        return nodes;
    }

    const std::vector<std::vector<NodeNeighboursType> >& getRegGraph() const
    {
        return hierarchy;
    }

    const std::vector<std::vector<Ptr<WarpNode> > >& getGraphNodes() const
    {
        return regGraphNodes;
    }

    size_t getNodesLen() const
    {
        return nodes.size();
    }

    size_t getRegNodesLen() const
    {
        size_t len = 0;
        for (auto level : regGraphNodes)
        {
            len += level.size();
        }
        return len;
    }

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > getNodeIndex() const
    {
        return nodeIndex;
    }  

    void setAllRT(Affine3f warpRT)
    {
        for (auto n : nodes)
        {
            n->transform = warpRT;
        }
    }

    int k; //k-nearest neighbours will be used
    size_t n_levels; // number of levels in the heirarchy

private:
    void removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind, AutoBuffer<bool>& supInd);

    std::vector<Ptr<WarpNode> > subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind,
                                               AutoBuffer<bool>& supInd, float res,
                                               Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex = nullptr);
    void constructRegGraph();

    void initTransforms(std::vector<Ptr<WarpNode> > nv);

    std::vector<Ptr<WarpNode> > nodes; //hierarchy level 0
    int maxNeighbours;

    // Starting resolution for 0th level building
    float baseRes;
    // Scale increase between hierarchy levels
    float resGrowthRate;

    /*
    Regularization graph nodes level by level, from coarse to fine
    Excluding level 0
    */
    std::vector<std::vector<Ptr<WarpNode> > > regGraphNodes;
    /*
    Regularization graph structure
    for each node of each level: nearest neighbours indices among nodes on next level
    */
    std::vector<std::vector<NodeNeighboursType> > hierarchy;

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > nodeIndex;

    Mat nodesPos;

};

bool PtCmp(cv::Point3f a, cv::Point3f b);
Mat getNodesPos(const std::vector<Ptr<WarpNode> >& nv);

} // namepsace dynafu
} // namespace cv

#endif
