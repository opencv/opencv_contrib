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
    WarpNode():
        pos(), radius(), transform(), place((size_t)-1), cachedJac()
    {}

    WarpNode(const WarpNode& wn) = default;

    float weight(Point3f x) const
    {
        Point3f diff = pos - x;
        float norm2 = diff.dot(diff);
        return expf(-norm2/(2.f*radius));
    }

    // Returns transform applied to node's center
    UnitDualQuaternion centeredRt() const
    {
        return transform.centered(pos);
    }

    // node's center
    Point3f pos;
    float radius;
    UnitDualQuaternion transform;
    // the rest fields are used at optimization stage
    // depending on parametrization, this field keeps:
    // - rotation in axis-by-angle format and translation
    // - dual and real part of dual quaternion logarithm
    cv::Vec3f arg[2];
    // where it is in params vector
    size_t place;
    // cached jacobian
    cv::Matx<float, 8, 6> cachedJac;
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
        std::vector<float> query = { queryPt.x, queryPt.y, queryPt.z };
        nodeIndex->knnSearch(query, indices, dists, k, cvflann::SearchParams());

        //TODO URGENT: preprocessing to get arrays with -1's

        throw "here we stop for debugging";
    }

    const std::vector<std::vector<NodeNeighboursType> >& getRegGraph() const
    {
        return hierarchy;
    }

    const std::vector<Ptr<WarpNode> >& getNodes() const
    {
        return nodes;
    }

    const std::vector<std::vector<Ptr<WarpNode> > >& getGraphNodes() const
    {
        return regGraphNodes;
    }

    // used at LevMarq
    void setNodes(const std::vector<Ptr<WarpNode>>& n)
    {
        nodes = n;
    }

    void setRegNodes(const std::vector<std::vector<Ptr<WarpNode>>>& g)
    {
        regGraphNodes = g;
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

    void setAllRT(Affine3f warpRt)
    {
        UnitDualQuaternion u(warpRt);
        for (auto n : nodes)
        {
            n->transform = u;
        }
    }

    std::vector<Ptr<WarpNode>> cloneNodes()
    {
        const auto& wn = getNodes();
        std::vector<Ptr<WarpNode>> v;
        v.reserve(wn.size());
        for (const auto& p : wn)
        {
            v.push_back(std::make_shared<WarpNode>(*p));
        }
        return v;
    }

    std::vector<std::vector<Ptr<WarpNode>>> cloneGraphNodes()
    {
        const auto& wgn = getGraphNodes();
        std::vector<std::vector<Ptr<WarpNode>>> vv;
        vv.reserve(wgn.size());
        for (const auto& vp : wgn)
        {
            std::vector<Ptr<WarpNode>> v;
            v.reserve(vp.size());
            for (const auto& p : vp)
            {
                v.push_back(std::make_shared<WarpNode>(*p));
            }
            vv.push_back(v);
        }
        return vv;
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
