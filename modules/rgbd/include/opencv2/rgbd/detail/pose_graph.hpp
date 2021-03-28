#ifndef OPENCV_RGBD_GRAPH_NODE_H
#define OPENCV_RGBD_GRAPH_NODE_H

#include <map>
#include <unordered_map>

#include "opencv2/core/affine.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "opencv2/core/eigen.hpp"
#endif

#include "sparse_block_matrix.hpp"
#include "opencv2/core/quaternion.hpp"

namespace cv
{
namespace kinfu
{
namespace detail
{

// ATTENTION! This class is used internally in Large KinFu.
// It has been pushed to publicly available headers for tests only.
// Source compatibility of this API is not guaranteed in the future.

// This class provides tools to solve so-called pose graph problem often arisen in SLAM problems
// The pose graph format, cost function and optimization techniques
// repeat the ones used in Ceres 3D Pose Graph Optimization:
// http://ceres-solver.org/nnls_tutorial.html#other-examples, pose_graph_3d.cc bullet
class CV_EXPORTS_W PoseGraph
{
public:
    CV_WRAP static Ptr<PoseGraph> create();
    virtual ~PoseGraph() = 0;

    // Node may have any id >= 0
    virtual void addNode(size_t _nodeId, const Affine3d& _pose, bool fixed) = 0;
    virtual bool isNodeExist(size_t nodeId) const = 0;
    virtual bool setNodeFixed(size_t nodeId, bool fixed) = 0;
    virtual bool isNodeFixed(size_t nodeId) = 0;
    virtual Affine3d getNodePose(size_t nodeId) = 0;
    virtual std::vector<size_t> getNodesIds() const = 0;
    virtual size_t getNumNodes() const = 0;

    // Edges have consequent indices starting from 0
    virtual void addEdge(size_t _sourceNodeId, size_t _targetNodeId, const Affine3f& _transformation,
                         const Matx66f& _information = Matx66f::eye()) = 0;
    virtual size_t getEdgeStart(size_t i) const = 0;
    virtual size_t getEdgeEnd(size_t i) const = 0;
    virtual size_t getNumEdges() const = 0;

    // checks if graph is connected and each edge connects exactly 2 nodes
    virtual bool isValid() const = 0;

    // Termination criteria are max number of iterations and min relative energy change to current energy
    // Returns number of iterations elapsed or -1 if max number of iterations was reached or failed to optimize
    virtual int optimize(const cv::TermCriteria& tc = cv::TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-6)) = 0;

    // calculate cost function based on current nodes parameters
    virtual double calcEnergy() const = 0;
};

}  // namespace detail
}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
