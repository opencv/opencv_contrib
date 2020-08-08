#ifndef OPENCV_RGBD_GRAPH_NODE_H
#define OPENCV_RGBD_GRAPH_NODE_H

#include <map>
#include <unordered_map>

#include "opencv2/core/affine.hpp"
#include "sparse_block_matrix.hpp"
namespace cv
{
namespace kinfu
{
/*! \class GraphNode
 *  \brief Defines a node/variable that is optimizable in a posegraph
 *
 *  Detailed description
 */
struct PoseGraphNode
{
   public:
    explicit PoseGraphNode(int _nodeId, const Affine3f& _pose) : nodeId(_nodeId), isFixed(false), pose(_pose) {}
    virtual ~PoseGraphNode() = default;

    int getId() const { return nodeId; }
    Affine3f getPose() const { return pose; }
    void setPose(const Affine3f& _pose) { pose = _pose; }
    void setFixed(bool val = true) { isFixed = val; }

   private:
    int nodeId;
    bool isFixed;
    Affine3f pose;
};

/*! \class PoseGraphEdge
 *  \brief Defines the constraints between two PoseGraphNodes
 *
 *  Detailed description
 */
struct PoseGraphEdge
{
   public:
    PoseGraphEdge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
                  const Matx66f& _information = Matx66f::eye())
        : sourceNodeId(_sourceNodeId),
          targetNodeId(_targetNodeId),
          transformation(_transformation),
          information(_information)
    {
    }
    virtual ~PoseGraphEdge() = default;

    int getSourceNodeId() const { return sourceNodeId; }
    int getTargetNodeId() const { return targetNodeId; }

   public:
    int sourceNodeId;
    int targetNodeId;
    Affine3f transformation;
    Matx66f information;
};

//! @brief Reference: A tutorial on SE(3) transformation parameterizations and on-manifold optimization
//! Jose Luis Blanco
//! Compactly represents the jacobian of the SE3 generator
// clang-format off
static const std::array<Matx44f, 6> generatorJacobian = {
    // alpha
    Matx44f(0, 0,  0, 0,
            0, 0, -1, 0,
            0, 1,  0, 0,
            0, 0,  0, 0),
    // beta
    Matx44f( 0, 0, 1, 0,
             0, 0, 0, 0,
            -1, 0, 0, 0,
             0, 0, 0, 0),
    // gamma
    Matx44f(0, -1, 0, 0,
            1,  0, 0, 0,
            0,  0, 0, 0,
            0,  0, 0, 0),
    // x
    Matx44f(0, 0, 0, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0),
    // y
    Matx44f(0, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 0,
            0, 0, 0, 0),
    // z
    Matx44f(0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 0, 0)
};
// clang-format on

class PoseGraph
{
   public:
    typedef std::vector<PoseGraphNode> NodeVector;
    typedef std::vector<PoseGraphEdge> EdgeVector;

    explicit PoseGraph(){};
    virtual ~PoseGraph() = default;

    //! PoseGraph can be copied/cloned
    PoseGraph(const PoseGraph& _poseGraph) = default;
    PoseGraph& operator=(const PoseGraph& _poseGraph) = default;

    void addNode(const PoseGraphNode& node) { nodes.push_back(node); }
    void addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

    bool nodeExists(int nodeId) const
    {
        return std::find_if(nodes.begin(), nodes.end(),
                            [nodeId](const PoseGraphNode& currNode) { return currNode.getId() == nodeId; }) != nodes.end();
    }

    bool isValid() const;

    PoseGraph update(const Mat& delta);

    int getNumNodes() const { return nodes.size(); }
    int getNumEdges() const { return edges.size(); }

    Mat getVector();

    //! @brief: Constructs a linear system and returns the residual of the current system
    float createLinearSystem(BlockSparseMat<float, 6, 6>& H, Mat& B);

   private:
    NodeVector nodes;
    EdgeVector edges;
};

namespace Optimizer
{
struct Params
{
    int maxNumIters;
    float minResidual;
    float maxAcceptableResIncre;

    // TODO: Refine these constants
    Params() : maxNumIters(40), minResidual(1e-3f), maxAcceptableResIncre(1e-2f){};
    virtual ~Params() = default;
};

void optimizeGaussNewton(const Params& params, PoseGraph& poseGraph);
void optimizeLevenberg(const Params& params, PoseGraph& poseGraph);
}  // namespace Optimizer

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
