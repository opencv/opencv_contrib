#ifndef OPENCV_RGBD_GRAPH_NODE_H
#define OPENCV_RGBD_GRAPH_NODE_H

#include "opencv2/core/affine.hpp"
#include <map>
#include <unordered_map>
namespace cv
{
namespace kinfu
{
/*!
 * \class BlockSparseMat
 * Naive implementation of Sparse Block Matrix
 */
template<typename _Tp, int blockM, int blockN>
struct BlockSparseMat
{
    typedef std::unordered_map<Point2i, Matx<_Tp, blockM, blockN>> IDtoBlockValueMap;
    static constexpr int blockSize = blockM * blockN;
    BlockSparseMat(int _nBlocks) :
        nBlocks(_nBlocks), ijValue()
    { }

    Matx66f& refBlock(int i, int j)
    {
        Point2i p(i, j);
        auto it = ijValue.find(p);
        if (it == ijValue.end())
        {
            it = ijValue.insert({ p, Matx<_Tp, blockM, blockN>()}).first;
        }
        return it->second;
    }

    float& refElem(int i, int j)
    {
        Point2i ib(i / blockSize, j / blockSize), iv(i % blockSize, j % blockSize);
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

    size_t nonZeroBlocks() const
    {
        return ijValue.size();
    }

    int nBlocks;
    IDtoBlockValueMap ijValue;
};


/*! \class GraphNode
 *  \brief Defines a node/variable that is optimizable in a posegraph
 *
 *  Detailed description
 */
struct PoseGraphNode
{
   public:
    explicit PoseGraphNode(int nodeId, const Affine3f& _pose) : pose(_pose) {}
    virtual ~PoseGraphNode();

    int getId() const { return nodeId; }
    Affine3f getPose() const { return pose; }
    void setPose(const Affine3f& _pose) { pose = _pose; }
   private:
    int nodeId;
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
        : sourceNodeId(_sourceNodeId), targetNodeId(_targetNodeId), transformation(_transformation), information(_information)
    {}
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
static const std::array<Affine3f, 6> generatorJacobian =
{   //alpha
    Affine3f(Matx44f(0, 0,  0, 0,
                     0, 0, -1, 0,
                     0, 1,  0, 0,
                     0, 0,  0, 0)),
    //beta
    Affine3f(Matx44f( 0, 0, 1, 0,
                      0, 0, 0, 0,
                     -1, 0, 0, 0,
                      0, 0, 0, 0)),
    //gamma
    Affine3f(Matx44f(0, -1, 0, 0,
                     1,  0, 0, 0,
                     0,  0, 0, 0,
                     0,  0, 0, 0)),
    //x
    Affine3f(Matx44f(0, 0, 0, 1,
                     0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 0)),
    //y
    Affine3f(Matx44f(0, 0, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 0, 0,
                     0, 0, 0, 0)),
    //z
    Affine3f(Matx44f(0, 0, 0, 0,
                     0, 0, 0, 0,
                     0, 0, 0, 1,
                     0, 0, 0, 0))
};

class PoseGraph
{
    public:
        typedef std::vector<PoseGraphNode> NodeVector;
        typedef std::vector<PoseGraphEdge> EdgeVector;

        PoseGraph() {};
        virtual ~PoseGraph() = default;

        void addNode(const PoseGraphNode& node);
        void addEdge(const PoseGraphEdge& edge);

        bool nodeExists(int nodeId);

        bool isValid();
        static cv::Ptr<PoseGraph> updatePoseGraph(const PoseGraph& poseGraphPrevious, const Mat& delta)
        static void optimize(PoseGraph& poseGraph, int numIters = 10, float min_residual = 1e-6);

    private:
        //! @brief: Constructs a linear system and returns the residual of the current system
        float createLinearSystem(BlockSparseMat<float, 6, 6>& H, Mat& B);

        NodeVector nodes;
        EdgeVector edges;

};  // namespace large_kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
