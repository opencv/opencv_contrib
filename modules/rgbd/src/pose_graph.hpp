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
#include "opencv2/core/dualquaternion.hpp"

namespace cv
{
namespace kinfu
{

class PoseGraph
{
public:
    struct Pose3d
    {
        Vec3d t;
        Quatd q;

        Pose3d() : t(), q(1, 0, 0, 0) { }

        Pose3d(const Matx33d& rotation, const Vec3d& translation)
            : t(translation), q(Quatd::createFromRotMat(rotation).normalize())
        { }

        explicit Pose3d(const Matx44d& pose) :
            Pose3d(pose.get_minor<3, 3>(0, 0), Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)))
        { }

        inline Pose3d operator*(const Pose3d& otherPose) const
        {
            Pose3d out(*this);
            out.t += q.toRotMat3x3(QUAT_ASSUME_UNIT) * otherPose.t;
            out.q = out.q * otherPose.q;
            return out;
        }

        Affine3d getAffine() const
        {
            return Affine3d(q.toRotMat3x3(QUAT_ASSUME_UNIT), t);
        }

        inline Pose3d inverse() const
        {
            Pose3d out;
            out.q = q.conjugate();
            out.t = -(out.q.toRotMat3x3(QUAT_ASSUME_UNIT) * t);
            return out;
        }

        inline void normalizeRotation()
        {
            q = q.normalize();
        }
    };

    /*! \class GraphNode
     *  \brief Defines a node/variable that is optimizable in a posegraph
     *
     *  Detailed description
     */
    struct Node
    {
    public:
        explicit Node(int _nodeId, const Affine3d& _pose)
            : nodeId(_nodeId), isFixed(false), pose(_pose.rotation(), _pose.translation())
        { }
        virtual ~Node() = default;

        int getId() const { return nodeId; }
        inline Affine3d getPose() const
        {
            return pose.getAffine();
        }
        void setPose(const Affine3d& _pose)
        {
            pose = Pose3d(_pose.rotation(), _pose.translation());
        }
        void setPose(const Pose3d& _pose)
        {
            pose = _pose;
        }
        void setFixed(bool val = true) { isFixed = val; }
        bool isPoseFixed() const { return isFixed; }

    public:
        int nodeId;
        bool isFixed;
        Pose3d pose;
    };

    /*! \class PoseGraphEdge
     *  \brief Defines the constraints between two PoseGraphNodes
     *
     *  Detailed description
     */
    struct Edge
    {
    public:
        Edge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
             const Matx66f& _information = Matx66f::eye());

        virtual ~Edge() = default;

        int getSourceNodeId() const { return sourceNodeId; }
        int getTargetNodeId() const { return targetNodeId; }

        bool operator==(const Edge& edge)
        {
            if ((edge.getSourceNodeId() == sourceNodeId && edge.getTargetNodeId() == targetNodeId) ||
                (edge.getSourceNodeId() == targetNodeId && edge.getTargetNodeId() == sourceNodeId))
                return true;
            return false;
        }

    public:
        int sourceNodeId;
        int targetNodeId;
        Pose3d pose;
        Matx66f sqrtInfo;
    };

   public:
    explicit PoseGraph() {};
    virtual ~PoseGraph() = default;

    //! PoseGraph can be copied/cloned
    PoseGraph(const PoseGraph&) = default;
    PoseGraph& operator=(const PoseGraph&) = default;

    // can be used for debugging
    PoseGraph(const std::string& g2oFileName);
    void readG2OFile(const std::string& g2oFileName);
    void writeToObjFile(const std::string& objFname) const;

    void addNode(const Node& node)
    {
        int id = node.getId();
        const auto& it = nodes.find(id);
        if (it != nodes.end())
        {
            std::cout << "duplicated node, id=" << id << std::endl;
            nodes.insert(it, { id, node });
        }
        else
        {
            nodes.insert({ id, node });
        }
    }
    void addEdge(const Edge& edge) { edges.push_back(edge); }

    bool nodeExists(int nodeId) const
    {
        return (nodes.find(nodeId) != nodes.end());
    }

    bool isValid() const;

    int getNumNodes() const { return int(nodes.size()); }
    int getNumEdges() const { return int(edges.size()); }

   public:

    void optimize();

    // used during optimization
    // nodes is a set of parameters to be used instead of contained in the graph
    double calcEnergy(const std::map<int, Node>& newNodes) const;

    std::map<int, Node> nodes;
    std::vector<Edge>   edges;
};

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
