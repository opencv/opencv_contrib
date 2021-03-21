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
/*! \class GraphNode
 *  \brief Defines a node/variable that is optimizable in a posegraph
 *
 *  Detailed description
 */



// TODO: put it to PoseGraph as a subtype
struct Pose3d
{
    Vec3d t;
    Vec4d vq;

    Pose3d() : t(), vq(1, 0, 0, 0) { }

    Pose3d(const Matx33d& rotation, const Vec3d& translation)
        : t(translation)
    {
        vq = Quatd::createFromRotMat(rotation).normalize().toVec();
    }

    explicit Pose3d(const Matx44d& pose) :
        Pose3d(pose.get_minor<3, 3>(0, 0), Vec3d(pose(0, 3), pose(1, 3), pose(2, 3)))
    { }

    // NOTE: Eigen overloads quaternion multiplication appropriately
    inline Pose3d operator*(const Pose3d& otherPose) const
    {
        Pose3d out(*this);
        out.t += Quatd(vq).toRotMat3x3(QUAT_ASSUME_UNIT) * otherPose.t;
        out.vq = (Quatd(out.vq) * Quatd(otherPose.vq)).toVec();
        return out;
    }

    Affine3d getAffine() const
    {
        return Affine3d(Quatd(vq).toRotMat3x3(QUAT_ASSUME_UNIT), t);
    }

    Quatd getQuat() const
    {
        return Quatd(vq);
    }

    inline Pose3d inverse() const
    {
        Pose3d out;
        out.vq = Quatd(vq).conjugate().toVec();
        out.t = - (Quatd(out.vq).toRotMat3x3(QUAT_ASSUME_UNIT) * t);
        return out;
    }

    inline void normalizeRotation()
    {
        vq = Quatd(vq).normalize().toVec();
    }
};

// TODO: put it to PoseGraph as a subtype
struct PoseGraphNode
{
   public:
    explicit PoseGraphNode(int _nodeId, const Affine3d& _pose)
        : nodeId(_nodeId), isFixed(false)
    {
        se3Pose = Pose3d(_pose.rotation(), _pose.translation());
    }
    virtual ~PoseGraphNode() = default;

    int getId() const { return nodeId; }
    inline Affine3d getPose() const
    {
        return se3Pose.getAffine();
    }
    void setPose(const Affine3d& _pose)
    {
        se3Pose = Pose3d(_pose.rotation(), _pose.translation());
    }
    void setPose(const Pose3d& _pose)
    {
        se3Pose = _pose;
    }
    void setFixed(bool val = true) { isFixed = val; }
    bool isPoseFixed() const { return isFixed; }

   public:
    int nodeId;
    bool isFixed;
    Pose3d se3Pose;
};

// TODO: put it to PoseGraph as a subtype
/*! \class PoseGraphEdge
 *  \brief Defines the constraints between two PoseGraphNodes
 *
 *  Detailed description
 */
struct PoseGraphEdge
{
   public:
    PoseGraphEdge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
                  const Matx66f& _information = Matx66f::eye());

    virtual ~PoseGraphEdge() = default;

    int getSourceNodeId() const { return sourceNodeId; }
    int getTargetNodeId() const { return targetNodeId; }

    bool operator==(const PoseGraphEdge& edge)
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
    Matx66f information;
    Matx66f sqrtInfo;
};

class PoseGraph
{
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

    void addNode(const PoseGraphNode& node)
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
    void addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

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
    double calcEnergy(const std::map<int, PoseGraphNode>& newNodes) const;

    std::map<int, PoseGraphNode> nodes;
    std::vector<PoseGraphEdge>   edges;
};

}  // namespace kinfu
}  // namespace cv
#endif /* ifndef OPENCV_RGBD_GRAPH_NODE_H */
