// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

#if defined(CERES_FOUND)
#include <ceres/ceres.h>
#endif

namespace cv
{
namespace kinfu
{
bool PoseGraph::isValid() const
{
    int numNodes = getNumNodes();
    int numEdges = getNumEdges();

    if (numNodes <= 0 || numEdges <= 0)
        return false;

    std::unordered_set<int> nodesVisited;
    std::vector<int> nodesToVisit;

    nodesToVisit.push_back(nodes.begin()->first);

    bool isGraphConnected = false;
    while (!nodesToVisit.empty())
    {
        int currNodeId = nodesToVisit.back();
        nodesToVisit.pop_back();
        //std::cout << "Visiting node: " << currNodeId << "\n";
        nodesVisited.insert(currNodeId);
        // Since each node does not maintain its neighbor list
        for (int i = 0; i < numEdges; i++)
        {
            const PoseGraphEdge& potentialEdge = edges.at(i);
            int nextNodeId                     = -1;

            if (potentialEdge.getSourceNodeId() == currNodeId)
            {
                nextNodeId = potentialEdge.getTargetNodeId();
            }
            else if (potentialEdge.getTargetNodeId() == currNodeId)
            {
                nextNodeId = potentialEdge.getSourceNodeId();
            }
            if (nextNodeId != -1)
            {
                //std::cout << "Next node: " << nextNodeId << " " << nodesVisited.count(nextNodeId)
                //          << std::endl;
                if (nodesVisited.count(nextNodeId) == 0)
                {
                    nodesToVisit.push_back(nextNodeId);
                }
            }
        }
    }

    isGraphConnected = (int(nodesVisited.size()) == numNodes);
    std::cout << "nodesVisited: " << nodesVisited.size()
              << " IsGraphConnected: " << isGraphConnected << std::endl;
    bool invalidEdgeNode = false;
    for (int i = 0; i < numEdges; i++)
    {
        const PoseGraphEdge& edge = edges.at(i);
        // edges have spurious source/target nodes
        if ((nodesVisited.count(edge.getSourceNodeId()) != 1) ||
            (nodesVisited.count(edge.getTargetNodeId()) != 1))
        {
            invalidEdgeNode = true;
            break;
        }
    }
    return isGraphConnected && !invalidEdgeNode;
}


// Taken from Ceres pose graph demo: https://ceres-solver.org/
PoseGraph::PoseGraph(const std::string& g2oFileName) :
    nodes(), edges()
{
    auto readAffine = [](std::istream& input) -> Affine3d
    {
        Vec3d p;
        Vec4d q;
        input >> p[0] >> p[1] >> p[2];
        input >> q[1] >> q[2] >> q[3] >> q[0];
        // Normalize the quaternion to account for precision loss due to
        // serialization.
        return Affine3d(Quatd(q).toRotMat3x3(), p);
    };

    // for debugging purposes
    int minId = 0, maxId = 1 << 30;

    std::ifstream infile(g2oFileName.c_str());
    if (!infile)
    {
        CV_Error(cv::Error::StsError, "failed to open file");
    }
    
    while (infile.good())
    {
        std::string data_type;
        // Read whether the type is a node or a constraint
        infile >> data_type;
        if (data_type == "VERTEX_SE3:QUAT")
        {
            int id;
            infile >> id;
            Affine3d pose = readAffine(infile);

            if (id < minId || id >= maxId)
                continue;

            kinfu::PoseGraphNode n(id, pose);
            if (id == minId)
                n.setFixed();

            // Ensure we don't have duplicate poses
            const auto& it = nodes.find(id);
            if (it != nodes.end())
            {
                std::cout << "duplicated node, id=" << id << std::endl;
                nodes.insert(it, { id, n });
            }
            else
            {
                nodes.insert({ id, n });
            }
        }
        else if (data_type == "EDGE_SE3:QUAT")
        {
            int startId, endId;
            infile >> startId >> endId;
            Affine3d pose = readAffine(infile);

            Matx66d info;
            for (int i = 0; i < 6 && infile.good(); ++i)
            {
                for (int j = i; j < 6 && infile.good(); ++j)
                {
                    infile >> info(i, j);
                    if (i != j)
                    {
                        info(j, i) = info(i, j);
                    }
                }
            }

            if ((startId >= minId && startId < maxId) && (endId >= minId && endId < maxId))
            {
                edges.push_back(PoseGraphEdge(startId, endId, pose, info));
            }
        }
        else
        {
            CV_Error(cv::Error::StsError, "unknown tag");
        }
        
        // Clear any trailing whitespace from the line
        infile >> std::ws;
    }
}



#if defined(CERES_FOUND) && defined(HAVE_EIGEN)

class MyQuaternionParameterization
    : public ceres::LocalParameterization {
public:
    virtual ~MyQuaternionParameterization() {}
    bool Plus(const double* x_ptr, const double* delta_ptr, double* x_plus_delta_ptr) const override
    {
        Vec4d vx(x_ptr);
        Quatd x(vx);
        Vec3d delta(delta_ptr);

        Quatd x_plus_delta = Quatd(0, delta[0], delta[1], delta[2]).exp() * x;

        *(Vec4d*)(x_plus_delta_ptr) = x_plus_delta.toVec();

        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Vec4d vx(x);
        Matx43d jm = Optimizer::expQuatJacobian(Quatd(vx));

        for (int ii = 0; ii < 12; ii++)
            jacobian[ii] = jm.val[ii];

        return true;
    }

    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
};

void Optimizer::createOptimizationProblem(PoseGraph& poseGraph, ceres::Problem& problem)
{
    int numEdges = poseGraph.getNumEdges();
    if (numEdges == 0)
    {
        CV_Error(Error::StsBadArg, "PoseGraph has no edges, no optimization to be done");
        return;
    }

    ceres::LossFunction* lossFunction = nullptr;
    // TODO: Experiment with SE3 parameterization
    ceres::LocalParameterization* quatLocalParameterization =
         new MyQuaternionParameterization;

    for (const PoseGraphEdge& currEdge : poseGraph.edges)
    {
        int sourceNodeId = currEdge.getSourceNodeId();
        int targetNodeId = currEdge.getTargetNodeId();
        Pose3d& sourcePose = poseGraph.nodes.at(sourceNodeId).se3Pose;
        Pose3d& targetPose = poseGraph.nodes.at(targetNodeId).se3Pose;

        ceres::CostFunction* costFunction = Pose3dAnalyticCostFunction::create(
            Vec3d(currEdge.transformation.translation()),
            Quatd::createFromRotMat(Matx33d(currEdge.transformation.rotation())),
            currEdge.information);

        problem.AddResidualBlock(costFunction, lossFunction,
            sourcePose.t.val, sourcePose.vq.val,
            targetPose.t.val, targetPose.vq.val);
        problem.SetParameterization(sourcePose.vq.val, quatLocalParameterization);
        problem.SetParameterization(targetPose.vq.val, quatLocalParameterization);
    }

    for (const auto& it : poseGraph.nodes)
    {
        const PoseGraphNode& node = it.second;
        if (node.isPoseFixed())
        {
            problem.SetParameterBlockConstant(node.se3Pose.t.val);
            problem.SetParameterBlockConstant(node.se3Pose.vq.val);
        }
    }
}
#endif

void Optimizer::optimize(PoseGraph& poseGraph)
{
    PoseGraph poseGraphOriginal = poseGraph;

    if (!poseGraphOriginal.isValid())
    {
        CV_Error(Error::StsBadArg,
                 "Invalid PoseGraph that is either not connected or has invalid nodes");
        return;
    }

    int numNodes = poseGraph.getNumNodes();
    int numEdges = poseGraph.getNumEdges();
    std::cout << "Optimizing PoseGraph with " << numNodes << " nodes and " << numEdges << " edges"
              << std::endl;

#if defined(CERES_FOUND) && defined(HAVE_EIGEN)
    ceres::Problem problem;
    createOptimizationProblem(poseGraph, problem);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << '\n';

    std::cout << "Is solution usable: " << summary.IsSolutionUsable() << std::endl;
#else
    CV_Error(Error::StsNotImplemented, "Ceres and Eigen required for Pose Graph optimization");
#endif
}

}  // namespace kinfu
}  // namespace cv
