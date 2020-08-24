// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

#include <limits>
#include <unordered_set>
#include <vector>

#if defined(CERES_FOUND)
#include "ceres/ceres.h"
#endif

namespace cv
{
namespace kinfu
{
void PoseGraph::addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

bool PoseGraph::isValid() const
{
    int numNodes = int(nodes.size());
    int numEdges = int(edges.size());

    if (numNodes <= 0 || numEdges <= 0)
        return false;

    std::unordered_set<int> nodesVisited;
    std::vector<int> nodesToVisit;

    nodesToVisit.push_back(nodes.at(0).getId());

    bool isGraphConnected = false;
    while (!nodesToVisit.empty())
    {
        int currNodeId = nodesToVisit.back();
        nodesToVisit.pop_back();
        std::cout << "Visiting node: " << currNodeId << "\n";
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
                std::cout << "Next node: " << nextNodeId << " " << nodesVisited.count(nextNodeId)
                          << std::endl;
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

#if defined(CERES_FOUND)
void Optimizer::createOptimizationProblem(PoseGraph& poseGraph, ceres::Problem& problem)
{
    int numEdges = poseGraph.getNumEdges();
    int numNodes = poseGraph.getNumNodes();
    if (numEdges == 0)
    {
        CV_Error(Error::StsBadArg, "PoseGraph has no edges, no optimization to be done");
        return;
    }

    ceres::LossFunction* lossFunction = nullptr;
    // TODO: Experiment with SE3 parameterization
    ceres::LocalParameterization* quatLocalParameterization =
        new ceres::EigenQuaternionParameterization;

    for (int currEdgeNum = 0; currEdgeNum < numEdges; ++currEdgeNum)
    {
        const PoseGraphEdge& currEdge = poseGraph.edges.at(currEdgeNum);
        int sourceNodeId              = currEdge.getSourceNodeId();
        int targetNodeId              = currEdge.getTargetNodeId();
        Pose3d& sourcePose            = poseGraph.nodes.at(sourceNodeId).pose;
        Pose3d& targetPose            = poseGraph.nodes.at(targetNodeId).pose;

        const Matx66f& informationMatrix = currEdge.information;

        ceres::CostFunction* costFunction =
            Pose3dErrorFunctor::create(currEdge.transformation, informationMatrix);

        problem.AddResidualBlock(costFunction, lossFunction, sourcePose.t.data(),
                                 sourcePose.r.coeffs().data(), targetPose.t.data(),
                                 targetPose.r.coeffs().data());
        problem.SetParameterization(sourcePose.r.coeffs().data(), quatLocalParameterization);
        problem.SetParameterization(targetPose.r.coeffs().data(), quatLocalParameterization);
    }

    for (int currNodeId = 0; currNodeId < numNodes; ++currNodeId)
    {
        PoseGraphNode& currNode = poseGraph.nodes.at(currNodeId);
        if (currNode.isPoseFixed())
        {
            problem.SetParameterBlockConstant(currNode.pose.t.data());
            problem.SetParameterBlockConstant(currNode.pose.r.coeffs().data());
        }
    }
}
#endif

void Optimizer::optimizeCeres(PoseGraph& poseGraph)
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

#if defined(CERES_FOUND)
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
    CV_Error(Error::StsNotImplemented, "Ceres required for Pose Graph optimization");
#endif
}

}  // namespace kinfu
}  // namespace cv
