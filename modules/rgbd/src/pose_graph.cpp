// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

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

        const double norm_delta = norm(delta);
        Quatd x_plus_delta;
        if (norm_delta > 0.0)
        {
            const double sin_delta_by_delta = std::sin(norm_delta) / norm_delta;

            // Note, in the constructor w is first.
            Quatd delta_q(std::cos(norm_delta),
                          sin_delta_by_delta * delta[0],
                          sin_delta_by_delta * delta[1],
                          sin_delta_by_delta * delta[2]);
            x_plus_delta = delta_q * x;
        }
        else
        {
            x_plus_delta = x;
        }

        Vec4d xpd = x_plus_delta.toVec();
        x_plus_delta_ptr[0] = xpd[0];
        x_plus_delta_ptr[1] = xpd[1];
        x_plus_delta_ptr[2] = xpd[2];
        x_plus_delta_ptr[3] = xpd[3];

        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        // clang-format off
        jacobian[0] = -x[1];  jacobian[1]  = -x[2];   jacobian[2]  = -x[3];
        jacobian[3] =  x[0];  jacobian[4]  =  x[3];   jacobian[5]  = -x[2];
        jacobian[6] = -x[3];  jacobian[7]  =  x[0];   jacobian[8]  =  x[1];
        jacobian[9] =  x[2];  jacobian[10] = -x[1];   jacobian[11] =  x[0];
        // clang-format on
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
        new ceres::EigenQuaternionParameterization;

    for (const PoseGraphEdge& currEdge : poseGraph.edges)
    {
        int sourceNodeId = currEdge.getSourceNodeId();
        int targetNodeId = currEdge.getTargetNodeId();
        Pose3d& sourcePose = poseGraph.nodes.at(sourceNodeId).se3Pose;
        Pose3d& targetPose = poseGraph.nodes.at(targetNodeId).se3Pose;

        // -------

        Eigen::Matrix<double, 6, 6> info;
        cv2eigen(Matx66d(currEdge.information), info);
        const Eigen::Matrix<double, 6, 6> sqrt_information = info.llt().matrixL();
        Matx66d sqrtInfo;
        eigen2cv(sqrt_information, sqrtInfo);

        ceres::CostFunction* costFunction = Pose3dErrorFunctor::create(
            Pose3d(currEdge.transformation.rotation(), currEdge.transformation.translation()),
            sqrtInfo);

        // -------

        ceres::CostFunction* costFunction2 = Pose3dAnalyticCostFunction::create(
            Vec3d(currEdge.transformation.translation()),
            Quatd::createFromRotMat(Matx33d(currEdge.transformation.rotation())),
            currEdge.information);

        // -------

        problem.AddResidualBlock(costFunction2, lossFunction,
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
