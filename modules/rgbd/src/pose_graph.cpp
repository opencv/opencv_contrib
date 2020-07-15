// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.h"

#include <unordered_set>
#include <vector>

#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"

namespace cv
{
namespace kinfu
{
bool PoseGraph::isValid()
{
    size_t numNodes = nodes.size();
    size_t numEdges = edges.size();

    if (numNodes <= 0 || numEdges <= 0)
        return false;

    std::unordered_set<int> nodeSet;
    std::vector<int> nodeList;

    nodeList.push_back(nodes.at(0)->getId());
    nodeSet.insert(nodes.at(0)->getId());

    bool isGraphConnected = false;
    if (!nodeList.empty())
    {
        int currNodeId = nodeList.back();
        nodeList.pop_back();

        for (size_t i = 0; i < numEdges; i++)
        {
            const PoseGraphEdge& potentialEdge = *edges.at(i);
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
                nodeList.push_back(nextNodeId);
                nodeSet.insert(nextNodeId);
            }
        }

        isGraphConnected = (nodeSet.size() == numNodes);

        bool invalidEdgeNode = false;
        for (size_t i = 0; i < numEdges; i++)
        {
            const PoseGraphEdge& edge = *edges.at(i);
            if ((nodeSet.count(edge.getSourceNodeId()) != 1) || (nodeSet.count(edge.getTargetNodeId()) != 1))
            {
                invalidEdgeNode = true;
                break;
            }
        }
        return isGraphConnected && !invalidEdgeNode;
    }
}

//! TODO: Reference: Open3D and linearizeOplus in g2o
void calcErrorJacobian() {}

float PoseGraph::createLinearSystem(BlockSparseMat<float, 6, 6>& H, Mat& B)
{
    int numNodes = int(nodes.size());
    int numEdges = int(edges.size());

    float residual = 0;

    //! Lmabda to calculate error jacobian for a particular constraint
    auto calcErrorJacobian = [](Matx66f& jacSource, Matx66f& jacTarget, const Affine3f& relativePoseMeas,
                                const Affine3f& sourcePose, const Affine3f& targetPose) -> void {
        const Affine3f relativePoseMeas_inv = relativePoseMeas.inv();
        const Affine3f targetPose_inv       = targetPose.inv();
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dSource = relativePoseMeas_inv * targetPose_inv * generatorJacobian[i] * sourcePose;
            Vec3f rot               = dError_dSource.rvec();
            Vec3f trans             = dError_dSource.translation();
            jacSource.col(i)        = Vec6f(rot(0), rot(1), rot(2), trans(0), trans(1), trans(2));
        }
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dSource = relativePoseMeas_inv * targetPose_inv * -generatorJacobian[i] * sourcePose;
            Vec3f rot               = dError_dSource.rvec();
            Vec3f trans             = dError_dSource.translation();
            jacTarget.col(i)        = Vec6f(rot(0), rot(1), rot(2), trans(0), trans(1), trans(2));
        }
    };

    //! Compute linear system of equations
    for (int currEdgeNum = 0; currEdgeNum < numEdges; currEdgeNum++)
    {
        const PoseGraphEdge& currEdge = edges.at(currEdgeNum);
        int sourceNodeId              = currEdge.getSourceNodeId();
        int targetNodeId              = currEdge.getTargetNodeId();
        const Affine3f& sourcePose    = nodes.at(sourceNodeId)->getPose();
        const Affine3f& targetPose    = nodes.at(targetNodeId)->getPose();

        Affine3f relativeSourceToTarget = targetPose.inv() * sourcePose;
        //! Edge transformation is source to target. (relativeConstraint should be identity if no error)
        Affine3f poseConstraint = currEdge.transformation.inv() * relativeSourceToTarget;

        Vec6f error;
        Vec3f rot   = poseConstraint.rvec();
        Vec3f trans = poseConstraint.translation();

        error[0] = rot[0];
        error[1] = rot[1];
        error[2] = rot[2];
        error[3] = trans[0];
        error[4] = trans[1];
        error[5] = trans[2];

        Matx66f jacSource, jacTarget;
        calcErrorJacobian(currEdge.transformation, sourcePose, targetPose, jacSource, jacTarget);

        Matx16f errorTransposeInfo     = error.t() * currEdge.information;
        Matx66f jacSourceTransposeInfo = jacSource.t() * currEdge.information;
        Matx66f jacTargetTransposeInfo = jacTarget.t() * currEdge.information;

        residual += (errorTransposeInfo * error)[0];

        H.refBlock(sourceNodeId, sourceNodeId) += jacSourceTransposeInfo * jacSource;
        H.refBlock(targetNodeId, targetNodeId) += jacTargetTransposeInfo * jacTarget;
        H.refBlock(sourceNodeId, targetNodeId) += jacSourceTransposeInfo * jacTarget;
        H.refBlock(targetNodeId, sourceNodeId) += jacTargetTransposeInfo * jacSource;

        B.rowRange(6 * sourceNodeId, 6 * (sourceNodeId + 1)) += errorTransposeInfo * jacSource;
        B.rowRange(6 * targetNodeId, 6 * (targetNodeId + 1)) += errorTransposeInfo * jacTarget;
    }

    return residual;
}

PoseGraph PoseGraph::update(const Mat& deltaVec)
{
    int numNodes = int(nodes.size());
    int numEdges = int(edges.size());

    //! Check: This should create a copy of the posegraph
    PoseGraph updatedPoseGraph(*this);

    for (int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        Vec6f delta          = deltaVec.rowRange(6 * currentNodeId, 6 * (currentNodeId + 1));
        Affine3f pose        = nodes.at(currentNodeId)->getPose();
        Affine3f currDelta   = Affine3f(Vec3f(delta.val), Vec3f(delta.val + 3));
        Affine3f updatedPose = currDelta * pose;

        updatedPoseGraph.nodes.at(currentNodeId)->setPose(updatedPose);
    }

    return updatedPoseGraph;
}

//! NOTE: We follow the left-composition for the infinitesimal pose update
void Optimizer::optimizeGaussNewton(const Optimizer::Params& params, PoseGraph& poseGraph)
{
    PoseGraph poseGraphOriginal = poseGraph;
    //! Check if posegraph is valid
    if (!poseGraphOriginal.isValid())
    //! Should print some error
    {
        CV_Error(Error::StsBadArg, "Invalid PoseGraph that is either not connected or has invalid nodes");
        return;
    }

    int numNodes = int(poseGraph.nodes.size());
    int numEdges = int(poseGraph.edges.size());

    std::cout << "Optimizing posegraph with nodes: " << numNodes << " edges: " << numEdges << std::endl;

    //! ApproxH = Approximate Hessian = J^T * information * J
    int iter = 0;
    //! converged is set to true when error/residual is within some limit
    bool converged     = false;
    float prevResidual = std::numeric_limits<float>::max();
    do
    {
        BlockSparseMat<float, 6, 6> ApproxH(numNodes);
        Mat B(6 * numNodes, 1, float);

        float currentResidual = poseGraph.createLinearSystem(ApproxH, B);

        Mat delta(6 * numNodes, 1, float);
        bool success = sparseSolve(ApproxH, B, delta);
        if (!success)
        {
            CV_Error(Error::StsNoConv, "Sparse solve failed");
            return;
        }
        //! Check delta

        //! TODO: Improve criterion and allow small increments in residual
        if((currentResidual - prevResidual) > params.maxAcceptableResIncre)
            break;
        if ((currentResidual - params.minResidual) < 0.00001f)
            converged = true;

        poseGraph = poseGraph.update(delta);
        std::cout << " Current residual: " << currentResidual << " Prev residual: " << prevResidual << "\n";

        prevResidual = currentResidual;

    } while (iter < params.maxNumIters && !converged)
}
}  // namespace kinfu
}  // namespace cv
