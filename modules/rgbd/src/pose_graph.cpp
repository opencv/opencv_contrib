// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

#include <limits>
#include <unordered_set>
#include <vector>

#include "opencv2/core/base.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"

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

float PoseGraph::createLinearSystem(BlockSparseMat<float, 6, 6>& hessian, Mat& B)
{
    int numEdges = int(edges.size());

    float residual = 0.0f;

    // Lambda to calculate the error Jacobians w.r.t source and target poses
    auto calcErrorJacobian = [](Matx66f& jacSource, Matx66f& jacTarget,
                                const Affine3f& relativePoseMeas, const Affine3f& sourcePose,
                                const Affine3f& targetPose) -> void {
        const Affine3f relativePoseMeas_inv = relativePoseMeas.inv();
        const Affine3f targetPose_inv       = targetPose.inv();
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dSource = relativePoseMeas_inv * targetPose_inv *
                                      cv::Affine3f(generatorJacobian[i]) * sourcePose;
            Vec3f rot       = dError_dSource.rvec();
            Vec3f trans     = dError_dSource.translation();
            jacSource(i, 0) = rot(0);
            jacSource(i, 1) = rot(1);
            jacSource(i, 2) = rot(2);
            jacSource(i, 3) = trans(0);
            jacSource(i, 4) = trans(1);
            jacSource(i, 5) = trans(2);
        }
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dTarget = relativePoseMeas_inv * targetPose_inv *
                                      cv::Affine3f(-generatorJacobian[i]) * sourcePose;
            Vec3f rot       = dError_dTarget.rvec();
            Vec3f trans     = dError_dTarget.translation();
            jacTarget(i, 0) = rot(0);
            jacTarget(i, 1) = rot(1);
            jacTarget(i, 2) = rot(2);
            jacTarget(i, 3) = trans(0);
            jacTarget(i, 4) = trans(1);
            jacTarget(i, 5) = trans(2);
        }
    };

    //! Compute linear system of equations
    for (int currEdgeNum = 0; currEdgeNum < numEdges; currEdgeNum++)
    {
        const PoseGraphEdge& currEdge = edges.at(currEdgeNum);
        int sourceNodeId              = currEdge.getSourceNodeId();
        int targetNodeId              = currEdge.getTargetNodeId();
        const Affine3f& sourcePose    = nodes.at(sourceNodeId).getPose();
        const Affine3f& targetPose    = nodes.at(targetNodeId).getPose();

        Affine3f relativeSourceToTarget = targetPose.inv() * sourcePose;
        //! Edge transformation is source to target. (relativeConstraint should be identity if no
        //! error)
        Affine3f poseConstraint = currEdge.transformation.inv() * relativeSourceToTarget;

        Matx61f error;
        Vec3f rot   = poseConstraint.rvec();
        Vec3f trans = poseConstraint.translation();

        error.val[0] = rot[0];
        error.val[1] = rot[1];
        error.val[2] = rot[2];
        error.val[3] = trans[0];
        error.val[4] = trans[1];
        error.val[5] = trans[2];

        Matx66f jacSource, jacTarget;
        calcErrorJacobian(jacSource, jacTarget, currEdge.transformation, sourcePose, targetPose);

        Matx16f errorTransposeInfo     = error.t() * currEdge.information;
        Matx66f jacSourceTransposeInfo = jacSource.t() * currEdge.information;
        Matx66f jacTargetTransposeInfo = jacTarget.t() * currEdge.information;

        residual += 0.5*(errorTransposeInfo * error).val[0];

        hessian.refBlock(sourceNodeId, sourceNodeId) += jacSourceTransposeInfo * jacSource;
        hessian.refBlock(targetNodeId, targetNodeId) += jacTargetTransposeInfo * jacTarget;
        hessian.refBlock(sourceNodeId, targetNodeId) += jacSourceTransposeInfo * jacTarget;
        hessian.refBlock(targetNodeId, sourceNodeId) += jacTargetTransposeInfo * jacSource;

        B.rowRange(6 * sourceNodeId, 6 * (sourceNodeId + 1)) +=
            (errorTransposeInfo * jacSource).reshape<6, 1>();
        B.rowRange(6 * targetNodeId, 6 * (targetNodeId + 1)) +=
            (errorTransposeInfo * jacTarget).reshape<6, 1>();
    }
    return residual;
}

PoseGraph PoseGraph::update(const Mat& deltaVec)
{
    int numNodes = int(nodes.size());

    //! Check: This should create a copy of the posegraph
    PoseGraph updatedPoseGraph(*this);

    for (int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        if (nodes.at(currentNodeId).isPoseFixed())
            continue;
        Vec6f delta          = deltaVec.rowRange(6 * currentNodeId, 6 * (currentNodeId + 1));
        Affine3f pose        = nodes.at(currentNodeId).getPose();
        Affine3f currDelta   = Affine3f(Vec3f(delta.val), Vec3f(delta.val + 3));
        Affine3f updatedPose = currDelta * pose;

        updatedPoseGraph.nodes.at(currentNodeId).setPose(updatedPose);
    }

    return updatedPoseGraph;
}

Mat PoseGraph::getVector()
{
    int numNodes = int(nodes.size());
    Mat vector(6 * numNodes, 1, CV_32F, Scalar(0));
    for (int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        Affine3f pose = nodes.at(currentNodeId).getPose();
        Vec3f rot     = pose.rvec();
        Vec3f trans   = pose.translation();
        vector.rowRange(6 * currentNodeId, 6 * (currentNodeId + 1)) =
            Vec6f(rot.val[0], rot.val[1], rot.val[2], trans.val[0], trans.val[1], trans.val[2]);
    }
    return vector;
}

float PoseGraph::computeResidual()
{
    int numEdges = int(edges.size());

    float residual = 0.0f;
    for (int currEdgeNum = 0; currEdgeNum < numEdges; currEdgeNum++)
    {
        const PoseGraphEdge& currEdge = edges.at(currEdgeNum);
        int sourceNodeId              = currEdge.getSourceNodeId();
        int targetNodeId              = currEdge.getTargetNodeId();
        const Affine3f& sourcePose    = nodes.at(sourceNodeId).getPose();
        const Affine3f& targetPose    = nodes.at(targetNodeId).getPose();

        Affine3f relativeSourceToTarget = targetPose.inv() * sourcePose;
        Affine3f poseConstraint         = currEdge.transformation.inv() * relativeSourceToTarget;

        Matx61f error;
        Vec3f rot   = poseConstraint.rvec();
        Vec3f trans = poseConstraint.translation();

        error.val[0] = rot[0];
        error.val[1] = rot[1];
        error.val[2] = rot[2];
        error.val[3] = trans[0];
        error.val[4] = trans[1];
        error.val[5] = trans[2];

        Matx16f errorTransposeInfo = error.t() * currEdge.information;
        residual += 0.5*(errorTransposeInfo * error).val[0];
    }
    return residual;
}

bool Optimizer::isStepSizeSmall(const Mat& delta, float minStepSize)
{
    float maxDeltaNorm = 0.0f;
    for (int i = 0; i < delta.rows; i++)
    {
        float val = abs(delta.at<float>(i, 0));
        if (val > maxDeltaNorm)
            maxDeltaNorm = val;
    }
    return maxDeltaNorm < minStepSize;
}

float Optimizer::stepQuality(float currentResidual, float prevResidual, const Mat& delta,
                             const Mat& B, const Mat& predB)
{
    float actualReduction    = prevResidual - currentResidual;
    float predictedReduction = 0.0f;
    for (int i = 0; i < predB.cols; i++)
    {
        predictedReduction -= (0.5 * predB.at<float>(i, 0) * delta.at<float>(i, 0) +
                               B.at<float>(i, 0) * delta.at<float>(i, 0));
    }
    std::cout << " Actual reduction: " << actualReduction
              << " Prediction reduction: " << predictedReduction << std::endl;
    if (predictedReduction < 0)
        return actualReduction / abs(predictedReduction);
    return actualReduction / predictedReduction;
}

//! NOTE: We follow left-composition for the infinitesimal pose update
void Optimizer::optimizeLevenberg(const Optimizer::Params& params, PoseGraph& poseGraph)
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
    int iter           = 0;
    float lambda       = 0.01;
    const float factor = 2;
    float prevResidual = std::numeric_limits<float>::max();

    while (iter < params.maxNumIters)
    {
        std::cout << "Current iteration: " << iter << std::endl;

        BlockSparseMat<float, 6, 6> hessian(numNodes);
        Mat B = cv::Mat::zeros(6 * numNodes, 1, CV_32F);

        prevResidual = poseGraph.createLinearSystem(hessian, B);
        //! H_LM = H + lambda * diag(H);
        Mat Hdiag = hessian.diagonal();
        for (int i = 0; i < Hdiag.rows; i++)
        {
            hessian.refElem(i, i) += lambda * Hdiag.at<float>(i, 0);
        }

        Mat delta(6 * numNodes, 1, CV_32F);
        Mat predB(6 * numNodes, 1, CV_32F);
        bool success = sparseSolve(hessian, B, delta, predB);
        if (!success)
        {
            CV_Error(Error::StsNoConv, "Sparse solve failed");
            return;
        }
        delta = -delta;

        //! If step size is too small break
        if (isStepSizeSmall(delta, params.minStepSize))
        {
            std::cout << "Step size is too small.\n";
            break;
        }

        PoseGraph poseGraphNew = poseGraph.update(delta);
        float currentResidual  = poseGraphNew.computeResidual();
        std::cout << " Current residual: " << currentResidual << " Prev residual: " << prevResidual
                  << "\n";

        float quality = stepQuality(currentResidual, prevResidual, delta, B, predB);
        std::cout << " Step Quality: " << quality << std::endl;
        /* float reduction = currentResidual - prevResidual; */
        bool stepSuccess = false;
        if (quality > 0.75)
        {
            //! Accept update
            lambda    = lambda / factor;
            poseGraph = poseGraphNew;
            std::cout << "Accepting update new lambda: " << lambda << std::endl;

            stepSuccess = true;
        }
        else if (quality > 0.25)
        {
            poseGraph = poseGraphNew;
            std::cout << "Accepting update no update to lambda: " << lambda << std::endl;
            stepSuccess = true;
        }
        else
        {
            //! Reject update
            lambda = lambda * 2 * factor;
            std::cout << "Rejecting update new lambda: " << lambda << std::endl;
            stepSuccess = false;
        }
        if(stepSuccess)
        {
            if (abs(currentResidual - prevResidual) < (prevResidual * params.minResidualDecrease))
            {
                std::cout << "Reduction in residual too small, converged "
                          << (prevResidual * params.minResidualDecrease) << std::endl;
                break;
            }
        }
        //! TODO: Change to another paramter, this is dummy
        prevResidual = currentResidual;
        iter++;
    }
}

}  // namespace kinfu
}  // namespace cv
