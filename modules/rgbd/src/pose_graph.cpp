// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

#include <unordered_set>
#include <vector>

#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"

namespace cv
{
namespace kinfu
{
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
            std::cout << "Next node: " << nextNodeId << std::endl;
            if (nextNodeId != -1)
            {
                if (nodesVisited.count(currNodeId) == 0)
                {
                    nodesVisited.insert(currNodeId);
                    nodesToVisit.push_back(nextNodeId);
                }
            }
        }
    }

    isGraphConnected = (int(nodesVisited.size()) == numNodes);
    std::cout << "IsGraphConnected: " << isGraphConnected << std::endl;
    bool invalidEdgeNode = false;
    for (int i = 0; i < numEdges; i++)
    {
        const PoseGraphEdge& edge = edges.at(i);
        // edges have spurious source/target nodes
        if ((nodesVisited.count(edge.getSourceNodeId()) != 1) || (nodesVisited.count(edge.getTargetNodeId()) != 1))
        {
            invalidEdgeNode = true;
            break;
        }
    }
    return isGraphConnected && !invalidEdgeNode;
}

float PoseGraph::createLinearSystem(BlockSparseMat<float, 6, 6>& H, Mat& B)
{
    int numEdges = int(edges.size());

    float residual = 0.0f;

    // Lambda to calculate the error Jacobians w.r.t source and target poses
    auto calcErrorJacobian = [](Matx66f& jacSource, Matx66f& jacTarget, const Affine3f& relativePoseMeas,
                                      const Affine3f& sourcePose, const Affine3f& targetPose) -> void
    {
        const Affine3f relativePoseMeas_inv = relativePoseMeas.inv();
        const Affine3f targetPose_inv       = targetPose.inv();
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dSource = relativePoseMeas_inv * targetPose_inv * cv::Affine3f(generatorJacobian[i]) * sourcePose;
            Vec3f rot   = dError_dSource.rvec();
            Vec3f trans = dError_dSource.translation();
            jacSource.val[i + 6*0] = rot(0);
            jacSource.val[i + 6*1] = rot(1);
            jacSource.val[i + 6*2] = rot(2);
            jacSource.val[i + 6*3] = trans(0);
            jacSource.val[i + 6*4] = trans(1);
            jacSource.val[i + 6*5] = trans(2);
        }
        for (int i = 0; i < 6; i++)
        {
            Affine3f dError_dSource = relativePoseMeas_inv * targetPose_inv * cv::Affine3f(-generatorJacobian[i]) * sourcePose;
            Vec3f rot   = dError_dSource.rvec();
            Vec3f trans = dError_dSource.translation();
            jacTarget.val[i + 6*0] = rot(0);
            jacTarget.val[i + 6*1] = rot(1);
            jacTarget.val[i + 6*2] = rot(2);
            jacTarget.val[i + 6*3] = trans(0);
            jacTarget.val[i + 6*4] = trans(1);
            jacTarget.val[i + 6*5] = trans(2);
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
        //! Edge transformation is source to target. (relativeConstraint should be identity if no error)
        Affine3f poseConstraint = currEdge.transformation.inv() * relativeSourceToTarget;

        std::cout << "RelativeSourceToTarget: \n" << relativeSourceToTarget.matrix << "\n";
        std::cout << "Edge estimated transformation: \n" << currEdge.transformation.matrix << "\n";
        std::cout << "PoseConstraint: \n" << poseConstraint.matrix << "\n";

        Matx61f error;
        Vec3f rot   = poseConstraint.rvec();
        Vec3f trans = poseConstraint.translation();

        error.val[0] = rot[0];
        error.val[1] = rot[1];
        error.val[2] = rot[2];
        error.val[3] = trans[0];
        error.val[4] = trans[1];
        error.val[5] = trans[2];

        std::cout << "Error vector: \n" << error << std::endl;

        Matx66f jacSource, jacTarget;
        calcErrorJacobian(jacSource, jacTarget, currEdge.transformation, sourcePose, targetPose);
        std::cout << "Source Jacobian: \n" << jacSource << std::endl;
        std::cout << "Target Jacobian: \n" << jacTarget << std::endl;

        Matx16f errorTransposeInfo     = error.t() * currEdge.information;
        Matx66f jacSourceTransposeInfo = jacSource.t() * currEdge.information;
        Matx66f jacTargetTransposeInfo = jacTarget.t() * currEdge.information;

        /* std::cout << "errorTransposeInfo: " << errorTransposeInfo << "\n"; */
        /* std::cout << "jacSourceTransposeInfo: " << jacSourceTransposeInfo << "\n"; */
        /* std::cout << "jacTargetTransposeInfo: " << jacTargetTransposeInfo << "\n"; */

        float res = std::sqrt((errorTransposeInfo * error).val[0]);
        residual += res;

        std::cout << "sourceNodeId: " << sourceNodeId << " targetNodeId: " << targetNodeId << std::endl;
        H.refBlock(sourceNodeId, sourceNodeId) += jacSourceTransposeInfo * jacSource;
        H.refBlock(targetNodeId, targetNodeId) += jacTargetTransposeInfo * jacTarget;
        H.refBlock(sourceNodeId, targetNodeId) += jacSourceTransposeInfo * jacTarget;
        H.refBlock(targetNodeId, sourceNodeId) += jacTargetTransposeInfo * jacSource;


        B.rowRange(6 * sourceNodeId, 6 * (sourceNodeId + 1)) += (errorTransposeInfo * jacSource).reshape<6, 1>();
        B.rowRange(6 * targetNodeId, 6 * (targetNodeId + 1)) += (errorTransposeInfo * jacTarget).reshape<6, 1>();
    }

    std::cout << "Residual value: " << residual << std::endl;
    return residual;
}

PoseGraph PoseGraph::update(const Mat& deltaVec)
{
    int numNodes = int(nodes.size());

    //! Check: This should create a copy of the posegraph
    PoseGraph updatedPoseGraph(*this);

    for (int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        Vec6f delta          = deltaVec.rowRange(6 * currentNodeId, 6 * (currentNodeId + 1));
        Affine3f pose        = nodes.at(currentNodeId).getPose();
        Affine3f currDelta   = Affine3f(Vec3f(delta.val), Vec3f(delta.val + 3));
        std::cout << "Current Delta for node ID: " << currentNodeId << " \n" << currDelta.matrix << std::endl;
        Affine3f updatedPose = currDelta * pose;

        updatedPoseGraph.nodes.at(currentNodeId).setPose(updatedPose);
    }

    return updatedPoseGraph;
}

Mat PoseGraph::getVector()
{
    int numNodes = int(nodes.size());
    Mat vector(6 * numNodes, 1, CV_32F, Scalar(0));
    for(int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        Affine3f pose = nodes.at(currentNodeId).getPose();
        Vec3f rot = pose.rvec();
        Vec3f trans = pose.translation();
        vector.rowRange(6 * currentNodeId, 6 * (currentNodeId+1)) = Vec6f(rot.val[0], rot.val[1], rot.val[2], trans.val[0], trans.val[1], trans.val[2]);
    }
    return vector;
}

//! NOTE: We follow left-composition for the infinitesimal pose update
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

    int numNodes = poseGraph.getNumNodes();
    int numEdges = poseGraph.getNumEdges();

    std::cout << "Optimizing posegraph with nodes: " << numNodes << " edges: " << numEdges << std::endl;

    int iter = 0;
    float prevResidual = std::numeric_limits<float>::max();
    do
    {
        std::cout << "Current iteration: " << iter << std::endl;
        //! ApproxH = Approximate Hessian = J^T * information * J
        BlockSparseMat<float, 6, 6> ApproxH(numNodes);
        Mat B = cv::Mat::zeros(6 * numNodes, 1, CV_32F);

        float currentResidual = poseGraph.createLinearSystem(ApproxH, B);
        std::cout << "Number of non-zero blocks in H: " << ApproxH.nonZeroBlocks() << "\n";
        Mat delta(6 * numNodes, 1, CV_32F);
        Mat X = poseGraph.getVector();
        bool success = sparseSolve(ApproxH, B, delta);
        if (!success)
        {
            CV_Error(Error::StsNoConv, "Sparse solve failed");
            return;
        }
        std::cout << "delta update: " << delta << "\n delta norm: " << cv::norm(delta) << " X norm: " << cv::norm(X) << std::endl;

        //! Check delta
        if(cv::norm(delta) < 1e-6f * (cv::norm(X) + 1e-6f))
        {
            std::cout << "Delta norm[" << cv::norm(delta) << "] < 1e-6f * X norm[" << cv::norm(X) <<"] + 1e-6f\n";
            break;
        }
        std::cout << " Current residual: " << currentResidual << " Prev residual: " << prevResidual << "\n";

        //! TODO: Improve criterion and allow small increments in residual
        if ((currentResidual - prevResidual) > params.maxAcceptableResIncre)
        {
            std::cout << "Current residual increased from prevResidual by at least " << params.maxAcceptableResIncre << std::endl;
            break;
        }
        //! If updates don't improve a lot means loss function basin has been reached
        if ((currentResidual - params.minResidual) < 1e-5f)
        {
            std::cout << "Gauss newton converged \n";
            break;
        }
        poseGraph = poseGraph.update(delta);
        prevResidual = currentResidual;
    } while (iter < params.maxNumIters);
}
}  // namespace kinfu
}  // namespace cv
