// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.h"
#include <Eigen/src/Core/arch/CUDA/Half.h>

#include <unordered_set>
#include <vector>

#include "opencv2/core/eigen.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#if defined(HAVE_EIGEN)
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#endif

namespace cv
{
namespace kinfu
{

//! Function to solve a sparse linear system of equations HX = B
//! Requires Eigen
static bool sparseSolve(const BlockSparseMat<float, 6, 6>& H, const Mat& B, Mat& X)
{
    const float matValThreshold = 0.001f;

    bool result = false;

#if defined(HAVE_EIGEN)

    std::cout << "starting eigen-insertion..." << std::endl;

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(H.ijValue.size() * H.blockSize * H.blockSize);
    for (auto ijValue : H.ijValue)
    {
        int xb = ijValue.first.x, yb = ijValue.first.y;
        Matx66f vblock = ijValue.second;
        for (int i = 0; i < H.blockSize; i++)
        {
            for (int j = 0; j < H.blockSize; j++)
            {
                float val = vblock(i, j);
                if (abs(val) >= matValThreshold)
                {
                    tripletList.push_back(Eigen::Triplet<double>(H.blockSize * xb + i, H.blockSize * yb + j, val));
                }
            }
        }
    }

    Eigen::SparseMatrix<float> bigA(H.blockSize * H.nBlocks, H.blockSize * H.nBlocks);
    bigA.setFromTriplets(tripletList.begin(), tripletList.end());

    // TODO: do we need this?
    bigA.makeCompressed();

    Eigen::VectorXf bigB;
    cv2eigen(B, bigB);

    if (!bigA.isApprox(bigA.transpose())
    {
        CV_Error(Error::StsBadArg, "Sparse Matrix is not symmetric");
        return result;
    }

    //!TODO: Check determinant of bigA

    // TODO: try this, LLT and Cholesky
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>, Eigen::NaturalOrdering<int>> solver;

    std::cout << "starting eigen-compute..." << std::endl;
    solver.compute(bigA);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        std::cout << "starting eigen-solve..." << std::endl;

        Eigen::VectorXf sx = solver.solve(bigB);
        if (solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            x.resize(jtb.size);
            eigen2cv(sx, x);
            result = true;
        }
    }

#else
    std::cout << "no eigen library" << std::endl;

    CV_Error(Error::StsNotImplemented, "Eigen library required for matrix solve, dense solver is not implemented");
#endif

    return result;
}

void PoseGraph::addNode(const PoseGraphNode& node) { nodes.push_back(node); }

void PoseGraph::addEdge(const PoseGraphEdge& edge) { edges.push_back(edge); }

bool PoseGraph::nodeExists(int nodeId) { return (nodes.find(nodeId) != nodes.end()); }

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
void calcJacobian(Matx66f& jacSource, Matx66f& jacTarget, const Affine3f& relativePoseMeas, const Affine3f& sourcePose,
                  const Affine3f& targetPose)
{
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
}

float PoseGraph::createLinearSystem(BlockSparseMat<float, 6, 6>& H, Mat& B)
{
    size_t numNodes = nodes.size();
    size_t numEdges = edges.size();

    float residual = 0;
    //! Compute linear system of equations
    for (int currEdgeNum = 0; currEdgeNum < numEdges; currEdgeNum++)
    {
        const PoseGraphEdge& currEdge = *edges.at(currEdgeNum);
        int sourceNodeId              = currEdge.getSourceNodeId();
        int targetNodeId              = currEdge.getTargetNodeId();
        const Affine3f& sourcePose    = nodes.at(sourceNodeId)->getPose();
        const Affine3f& targetPose    = nodes.at(targetNodeId)->getPose();

        Affine3f relativeSourceToTarget = targetPose.inv() * sourcePose;
        //! Edge transformation is source to target. (relativeConstraint should be identity if no error)
        Affine3f relativeConstraint = currEdge.transformation.inv() * relativeSourceToTarget;

        Vec6f error;
        Vec3f rot    = relativeConstraint.rvec();
        Vec3f trans = relativeConstraint.translation();

        error[0] = rot[0];
        error[1] = rot[1];
        error[2] = rot[2];
        error[3] = trans[0];
        error[4] = trans[1];
        error[5] = trans[2];

        Matx66f jacSource, jacTarget;
        calcErrorJacobian(currEdge.transformation, sourcePose, targetPose, jacSource, jacTarget);

        Vec6f errorTransposeInfo       = error.t() * currEdge.transformation;
        Matx66f jacSourceTransposeInfo = jacSource.t() * currEdge.information;
        Matx66f jacTargetTransposeInfo = jacTarget.t() * currEdge.information;

        residual += errorTransposeInfo * error;

        ApproxH.refBlock(sourceNodeId, sourceNodeId) += jacSourceTransposeInfo * jacSource;
        ApproxH.refBlock(targetNodeId, targetNodeId) += jacTargetTransposeInfo * jacTarget;
        ApproxH.refBlock(sourceNodeId, targetNodeId) += jacSourceTransposeInfo * jacTarget;
        ApproxH.refBlock(targetNodeId, sourceNodeId) += jacTargetTransposeInfo * jacSource;

        B.rowRange(6*sourceNodeId, 6*(sourceNodeId+1)) += errorTransposeInfo * jacSource;
        B.rowRange(6*targetNodeId, 6*(targetNodeId+1)) += errorTransposeInfo * jacTarget;
    }

    return residual;
}

PoseGraph PoseGraph::updatePoseGraph(const PoseGraph& poseGraphPrev, const Mat& delta)
{
    size_t numNodes = poseGraphPrev.nodes.size();
    size_t numEdges = poseGraphPrev.edges.size();

    //!Check: This should create a copy of the posegraph
    PoseGraph updatedPoseGraph(poseGraphPrev);

    for(int currentNodeId = 0; currentNodeId < numNodes; currentNodeId++)
    {
        Vec6f delta = delta.rowRange(6*currentNodeId, 6*(currentNodeId+1));
        Affine3f pose = poseGraphPrev.nodes.at(currentNodeId)->getPose();
        Affine3f currDelta = Affine3f(Vec3f(delta.val), Vec3f(delta.val+3));
        Affine3f updatedPose = currDelta * pose;

        updatePoseGraph.nodes.at(currentNodeId)->setPose(updatedPose);
    }

    return updatePoseGraph;
}

//! NOTE: We follow the left-composition for the infinitesimal pose update
void PoseGraph::optimize(PoseGraph& poseGraph, int numIter, int minResidual)
{
    PoseGraph poseGraphOriginal = poseGraph;
    //! Check if posegraph is valid
    if (!isValid())
        //! Should print some error
        return;

    size_t numNodes = nodes.size();
    size_t numEdges = edges.size();

    //! ApproxH = Approximate Hessian = J^T * information * J
    int iter = 0;
    //! converged is set to true when error/residual is within some limit
    bool converged = false;
    float prevResidual = std::numeric_limits<float>::max();

    //! TODO: Try LM instead of GN
    do {
        BlockSparseMat<float, 6, 6> ApproxH(numNodes);
        Mat B(6 * numNodes, 1, float);

        float currentResidual = poseGraph.createLinearSystem(ApproxH, B);

        Mat delta(6*numNodes, 1, float);
        bool success = sparseSolve(ApproxH, B, delta);
        if(!success)
        {
            CV_Error(Error::StsNoConv, "Sparse solve failed");
            return;
        }
        //! Check delta
        poseGraph = updatePoseGraph(poseGraph, delta);

        std::cout << " Current residual: " << currentResidual << " Prev residual: " << prevResidual << "\n";

        //!TODO: Improve criterion and allow small increments in residual
        if(currentResidual - prevResidual > 0f)
            break;
        if((currentResidual - minResidual) < 0.00001f)
            converged = true;

    } while (iter < numIter && !converged)
}
}  // namespace kinfu
}  // namespace cv
