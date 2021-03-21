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

        ceres::CostFunction* costFunction = Pose3dAnalyticCostFunction::create(Vec3d(currEdge.pose.t), currEdge.pose.getQuat(), currEdge.information);

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

//DEBUG
void writePg(const PoseGraph& pg, std::string fname)
{
    std::fstream of(fname, std::fstream::out);
    for (const auto& n : pg.nodes)
    {
        Point3d d = n.second.getPose().translation();
        of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
    }
    for (const auto& e : pg.edges)
    {
        int sid = e.sourceNodeId, tid = e.targetNodeId;
        of << "l " << sid + 1 << " " << tid + 1 << std::endl;
    }
    of.close();
};


void Optimizer::optimize(PoseGraph& poseGraph)
{
    //TODO: no copying
    PoseGraph poseGraphOriginal = poseGraph;

    if (!poseGraphOriginal.isValid())
    {
        CV_Error(Error::StsBadArg,
            "Invalid PoseGraph that is either not connected or has invalid nodes");
        return;
    }

    int numNodes = poseGraph.getNumNodes();
    int numEdges = poseGraph.getNumEdges();

    // Allocate indices for nodes
    std::vector<int> placesIds;
    std::map<int, int> idToPlace;
    for (const auto& ni : poseGraph.nodes)
    {
        if (!ni.second.isPoseFixed())
        {
            idToPlace[ni.first] = placesIds.size();
            placesIds.push_back(ni.first);
        }
    }

    int nVarNodes = placesIds.size();
    if (!nVarNodes)
    {
        CV_Error(Error::StsBadArg, "PoseGraph has no non-constant nodes, no optimization to be done");
        return;
    }

    if (numEdges == 0)
    {
        CV_Error(Error::StsBadArg, "PoseGraph has no edges, no optimization to be done");
        return;
    }

    std::cout << "Optimizing PoseGraph with " << numNodes << " nodes and " << numEdges << " edges" << std::endl;

    size_t nVars = nVarNodes * 6;
    BlockSparseMat<double, 6, 6> jtj(nVarNodes);
    std::vector<double> jtb(nVars);

    // estimate current energy
    auto calcEnergy = [&poseGraph](const std::map<int, PoseGraphNode>& nodes) -> double
    {
        double totalErr = 0;
        for (const auto& e : poseGraph.edges)
        {
            Pose3d srcP = nodes.at(e.getSourceNodeId()).se3Pose;
            Pose3d tgtP = nodes.at(e.getTargetNodeId()).se3Pose;

            Vec6d res;
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;
            double err = poseError(srcP.getQuat(), srcP.t, tgtP.getQuat(), tgtP.t,
                                   e.pose.getQuat(), e.pose.t, e.sqrtInfo, /* needJacobians = */ false,
                                   sqj, stj, tqj, ttj, res);

            totalErr += err;
        }
        return totalErr*0.5;
    };
    
    // from Ceres, equation energy change:
    // eq. energy = 1/2 * (residuals + J * step)^2 =
    // 1/2 * ( residuals^2 + 2 * residuals^T * J * step + (J*step)^T * J * step)
    // eq. energy change = 1/2 * residuals^2 - eq. energy =
    // residuals^T * J * step + 1/2 * (J*step)^T * J * step =
    // (residuals^T * J + 1/2 * step^T * J^T * J) * step =
    // step^T * ((residuals^T * J)^T + 1/2 * (step^T * J^T * J)^T) =
    // 1/2 * step^T * (2 * J^T * residuals + J^T * J * step) =
    // 1/2 * step^T * (2 * J^T * residuals + (J^T * J + LMDiag - LMDiag) * step) =
    // 1/2 * step^T * (2 * J^T * residuals + (J^T * J + LMDiag) * step - LMDiag * step) =
    // 1/2 * step^T * (J^T * residuals - LMDiag * step) =
    // 1/2 * x^T * (jtb - lmDiag^T * x)
    auto calcJacCostChange = [nVars, &jtb](const std::vector<double>& x, const std::vector<double>& lmDiag) -> double
    {
        double jdiag = 0.0;
        for (int i = 0; i < nVars; i++)
        {
            jdiag += x[i] * (jtb[i] - lmDiag[i] * x[i]);
        }
        double costChange = jdiag * 0.5;
        return costChange;
    };

    double energy = calcEnergy(poseGraph.nodes);
    double startEnergy = energy;
    double oldEnergy = energy;

    std::cout << "#s" << " energy: " << energy << std::endl;

    // options
    // stop conditions
    const unsigned int maxIterations = 100;
    const double maxGradientTolerance = 1e-6;
    const double stepNorm2Tolerance = 1e-6;
    const double relEnergyDeltaTolerance = 1e-6;
    // normalize jac columns for better conditioning
    const bool jacobiScaling = true;
    const double minDiag = 1e-6;
    const double maxDiag = 1e32;

    const double initialLambdaLevMarq = 0.0001;
    const double initialLmUpFactor = 2.0;
    const double initialLmDownFactor = 3.0;

    // finish reasons
    bool tooLong = false; // => not found
    bool smallGradient = false; // => found
    bool smallStep = false; // => found
    bool smallEnergyDelta = false; // => found

    // column scale inverted, for jacobi scaling
    std::vector<double> di(nVars);

    double lmUpFactor = initialLmUpFactor;
    double decreaseFactorLevMarq = 2.0;
    double lambdaLevMarq = initialLambdaLevMarq;

    unsigned int iter = 0;
    bool done = false;
    while (!done)
    {
        jtj.clear();
        std::fill(jtb.begin(), jtb.end(), 0.0);

        // caching nodes jacobians
        std::vector<cv::Matx<double, 7, 6>> cachedJac;
        for (auto id : placesIds)
        {
            Pose3d p = poseGraph.nodes.at(id).se3Pose;
            Matx43d qj = expQuatJacobian(p.getQuat());
            // x node layout is (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
            // pose layout is (q_w, q_x, q_y, q_z, trans_x, trans_y, trans_z)
            Matx<double, 7, 6> j = concatVert(concatHor(qj, Matx43d()),
                                              concatHor(Matx33d(), Matx33d::eye()));
            cachedJac.push_back(j);
        }

        // fill jtj and jtb
        for (const auto& e : poseGraph.edges)
        {
            int srcId = e.getSourceNodeId(), dstId = e.getTargetNodeId();
            const PoseGraphNode& srcNode = poseGraph.nodes.at(srcId);
            const PoseGraphNode& dstNode = poseGraph.nodes.at(dstId);

            Pose3d srcP = srcNode.se3Pose;
            Pose3d tgtP = dstNode.se3Pose;
            bool srcFixed = srcNode.isPoseFixed();
            bool dstFixed = dstNode.isPoseFixed();

            Vec6d res;
            Matx<double, 6, 3> stj, ttj;
            Matx<double, 6, 4> sqj, tqj;
            double err = poseError(srcP.getQuat(), srcP.t, tgtP.getQuat(), tgtP.t,
                                   e.pose.getQuat(), e.pose.t, e.sqrtInfo, /* needJacobians = */ true,
                                   sqj, stj, tqj, ttj, res);

            size_t srcPlace, dstPlace;
            Matx66d sj, tj;
            if (!srcFixed)
            {
                srcPlace = idToPlace.at(srcId);
                sj = concatHor(sqj, stj) * cachedJac[srcPlace];

                jtj.refBlock(srcPlace, srcPlace) += sj.t() * sj;

                Vec6f jtbSrc = sj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtb[6 * srcPlace + i] += - jtbSrc[i];
                }
            }

            if (!dstFixed)
            {
                dstPlace = idToPlace.at(dstId);
                tj = concatHor(tqj, ttj) * cachedJac[dstPlace];
                
                jtj.refBlock(dstPlace, dstPlace) += tj.t() * tj;

                Vec6f jtbDst = tj.t() * res;
                for (int i = 0; i < 6; i++)
                {
                    jtb[6 * dstPlace + i] += -jtbDst[i];
                }
            }
            
            if (!(srcFixed || dstFixed))
            {
                Matx66d sjttj = sj.t() * tj;
                jtj.refBlock(srcPlace, dstPlace) += sjttj;
                jtj.refBlock(dstPlace, srcPlace) += sjttj.t();
            }
        }

        std::cout << "#LM#s" << " energy: " << energy << std::endl;

        // do the jacobian conditioning improvement used in Ceres
        if (jacobiScaling)
        {
            // L2-normalize each jacobian column
            // vec d = {d_j = sum(J_ij^2) for each column j of J} = get_diag{ J^T * J }
            // di = { 1/(1+sqrt(d_j)) }, extra +1 to avoid div by zero
            if (iter == 0)
            {
                for (int i = 0; i < nVars; i++)
                {
                    double ds = sqrt(jtj.valElem(i, i)) + 1.0;
                    di[i] = 1.0 / ds;
                }
            }

            // J := J * d_inv, d_inv = make_diag(di)
            // J^T*J := (J * d_inv)^T * J * d_inv = diag(di)* (J^T * J)* diag(di) = eltwise_mul(J^T*J, di*di^T)
            // J^T*b := (J * d_inv)^T * b = d_inv^T * J^T*b = eltwise_mul(J^T*b, di)

            // scaling J^T*J
            for (auto& ijv : jtj.ijValue)
            {
                Point2i bpt = ijv.first;
                Matx66d& m = ijv.second;
                for (int i = 0; i < 6; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        Point2i pt(bpt.x * 6 + i, bpt.y * 6 + j);
                        m(i, j) *= di[pt.x] * di[pt.y];
                    }
                }
            }

            // scaling J^T*b
            for (int i = 0; i < nVars; i++)
            {
                jtb[i] *= di[i];
            }
        }

        double gradientMax = 0.0;
        // gradient max
        for (int i = 0; i < nVars; i++)
        {
            gradientMax = std::max(gradientMax, abs(jtb[i]));
        }

        // Save original diagonal of jtj matrix for LevMarq
        std::vector<double> diag(nVars);
        for (int i = 0; i < nVars; i++)
        {
            diag[i] = jtj.valElem(i, i);
        }
        
        // Solve using LevMarq and get delta transform
        bool enough = false;

        decltype(poseGraph.nodes) tempNodes = poseGraph.nodes;

        while (!enough && !done)
        {
            // form LevMarq matrix
            std::vector<double> lmDiag(nVars);
            for (int i = 0; i < nVars; i++)
            {
                double v = diag[i];

                //double ld = lambdaLevMarq * (v + coeffILM);
                double ld = std::min(max(v * lambdaLevMarq, minDiag), maxDiag);

                lmDiag[i] = ld;

                jtj.refElem(i, i) = v + ld;
            }

            std::cout << std::endl;

            std::cout << "sparse solve...";

            // use double or convert everything to float
            std::vector<double> x;
            bool solved = kinfu::sparseSolve(jtj, Mat(jtb), x);

            std::cout << "solve finished: " << std::endl;

            double costChange = 0.0;
            double jacCostChange = 0.0;
            double stepQuality = 0.0;
            double xNorm2 = 0.0;
            if (solved)
            {
                jacCostChange = calcJacCostChange(x, lmDiag);

                // x squared norm
                for (int i = 0; i < nVars; i++)
                {
                    xNorm2 += x[i] * x[i];
                }

                // undo jacobi scaling
                if (jacobiScaling)
                {
                    for (int i = 0; i < nVars; i++)
                    {
                        x[i] *= di[i];
                    }
                }

                tempNodes = poseGraph.nodes;

                // Update temp nodes using x
                for (int i = 0; i < nVarNodes; i++)
                {
                    Vec6d dx(&x[i * 6]);
                    Vec3d deltaRot(dx[0], dx[1], dx[2]), deltaTrans(dx[3], dx[4], dx[5]);
                    Pose3d& p = tempNodes.at(placesIds[i]).se3Pose;
                    
                    p.vq = (Quatd(0, deltaRot[0], deltaRot[1], deltaRot[2]).exp() * p.getQuat()).toVec();
                    p.t += deltaTrans;
                }

                // calc energy with temp nodes
                energy = calcEnergy(tempNodes);

                costChange = oldEnergy - energy;

                stepQuality = costChange / jacCostChange;

                std::cout << "#LM#" << iter;
                std::cout << " energy: " << energy;
                std::cout << " deltaEnergy: " << costChange;
                std::cout << " deltaEqEnergy: " << jacCostChange;
                std::cout << " max(J^T*b): " << gradientMax;
                std::cout << " norm2(x): " << xNorm2;
                std::cout << " deltaEnergy/energy: " << costChange / energy;
            }
            else
            {
                std::cout << "not solved" << std::endl;
            }

            std::cout << std::endl;

            if (!solved || costChange < 0)
            {
                // failed to optimize, increase lambda and repeat

                lambdaLevMarq *= lmUpFactor;
                lmUpFactor *= 2.0;

                std::cout << "LM up: " << lambdaLevMarq << ", old energy = " << oldEnergy << std::endl;
            }
            else
            {
                // optimized successfully, decrease lambda and set variables for next iteration
                enough = true;

                lambdaLevMarq *= std::max(1.0 / 3.0, 1.0 - pow(2.0 * stepQuality - 1.0, 3));
                lmUpFactor = initialLmUpFactor;

                smallGradient = (gradientMax < maxGradientTolerance);
                smallStep = (xNorm2 < stepNorm2Tolerance);
                smallEnergyDelta = (costChange / energy < relEnergyDeltaTolerance);

                poseGraph.nodes = tempNodes;

                std::cout << "#" << iter;
                std::cout << " energy: " << energy;
                std::cout << std::endl;

                oldEnergy = energy;

                std::cout << "LM down: " << lambdaLevMarq;
                std::cout << " step quality: " << stepQuality;
                std::cout << std::endl;
            }

            iter++;

            tooLong = (iter >= maxIterations);

            done = tooLong || smallGradient || smallStep || smallEnergyDelta;
        }

        }
        //writePg(poseGraph, format("C:\\Temp\\g2opt\\it%03d.obj", iter));


    }

    // TODO: set timers & produce report
    bool found = smallGradient || smallStep || smallEnergyDelta;

    std::cout << "Finished:";
    if (!found)
        std::cout << " not";
    std::cout << " found" << std::endl;
    std::vector < std::string > txtFlags;
    if (smallGradient)
        txtFlags.push_back("smallGradient");
    if (smallStep)
        txtFlags.push_back("smallStep");
    if (smallEnergyDelta)
        txtFlags.push_back("smallEnergyDelta");
    if (tooLong)
        txtFlags.push_back("tooLong");

    std::cout << "(";
    for (const auto& t : txtFlags)
        std::cout << " " << t;
    std::cout << ")" << std::endl;
}

void Optimizer::CeresOptimize(PoseGraph& poseGraph)
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
