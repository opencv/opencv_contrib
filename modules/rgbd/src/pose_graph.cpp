// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "pose_graph.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>


namespace cv
{
namespace kinfu
{

// Cholesky decomposition of symmetrical 6x6 matrix
static inline cv::Matx66d llt6(Matx66d m)
{
    Matx66d L;
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < (i + 1); j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += L(i, k) * L(j, k);

            if (i == j)
                L(i, i) = sqrt(m(i, i) - sum);
            else
                L(i, j) = (1.0 / L(j, j) * (m(i, j) - sum));
        }
    }
    return L;
}

PoseGraphEdge::PoseGraphEdge(int _sourceNodeId, int _targetNodeId, const Affine3f& _transformation,
                             const Matx66f& _information) :
                             sourceNodeId(_sourceNodeId),
                             targetNodeId(_targetNodeId),
                             pose(_transformation.rotation(), _transformation.translation()),
                             information(_information),
                             sqrtInfo(llt6(_information))
{ }

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
                if (nodesVisited.count(nextNodeId) == 0)
                {
                    nodesToVisit.push_back(nextNodeId);
                }
            }
        }
    }

    isGraphConnected = (int(nodesVisited.size()) == numNodes);
    //std::cout << "nodesVisited: " << nodesVisited.size();
    //std::cout << " IsGraphConnected: " << isGraphConnected << std::endl;
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

PoseGraph::PoseGraph(const std::string& g2oFileName) :
    nodes(), edges()
{
    readG2OFile(g2oFileName);
}

static Affine3d readAffine(std::istream& input)
{
    Vec3d p;
    Vec4d q;
    input >> p[0] >> p[1] >> p[2];
    input >> q[1] >> q[2] >> q[3] >> q[0];
    // Normalize the quaternion to account for precision loss due to
    // serialization.
    return Affine3d(Quatd(q).toRotMat3x3(), p);
};

// Rewritten from Ceres pose graph demo: https://ceres-solver.org/
void PoseGraph::readG2OFile(const std::string& g2oFileName)
{
    nodes.clear(); edges.clear();

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


// Writes edge-only model of how nodes are located in space
void PoseGraph::writeToObjFile(const std::string& fname) const
{
    std::fstream of(fname, std::fstream::out);
    for (const auto& n : nodes)
    {
        Point3d d = n.second.getPose().translation();
        of << "v " << d.x << " " << d.y << " " << d.z << std::endl;
    }
    for (const auto& e : edges)
    {
        int sid = e.sourceNodeId, tid = e.targetNodeId;
        of << "l " << sid + 1 << " " << tid + 1 << std::endl;
    }
    of.close();
};

//////////////////////////
// Optimization itself //
////////////////////////

// matrix form of Im(a)
const Matx44d M_Im{ 0, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1 };

// matrix form of conjugation
const Matx44d M_Conj{ 1,  0,  0,  0,
                      0, -1,  0,  0,
                      0,  0, -1,  0,
                      0,  0,  0, -1 };

// matrix form of quaternion multiplication from left side
static inline Matx44d m_left(Quatd q)
{
    // M_left(a)* V(b) =
    //    = (I_4 * a0 + [ 0 | -av    [    0 | 0_1x3
    //                   av | 0_3] +  0_3x1 | skew(av)]) * V(b)

    float w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w, -z,  y,
             y,  z,  w, -x,
             z, -y,  x,  w };
}

// matrix form of quaternion multiplication from right side
static inline Matx44d m_right(Quatd q)
{
    // M_right(b)* V(a) =
    //    = (I_4 * b0 + [ 0 | -bv    [    0 | 0_1x3
    //                   bv | 0_3] +  0_3x1 | skew(-bv)]) * V(a)

    float w = q.w, x = q.x, y = q.y, z = q.z;
    return { w, -x, -y, -z,
             x,  w,  z, -y,
             y, -z,  w,  x,
             z,  y, -x,  w };
}

static inline Matx43d expQuatJacobian(Quatd q)
{
    double w = q.w, x = q.x, y = q.y, z = q.z;
    return Matx43d(-x, -y, -z,
                    w,  z, -y,
                   -z,  w,  x,
                    y, -x,  w);
}

// concatenate matrices vertically
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m + k, n> concatVert(const Matx<_Tp, m, n>& a, const Matx<_Tp, k, n>& b)
{
    Matx<_Tp, m + k, n> res;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(m + i, j) = b(i, j);
        }
    }
    return res;
}

// concatenate matrices horizontally
template<typename _Tp, int m, int n, int k> static inline
Matx<_Tp, m, n + k> concatHor(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, k>& b)
{
    Matx<_Tp, m, n + k> res;

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            res(i, j) = a(i, j);
        }
    }
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < k; j++)
        {
            res(i, n + j) = b(i, j);
        }
    }
    return res;
}


static inline double poseError(Quatd sourceQuat, Vec3d sourceTrans, Quatd targetQuat, Vec3d targetTrans,
                               Quatd rotMeasured, Vec3d transMeasured, Matx66d sqrtInfoMatrix, bool needJacobians,
                               Matx<double, 6, 4>& sqj, Matx<double, 6, 3>& stj,
                               Matx<double, 6, 4>& tqj, Matx<double, 6, 3>& ttj,
                               Vec6d& res)
{
    // err_r = 2*Im(conj(rel_r) * measure_r) = 2*Im(conj(target_r) * source_r * measure_r)
    // err_t = conj(source_r) * (target_t - source_t) * source_r - measure_t

    Quatd sourceQuatInv = sourceQuat.conjugate();
    Vec3d deltaTrans = targetTrans - sourceTrans;

    Quatd relativeQuat = sourceQuatInv * targetQuat;
    Vec3d relativeTrans = sourceQuatInv.toRotMat3x3(cv::QUAT_ASSUME_UNIT) * deltaTrans;

    //! Definition should actually be relativeQuat * rotMeasured.conjugate()
    Quatd deltaRot = relativeQuat.conjugate() * rotMeasured;

    Vec3d terr = relativeTrans - transMeasured;
    Vec3d rerr = 2.0 * Vec3d(deltaRot.x, deltaRot.y, deltaRot.z);
    Vec6d rterr(terr[0], terr[1], terr[2], rerr[0], rerr[1], rerr[2]);

    res = sqrtInfoMatrix * rterr;

    if (needJacobians)
    {
        // d(err_r) = 2*Im(d(conj(target_r) * source_r * measure_r)) = < measure_r is constant > =
        // 2*Im((conj(d(target_r)) * source_r + conj(target_r) * d(source_r)) * measure_r)
        // d(target_r) == 0:
        //  # d(err_r) = 2*Im(conj(target_r) * d(source_r) * measure_r)
        //  # V(d(err_r)) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r)) * V(d(source_r))
        //  # d(err_r) / d(source_r) = 2 * M_Im * M_right(measure_r) * M_left(conj(target_r))
        Matx34d drdsq = 2.0 * (m_right(rotMeasured) * m_left(targetQuat.conjugate())).get_minor<3, 4>(1, 0);

        // d(source_r) == 0:
        //  # d(err_r) = 2*Im(conj(d(target_r)) * source_r * measure_r)
        //  # V(d(err_r)) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj * V(d(target_r))
        //  # d(err_r) / d(target_r) = 2 * M_Im * M_right(source_r * measure_r) * M_Conj
        Matx34d drdtq = 2.0 * (m_right(sourceQuat * rotMeasured) * M_Conj).get_minor<3, 4>(1, 0);

        // d(err_t) = d(conj(source_r) * (target_t - source_t) * source_r) =
        // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
        // conj(d(source_r)) * (target_t - source_t) * source_r +
        // conj(source_r) * (target_t - source_t) * d(source_r) =
        // <conj(a*b) == conj(b)*conj(a), conj(target_t - source_t) = - (target_t - source_t), 2 * Im(x) = (x - conj(x))>
        // conj(source_r) * (d(target_t) - d(source_t)) * source_r +
        // 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
        // d(*_t) == 0:
        //  # d(err_t) = 2 * Im(conj(source_r) * (target_t - source_t) * d(source_r))
        //  # V(d(err_t)) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t)) * V(d(source_r))
        //  # d(err_t) / d(source_r) = 2 * M_Im * M_left(conj(source_r) * (target_t - source_t))
        Matx34d dtdsq = 2 * m_left(sourceQuatInv * Quatd(0, deltaTrans[0], deltaTrans[1], deltaTrans[2])).get_minor<3, 4>(1, 0);
        // deltaTrans is rotated by sourceQuatInv, so the jacobian is rot matrix of sourceQuatInv by +1 or -1
        Matx33d dtdtt = sourceQuatInv.toRotMat3x3(QUAT_ASSUME_UNIT);
        Matx33d dtdst = -dtdtt;

        Matx33d z;
        sqj = concatVert(dtdsq, drdsq);
        tqj = concatVert(Matx34d(), drdtq);
        stj = concatVert(dtdst, z);
        ttj = concatVert(dtdtt, z);

        stj = sqrtInfoMatrix * stj;
        ttj = sqrtInfoMatrix * ttj;
        sqj = sqrtInfoMatrix * sqj;
        tqj = sqrtInfoMatrix * tqj;
    }

    return res.ddot(res);
}


// estimate current energy
double PoseGraph::calcEnergy(const std::map<int, PoseGraphNode>& newNodes) const
{
    double totalErr = 0;
    for (const auto& e : edges)
    {
        Pose3d srcP = newNodes.at(e.getSourceNodeId()).se3Pose;
        Pose3d tgtP = newNodes.at(e.getTargetNodeId()).se3Pose;

        Vec6d res;
        Matx<double, 6, 3> stj, ttj;
        Matx<double, 6, 4> sqj, tqj;
        double err = poseError(srcP.getQuat(), srcP.t, tgtP.getQuat(), tgtP.t,
                               e.pose.getQuat(), e.pose.t, e.sqrtInfo, /* needJacobians = */ false,
                               sqj, stj, tqj, ttj, res);

        totalErr += err;
    }
    return totalErr * 0.5;
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
static inline double calcJacCostChange(const std::vector<double>& jtb,
                                       const std::vector<double>& x,
                                       const std::vector<double>& lmDiag)
{
    double jdiag = 0.0;
    for (size_t i = 0; i < x.size(); i++)
    {
        jdiag += x[i] * (jtb[i] - lmDiag[i] * x[i]);
    }
    double costChange = jdiag * 0.5;
    return costChange;
};


void PoseGraph::optimize()
{
    if (!isValid())
    {
        CV_Error(Error::StsBadArg,
            "Invalid PoseGraph that is either not connected or has invalid nodes");
        return;
    }

    int numNodes = getNumNodes();
    int numEdges = getNumEdges();

    // Allocate indices for nodes
    std::vector<int> placesIds;
    std::map<int, int> idToPlace;
    for (const auto& ni : nodes)
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
        std::cout << "PoseGraph contains no non-constant nodes, skipping optimization" << std::endl;
        return;
    }

    if (numEdges == 0)
    {
        std::cout << "PoseGraph has no edges, no optimization to be done" << std::endl;
        return;
    }

    std::cout << "Optimizing PoseGraph with " << numNodes << " nodes and " << numEdges << " edges" << std::endl;

    size_t nVars = nVarNodes * 6;
    BlockSparseMat<double, 6, 6> jtj(nVarNodes);
    std::vector<double> jtb(nVars);
 
    double energy = calcEnergy(nodes);
    double startEnergy = energy;
    double oldEnergy = energy;

    std::cout << "#s" << " energy: " << energy << std::endl;

    // options
    // stop conditions
    const unsigned int maxIterations = 100;
    const double minGradientTolerance = 1e-6;
    const double stepNorm2Tolerance = 1e-6;
    const double relEnergyDeltaTolerance = 1e-6;
    // normalize jacobian columns for better conditioning
    const bool jacobiScaling = true;
    const double minDiag = 1e-6;
    const double maxDiag = 1e32;

    const double initialLambdaLevMarq = 0.0001;
    const double initialLmUpFactor = 2.0;
    const double initialLmDownFactor = 3.0;

    // finish reasons
    bool tooLong          = false; // => not found
    bool smallGradient    = false; // => found
    bool smallStep        = false; // => found
    bool smallEnergyDelta = false; // => found

    // column scale inverted, for jacobian scaling
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
            Pose3d p = nodes.at(id).se3Pose;
            Matx43d qj = expQuatJacobian(p.getQuat());
            // x node layout is (rot_x, rot_y, rot_z, trans_x, trans_y, trans_z)
            // pose layout is (q_w, q_x, q_y, q_z, trans_x, trans_y, trans_z)
            Matx<double, 7, 6> j = concatVert(concatHor(qj, Matx43d()),
                                              concatHor(Matx33d(), Matx33d::eye()));
            cachedJac.push_back(j);
        }

        // fill jtj and jtb
        for (const auto& e : edges)
        {
            int srcId = e.getSourceNodeId(), dstId = e.getTargetNodeId();
            const PoseGraphNode& srcNode = nodes.at(srcId);
            const PoseGraphNode& dstNode = nodes.at(dstId);

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

        decltype(nodes) tempNodes = nodes;

        while (!enough && !done)
        {
            // form LevMarq matrix
            std::vector<double> lmDiag(nVars);
            for (int i = 0; i < nVars; i++)
            {
                double v = diag[i];
                double ld = std::min(max(v * lambdaLevMarq, minDiag), maxDiag);
                lmDiag[i] = ld;
                jtj.refElem(i, i) = v + ld;
            }

            std::cout << std::endl;

            std::cout << "sparse solve...";

            // use double or convert everything to float
            std::vector<double> x;
            bool solved = kinfu::sparseSolve(jtj, jtb, x, false);

            std::cout << (solved ? "OK" : "FAIL") << std::endl;

            double costChange = 0.0;
            double jacCostChange = 0.0;
            double stepQuality = 0.0;
            double xNorm2 = 0.0;
            if (solved)
            {
                jacCostChange = calcJacCostChange(jtb, x, lmDiag);

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

                tempNodes = nodes;

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
                std::cout << std::endl;
            }

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

                smallGradient = (gradientMax < minGradientTolerance);
                smallStep = (xNorm2 < stepNorm2Tolerance);
                smallEnergyDelta = (costChange / energy < relEnergyDeltaTolerance);

                nodes = tempNodes;

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

    }

    // TODO: set timers
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
    std::cout << " )" << std::endl;
}

}  // namespace kinfu
}  // namespace cv
