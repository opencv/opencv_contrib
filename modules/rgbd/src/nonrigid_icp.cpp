// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include "precomp.hpp"
#include "nonrigid_icp.hpp"

#define MAD_SCALE 1.4826f
#define TUKEY_B 4.6851f
#define HUBER_K 1.345f

namespace cv
{
namespace dynafu
{
using namespace kinfu;

NonRigidICP::NonRigidICP(const Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations) :
iterations(_iterations), volume(_volume), intrinsics(_intrinsics)
{}

class ICPImpl : public NonRigidICP
{
public:
    ICPImpl(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations);

    virtual bool estimateWarpNodes(WarpField& warp, const Affine3f &pose,
                                   InputArray vertImage, InputArray normImage,
                                   InputArray oldPoints, InputArray oldNormals,
                                   InputArray newPoints, InputArray newNormals) const override;

    virtual ~ICPImpl() {}
};


ICPImpl::ICPImpl(const Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations) :
NonRigidICP(_intrinsics, _volume, _iterations)
{}


static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}


//TODO: optimize it and (or) madEstimate
static float median(std::vector<float>& v)
{
    size_t n = v.size()/2;
    if(n == 0) return 0;

    std::nth_element(v.begin(), v.begin()+n, v.end());
    float vn = v[n];

    if(n%2 == 0)
    {
        std::nth_element(v.begin(), v.begin()+n-1, v.end());
        return (vn+v[n-1])/2.f;
    }
    else
    {
        return vn;
    }
}

static float madEstimate(std::vector<float>& v)
{
    float med = median(v);
    std::for_each(v.begin(), v.end(), [med](float& x) {x = std::abs(x - med); });
    return MAD_SCALE * median(v);
}


static float tukeyWeight(float v, float sigma = 1.f)
{
    v /= sigma;
    const float b2 = TUKEY_B * TUKEY_B;
    if (std::abs(v) <= TUKEY_B)
    {
        float y = 1.f - (v*v) / b2;
        return y*y;
    }
    else return 0;
}


// Works the same as above but takes squared norm
// Can be optimized in the future to interpolated LUTs
static float tukeyWeightSq(float vv, float sigma = 1.f)
{
    vv /= sigma*sigma;
    const float b2 = TUKEY_B * TUKEY_B;
    if (vv <= b2)
    {
        float y = 1.f - vv / b2;
        return y * y;
    }
    else return 0;
}


static float huberWeight(float vnorm, float sigma = 1.f)
{
    if (std::abs(sigma) < 0.001f) return 0.f;
    float x = (float)std::abs(vnorm/sigma);
    return (x > HUBER_K) ? HUBER_K / x : 1.f;
}


// Works the same as above but takes squared norm
// Can be optimized in the future to interpolated LUTs
static float huberWeightSq(float vnorm2, float sigma = 1.f)
{
    if (std::abs(sigma) < 0.001f) return 0.f;
    float vnorm = std::sqrt(vnorm2);
    float x = (float)std::abs(vnorm / sigma);
    return (x > HUBER_K) ? HUBER_K / x : 1.f;
}

/*
TODO: remove it when everything works
-----------------------------
-----------------------------

static void fillRegularization(Mat_<float>& A_reg, Mat_<float>& b_reg, WarpField& warp,
                               int totalNodes, std::vector<int> baseIndices)
{
    int nLevels = warp.n_levels;
    int k = warp.k;

    const std::vector<Ptr<WarpNode> >& warpNodes = warp.getNodes();
    // Accumulate regularisation term for each node in the heiarchy
    const std::vector<std::vector<Ptr<WarpNode> > >& regNodes = warp.getGraphNodes();
    const std::vector<std::vector<NodeNeighboursType> >& regGraph = warp.getRegGraph();

    // populate residuals for each edge in the graph to calculate sigma
    std::vector<float> reg_residuals;
    float RegEnergy = 0;
    int numEdges = 0;
    for(int l = 0; l < (nLevels-1); l++)
    {
        const std::vector<NodeNeighboursType>& level = regGraph[l];

        const std::vector<Ptr<WarpNode> >& currentLevelNodes = (l == 0)? warpNodes : regNodes[l-1];

        const std::vector<Ptr<WarpNode> >& nextLevelNodes = regNodes[l];

        std::cout << currentLevelNodes.size() << " " << nextLevelNodes.size() << std::endl;


        for(size_t node = 0; node < level.size(); node++)
        {
            const NodeNeighboursType& children = level[node];
            Vec3f nodePos = currentLevelNodes[node]->pos;
            Affine3f nodeTransform = currentLevelNodes[node]->transform;

            for(int c = 0; c < k; c++)
            {
                const int child = children[c];
                Vec3f childPos = nextLevelNodes[child]->pos;
                Vec3f childTranslation = nextLevelNodes[child]->transform.translation();

                Vec3f re = nodeTransform * (childPos - nodePos) + nodePos
                           - (childTranslation + childPos);
                numEdges++;

                reg_residuals.push_back((float)norm(re));
                RegEnergy += (float)norm(re);
            }
        }
    }

    Mat_<float> J_reg(6*numEdges, 6*totalNodes, 0.f);

    std::cout << "Total reg energy: " << RegEnergy << ", Average: " << RegEnergy/numEdges << std::endl;

    float reg_med = median(reg_residuals);
    std::for_each(reg_residuals.begin(), reg_residuals.end(),
                  [reg_med](float& x)
                  {
                      x = std::abs(x-reg_med);
                  });

    float reg_sigma = MAD_SCALE * median(reg_residuals);
    std::cout << "[Reg] Sigma: " << reg_sigma << " from " << reg_residuals.size() << " residuals " << std::endl;

    for(int l = 0; l < (nLevels-1); l++)
    {
        const std::vector<NodeNeighboursType>& level = regGraph[l];

        const NodeVectorType& currentLevelNodes = (l == 0)? warpNodes : regNodes[l-1];
        const NodeVectorType& nextLevelNodes = regNodes[l];

        for(size_t node = 0; node < level.size(); node++)
        {
            const NodeNeighboursType& children = level[node];
            Vec3f nodePos = currentLevelNodes[node]->pos;
            Affine3f nodeTransform = currentLevelNodes[node]->transform;

            int parentIndex = baseIndices[l]+6*(int)node;

            for(int edge = 0; edge < k; edge++)
            {
                const int child = children[edge];
                const Ptr<WarpNode> childNode = nextLevelNodes[child];
                Vec3f childTranslation = childNode->transform.translation();

                Vec3f childPos = childNode->pos;
                Vec3f transformedChild = nodeTransform * (childPos - nodePos);
                Vec3f r_edge = transformedChild + nodePos - (childTranslation + childPos);

                if(norm(r_edge) > 0.01) continue;

                float robustWeight = huberWeight(r_edge, reg_sigma);

                // take sqrt since radius is stored as squared distance
                float edgeWeight = sqrt(min(childNode->radius, currentLevelNodes[node]->radius));

                Vec3f v1 = transformedChild.cross(r_edge);


                float w = 1 * robustWeight * edgeWeight;
                b_reg(parentIndex+0) += -w * v1[0];
                b_reg(parentIndex+1) += -w * v1[1];
                b_reg(parentIndex+2) += -w * v1[2];
                b_reg(parentIndex+3) += -w * r_edge[0];
                b_reg(parentIndex+4) += -w * r_edge[1];
                b_reg(parentIndex+5) += -w * r_edge[2];

                int childIndex = baseIndices[l+1]+6*child;
                Vec3f v2 = childTranslation.cross(r_edge);
                b_reg(childIndex+0) += w * v2[0];
                b_reg(childIndex+1) += w * v2[1];
                b_reg(childIndex+2) += w * v2[2];
                b_reg(childIndex+3) += w * r_edge[0];
                b_reg(childIndex+4) += w * r_edge[1];
                b_reg(childIndex+5) += w * r_edge[2];


                Matx33f Tj_Vj_Vi_cross(0, -transformedChild[2], transformedChild[1],
                                       transformedChild[2], 0, -transformedChild[0],
                                       -transformedChild[1], transformedChild[0], 0);
                Matx33f tj_cross(0, -childTranslation[2], childTranslation[1],
                                 childTranslation[2], 0, -childTranslation[0],
                                 -childTranslation[1], childTranslation[0], 0);

                // place top left elements
                Matx33f top_left = Tj_Vj_Vi_cross * tj_cross;
                for(int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                    {
                        A_reg(parentIndex+i, childIndex+j) += w * top_left(i, j);
                        A_reg(childIndex+i, parentIndex+j) += w * top_left(i, j);
                    }

                // place top right elements
                for(int i = 0; i < 3; i++)
                    for(int j = 0; j < 3; j++)
                    {
                        A_reg(parentIndex+i, childIndex+j+3) += - w * Tj_Vj_Vi_cross(i, j);
                        A_reg(childIndex+i, parentIndex+j+3) += - w * Tj_Vj_Vi_cross(i, j);
                    }

                // place bottom left elements
                for(int i = 0; i < 3; i++)
                    for(int j = 0; j < 3; j++)
                    {
                        A_reg(parentIndex+i+3, childIndex+j) += w * tj_cross(i, j);
                        A_reg(childIndex+i+3, parentIndex+j) += w * tj_cross(i, j);
                    }

                // place bottom right elements (which is -I_3)
                for(int i = 0; i < 3; i++)
                {
                    A_reg(parentIndex+i+3, childIndex+i+3) += -w;
                    A_reg(childIndex+i+3, parentIndex+i+3) += -w;
                }

            }
        }
    }
}

-----------------------------
-----------------------------
*/

// GOOD CODE STARTS HERE --------------------------------------------------------------------------

inline std::pair<bool, Point3f> interpolateP3f(Point2f pt, const ptype* data, size_t width)
{
    // bilinearly interpolate newPoints under newCoords point
    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);
    float tx = pt.x - xi, ty = pt.y - yi;

    const ptype* prow0 = data + width *  yi;
    const ptype* prow1 = data + width * (yi + 1);

    Point3f p00 = fromPtype(prow0[xi + 0]);
    Point3f p01 = fromPtype(prow0[xi + 1]);
    Point3f p10 = fromPtype(prow1[xi + 0]);
    Point3f p11 = fromPtype(prow1[xi + 1]);

    //do not fix missing data
    if (!(fastCheck(p00) && fastCheck(p01) &&
          fastCheck(p10) && fastCheck(p11)))
        return { false, Point3f() };

    Point3f p0 = p00 + tx * (p01 - p00);
    Point3f p1 = p10 + tx * (p11 - p10);

    return { true, (p0 + ty * (p1 - p0)) };
}

// Per-pixel structure to keep neighbours and weights
// neighbours[i] < 0 indicates last neighbour
struct WeightsNeighbours
{
    WeightsNeighbours()
    {
        for (int i = 0; i < DYNAFU_MAX_NEIGHBOURS; i++)
        {
            neighbours[i] = -1;
            weights[i] = 0;
        }
    }
    int    neighbours[DYNAFU_MAX_NEIGHBOURS];
    float     weights[DYNAFU_MAX_NEIGHBOURS];
};


static DualQuaternion dampedDQ(int nNodes, float wsum, float coeff)
{
    float wdamp = nNodes - wsum;
    return UnitDualQuaternion().dq() * wdamp * coeff;
}


// residual norm for sigma estimation
static float resNormEdge(const WarpNode& actNode, const WarpNode& pasNode, const bool disableCentering)
{
    UnitDualQuaternion dqAct = actNode.transform;
    UnitDualQuaternion dqPas = pasNode.transform;
    Point3f cAct = actNode.pos;
    Point3f cPas = pasNode.pos;
    // int nAct = actNode.place, nPas = pasNode.place;

    UnitDualQuaternion dqActCentered, dqPasCentered;
    Point3f tPas = dqPas.getT();
    if (disableCentering)
    {
        dqActCentered = dqAct;
        dqPasCentered = dqPas;
    }
    else
    {
        dqActCentered = dqAct.centered(cAct);
        dqPasCentered = dqPas.centered(cPas);
    }

    Vec3f vMoveActC = dqActCentered.apply(cPas);

    Vec3f vMovePasC;
    if (disableCentering)
    {
        vMovePasC = dqPasCentered.apply(cPas);
    }
    else
    {
        vMovePasC = cPas + tPas;
    }

    // -delta_x from Gauss-Newton
    Vec3f vDiffActC = -vMoveActC + vMovePasC;

    float resNorm = std::sqrt(vDiffActC.dot(vDiffActC));

    return resNorm;
}


// Passive node's center is moved by both active and passive nodes
static void fillEdge(BlockSparseMat<float, 6, 6>& jtj, std::vector<float>& jtb,
                     const WarpNode& actNode, const WarpNode& pasNode,
                     const float regTermWeight, const float huberSigma,
                     const bool disableCentering, const bool useExp, const bool useNormApply)
{
    UnitDualQuaternion dqAct = actNode.transform;
    UnitDualQuaternion dqPas = pasNode.transform;
    Point3f cAct = actNode.pos;
    Point3f cPas = pasNode.pos;
    size_t nAct = actNode.place, nPas = pasNode.place;
    cv::Matx<float, 8, 6> jAct = actNode.cachedJac;
    cv::Matx<float, 8, 6> jPas = pasNode.cachedJac;

    UnitDualQuaternion dqActCentered, dqPasCentered;
    Point3f tPas = dqPas.getT();
    if (disableCentering)
    {
        dqActCentered = dqAct;
        dqPasCentered = dqPas;
    }
    else
    {
        dqActCentered = dqAct.centered(cAct);
        dqPasCentered = dqPas.centered(cPas);
    }
    
    Matx<float, 3, 8> jNormApplyActC, jNormApplyPasC, jUnitApplyActC, jUnitApplyPasC;

    jNormApplyActC = dqActCentered.dq().j_normapply(cPas);
    jUnitApplyActC = dqActCentered.j_apply(cPas);
    jNormApplyPasC = dqPasCentered.dq().j_normapply(cPas);
    jUnitApplyPasC = dqPasCentered.j_apply(cPas);

    Matx<float, 3, 8> jApplyActC, jApplyPasC;
    if (useNormApply)
    {
        jApplyActC = jNormApplyActC;
        jApplyPasC = jNormApplyPasC;
    }
    else
    {
        jApplyActC = jUnitApplyActC;
        jApplyPasC = jUnitApplyPasC;
    }

    Matx<float, 3, 6> jMoveActC = jApplyActC * jAct;
    Vec3f vMoveActC = dqActCentered.apply(cPas);

    Matx<float, 3, 6> jMovePasC;
    Vec3f vMovePasC;
    if (disableCentering)
    {
        vMovePasC = dqPasCentered.apply(cPas);
        jMovePasC = jApplyPasC * jPas;
    }
    else
    {
        vMovePasC = cPas + tPas;
        if (!useExp)
        {
            jMovePasC = cv::Matx<float, 3, 6>({ 0, 0, 0, 1, 0, 0,
                                                0, 0, 0, 0, 1, 0,
                                                0, 0, 0, 0, 0, 1 });
        }
        else
        {
            jMovePasC = jApplyPasC * jPas;
        }
    }

    // -delta_x from Gauss-Newton
    Vec3f vDiffActC = - vMoveActC + vMovePasC;

    float weight = 1.f;
    if (huberSigma >= 0.f)
    {
        // TODO: try real values of sigma, gather histogram
        // maybe split by levels
        weight = huberWeightSq(vDiffActC.dot(vDiffActC), huberSigma);
    }

    // Building J^T * J and J^T*b

    //TODO: add node's own weight
    //based on how many pts it covers
    //TODO: less level, less difference is important
    //on the highest level we shouldn't care about that difference at all
    //DEBUG
    weight *= regTermWeight;
    //weight = weight*reg_term_weight*min(act_node.amt, c_node["amt"])/nPts
    weight *= min(actNode.radius, pasNode.radius);

    // emulating jtjEdge = jEdge.t() * jEdge
    // where jEdge is a jacobian of an edge
    // act node goes positive, c node goes negative

    jtj.refBlock(nPas, nPas) +=   weight * jMovePasC.t() * jMovePasC;
    jtj.refBlock(nAct, nAct) +=   weight * jMoveActC.t() * jMoveActC;
    jtj.refBlock(nAct, nPas) += - weight * jMoveActC.t() * jMovePasC;
    jtj.refBlock(nPas, nAct) += - weight * jMovePasC.t() * jMoveActC;

    // emulating jEdge.t() * vDiffActC
    Vec6f jtbPas, jtbAct;
    jtbPas = - jMovePasC.t() * vDiffActC;
    jtbAct =   jMoveActC.t() * vDiffActC;

    for (int i = 0; i < 6; i++)
    {
        jtb[6 * nPas + i] += weight * jtbPas[i];
        jtb[6 * nAct + i] += weight * jtbAct[i];
    }
}


static void fillVertex(BlockSparseMat<float, 6, 6>& jtj,
                       std::vector<float>& jtb,
                       const std::vector<WarpNode>& nodes,
                       WeightsNeighbours knns,
                       // per-vertex values
                       const Point3f& inp,
                       const float pointPlaneDistance,
                       const Point3f& outVolN,
                       const DualQuaternion& dqsum,
                       const float weight,
                       // algorithm params
                       const float normPenalty,
                       const bool decorrelate)
{
    size_t knn = nodes.size();

    // run through DQB
    float wsum = 0;
     
    std::vector<size_t> places(knn);
    std::vector<Matx<float, 8, 6>> jPerNodeWeighted(knn);
    for (int k = 0; k < knn; k++)
    {
        //int nei = knns.neighbours[k];
        float w = knns.weights[k];

        const WarpNode& node = nodes[k];

        places[k] = node.place;
        jPerNodeWeighted[node.place] = w * node.cachedJac;
        wsum += w;
    }

    // jacobian of normalization+application to a point
    Matx<float, 3, 8> jNormApply = dqsum.j_normapply(inp);

    // jacobian of norm penalty
    Vec2f norm = dqsum.norm();
    Matx<float, 2, 8> jNormPenalty = dqsum.j_norm();
    Vec2f normDiff(1 - norm[0], 0 - norm[1]);

    std::vector<Matx<float, 3, 6>> jPerNode(knn);

    std::vector<Matx<float, 2, 6>> jNodeNormPenalty(knn);

    // emulating jNormApply * jPerNodeWeighted
    for (int k = 0; k < knn; k++)
    {
        jPerNode[k] = jNormApply * jPerNodeWeighted[k];
        jNodeNormPenalty[k] = jNormPenalty * jPerNodeWeighted[k];
    }

    // no need to pass diff, pointPlaneDistance is enough
    //float pointPlaneDistance = outVolN.dot(diff);

    // for point-plane distance jacobian
    Matx33f nnt = Vec3f(outVolN) * Vec3f(outVolN).t();

    // emulate jVertex^T * jVertex and jVertex^T*b
    for (int k = 0; k < knn; k++)
    {
        size_t kplace = places[k];
        if (decorrelate)
        {
            Matx66f block = jPerNode[k].t() * nnt * jPerNode[k];
            block += normPenalty * jNodeNormPenalty[k].t() * jNodeNormPenalty[k];
            jtj.refBlock(kplace, kplace) += weight * block;
        }
        else
        {
            for (int l = 0; l < knn; l++)
            {
                Matx66f block = jPerNode[k].t() * nnt * jPerNode[l];
                block += normPenalty * jNodeNormPenalty[k].t() * jNodeNormPenalty[l];
                size_t lplace = places[l];
                jtj.refBlock(kplace, lplace) += weight * block;
            }
        }

        // no need to pass diff, pointPlaneDistance is enough
        //Vec6f jtbBlock = jPerNode[k].t() * nnt * Vec3f(diff);
        Vec6f jtbBlock = jPerNode[k].t() * Vec3f(outVolN) * pointPlaneDistance;
        jtbBlock += normPenalty * jNodeNormPenalty[k].t() * normDiff;
        for (int i = 0; i < 6; i++)
        {
            jtb[6 * kplace + i] += weight * jtbBlock[i];
        }
    }
}


// nodes = warp.getNodes();
DualQuaternion warpForVertex(const WeightsNeighbours& knns, const std::vector<Ptr<WarpNode>>& nodes,
                             const int knn, const float damping, const bool disableCentering)
{
    DualQuaternion dqsum;
    float wsum = 0; int nValid = 0;
    for (int k = 0; k < knn; k++)
    {
        int ixn = knns.neighbours[k];
        if (ixn >= 0)
        {
            float w = knns.weights[k];

            Ptr<WarpNode> node = nodes[ixn];

            // center(x) := (1+e*1/2*c)*x*(1-e*1/2*c)
            UnitDualQuaternion dqi = disableCentering ?
                node->transform :
                node->transform.centered(node->pos);

            dqsum += w * dqi.dq();
            wsum += w;
            nValid++;
        }
    }

    dqsum += dampedDQ(knn, wsum, damping);
    return dqsum;
}


// MAD sigma estimation
float estimateVertexSigma(const Mat_<float>& cachedResiduals)
{
    Size size = cachedResiduals.size();
    std::vector<float> vertResiduals;
    vertResiduals.reserve(size.area());
    for (int y = 0; y < size.height; y++)
    {
        auto row = cachedResiduals[y];
        for (int x = 0; x < size.width; x++)
        {
            float pointPlaneDistance = row[x];
            if (std::isnan(pointPlaneDistance))
                continue;

            vertResiduals.push_back(pointPlaneDistance);
        }
    }

    return madEstimate(vertResiduals);
}


float estimateVertexEnergy(const Mat_<float>& cachedResiduals,
                           const Mat_<float>& cachedWeights)
{
    Size size = cachedResiduals.size();
    float energy = 0.f;
    for (int y = 0; y < size.height; y++)
    {
        auto resrow = cachedResiduals[y];
        auto wrow   = cachedWeights  [y];
        for (int x = 0; x < size.width; x++)
        {
            float pointPlaneDistance = resrow[x];
            if (std::isnan(pointPlaneDistance))
                continue;
            float weight = wrow[x];
            energy += weight * pointPlaneDistance;
        }
    }
    return energy;
}


float estimateRegEnergy(const std::vector<std::vector<NodeNeighboursType>>& graph,
                        const std::vector<Ptr<WarpNode>>& nodes,
                        const std::vector<std::vector<Ptr<WarpNode>>>& regNodes,
                        const std::vector<float>& regSigmas,
                        const bool useHuber,
                        const bool disableCentering,
                        const bool parentMovesChild, const bool childMovesParent)
{
    float energy = 0.f;

    for (int level = 0; level < graph.size(); level++)
    {
        auto childLevelNodes = (level == 0) ? nodes : regNodes[level - 1];
        auto levelNodes = regNodes[level];
        auto levelChildIdx = graph[level];
        float sigma = regSigmas[level];

        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];

            auto children = levelChildIdx[ixn];

            for (int ixc = 0; ixc < children.size(); ixc++)
            {
                Ptr<WarpNode> child = childLevelNodes[children[ixc]];

                if (parentMovesChild)
                {
                    float rn = resNormEdge(*node, *child, disableCentering);
                    float weight = 1.f;
                    if (useHuber)
                    {
                        // TODO: try real values of sigma, gather histogram
                        // maybe split by levels
                        weight = huberWeight(rn, sigma);
                    }
                    energy += rn * weight;
                }
                if (childMovesParent)
                {
                    float rn = resNormEdge(*child, *node, disableCentering);
                    float weight = 1.f;
                    if (useHuber)
                    {
                        // TODO: try real values of sigma, gather histogram
                        // maybe split by levels
                        weight = huberWeight(rn, sigma);
                    }
                    energy += rn * weight;
                }
            }
        }
    }

    return energy;
}

std::vector<float> estimateRegSigmas(const std::vector<std::vector<NodeNeighboursType>>& graph,
                                     const std::vector<Ptr<WarpNode>>& nodes,
                                     const std::vector<std::vector<Ptr<WarpNode>>>& regNodes,
                                     const bool disableCentering,
                                     const bool parentMovesChild, const bool childMovesParent)
{
    std::vector<float> regSigmas(graph.size(), 1.f);
    for (int level = 0; level < graph.size(); level++)
    {
        auto childLevelNodes = (level == 0) ? nodes : regNodes[level - 1];
        auto levelNodes = regNodes[level];
        auto levelChildIdx = graph[level];

        std::vector<float> regLevResiduals;
        regLevResiduals.reserve(levelNodes.size());

        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];

            auto children = levelChildIdx[ixn];

            for (int ixc = 0; ixc < children.size(); ixc++)
            {
                Ptr<WarpNode> child = childLevelNodes[children[ixc]];

                if (parentMovesChild)
                {
                    float rn = resNormEdge(*node, *child, disableCentering);
                    regLevResiduals.push_back(rn);
                }
                if (childMovesParent)
                {
                    float rn = resNormEdge(*child, *node, disableCentering);
                    regLevResiduals.push_back(rn);
                }
            }
        }

        regSigmas[level] = madEstimate(regLevResiduals);
    }

    return regSigmas;
}


// use new x as delta to update cost function values
void updateNodes(const std::vector<float>& x,
                 std::vector<Ptr<WarpNode> >& nodes,
                 std::vector<std::vector<Ptr<WarpNode> > >& regNodes,
                 const bool additiveDerivative,
                 const bool useExp,
                 const bool signFix,
                 const bool signFixRelative)
{
    // for sign fix
    UnitDualQuaternion ref;
    float refnorm = std::numeric_limits<float>::max();

    for (int level = 0; level < regNodes.size() + 1; level++)
    {
        auto levelNodes = (level == 0) ? nodes : regNodes[level - 1];
        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];
            size_t place = node->place;

            // blockNode = x[6 * n:6 * (n + 1)]
            float blockNode[6];
            for (int i = 0; i < 6; i++)
            {
                blockNode[i] = x[6 * place + i];
            }

            UnitDualQuaternion dq = node->transform;
            Point3d c = node->pos;

            UnitDualQuaternion dqnew;
            if (additiveDerivative)
            {
                if (useExp)
                {
                    // the order is opposite
                    Vec3f dualx(blockNode[0], blockNode[1], blockNode[2]);
                    Vec3f realx(blockNode[3], blockNode[4], blockNode[5]);

                    node->arg[0] += dualx;
                    node->arg[1] += realx;

                    // TODO: no normalization, this dq is already normal
                    // SIC! opposite order
                    dqnew = DualQuaternion(Quaternion(0.f, node->arg[1]),
                                           Quaternion(0.f, node->arg[0])).exp().normalized();
                }
                else
                {
                    Vec3f rotparams(blockNode[0], blockNode[1], blockNode[2]);
                    Vec3f transparams(blockNode[3], blockNode[4], blockNode[5]);

                    node->arg[0] += rotparams;
                    node->arg[1] += transparams;

                    dqnew = UnitDualQuaternion(Affine3f(node->arg[0], node->arg[1]));
                }
            }
            else
            {
                UnitDualQuaternion dqit;
                if (useExp)
                {
                    // the order is the opposite to !useExp
                    Quaternion dualx(0, blockNode[0], blockNode[1], blockNode[2]);
                    Quaternion realx(0, blockNode[3], blockNode[4], blockNode[5]);

                    //TODO: no normalization, this dq is already normal
                    dqit = DualQuaternion(realx, dualx).exp().normalized();
                }
                else
                {
                    Vec3f rotParams(blockNode[0], blockNode[1], blockNode[2]);
                    Vec3f transParams(blockNode[3], blockNode[4], blockNode[5]);

                    // this function expects input vector to be a norm of full angle
                    // while it contains just 1 / 2 of an angle

                    //TODO: dqit = UnitDualQuaternion::fromRt(rotParams*2, transParams)
                    Affine3f aff(rotParams * 2, transParams);
                    dqit = UnitDualQuaternion(aff);
                }

                // sic! (2nd transform) * (1st transform)
                dqnew = dqit * dq;
            }

            node->transform = dqnew;

            if (signFixRelative && level == 0)
            {
                float norm = dqnew.dq().dot(dqnew.dq());
                if (norm < refnorm)
                {
                    refnorm = norm;
                    ref = dqnew;
                }
            }
        }

        if (signFix && level == 0)
        {
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                UnitDualQuaternion& dq = node->transform;

                dq = ref.dq().dot(dq.dq()) >= 0 ? dq : -dq;
            }
        }
    }
}


// Find a place for each node params in x vector
void placeNodesInX(const std::vector<Ptr<WarpNode>>& nodes,
                   const std::vector<std::vector<Ptr<WarpNode>>>& regNodes)
{
    size_t idx = 0;
    for (int level = 0; level < regNodes.size() + 1; level++)
    {
        auto levelNodes = (level == 0) ? nodes : regNodes[level - 1];
        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];
            node->place = idx++;
        }
    }
}


// For each node find its pose representation as rotation+translation vectors
// or as se(3) logarithm
void calcArgs(const std::vector<Ptr<WarpNode>>& nodes,
              const std::vector<std::vector<Ptr<WarpNode>>>& regNodes,
              const bool useExp)
{
    for (int level = 0; level < regNodes.size() + 1; level++)
    {
        auto levelNodes = (level == 0) ? nodes : regNodes[level - 1];
        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];
            if (useExp)
            {
                DualQuaternion log = node->transform.dq().log();
                // SIC! the opposite order
                node->arg[0] = log.dual().vec();
                node->arg[1] = log.real().vec();
            }
            else
            {
                Affine3f rt = node->transform.getRt();
                node->arg[0] = rt.rvec();
                node->arg[1] = rt.translation();
            }
        }
    }
}


void buildInWarpedInitial(const Mat_<ptype>& oldPoints,
                          const Mat_<ptype>& oldNormals,
                          const Affine3f& cam2vol,
                          // output params
                          Mat_<ptype>& ptsInWarped,
                          Mat_<ptype>& ptsInWarpedNormals,
                          Mat_<ptype>& ptsInWarpedRendered,
                          Mat_<ptype>& ptsInWarpedRenderedNormals)
{
    Size size = oldPoints.size();
    ptsInWarped.create(size);
    ptsInWarpedNormals.create(size);
    ptsInWarpedRendered.create(size);
    ptsInWarpedRenderedNormals.create(size);

    for (int y = 0; y < size.height; y++)
    {
        auto oldPointsRow  = oldPoints [y];
        auto oldNormalsRow = oldNormals[y];

        auto ptsInWarpedRenderedRow = ptsInWarpedRendered[y];
        auto ptsInWarpedRow         = ptsInWarped[y];
        auto ptsInWarpedNormalsRow  = ptsInWarpedNormals[y];
        auto ptsInWarpedRenderedNormalsRow = ptsInWarpedRenderedNormals[y];

        for (int x = 0; x < size.width; x++)
        {
            // Since oldPoints are warped and rendered data,
            // we can get warped data in volume coords just by cam2vol transformation,
            // w/o applying warp to inp
            Point3f inWarpedRendered = fromPtype(oldPointsRow[x]);
            ptsInWarpedRenderedRow[x] = toPtype(inWarpedRendered);
            Point3f inWarped = cam2vol * inWarpedRendered;
            ptsInWarpedRow[x] = toPtype(inWarped);
            Point3f inrn = fromPtype(oldNormalsRow[x]);
            ptsInWarpedRenderedNormalsRow[x] = toPtype(inrn);
            ptsInWarpedNormalsRow[x] = toPtype(cam2vol.rotation() * inrn);
        }
    }
}


void buildInProjected(const Mat_<ptype>& ptsInWarpedRendered,
                      const cv::kinfu::Intr::Projector& proj,
                      // output param
                      Mat_<Point2f>& ptsInProjected)
{
    Size size = ptsInWarpedRendered.size();
    ptsInProjected.create(size);
    Rect inside(Point(), size);
    for (int y = 0; y < size.height; y++)
    {
        auto ptsInWarpedRenderedRow = ptsInWarpedRendered[y];
        auto ptsInProjectedRow      = ptsInProjected[y];
        for (int x = 0; x < size.width; x++)
        {
            Point3f inWarpedRendered = fromPtype(ptsInWarpedRenderedRow[x]);
                        
            Point2f outXY = proj(inWarpedRendered);
            if (! inside.contains(outXY))
                outXY = Point2f(qnan, qnan);
            ptsInProjectedRow[x] = outXY;
        }
    }
}


void buildShaded(const Mat_<ptype>& vertImage, const Mat_<ptype>& normImage,
                Point3f volSize,
                 // output params
                 Mat_<ptype>& ptsIn,
                 Mat_<ptype>& nrmIn)
{
    Size size = vertImage.size();
    ptsIn.create(size);
    nrmIn.create(size);
    for (int y = 0; y < size.height; y++)
    {
        auto ptsInRow = ptsIn[y];
        auto nrmInRow = nrmIn[y];
        auto vertImageRow = vertImage[y];
        auto normImageRow = normImage[y];
        for (int x = 0; x < size.width; x++)
        {
            // Get ptsIn from shaded data
            Point3f vshad = fromPtype(vertImageRow[x]);
            Point3f inp(vshad.x * volSize.x,
                        vshad.y * volSize.y,
                        vshad.z * volSize.z);
            ptsInRow[x] = toPtype(inp);
            // Get normals from shaded data
            Point3f nshad = fromPtype(normImageRow[x]);
            Point3f inn(nshad * 2.f - Point3f(1.f, 1.f, 1.f));
            nrmInRow[x] = toPtype(inn);
        }
    }
}


// Transform data-to-fit from camera to volume coordinate system
void buildOut(const Mat_<ptype>& newPoints,
              const Mat_<ptype>& newNormals,
              const Affine3f& cam2vol,
              // output params
              Mat_<ptype>& ptsOutVolP,
              Mat_<ptype>& ptsOutVolN)
{
    Size size = newPoints.size();
    ptsOutVolP.create(size);
    ptsOutVolN.create(size);

    for (int y = 0; y < size.height; y++)
    {
        auto newPointsRow  = newPoints [y];
        auto newNormalsRow = newNormals[y];
        auto ptsOutVolProw = ptsOutVolP[y];
        auto ptsOutVolNrow = ptsOutVolN[y];
        
        for (int x = 0; x < size.width; x++)
        {
            // Get newPoint and newNormal
            Point3f outp = fromPtype(newPointsRow [x]);
            Point3f outn = fromPtype(newNormalsRow[x]);
            // Transform them to coords in volume
            Point3f outVolP = cam2vol * outp;
            Point3f outVolN = cam2vol.rotation() * outn;
            ptsOutVolProw[x] = toPtype(outVolP);
            ptsOutVolNrow[x] = toPtype(outVolN);
        }
    }
}


// Precalculate knns and dists
void buildKnns(const WarpField& warp, const Mat_<ptype>& ptsIn,
               // output param
               Mat& cachedKnns)
{
    Size size = ptsIn.size();
    cachedKnns.create(size, rawType<WeightsNeighbours>());

    for (int y = 0; y < size.height; y++)
    {
        auto ptsInRow = ptsIn[y];
        auto cachedKnnsRow = cachedKnns.ptr<WeightsNeighbours>(y);

        for (int x = 0; x < size.width; x++)
        {
            // Get ptsIn from shaded data
            Point3f inp = fromPtype(ptsInRow[x]);

            std::vector<float> dists;
            std::vector<int> indices;

            WeightsNeighbours wn;
            //TODO: maybe use this: volume->getVoxelNeighbours(p, n)
            warp.findNeighbours(inp, indices, dists);
            int k = 0;
            for (size_t i = 0; i < indices.size(); i++)
            {
                if (std::isnan(dists[i]))
                    continue;

                wn.neighbours[k] = indices[i];
                wn.weights[k] = dists[i];
                k++;
            }
            cachedKnnsRow[x] = wn;
        }
    }
}


void buildCachedDqSums(const Mat& cachedKnns,
                       const std::vector<Ptr<WarpNode>>& nodes,
                       const int knn, const float damping, const bool disableCentering,
                       // output params
                       Mat& cachedDqSums)
{
    Size size = cachedKnns.size();
    cachedDqSums.create(size, rawType<DualQuaternion>());

    for (int y = 0; y < size.height; y++)
    {
        auto cachedKnnsRow  = cachedKnns.ptr<WeightsNeighbours>(y);
        auto cachedDqSumsRow = cachedDqSums.ptr<DualQuaternion>(y);

        for (int x = 0; x < size.width; x++)
        {
            WeightsNeighbours knns = cachedKnnsRow[x];
            DualQuaternion dqrt = warpForVertex(knns, nodes, knn, damping, disableCentering);
            cachedDqSumsRow[x] = dqrt;
        }
    }
}


void buildWarped(const Mat_<ptype>& ptsIn, const Mat_<ptype>& nrmIn, const Mat& cachedDqSums,
                 const Affine3f& vol2cam,
                 // output params
                 Mat_<ptype>& ptsInWarped,
                 Mat_<ptype>& ptsInWarpedRendered,
                 Mat_<ptype>& ptsInWarpedNormals,
                 Mat_<ptype>& ptsInWarpedRenderedNormals)
{
    Size size = ptsIn.size();
    ptsInWarped.create(size);
    ptsInWarpedRendered.create(size);
    ptsInWarpedNormals.create(size);
    ptsInWarpedRenderedNormals.create(size);
    
    for (int y = 0; y < size.height; y++)
    {
        auto ptsInRow = ptsIn[y];
        auto nrmInRow = nrmIn[y];
        auto ptsInWarpedRow = ptsInWarped[y];
        auto ptsInWarpedRenderedRow = ptsInWarpedRendered[y];
        auto ptsInWarpedNormalsRow = ptsInWarpedNormals[y];
        auto ptsInWarpedRenderedNormalsRow = ptsInWarpedRenderedNormals[y];
        auto cachedDqSumsRow = cachedDqSums.ptr<DualQuaternion>(y);
        
        for (int x = 0; x < size.width; x++)
        {
            // Get ptsIn from shaded data
            Point3f inp = fromPtype(ptsInRow[x]);
            // Get initial normals for transformation
            Point3f inn = fromPtype(nrmInRow[x]);

            DualQuaternion dqsum = cachedDqSumsRow[x];
            // We don't use commondq here, it's done at other stages of pipeline
            //UnitDualQuaternion dqfull = dqn; // dqfull = dqn * commondq;
            Affine3f rt = dqsum.getRt();
            Point3f warpedP = rt * inp;
            ptsInWarpedRow[x] = toPtype(warpedP);
            Point3f inWarpedRendered = vol2cam * warpedP;
            ptsInWarpedRenderedRow[x] = toPtype(inWarpedRendered);
            
            // Fill transformed normals
            Point3f warpedN = rt.rotation() * inn;
            ptsInWarpedNormalsRow[x] = toPtype(warpedN);
            Point3f inrn = vol2cam.rotation() * warpedN;
            ptsInWarpedRenderedNormalsRow[x] = toPtype(inrn);
        }
    }
}


// vertex residuals, point-plane metrics given warped data
void buildVertexResiduals(const Mat_<Point2f>& ptsInProjected,
                          const Mat_<ptype>& ptsInWarped,
                          const Mat_<ptype>& ptsInWarpedNormals,
                          const Mat_<ptype>& ptsOutVolP,
                          const Mat_<ptype>& ptsOutVolN,
                          const float diffThreshold, const float critAngleCos,
                          // output params
                          Mat_<float>& cachedResiduals,
                          Mat_<ptype>& cachedOutVolN)
{
    Size size = ptsInProjected.size();
    cachedResiduals.create(size);
    cachedOutVolN.create(size);

    for (int y = 0; y < size.height; y++)
    {
        auto ptsInProjectedRow = ptsInProjected[y];
        auto ptsInWarpedRow = ptsInWarped[y];
        auto ptsInWarpedNormalsRow = ptsInWarpedNormals[y];

        auto cachedResidualsRow = cachedResiduals[y];
        auto cachedOutVolNRow = cachedOutVolN[y];

        for (int x = 0; x < size.width; x++)
        {
            bool goodv = false;
            float pointPlaneDistance = 0.f;
            Point3f outVolP, outVolN;
            // ptsIn warped and rendered from camera
            // Point3f inrp = fromPtype(ptsInWarpedRenderedRow[x]);

            // Project it to screen to get corresponding out point to calc delta
            Point2f outXY = ptsInProjectedRow[x];
            if (!(cvIsNaN(outXY.x) || cvIsNaN(outXY.y)))
            {
                // Get newPoint and newNormal
                bool hasP, hasN;
                std::tie(hasP, outVolP) = interpolateP3f(outXY, ptsOutVolP[0], size.width);
                std::tie(hasN, outVolN) = interpolateP3f(outXY, ptsOutVolN[0], size.width);
                if (hasP && hasN)
                {
                    // Interpolated normal is not normalized usually; fix it
                    outVolN = outVolN / norm(outVolN);

                    // Get ptsInWarped (in volume coords)
                    Point3f inWarped = fromPtype(ptsInWarpedRow[x]);

                    // Get normals for filtering out
                    //Point3f inrn = fromPtype(ptsInWarpedRenderedNormals(x, y));
                    Point3f inVolumeN = fromPtype(ptsInWarpedNormalsRow[x]);

                    Point3f diff = outVolP - inWarped;

                    pointPlaneDistance = outVolN.dot(diff);

                    goodv = (diff.dot(diff) <= diffThreshold) &&
                            (abs(inVolumeN.dot(outVolN)) >= critAngleCos) &&
                            (!(cvIsInf(pointPlaneDistance) || cvIsNaN(pointPlaneDistance)));
                }
            }

            if (goodv)
            {
                cachedResidualsRow[x] = pointPlaneDistance;
                cachedOutVolNRow[x] = toPtype(outVolN);
            }
            else
            {
                cachedResidualsRow[x] = qnan;
                cachedOutVolNRow[x] = toPtype(nan3);
            }
        }
    }
}


void buildWeights(const Mat_<float>& cachedResiduals,
                  const float vertSigma,
                  // output params
                  Mat_<float>& cachedWeights)
{
    Size size = cachedResiduals.size();
    cachedWeights.create(size);
    
    for (int y = 0; y < size.height; y++)
    {
        auto cachedResidualsRow = cachedResiduals[y];
        auto cachedWeightsRow = cachedWeights[y];
        for (int x = 0; x < size.width; x++)
        {
            float dist = cachedResidualsRow[x];
            float weight = tukeyWeightSq(dist * dist, vertSigma);
            cachedWeightsRow[x] = weight;
        }
    }
}


void fillJacobianReg(BlockSparseMat<float, 6, 6>& jtj, std::vector<float>& jtb,
                     const std::vector<std::vector<NodeNeighboursType> >& graph,
                     const std::vector<Ptr<WarpNode>>& nodes,
                     const std::vector<std::vector<Ptr<WarpNode>>>& regNodes,
                     const std::vector<float>& regSigmas,
                     const float regTermWeight, const bool useHuber,
                     const bool disableCentering, const bool useExp, const bool useNormApply,
                     const bool parentMovesChild, const bool childMovesParent)
{
    for (int level = 0; level < graph.size(); level++)
    {
        auto childLevelNodes = (level == 0) ? nodes : regNodes[level - 1];
        auto levelNodes = regNodes[level];
        auto levelChildIdx = graph[level];

        for (int ixn = 0; ixn < levelNodes.size(); ixn++)
        {
            Ptr<WarpNode> node = levelNodes[ixn];

            auto children = levelChildIdx[ixn];

            for (int ixc = 0; ixc < children.size(); ixc++)
            {
                Ptr<WarpNode> child = childLevelNodes[children[ixc]];

                if (parentMovesChild)
                {
                    fillEdge(jtj, jtb, *node, *child,
                             regTermWeight, (useHuber ? regSigmas[level] : -1.f),
                             disableCentering, useExp, useNormApply);
                }
                if (childMovesParent)
                {
                    fillEdge(jtj, jtb, *child, *node,
                             regTermWeight, (useHuber ? regSigmas[level] : -1.f),
                             disableCentering, useExp, useNormApply);
                }
            }
        }
    }
}


void fillJacobianData(BlockSparseMat<float, 6, 6>& jtj, std::vector<float>& jtb,
                      const Mat_<float>& cachedResiduals,
                      const Mat_<ptype>& cachedOutVolN,
                      const Mat& cachedDqSums,
                      const Mat_<float>& cachedWeights,
                      const Mat_<ptype>& ptsIn,
                      const Mat& cachedKnns,
                      const std::vector<Ptr<WarpNode>>& warpNodes,
                      const bool useTukey,
                      const float normPenalty,
                      const bool decorrelate)
{
    Size size = cachedResiduals.size();
    for (int y = 0; y < size.height; y++)
    {
        for (int x = 0; x < size.width; x++)
        {
            Point pt(x, y);

            float pointPlaneDistance = cachedResiduals(pt);
            Point3f outVolN = fromPtype(cachedOutVolN(pt));
            DualQuaternion dqsum = cachedDqSums.at<DualQuaternion>(pt);

            if (std::isnan(pointPlaneDistance))
                continue;

            float weight = 1.f;
            if (useTukey)
            {
                weight = cachedWeights(pt);
            }

            // Get ptsIn from shaded data
            Point3f inp = fromPtype(ptsIn(pt));

            WeightsNeighbours knns = cachedKnns.at<WeightsNeighbours>(pt);

            std::vector<WarpNode> nodes;
            for (int k = 0; k < DYNAFU_MAX_NEIGHBOURS; k++)
            {
                int nei = knns.neighbours[k];
                if (nei < 0)
                    break;
                const WarpNode node = *(warpNodes[nei]);
                nodes.push_back(node);
            }

            fillVertex(jtj, jtb, nodes, knns,
                       inp, pointPlaneDistance, outVolN, dqsum, weight,
                       normPenalty, decorrelate);
        }
    }
}


bool ICPImpl::estimateWarpNodes(WarpField& warp, const Affine3f &pose,
                                InputArray _vertImage, InputArray _normImage,
                                InputArray _oldPoints, InputArray _oldNormals,
                                InputArray _newPoints, InputArray _newNormals) const
{
    CV_Assert(_vertImage.isMat());
    CV_Assert(_oldPoints.isMat());
    CV_Assert(_newPoints.isMat());
    CV_Assert(_newNormals.isMat());

    CV_Assert(_vertImage.type()  == cv::DataType<ptype>::type);
    CV_Assert(_normImage.type()  == cv::DataType<ptype>::type);
    CV_Assert(_oldPoints.type()  == cv::DataType<ptype>::type);
    CV_Assert(_oldNormals.type() == cv::DataType<ptype>::type);
    CV_Assert(_newPoints.type()  == cv::DataType<ptype>::type);
    CV_Assert(_newNormals.type() == cv::DataType<ptype>::type);

    Mat vertImage = _vertImage.getMat();
    Mat normImage = _normImage.getMat();
    Mat oldPoints = _oldPoints.getMat();
    Mat newPoints = _newPoints.getMat();
    Mat newNormals = _newNormals.getMat();
    Mat oldNormals = _oldNormals.getMat();

    CV_Assert(!vertImage.empty());
    CV_Assert(!normImage.empty());
    CV_Assert(!oldPoints.empty());
    CV_Assert(!newPoints.empty());
    CV_Assert(!newNormals.empty());

    Size size = vertImage.size();

    CV_Assert(normImage.size() == size);
    CV_Assert(oldPoints.size() == size);
    CV_Assert(newPoints.size() == size);
    CV_Assert(newNormals.size() == size);

    // newPoints: points from camera to match to
    // vertImage: [0-1] in volumeDims

    //TODO: check this
    //possibly this should be connected to baseRadius
    //const float csize = 1.0f;
    //TODO: move all params to one place;

    // Can be used to fade transformation to identity far from nodes centers
    // To make them local even w/o knn nodes choice
    constexpr float damping = 0.f;

    uint32_t nIter = this->iterations;

    // calc j_rt params :
    const bool atZero = false;
    const bool useExp = false;
    const bool additiveDerivative = true;
    // TODO: we can calculate first T, then R, then RT both
    const bool needR = true;
    const bool needT = true;
    const bool disableCentering = false;
    // reg params :
    const bool needReg = true;
    const bool parentMovesChild = true;
    const bool childMovesParent = true;
    const bool useNormApply = false;
    const bool useHuber = true;
    const bool precalcHuberSigma = true;

    // data term params:
    const bool needData = false;
    const bool useTukey = true;
    const bool precalcTukeySigma = true;

    // tries to fix dual quaternions before DQB so that they will form shortest paths
    // the algorithm is described in [Kavan and Zara 2005], Kavan'08
    // to make it deterministic we choose a reference dq according to relative flag
    const bool signFix = false;
    const bool signFixRelative = false;

    // solve params
    // Used in DynaFu paper, simplifies calculation
    const bool decorrelate = false;
    const bool tryLevMarq = true;
    const bool addItoLM = true;
    const float initialLambdaLevMarq = 0.1f;
    const float lmUpFactor = 2.f;
    const float lmDownFactor = 3.f;
    const float coeffILM = 0.1f;

    const float factorCommon = false;

    // TODO: check and find good one
    const float normPenalty = 0.0f;

    //TODO: find good one
    const float reg_term_weight = size.area() * 0.0001;
    //TODO: find good one
    const float diffThreshold = 50.f;
    //TODO: find good one
    const float critAngleCos = cos((float)CV_PI / 2);
    

    cv::kinfu::Intr::Projector proj = intrinsics.makeProjector();

    Affine3f cam2vol = volume->pose.inv() * pose;
    Affine3f vol2cam = pose * volume->pose.inv();

    // Precalculate values:
    // - canonical points from shaded images
    // ptsIn = (vertImage*volumeSize) = (vertImage*volumeDims*voxelSize)
    // nrmIn: Normals from shaded data
    Mat_<ptype> ptsIn, nrmIn;
    buildShaded(vertImage, normImage, volume->volSize, ptsIn, nrmIn);

    // - knns and distances to nodes' centers
    Mat cachedKnns;
    buildKnns(warp, ptsIn, cachedKnns);

    // - output points from camera to align to
    // Camera points (and normals) in volume coordinates
    // Used for delta calculation as y_i
    // ptsOutVol = (invVolPose * camPose) * newPoints
    Mat_<ptype> ptsOutVolP, ptsOutVolN;
    buildOut(newPoints, newNormals, cam2vol, ptsOutVolP, ptsOutVolN);

    // - current warped points from rendered data
    // Warped points in camera coordinates
    // Used for correspondence search
    // oldPoints = ptsInWarpedRendered = invCamPose * volPose * ptsInWarped
    // oldNormals: used to filter out bad points based on their normals
    Mat_<ptype> ptsInWarpedRendered, ptsInWarpedRenderedNormals;
    // Warped points in volume coordinates
    // Used for delta calculation as f(x_i)
    // ptsInWarped = warped(ptsIn) = (invVolPose * camPose) * ptsInWarpedRendered
    Mat_<ptype> ptsInWarped, ptsInWarpedNormals;
    buildInWarpedInitial(oldPoints, oldNormals, cam2vol,
                         ptsInWarped, ptsInWarpedNormals,
                         ptsInWarpedRendered, ptsInWarpedRenderedNormals);

    // - current warped points projected onto image plane
    // Projected onto 2d plane, for point-plane metrics calculation
    Mat_<Point2f> ptsInProjected;
    buildInProjected(ptsInWarpedRendered, proj, ptsInProjected);

    // - per-vertex residuals before the optimization
    // Cached point-to-plane distances per vertex
    Mat_<float> cachedResiduals;
    // Cached ground truth normals from projected pixels
    Mat_<ptype> cachedOutVolN;
    buildVertexResiduals(ptsInProjected, ptsInWarped, ptsInWarpedNormals,
                         ptsOutVolP, ptsOutVolN,
                         diffThreshold, critAngleCos,
                         cachedResiduals, cachedOutVolN);

    // Will be calculated at first energy evaluation
    // Cached dq sums
    Mat cachedDqSums;

    // MAD sigma estimation
    float vertSigma = estimateVertexSigma(cachedResiduals);

    // - per-vertex weights before optimization (based on sigma we got)
    Mat_<float> cachedWeights;
    buildWeights(cachedResiduals, vertSigma, cachedWeights);

    float vertexEnergy = estimateVertexEnergy(cachedResiduals, cachedWeights);

    auto graph = warp.getRegGraph();
    const std::vector<Ptr<WarpNode>>& warpNodes = warp.getNodes();
    const std::vector<std::vector<Ptr<WarpNode>>>& regNodes = warp.getGraphNodes();
    std::vector<float> regSigmas = estimateRegSigmas(graph, warpNodes, regNodes,
                                                     disableCentering, parentMovesChild, childMovesParent);

    float regEnergy = estimateRegEnergy(graph, warpNodes, regNodes, regSigmas,
                                        useHuber, disableCentering, parentMovesChild, childMovesParent);

    float energy = vertexEnergy + reg_term_weight * regEnergy;
    float oldEnergy = energy;

    const int knn = warp.k;

    size_t nNodesAll = warp.getNodesLen() + warp.getRegNodesLen();

    // Find a place for each node params in x vector
    placeNodesInX(warpNodes, regNodes);

    // calc args for additive derivative
    if (additiveDerivative)
    {
        calcArgs(warpNodes, regNodes, useExp);
    }
    
    // LevMarq iterations themselves
    float lambdaLevMarq = initialLambdaLevMarq;
    unsigned int it = 0;
    while (it < nIter)
    {
        BlockSparseMat<float, 6, 6> jtj(nNodesAll);
        std::vector<float> jtb(nNodesAll * 6);
       
        // j_rt caching
        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warpNodes : regNodes[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                node->cachedJac = node->transform.jRt(node->pos, atZero, additiveDerivative,
                                                      useExp, needR, needT, disableCentering);
            }
        }

        // regularization
        if (needReg)
        {
            // Sigmas are to be updated before each iteration only
            regSigmas = estimateRegSigmas(graph, warpNodes, regNodes, disableCentering,
                                          parentMovesChild, childMovesParent);

            fillJacobianReg(jtj, jtb, graph, warpNodes, regNodes, regSigmas,  reg_term_weight,
                            useHuber, disableCentering, useExp, useNormApply,
                            parentMovesChild, childMovesParent);
        }

        if (needData)
        {
            // Sigmas are to be updated before each iteration only
            // At 0th iteration we had sigma and weights estimated
            if (it > 0)
            {
                vertSigma = estimateVertexSigma(cachedResiduals);
                // re-weighting based on a sigma
                buildWeights(cachedResiduals, vertSigma, cachedWeights);
            }
            else // there's no cachedDqSums at 0th iteration, fixing it
            {
                buildCachedDqSums(cachedKnns, warpNodes, knn, damping, disableCentering,
                                  cachedDqSums);
            }

            fillJacobianData(jtj, jtb, cachedResiduals, cachedOutVolN, cachedDqSums, cachedWeights,
                             ptsIn, cachedKnns, warpNodes, useTukey, normPenalty,
                             decorrelate);
        }

        // Solve and get delta transform
        if (tryLevMarq)
        {
            bool enough = false;

            // save original diagonal of jtj matrix
            std::vector<float> diag(nNodesAll);
            for (int i = 0; i < nNodesAll; i++)
            {
                diag[i] = jtj.refElem(i, i);
            }

            std::vector<Ptr<WarpNode>> tempWarpNodes;
            std::vector<std::vector<Ptr<WarpNode>>> tempRegNodes;
            Mat_<float> tempCachedResiduals;
            Mat_<ptype> tempCachedOutVolN, tempPtsInWarped;
            Mat tempCachedDqSums;
            while (!enough && it < nIter)
            {
                // form LevMarq matrix
                for (int i = 0; i < nNodesAll; i++)
                {
                    float v = diag[i];
                    jtj.refElem(i, i) = v + lambdaLevMarq * (v + coeffILM);
                }

                std::vector<float> x;
                bool solved = kinfu::sparseSolve(jtj, Mat(jtb), x);

                //DEBUG
                std::cout << "#" << nIter;

                if (solved)
                {
                    tempWarpNodes = warp.cloneNodes();
                    tempRegNodes = warp.cloneGraphNodes();

                    // Update nodes using x

                    updateNodes(x, tempWarpNodes, tempRegNodes, additiveDerivative,
                                useExp, signFix, signFixRelative);

                    // Warping nodes
                    buildCachedDqSums(cachedKnns, tempWarpNodes, knn, damping, disableCentering,
                                      tempCachedDqSums);
                    buildWarped(ptsIn, nrmIn, tempCachedDqSums, vol2cam,
                                tempPtsInWarped, ptsInWarpedRendered,
                                ptsInWarpedNormals, ptsInWarpedRenderedNormals);
                    buildInProjected(ptsInWarpedRendered, proj, ptsInProjected);
                    buildVertexResiduals(ptsInProjected, tempPtsInWarped, ptsInWarpedNormals,
                                         ptsOutVolP, ptsOutVolN,
                                         diffThreshold, critAngleCos,
                                         tempCachedResiduals, tempCachedOutVolN);
                    buildWeights(tempCachedResiduals, vertSigma, cachedWeights);

                    vertexEnergy = estimateVertexEnergy(tempCachedResiduals, cachedWeights);

                    regEnergy = estimateRegEnergy(graph, tempWarpNodes, tempRegNodes, regSigmas,
                                                  useHuber, disableCentering, parentMovesChild, childMovesParent);

                    energy = vertexEnergy + reg_term_weight * regEnergy;

                    //TODO: visualize iteration by iteration

                    //DEBUG
                    std::cout << " energy: " << energy;
                    std::cout << " = " << vertexEnergy << " + " << reg_term_weight << " * " << regEnergy;
                    std::cout << ", vertSigma: " << vertSigma;
                    std::cout << ", regSigmas: ";
                    for (auto f : regSigmas) std::cout << f;
                    
                }
                else
                {
                    //DEBUG
                    std::cout << " not solved";
                }

                //DEBUG
                std::cout << std::endl;
                                
                if (!solved || (energy > oldEnergy))
                {
                    lambdaLevMarq *= lmUpFactor;
                    it++;

                    //DEBUG
                    std::cout << "LM up" << std::endl;
                }
                else
                {
                    enough = true;

                    //DEBUG
                    std::cout << "LM down" << std::endl;
                }
            }

            lambdaLevMarq /= lmDownFactor;
            warp.setNodes(tempWarpNodes);
            warp.setRegNodes(tempRegNodes);
            oldEnergy = energy;
            // these things will be reused at next stages
            ptsInWarped = tempPtsInWarped;
            cachedResiduals = tempCachedResiduals;
            cachedOutVolN = tempCachedOutVolN;
            cachedDqSums = tempCachedDqSums;
        }
        else
        // TODO: remove this branch when LevMarq works
        {
            std::vector<float> x;
            if (!kinfu::sparseSolve(jtj, Mat(jtb), x))
                break;

            std::vector<Ptr<WarpNode>> tempWarpNodes = warp.cloneNodes();
            std::vector<std::vector<Ptr<WarpNode>>> tempRegNodes = warp.cloneGraphNodes();

            updateNodes(x, tempWarpNodes, tempRegNodes, additiveDerivative, useExp, signFix,
                        signFixRelative);

            warp.setNodes(tempWarpNodes);
            warp.setRegNodes(tempRegNodes);

            // Warping nodes
            buildCachedDqSums(cachedKnns, warpNodes, knn, damping, disableCentering,
                              cachedDqSums);
            buildWarped(ptsIn, nrmIn, cachedDqSums, vol2cam,
                        ptsInWarped, ptsInWarpedRendered,
                        ptsInWarpedNormals, ptsInWarpedRenderedNormals);
            buildInProjected(ptsInWarpedRendered, proj, ptsInProjected);
            buildVertexResiduals(ptsInProjected, ptsInWarped, ptsInWarpedNormals,
                                 ptsOutVolP, ptsOutVolN,
                                 diffThreshold, critAngleCos,
                                 cachedResiduals, cachedOutVolN);
            buildWeights(cachedResiduals, vertSigma, cachedWeights);

            vertexEnergy = estimateVertexEnergy(cachedResiduals, cachedWeights);

            regEnergy = estimateRegEnergy(graph, warpNodes, regNodes, regSigmas, useHuber,
                                          disableCentering, parentMovesChild, childMovesParent);

            energy = vertexEnergy + reg_term_weight * regEnergy;

            //TODO: visualize iteration by iteration

            //DEBUG
            std::cout << "#" << nIter << " energy: " << energy;
            std::cout << " = " << vertexEnergy << " + " << reg_term_weight << " * " << regEnergy;
            std::cout << ", vertSigma: " << vertSigma;
            std::cout << ", regSigmas: ";
            for (auto f : regSigmas) std::cout << f;
            std::cout << std::endl;
        }

        it++;
    }

    if(factorCommon)
    {
        Matx44f commonM;
        estimateAffine3D(ptsIn, ptsInWarped, commonM, noArray());
        Affine3f af(commonM);
        UnitDualQuaternion common(af);

        // Looks like procedure is the same for all levels
        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warpNodes : regNodes[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                node->transform = node->transform.factoredOut(common, node->pos);
            }
        }
    }

    if (it != nIter)
    {
        //DEBUG
        std::cout << "Failed at iteration #" << it << "/" << nIter << std::endl;
        return false;
    }
    else
    {
        return true;
    }




    //TODO: discard this when refactoring is over
    /*
    ---------------------------------------------------------------------------------------------------- 

    const NodeVectorType& warpNodes = warp.getNodes();

    Affine3f T_lw = pose.inv() * volume->pose;

    // Accumulate regularisation term for each node in the heiarchy
    const std::vector<NodeVectorType>& regNodes = warp.getGraphNodes();

    int totalNodes = (int)warpNodes.size();
    for(const auto& nodes: regNodes) totalNodes += (int)nodes.size();

    // level-wise regularisation components of A and b (from Ax = b) for each node in heirarchy
    Mat_<float> b_reg(6*totalNodes, 1, 0.f);
    Mat_<float> A_reg(6*totalNodes, 6*totalNodes, 0.f);

    // indices for each node block to A,b matrices. It determines the order
    // in which paramters are laid out

    std::vector<int> baseIndices(warp.n_levels, 0);

    for(int l = warp.n_levels-2; l >= 0; l--)
    {
        baseIndices[l] = baseIndices[l+1]+6*((int)regNodes[l].size());
    }

    for(const int& i: baseIndices) std::cout << i << ", ";
    std::cout << std::endl;

    fillRegularization(A_reg, b_reg, warp, totalNodes, baseIndices);

    std::vector<float> residuals;

    Mat Vg(oldPoints.size(), CV_32FC3, nan3);

    Mat Vc(oldPoints.size(), CV_32FC3, nan3);
    Mat Nc(oldPoints.size(), CV_32FC3, nan3);
    cv::kinfu::Intr::Projector proj = intrinsics.makeProjector();

    for (int y = 0; y < oldPoints.size().height; y++)
    {
        for (int x = 0; x < oldPoints.size().width; x++)
        {
            // Obtain correspondence by projecting Tu_Vg
            Vec3f curV = oldPoints.at<Vec3f>(y, x);
            if (curV == Vec3f::all(0) || cvIsNaN(curV[0]) || cvIsNaN(curV[1]) || cvIsNaN(curV[2]))
                continue;

            Point2f newCoords = proj(Point3f(curV));
            if(!(newCoords.x >= 0 && newCoords.x < newPoints.cols - 1 &&
                 newCoords.y >= 0 && newCoords.y < newPoints.rows - 1))
                continue;

            // TODO: interpolate Vg instead of simply converting projected coords to int
            Vg.at<Vec3f>(y, x) = vertImage.at<Vec3f>((int)newCoords.y, (int)newCoords.x);

            // bilinearly interpolate newPoints under newCoords point
            int xi = cvFloor(newCoords.x), yi = cvFloor(newCoords.y);
            float tx  = newCoords.x - xi, ty = newCoords.y - yi;

            const ptype* prow0 = newPoints.ptr<ptype>(yi+0);
            const ptype* prow1 = newPoints.ptr<ptype>(yi+1);

            Point3f p00 = fromPtype(prow0[xi+0]);
            Point3f p01 = fromPtype(prow0[xi+1]);
            Point3f p10 = fromPtype(prow1[xi+0]);
            Point3f p11 = fromPtype(prow1[xi+1]);

            //do not fix missing data
            if(!(fastCheck(p00) && fastCheck(p01) &&
                 fastCheck(p10) && fastCheck(p11)))
                continue;

            Point3f p0 = p00 + tx*(p01 - p00);
            Point3f p1 = p10 + tx*(p11 - p10);
            Point3f newP = (p0 + ty*(p1 - p0));

            const ptype* nrow0 = newNormals.ptr<ptype>(yi+0);
            const ptype* nrow1 = newNormals.ptr<ptype>(yi+1);

            Point3f n00 = fromPtype(nrow0[xi+0]);
            Point3f n01 = fromPtype(nrow0[xi+1]);
            Point3f n10 = fromPtype(nrow1[xi+0]);
            Point3f n11 = fromPtype(nrow1[xi+1]);

            if(!(fastCheck(n00) && fastCheck(n01) &&
                 fastCheck(n10) && fastCheck(n11)))
                continue;

            Point3f n0 = n00 + tx*(n01 - n00);
            Point3f n1 = n10 + tx*(n11 - n10);
            Point3f newN = n0 + ty*(n1 - n0);

            Vc.at<Point3f>(y, x) = newP;
            Nc.at<Point3f>(y, x) = newN;

            Vec3f diff = oldPoints.at<Vec3f>(y, x) - Vec3f(newP);
            if(diff.dot(diff) > 0.0004f) continue;
            if(abs(newN.dot(oldNormals.at<Point3f>(y, x))) < cos((float)CV_PI / 2)) continue;

            float rd = newN.dot(diff);

            residuals.push_back(rd);
        }
    }

    float med = median(residuals);
    std::for_each(residuals.begin(), residuals.end(), [med](float& x){x =  std::abs(x-med);});
    std::cout << "median: " << med << " from " << residuals.size() << " residuals " << std::endl;
    float sigma = MAD_SCALE * median(residuals);

    float total_error = 0;
    int pix_count = 0;

    for(int y = 0; y < oldPoints.size().height; y++)
    {
        for(int x = 0; x < oldPoints.size().width; x++)
        {
            Vec3f curV = oldPoints.at<Vec3f>(y, x);
            if (curV == Vec3f::all(0) || cvIsNaN(curV[0]))
                continue;

            Vec3f V = Vg.at<Vec3f>(y, x);
            if (V == Vec3f::all(0) || cvIsNaN(V[0]))
                continue;

            V[0] *= volume->volResolution.x;
            V[1] *= volume->volResolution.y;
            V[2] *= volume->volResolution.z;

            if(!fastCheck(Vc.at<Point3f>(y, x)))
                continue;

            if(!fastCheck(Nc.at<Point3f>(y, x)))
                continue;

            Point3i p((int)V[0], (int)V[1], (int)V[2]);
            Vec3f diff = oldPoints.at<Vec3f>(y, x) - Vc.at<Vec3f>(y, x);

            float rd = Nc.at<Vec3f>(y, x).dot(diff);

            total_error += tukeyWeight(rd, sigma) * rd * rd;
            pix_count++;

            int n;
            NodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
            float totalNeighbourWeight = 0.f;
            float neighWeights[DYNAFU_MAX_NEIGHBOURS];
            for (int i = 0; i < n; i++)
            {
                int neigh = neighbours[i];
                neighWeights[i] = warpNodes[neigh]->weight(Point3f(V)*volume->voxelSize);

                totalNeighbourWeight += neighWeights[i];
            }

            if(totalNeighbourWeight < 1e-5) continue;

            for (int i = 0; i < n; i++)
            {
                if(neighWeights[i] < 0.01) continue;
                int neigh = neighbours[i];

                Vec3f Tj_Vg_Vj = (warpNodes[neigh]->transform *
                                 (Point3f(V)*volume->voxelSize - warpNodes[neigh]->pos));

                Matx33f Tj_Vg_Vj_x(0, -Tj_Vg_Vj[2], Tj_Vg_Vj[1],
                                   Tj_Vg_Vj[2], 0, -Tj_Vg_Vj[0],
                                   -Tj_Vg_Vj[1], Tj_Vg_Vj[0], 0);

                Vec3f v1 = (Tj_Vg_Vj_x * T_lw.rotation().t()) * Nc.at<Vec3f>(y, x);
                Vec3f v2 = T_lw.rotation().t() * Nc.at<Vec3f>(y, x);

                Matx61f J_dataT(v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]);
                Matx16f J_data = J_dataT.t();
                Matx66f H_data = J_dataT * J_data;

                float w = (neighWeights[i] / totalNeighbourWeight);

                float robustWeight = tukeyWeight(rd, sigma);

                //DEBUG:
                //std::cout << "robw: " << robustWeight << std::endl;
                robustWeight = 1.0f;

                int blockIndex = baseIndices[0]+6*neigh;
                for(int row = 0; row < 6; row++)
                    for(int col = 0; col < 6; col++)
                        A_reg(blockIndex+row, blockIndex+col) += robustWeight * w * w * H_data(row, col);

                for(int row = 0; row < 6; row++)
                    b_reg(blockIndex+row) += -robustWeight * rd * w * J_dataT(row);
            }

        }
    }
    -------------------------------------------
    */
}

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations)
{
    return makePtr<ICPImpl>(_intrinsics, _volume, _iterations);
}

} // namespace dynafu
} // namespace cv
