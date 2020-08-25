// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <algorithm>
#include <unordered_map>

#include "precomp.hpp"
#include "nonrigid_icp.hpp"

#if defined(HAVE_EIGEN)
#    include <Eigen/Sparse>
//#    include <Eigen/SparseCholesky>
#    include <Eigen/SparseQR>
#endif


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


// TODO: the following code was used like that, maybe reuse it
/*
data:

    float med = median(residuals);
    std::for_each(residuals.begin(), residuals.end(), [med](float& x) {x = std::abs(x - med); });
    float sigma = MAD_SCALE * median(residuals);

    float robustWeight = tukeyWeight(rd, sigma);

reg:

    float reg_sigma = MAD_SCALE * median(reg_residuals);
    float robustWeight = huberWeight(r_edge, reg_sigma);
*/


static float median(std::vector<float> v)
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
TODO: remove it in the end
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

bool interpolateP3f(Point2f pt, const ptype* data, size_t width, Point3f& out)
{
    // bilinearly interpolate newPoints under newCoords point
    int xi = cvFloor(pt.x), yi = cvFloor(pt.y);
    float tx = pt.x - xi, ty = pt.y - yi;

    const ptype* prow0 = data + yi*width;
    const ptype* prow1 = data + (yi + 1)*width;

    Point3f p00 = fromPtype(prow0[xi + 0]);
    Point3f p01 = fromPtype(prow0[xi + 1]);
    Point3f p10 = fromPtype(prow1[xi + 0]);
    Point3f p11 = fromPtype(prow1[xi + 1]);

    //do not fix missing data
    if (!(fastCheck(p00) && fastCheck(p01) &&
          fastCheck(p10) && fastCheck(p11)))
        return false;

    Point3f p0 = p00 + tx * (p01 - p00);
    Point3f p1 = p10 + tx * (p11 - p10);
    out = (p0 + ty * (p1 - p0));
    
    return true;
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

// keeps jtj matrix between building and solving
struct BlockSparseMat
{
    static const int blockSize = 6;
    BlockSparseMat(int _nBlocks) :
        nBlocks(_nBlocks), ijv()
    { }
       
    Matx66f& refBlock(int i, int j)
    {
        Point2i p(i, j);
        auto it = ijv.find(p);
        if (it == ijv.end())
        {
            it = ijv.insert({ p, Matx66f()}).first;
        }
        return it->second;
    }

    float& refElem(int i, int j)
    {
        Point2i ib(i / blockSize, j / blockSize), iv(i % blockSize, j % blockSize);
        return refBlock(ib.x, ib.y)(iv.x, iv.y);
    }

    int nBlocks;
    std::unordered_map< Point2i, Matx66f > ijv;
};

static bool sparseSolve(const BlockSparseMat& jtj, const std::vector<float>& jtb, std::vector<float>& x)
{
    const float matValThreshold = 0.001f;

    bool result = false;

#if defined(HAVE_EIGEN)

    std::cout << "starting eigen-insertion..." << std::endl;

    //TODO: Consider COLAMD column reordering before solving matrix. This improves speed by a significant amount

    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(jtj.ijv.size()*jtj.blockSize*jtj.blockSize);
    for (auto ijv : jtj.ijv)
    {
        int xb = ijv.first.x, yb = ijv.first.y;
        Matx66f vblock = ijv.second;
        for (int i = 0; i < jtj.blockSize; i++)
        {
            for (int j = 0; j < jtj.blockSize; j++)
            {
                float val = vblock(i, j);
                if (abs(val) >= matValThreshold)
                {
                    tripletList.push_back(Eigen::Triplet<double>(jtj.blockSize * xb + i,
                                                                 jtj.blockSize * yb + j,
                                                                 val));
                }
            }
        }
    }
    
    Eigen::SparseMatrix<float> abig(jtj.blockSize * jtj.nBlocks, jtj.blockSize * jtj.nBlocks);
    abig.setFromTriplets(tripletList.begin(), tripletList.end());

    // TODO: do we need this?
    abig.makeCompressed();

    Eigen::VectorXf bBig(jtb);

    //TODO: try this, LLT and Cholesky
    //Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> solver;
    Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::NaturalOrdering<int>> solver;

    std::cout << "starting eigen-compute..." << std::endl;
    solver.compute(abig);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        std::cout << "starting eigen-solve..." << std::endl;

        Eigen::VectorXf sx = solver.solve(bBig);
        if (solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            x.resize(jtb.size);
            for (size_t i = 0; i < x.size(); i++)
            {
                x[i] = sx[i];
            }
            result = true;
        }
    }

#else
    std::cout << "no eigen library" << std::endl;

    CV_Error(Error::StsNotImplemented,
             "Eigen library required for matrix solve, dense solver is not implemented");
#endif

    return result;
}

static DualQuaternion dampedDQ(int nNodes, float wsum, float coeff)
{
    float wdamp = nNodes - wsum;
    //TODO URGENT: shortcut for dq
    return UnitDualQuaternion().dq() * wdamp * coeff;
}


// residual norm for sigma estimation
static float resNormEdge(const WarpNode& actNode, const WarpNode& pasNode, const bool disableCentering)
{
    UnitDualQuaternion dqAct = actNode.transform;
    UnitDualQuaternion dqPas = pasNode.transform;
    Point3f cAct = actNode.pos;
    Point3f cPas = pasNode.pos;
    int nAct = actNode.place, nPas = pasNode.place;

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
static void fillEdge(BlockSparseMat& jtj, std::vector<float>& jtb,
                     const WarpNode& actNode, const WarpNode& pasNode,
                     const float regTermWeight, const float huberSigma,
                     const bool disableCentering, const bool useExp, const bool useNormApply)
{
    UnitDualQuaternion dqAct = actNode.transform;
    UnitDualQuaternion dqPas = pasNode.transform;
    Point3f cAct = actNode.pos;
    Point3f cPas = pasNode.pos;
    int nAct = actNode.place, nPas = pasNode.place;
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
    BlockSparseMat jtjNode(jtj.nBlocks);
    
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


static void fillVertex(BlockSparseMat& jtj, std::vector<float>& jtb,
                       const std::vector<WarpNode>& nodes, WeightsNeighbours knns,
                       // no need to pass diff, pointPlaneDistance is enough
                       //Point3f inp, Point3f diff, Point3f outVolN,
                       Point3f inp, float pointPlaneDistance, Point3f outVolN, 
                       float damping, float tukeySigma, bool decorrelate, bool disableCentering)
{
    size_t knn = nodes.size();

    // run through DQB
    DualQuaternion dqsum;
    float wsum = 0;
     
    std::vector<int> places(knn);
    std::vector<Matx<float, 8, 6>> jPerNodeWeighted(knn);
    for (int k = 0; k < knn; k++)
    {
        int nei = knns.neighbours[k];
        float w = knns.weights[k];

        const WarpNode& node = nodes[k];

        Point3f c;
        if (!disableCentering)
        {
            c = node.pos;
        }

        // center(x) := (1+e*1/2*c)*x*(1-e*1/2*c)
        UnitDualQuaternion centered = node.transform.centered(c);
        places[k] = node.place;
        jPerNodeWeighted[node.place] = w * node.cachedJac;
        dqsum += w * centered.dq();
        wsum += w;
    }

    dqsum += dampedDQ(knn, wsum, damping);

    // jacobian of normalization+application to a point
    Matx<float, 3, 8> jNormApply = dqsum.j_normapply(inp);

    std::vector<Matx<float, 3, 6>> jPerNode(knn);

    // emulating jNormApply * jPerNodeWeighted
    for (int k = 0; k < knn; k++)
    {
        jPerNode[k] = jNormApply * jPerNodeWeighted[k];
    }    

    // no need to pass diff, pointPlaneDistance is enough
    //float pointPlaneDistance = outVolN.dot(diff);

    float weight = 1.f;
    if (tukeySigma >= 0.f)
    {
        weight = tukeyWeightSq(pointPlaneDistance * pointPlaneDistance, tukeySigma);
    }

    // for point-plane distance jacobian
    Matx33f nnt = Vec3f(outVolN) * Vec3f(outVolN).t();

    // emulate jVertex^T * jVertex and jVertex^T*b
    for (int k = 0; k < knn; k++)
    {
        int kplace = places[k];
        if (decorrelate)
        {
            jtj.refBlock(kplace, kplace) += weight * jPerNode[k].t() * nnt * jPerNode[k];
        }
        else
        {
            for (int l = 0; l < knn; l++)
            {
                int lplace = places[l];
                jtj.refBlock(kplace, lplace) += weight * jPerNode[k].t() * nnt * jPerNode[l];
            }
        }

        // no need to pass diff, pointPlaneDistance is enough
        //Vec6f jtbBlock = jPerNode[k].t() * nnt * Vec3f(diff);
        Vec6f jtbBlock = jPerNode[k].t() * Vec3f(outVolN) * pointPlaneDistance;
        for (int i = 0; i < 6; i++)
        {
            jtb[6 * kplace + i] += weight * jtbBlock[i];
        }
    }
}


// TODO URGENT THINGS: things are to be done before expecting that stuff is compiled
bool ICPImpl::estimateWarpNodes(WarpField& warp, const Affine3f &pose,
                                InputArray _vertImage, InputArray _normImage,
                                InputArray _oldPoints, InputArray _oldNormals,
                                InputArray _newPoints, InputArray _newNormals) const
{
    CV_Assert(_vertImage.isMat());
    CV_Assert(_oldPoints.isMat());
    CV_Assert(_newPoints.isMat());
    CV_Assert(_newNormals.isMat());

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
    int nPts = size.area();

    CV_Assert(normImage.size() == size);
    CV_Assert(oldPoints.size() == size);
    CV_Assert(newPoints.size() == size);
    CV_Assert(newNormals.size() == size);

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

    //TODO: find good one
    const float reg_term_weight = nPts * 0.0001;
    //TODO: find good one
    const float diffThreshold = 50.f;
    //TODO: find good one
    const float critAngleCos = cos((float)CV_PI / 2);

    // data term params :
    const bool needData = false;
    const bool useTukey = true;
    const bool precalcTukeySigma = true;

    // tries to fix dual quaternions before DQB so that they will form shortest paths
    // the algorithm is described in [Kavan and Zara 2005], Kavan'08
    // to make it deterministic we choose a reference dq according to relative flag
    const bool signFix = false;
    const bool signFixRelative = true;

    // solve params
    // Used in DynaFu paper, simplifies calculation
    const bool decorrelate = false;
    const bool tryLevMarq = true;
    const bool addItoLM = true;
    const float lambdaLevMarq = 0.1f;
    const float coeffILM = 0.1f;
    
    /*
    newPoints: points from camera to match to
    vertImage: [0-1] in volumeDims
    ptsIn = (vertImage*volumeSize) = (vertImage*volumeDims*voxelSize)
    */

    // Warped points in camera coordinates
    // Used for correspondence search
    // oldPoints = ptsInWarpedRendered = invCamPose * volPose * ptsInWarped
    cv::AutoBuffer<ptype> ptsInWarpedRendered(nPts);
    // Used to filter out bad points based on their normals
    cv::AutoBuffer<ptype> ptsInWarpedRenderedNormals(nPts);
    // Warped points in volume coordinates
    // Used for delta calculation as f(x_i)
    // ptsInWarped = warped(ptsIn) = (invVolPose * camPose) * ptsInWarpedRendered
    cv::AutoBuffer<ptype> ptsInWarped(nPts);
    cv::AutoBuffer<ptype> ptsInWarpedNormals(nPts);
    // Camera points (and normals) in volume coordinates
    // Used for delta calculation as y_i
    // ptsOutVol = (invVolPose * camPose) * newPoints
    cv::AutoBuffer<ptype> ptsOutVolP(nPts);
    cv::AutoBuffer<ptype> ptsOutVolN(nPts);

    // for data calculation
    //TODO URGENT: this
    //inp, pointPlaneDistance, outVolN

    cv::kinfu::Intr::Projector proj = intrinsics.makeProjector();

    Affine3f cam2vol = volume->pose.inv() * pose;
    Affine3f vol2cam = pose * volume->pose.inv();

    // Precalculate knns and dists
    const int knn = warp.k;
    cv::AutoBuffer<WeightsNeighbours> cachedKnns(nPts);

    // For MAD sigma estimation
    std::vector<float> vertResiduals;
    vertResiduals.reserve(nPts);

    for (int y = 0; y < size.height; y++)
    {
        for (int x = 0; x < size.width; x++)
        {
            // Fill ptsInWarped and ptsInWarpedRendered initially

            //TODO: Mat::ptr() instead
            Point3f inrp = fromPtype(oldPoints.at<ptype>(y, x));
            ptsInWarpedRendered[y * size.width + x] = toPtype(inrp);
            Point3f inWarped = cam2vol * inrp;
            ptsInWarped[y * size.width + x] = toPtype(inWarped);
            Point3f inrn = fromPtype(oldNormals.at<ptype>(y, x));
            ptsInWarpedRenderedNormals[y * size.width + x] = toPtype(inrn);
            ptsInWarpedNormals[y * size.width + x] = toPtype(cam2vol.rotation() * inrn);

            // Get newPoint and newNormal
            Point3f outp = fromPtype(newPoints.at<ptype>(y, x));
            Point3f outn = fromPtype(newNormals.at<ptype>(y, x));
            // Transform them to coords in volume
            Point3f outVolP = cam2vol * outp;
            Point3f outVolN = cam2vol.rotation() * outn;
            ptsOutVolP[y * size.width + x] = toPtype(outVolP);
            ptsOutVolN[y * size.width + x] = toPtype(outVolN);

            // Precalculate knns and dists

            // Get ptsIn from shaded data
            Point3f vshad = vertImage.at<Point3f>(y, x);
            Point3f inp(vshad.x * volume->volSize.x,
                        vshad.y * volume->volSize.y,
                        vshad.z * volume->volSize.z);

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

            cachedKnns[y * size.width + x] = wn;

            {
                // ptsIn warped and rendered from camera
                // Point3f inrp;

                // Project it to screen to get corresponding out point to calc delta
                Point2f outXY = proj(inrp);

                if (!(outXY.x >= 0 && outXY.x < size.width  - 1 &&
                      outXY.y >= 0 && outXY.y < size.height - 1))
                    continue;

                // Get newPoint and newNormal
                // (here's volume coords)
                if(!(interpolateP3f(outXY, ptsOutVolP.data(), size.width, outVolP) &&
                     interpolateP3f(outXY, ptsOutVolN.data(), size.width, outVolN)))
                    continue;

                // Normalize
                outVolN = outVolN / norm(outVolN);

                // Get ptsInWarped (in volume coords)
                // Point3f inWarped;

                // Get ptsIn from shaded data
                // Point3f vshad, inp;

                Point3f diff = outVolP - inWarped;

                float pointPlaneDistance = outVolN.dot(diff);

                if (!cvIsInf(pointPlaneDistance) && !cvIsNaN(pointPlaneDistance))
                    vertResiduals.push_back(pointPlaneDistance);
            }
        }
    }

    float vertSigma = 1.f;
    {
        float vertMed = median(vertResiduals);
        std::for_each(vertResiduals.begin(), vertResiduals.end(),
                      [vertMed](float& x) {x = std::abs(x - vertMed); });
        vertSigma = MAD_SCALE * median(vertResiduals);
    }
    
    auto graph = warp.getRegGraph();
    
    std::vector<float> regSigmas(graph.size(), 1.f);

    {
        for (int level = 0; level < graph.size(); level++)
        {
            auto childLevelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            auto levelNodes = warp.getGraphNodes()[level];
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

            float regMed = median(regLevResiduals);
            std::for_each(regLevResiduals.begin(), regLevResiduals.end(),
                          [regMed](float& x) {x = std::abs(x - regMed); });
            regSigmas[level] = MAD_SCALE * median(regLevResiduals);
        }
    }
    
    int nNodes = warp.getNodesLen();
    int nNodesAll = warp.getNodesLen() + warp.getRegNodesLen();

    // Find a place for each node params in x vector
    {
        size_t idx = 0;
        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                node->place = idx++;
            }
        }
    }
    
    // Gauss-Newton iteration themselves
    unsigned int it;
    for (it = 0; it < nIter; it++)
    {
        BlockSparseMat jtj(nNodesAll);
        std::vector<float> jtb(nNodesAll * 6);

        // j_rt caching
        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                int place = node->place;

                node->cachedJac = node->transform.jRt(node->pos, atZero, disableCentering, useExp, needR, needT);
            }
        }

        // regularization
        if (needReg)
        {
            for (int level = 0; level < graph.size(); level++)
            {
                auto childLevelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
                auto levelNodes = warp.getGraphNodes()[level];
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
                                     reg_term_weight, (useHuber ? regSigmas[level] : -1.f),
                                     disableCentering, useExp, useNormApply);
                        }
                        if (childMovesParent)
                        {
                            fillEdge(jtj, jtb, *child, *node,
                                     reg_term_weight, (useHuber ? regSigmas[level] : -1.f),
                                     disableCentering, useExp, useNormApply);
                        }
                    }
                }
            }
        }

        if (needData)
        {
            int usedPixels = 0;
            for (int y = 0; y < size.height; y++)
            {
                for (int x = 0; x < size.width; x++)
                {
                    //TODO: Mat::ptr() instead
                    // ptsIn warped and rendered from camera
                    Point3f inrp = fromPtype(ptsInWarpedRendered[y * size.width + x]);

                    // Project it to screen to get corresponding out point to calc delta
                    Point2f outXY = proj(inrp);

                    if (!(outXY.x >= 0 && outXY.x < size.width  - 1 &&
                          outXY.y >= 0 && outXY.y < size.height - 1))
                        continue;

                    // Get newPoint and newNormal
                    Point3f outVolP, outVolN;
                    // (here's volume coords)
                    if(!(interpolateP3f(outXY, ptsOutVolP.data(), size.width, outVolP) &&
                         interpolateP3f(outXY, ptsOutVolN.data(), size.width, outVolN)))
                        continue;

                    // Normalize
                    outVolN = outVolN / norm(outVolN);

                    // Get ptsInWarped (in volume coords)
                    Point3f inWarped = fromPtype(ptsInWarped[y * size.width + x]);
                    // Get ptsIn from shaded data
                    Point3f vshad = vertImage.at<Point3f>(y, x);
                    Point3f inp(vshad.x * volume->volSize.x,
                                vshad.y * volume->volSize.y,
                                vshad.z * volume->volSize.z);

                    // Get normals for filtering out
                    //Point3f inrn = fromPtype(ptsInWarpedRenderedNormals[y * size.width + x]);
                    Point3f inVolumeN = fromPtype(ptsInWarpedNormals[y * size.width + x]);

                    Point3f diff = outVolP - inWarped;
                    if (diff.dot(diff) > diffThreshold)
                        continue;
                    if (abs(inVolumeN.dot(outVolN) < critAngleCos))
                        continue;

                    float pointPlaneDistance = outVolN.dot(diff);

                    WeightsNeighbours knns = cachedKnns[y*size.width + x];

                    std::vector<WarpNode> nodes;
                    for (int k = 0; k < knn; k++)
                    {
                        int nei = knns.neighbours[k];
                        if (nei < 0)
                            break;
                        const WarpNode node = *(warp.getNodes()[nei]);
                        nodes.push_back(node);
                    }

                    fillVertex(jtj, jtb, nodes, knns, inp, pointPlaneDistance, outVolN,
                               damping, (useTukey ? vertSigma : -1.f), decorrelate, disableCentering);
                    usedPixels++;
                }
            }
        }

        // Solve and get delta transform
        if (tryLevMarq)
        {
            for (int i = 0; i < nNodesAll; i++)
            {
                float& v = jtj.refElem(i, i);
                v += lambdaLevMarq * (v + coeffILM);
            }
            //TODO: try LevMarq changes lambda according to error
            //and falls back to prev if error is too big, see g2o docs about it
            // like that: If (E < error(x)) { x = xold; lambda *= 2; } else { lambda /= 2; }
        }

        std::vector<float> x;
        if (!sparseSolve(jtj, jtb, x))
        {
            break;
        }

        // Update nodes using x

        // for sign fix
        UnitDualQuaternion ref;
        float refnorm = std::numeric_limits<float>::max();

        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                int place = node->place;

                // blockNode = x[6 * n:6 * (n + 1)]
                float blockNode[6];
                for (int i = 0; i < 6; i++)
                {
                    blockNode[i] = x[6 * place + i];
                }

                UnitDualQuaternion dq = node->transform;
                Point3d c = node->pos;

                //TODO: maybe calc Jacobians w/o exp and then apply using exp?? highly experimental
                UnitDualQuaternion dqit;
                if (useExp)
                {
                    // the order is the opposite to !useExp
                    Quaternion dualx(0, blockNode[0], blockNode[1], blockNode[2]);
                    Quaternion realx(0, blockNode[3], blockNode[4], blockNode[5]);

                    //TODO URGENT: this
                    dqit = UnitDualQuaternion(realx, dualx).exp();
                }
                else
                {
                    Vec3f rotParams(blockNode[0], blockNode[1], blockNode[2]);
                    Vec3f transParams(blockNode[3], blockNode[4], blockNode[5]);

                    // this function expects input vector to be a norm of full angle
                    // while it contains just 1 / 2 of an angle

                    //TODO URGENT: this using DQs
                    Affine3f aff(rotParams * 2, transParams);

                    dqit = UnitDualQuaternion(aff);
                }
                
                // sic! (2nd transform) * (1st transform)
                UnitDualQuaternion dqnew = dqit * dq;
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

        // Warping and error calculation
        double sumError = 0, sumSqError = 0;
        int nPixelsError = 0;
        vertResiduals.clear();
        vertResiduals.reserve(nPts);

        auto nodes = warp.getNodes();
        for (int y = 0; y < size.height; y++)
        {
            for (int x = 0; x < size.width; x++)
            {
                //TODO: Mat::ptr() instead
                // Get ptsIn from shaded data
                Point3f vshad = vertImage.at<Point3f>(y, x);
                Point3f inp(vshad.x * volume->volSize.x,
                            vshad.y * volume->volSize.y,
                            vshad.z * volume->volSize.z);
                // Get initial normals for transformation
                Point3f nshad = normImage.at<Point3f>(y, x);
                Point3f inn(nshad*2.f - Point3f(1.f, 1.f, 1.f));

                WeightsNeighbours knns = cachedKnns[y*size.width+x];

                DualQuaternion dqsum = DualQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0));
                float wsum = 0; int nValid = 0;
                for (int k = 0; k < knn; k++)
                {
                    int ixn = knns.neighbours[k];
                    if (ixn >= 0)
                    {
                        float w = knns.weights[k];

                        Ptr<WarpNode> node = nodes[ixn];
                        Point3f c = node->pos;

                        //TODO URGENT: centered
                        UnitDualQuaternion dqi = node->transform.centered(node->pos);

                        dqsum += w * dqi.dq();
                        wsum += w;
                        nValid++;
                    }
                }

                dqsum += dampedDQ(knn, wsum, damping);

                UnitDualQuaternion dqn = dqsum.normalized();
                // We don't use commondq here, it's done at other stages of pipeline
                UnitDualQuaternion dqfull = dqn; // dqfull = dqn * commondq;

                Affine3f rt = dqfull.getRt();

                Point3f warpedP = rt * inp;
                ptsInWarped[y * size.width + x] = toPtype(warpedP);
                Point3f inrp = vol2cam * warpedP;
                ptsInWarpedRendered[y*size.width+x] = toPtype(inrp);

                // Fill transformed normals
                Point3f warpedN = rt.rotation() * inn;
                ptsInWarpedNormals[y * size.width + x] = toPtype(warpedN);
                Point3f inrn = vol2cam.rotation() * warpedN;
                ptsInWarpedRenderedNormals[y * size.width + x] = toPtype(inrn);

                // Calculate current step error
                {
                    // Project it to screen to get corresponding out point to calc delta
                    Point2f outXY = proj(inrp);
                    // Get newPoint and newNormal
                    Point3f outVolP, outVolN;
                    if (!(interpolateP3f(outXY, ptsOutVolP.data(), size.width, outVolP) &&
                          interpolateP3f(outXY, ptsOutVolN.data(), size.width, outVolN)))
                        continue;

                    float pointToPlaneDistance = outVolN.dot(outVolP - warpedP);

                    if (!cvIsInf(pointToPlaneDistance) && !cvIsNaN(pointToPlaneDistance))
                        vertResiduals.push_back(pointToPlaneDistance);

                    sumError += pointToPlaneDistance;
                    sumSqError += pointToPlaneDistance * pointToPlaneDistance;
                    nPixelsError++;
                }
            }
        }

        vertSigma = 1.f;
        {
            float vertMed = median(vertResiduals);
            std::for_each(vertResiduals.begin(), vertResiduals.end(),
                          [vertMed](float& x) {x = std::abs(x - vertMed); });
            vertSigma = MAD_SCALE * median(vertResiduals);
        }

        double meanError = sumError / nPixelsError;
        double stddevError = sqrt(sumSqError / nPixelsError - meanError);

        // Calculate residuals by edges
        regSigmas = std::vector<float>(graph.size(), 1.f);
        for (int level = 0; level < graph.size(); level++)
        {
            auto childLevelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            auto levelNodes = warp.getGraphNodes()[level];
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

            float regMed = median(regLevResiduals);
            std::for_each(regLevResiduals.begin(), regLevResiduals.end(),
                          [regMed](float& x) {x = std::abs(x - regMed); });
            regSigmas[level] = MAD_SCALE * median(regLevResiduals);
        }

        //TODO: visualize iteration by iteration

        //DEBUG
        std::cout << "#" << nIter << " mean: " << meanError << ", std: " << stddevError;
        std::cout << ", vertSigma: " << vertSigma;
        std::cout << ", regSigmas: ";
        for (auto f : regSigmas) std::cout << f;
        std::cout << std::endl;
    }

    //TODO URGENT: factor out common Rt
    {
        std::vector<Point3f> inPts, outPts;
        inPts.reserve(nPts); outPts.reserve(nPts);
        for (int y = 0; y < size.height; y++)
        {
            for (int x = 0; x < size.width; x++)
            {
                //TODO: Mat::ptr() instead
                // Get ptsIn from shaded data
                Point3f vshad = vertImage.at<Point3f>(y, x);
                Point3f inp(vshad.x * volume->volSize.x,
                            vshad.y * volume->volSize.y,
                            vshad.z * volume->volSize.z);
                Point3f outp = fromPtype(ptsInWarped[y * size.width + x]);
                if (fastCheck(inp) && fastCheck(outp))
                {
                    inPts.push_back(inp);
                    outPts.push_back(outp);
                }
            }
        }

        Matx44f commonM;
        estimateAffine3D(inPts, outPts, commonM, noArray());
        Affine3f common(commonM);

        // Looks like procedure is the same for all levels
        for (int level = 0; level < graph.size(); level++)
        {
            auto levelNodes = (level == 0) ? warp.getNodes() : warp.getGraphNodes()[level - 1];
            for (int ixn = 0; ixn < levelNodes.size(); ixn++)
            {
                Ptr<WarpNode> node = levelNodes[ixn];
                node->transform = node->transform.factoredOut(node->transform, node->pos);
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


    //TODO: where to factor out?
    //let's do it after the optimization


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
