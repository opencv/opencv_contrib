// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <algorithm>
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

    virtual bool estimateWarpNodes(WarpField& currentWarp, const Affine3f &pose,
                                   InputArray vertImage, InputArray oldPoints,
                                   InputArray oldNormals, InputArray newPoints,
                                   InputArray newNormals) const override;

    virtual ~ICPImpl() {}
};

ICPImpl::ICPImpl(const Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume, int _iterations) :
NonRigidICP(_intrinsics, _volume, _iterations)
{}

static inline bool fastCheck(const Point3f& p)
{
    return !cvIsNaN(p.x);
}

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

static float tukeyWeight(float r, float sigma)
{
    float x = r/sigma;
    if(std::abs(x) <= TUKEY_B)
    {
        float y = 1 - (x*x)/(TUKEY_B * TUKEY_B);
        return y*y;

    } else return 0;
}

static float huberWeight(Vec3f v, float sigma)
{
    if(sigma == 0) return 0.f;
    float x = (float)std::abs(norm(v)/sigma);
    return (x > HUBER_K)? HUBER_K/x : 1.f;
}

static void fillRegularization(Mat_<float>& A_reg, Mat_<float>& b_reg, WarpField& currentWarp,
                               int totalNodes, std::vector<int> baseIndices)
{
    int nLevels = currentWarp.n_levels;
    int k = currentWarp.k;

    const NodeVectorType& warpNodes = currentWarp.getNodes();
    // Accumulate regularisation term for each node in the heiarchy
    const std::vector<NodeVectorType>& regNodes = currentWarp.getGraphNodes();
    const hierarchyType& regGraph = currentWarp.getRegGraph();

    // populate residuals for each edge in the graph to calculate sigma
    std::vector<float> reg_residuals;
    float RegEnergy = 0;
    int numEdges = 0;
    for(int l = 0; l < (nLevels-1); l++)
    {
        const std::vector<nodeNeighboursType>& level = regGraph[l];

        const NodeVectorType& currentLevelNodes = (l == 0)? warpNodes : regNodes[l-1];

        const NodeVectorType& nextLevelNodes = regNodes[l];

        std::cout << currentLevelNodes.size() << " " << nextLevelNodes.size() << std::endl;


        for(size_t node = 0; node < level.size(); node++)
        {
            const nodeNeighboursType& children = level[node];
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
        const std::vector<nodeNeighboursType>& level = regGraph[l];

        const NodeVectorType& currentLevelNodes = (l == 0)? warpNodes : regNodes[l-1];
        const NodeVectorType& nextLevelNodes = regNodes[l];

        for(size_t node = 0; node < level.size(); node++)
        {
            const nodeNeighboursType& children = level[node];
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


bool ICPImpl::estimateWarpNodes(WarpField& currentWarp, const Affine3f &pose,
                                InputArray _vertImage, InputArray _oldPoints,
                                InputArray _oldNormals, InputArray _newPoints,
                                InputArray _newNormals) const
{
    CV_Assert(_vertImage.isMat());
    CV_Assert(_oldPoints.isMat());
    CV_Assert(_newPoints.isMat());
    CV_Assert(_newNormals.isMat());

    Mat vertImage = _vertImage.getMat();
    Mat oldPoints = _oldPoints.getMat();
    Mat newPoints = _newPoints.getMat();
    Mat newNormals = _newNormals.getMat();
    Mat oldNormals = _oldNormals.getMat();

    CV_Assert(!vertImage.empty());
    CV_Assert(!oldPoints.empty());
    CV_Assert(!newPoints.empty());
    CV_Assert(!newNormals.empty());



    // let's start from porting Python script to c++ code
    float damping = 0.f;

    uint32_t nIter = 0;
    bool decorrelate = false;
    bool useTukey = true;
    bool bendInitial = true;
    bool signFix = false;
    bool disableCentering = false; //TODO
    bool atZero = false;
    bool useExp = false;
    bool tryLevMarq = true;

    double lambdaLevMarq = 0.1;
    uint32_t nNodes = len(nodesParams);
    std::vector<Point3f> ptsInitial(oldPoints.total());
    oldPoints.copyTo(ptsInitial);
    std::vector<Point3f> ptsOut(newPoints.total());
    newPoints.copyTo(ptsOut);

    std::vector<Point3f> ptsIn = ptsInitial;
    std::vector<DualQuaternion> dqcum(nNodes, DualQuaternion());
    std::vector< std::map<int, float> > weights;
    // Calculate distance-based weights
    for (unsigned int i = 0; i < ptsInitial.size(); i++)
    {
        unsigned int npnodes; // TODO: this
        unsigned int pnodes[DYNAFU_MAX_NEIGHBOURS]; //TODO: this
        for (unsigned int wi = 0; wi < npnodes; wi++)
        {
            unsigned int nodeIdx = pnodes[wi];
            //TODO
            float distWeight = dist_weight(ptsInitial[i], nodes[nodeIdx].center, csize);
            weights[i][nodeIdx] = distWeight;
        }
    }

    for (unsigned int it = 0; it < nIter; it++)
    {
        std::vector<Point3f> ptsDelta(ptsInitial.size());
        for (unsigned int i = 0; i < ptsInitial.size(); i++)
        {
            ptsDelta[i] = ptsOut[i] - ptsIn[i];
        }

        const int jacobDataType = CV_64FC1;
        const int block = 6;
        Mat jtj(block*nNodes, block*nNodes, jacobDataType); // 64F sic!
        Mat jtb(block*nNodes, 1, jacobDataType);

        // Build per-node jacobians (don't depend on vertices)
        std::vector<Mat> jpernode(nNodes);
        for(unsigned int i = 0; i < nNodes; i++)
        {
            jpernode[i] = Mat::zeros(8, block, CV_32FC1);
            // Current node's center
            Point3f c = nodes[i].center; // TODO
            // Current rotation, translation
            DualQuaternion dq = atZero ? DualQuaternion() : nodes[i].rt;
            Affine3f rt = dq.getAffineUnit(); // TODO

            Mat jpn;
            if(useExp)
            {
                jpn = j_centered(c) * j_dq_exp_val(dq);
            }
            else
            {
                jpn = j_pernode(rt, c);
            }
            jpernode[i] = jpn;
        }

        //TODO: BIG ONE regularization

        // Build vertex-dependent part of jacobian
        for(unsigned int i = 0; i < ptsInitial.size(); i++)
        {
            //TODO: BIG ONES projection, normals


            Point3f v = ptsIn[i];
            Point3f d = ptsDelta[i];
            DualQuaternion dqsum = DualQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0));
            float wsum = 0.f;
            std::vector<Mat> jnode, jpernodeweighted;
            //TODO: take only nodes that belong to a point
            for(unsigned int node = 0; node < nNodes; node++)
            {
                jnode[node] = Mat::zeros(3, block, CV_32FC1);
                jpernodeweighted[node] = Mat::zeros(8, block, CV_32FC1);
                // Current node's center
                Point3f c = nodes[node].center; // TODO
                // Current weight
                float w = weights[i][node];
                // Current rotation, translation
                DualQuaternion dq = nodes[i].rt; // TODO
                Affine3f rt = dq.getAffineUnit();

                // center(x) := (1+e*1/2*с)*x*(1-e*1/2*c)
                // TODO: or dq.centered(c);
                DualQuaternion centered = DualQuaternion::from_rt_centered(rt, c);

                jpernodeweighted[node] = w*jpernode[node];
                dqsum += w*centered;
                wsum += w;
            }

            dqsum = DualQuaternion::damped_dqsum(dqsum, nNodes, wsum, damping);
            Quaternion a = dqsum.real(), b = dqsum.dual();

            // Jacobian of normalization+application to a point
            Mat jnormapply = Mat::zeros(3, 8, CV_32FC1);
            jnormapply = j_normapply(a, b, v);
            //TODO: take only nodes that belong to a point
            for(unsigned int node = 0; node < nNodes; node++)
            {
                jnode[node] = jnormapply * jpernodeweighted[node];
            }

            // TODO: maybe less memory?
            Mat jm = Mat::zeros(3, block*nNodes, CV_32FC1);
            for(unsigned int node = 0; node < nNodes; node++)
            {
                jnode[node].copyTo(jm(Rect(block*node, 0, block*(node+1), 3)));
            }

            // TODO: maybe more effective?
            Mat left  = jm.t() * jm;
            Mat right = jm.t() * Matx31f(d);

            float lsweight = useTukey ? tukey(norm(d)) : 1.f;

            jtj += left*lsweight;
            jtb += right*lsweight;
        }

        // Used in DynaFu, simplifies calculation
        if(decorrelate)
        {
            jtj = decorrelated(jtj, nNodes);
        }

        if(tryLevMarq)
        {
            for (int row = 0; row < jtj.rows; row++)
            {
                jtj.at<float>(row, row) = jtj.at<float>(row, row)*(1.0 + lambdaLevMarq);
            }
        }

        // Solve and get delta transform
        x = solve(jtj, jtb);


    /*



    # Build transforms from params
    dqits = []
    for node in range(nNodes):
        blockNode = x[block*node : block*(node+1)]

        if useExp:
            # the order is the opposite
            dualx = blockNode[0:3]
            realx = blockNode[3:6]

            qrx = np.quaternion(0, *realx)
            qdx = np.quaternion(0, *dualx)
            # delta dq
            dqit = DualQuaternion(qrx, qdx).exp()

        else:
            rotparams   = blockNode[0:3]
            transparams = blockNode[3:6]

            # this function expects input vector to be a norm of full angle
            # while it contains just 1/2 of an angle
            qx = quaternion.from_rotation_vector(rotparams*2)
            tx = quaternion.from_float_array([0, *transparams])
            # delta dq
            dqit = DualQuaternion.from_rt(qx, tx)

        dqits.append(dqit)

        # dqcum_i = dqx_i * dqcum_i
        dqc = dqcum[node]
        dqc = DualQuaternion(dqit.real * dqc.real, dqit.real*dqc.dual + dqit.dual*dqc.real)
        dqcum[node] = dqc

    # Sign fix for DQB
    if signFix:
        dqcum = my_dq.dqb_sign_fixed(dqcum)
        dqits = my_dq.dqb_sign_fixed(dqits)

    # build effective (centered) DQs for dq_bend()
    dqit_effective = []
    dqcum_effective = []
    for node in range(nNodes):
        c = cs[node]
        dqcum_effective.append(dqcum[node].centered(c))
        dqit_effective.append(dqits[node].centered(c))

        #rcum1, tcum1 = dqcum1.get_rt()
        #vx1 = quaternion.as_rotation_vector(rcum1)
        #vxnorm1 = math.sqrt(np.dot(vx1, vx1))
        #print('vx1 a/a:', vxnorm1, vx1/vxnorm1)

        if it == nIter - 1:
            draw_dq(dqcum[node], c, 0.95*it/nIter, 'magenta')
        else:
            #draw_dq(dqcum[node], c, 0.95*it/nIter, 'magenta')
            pass

    # Transform vertices
    if bendInitial:
        ptsi = ptsInitial.copy()
        dq_effective = dqcum_effective
    else:
        ptsi = ptsin.copy()
        dq_effective = dqit_effective
    for i in range(ptsin.shape[0]):
        inp = ptsi[i]
        #DEBUG
        #ptsi[i] = my_dq.dq_bend(inp, dq_effective, cs, csize, damping)
        ptsi[i] = my_dq.dq_bend(inp, dq_effective, cs, csize, damping, sign_fix=False)
    ptsin = ptsi

    diff = np.linalg.norm(ptsout - ptsin, axis=1)
    print('mean, stddev:', np.mean(diff), np.std(diff))

    if it == nIter - 1:
        scati = ax.scatter(ptsin[:, 0], ptsin[:, 1], ptsin[:, 2], zdir = 'z', s=sz, c = colors[it])
        pass
    else:
        #scati = ax.scatter(ptsin[:, 0], ptsin[:, 1], ptsin[:, 2], zdir = 'z', s=sz, c = colors[it])
        pass
*/
    }

    //TODO: visualize iteration by iteration
    //TODO: less code duplication

    const NodeVectorType& warpNodes = currentWarp.getNodes();

    Affine3f T_lw = pose.inv() * volume->pose;

    // Accumulate regularisation term for each node in the heiarchy
    const std::vector<NodeVectorType>& regNodes = currentWarp.getGraphNodes();

    int totalNodes = (int)warpNodes.size();
    for(const auto& nodes: regNodes) totalNodes += (int)nodes.size();

    // level-wise regularisation components of A and b (from Ax = b) for each node in heirarchy
    Mat_<float> b_reg(6*totalNodes, 1, 0.f);
    Mat_<float> A_reg(6*totalNodes, 6*totalNodes, 0.f);

    // indices for each node block to A,b matrices. It determines the order
    // in which paramters are laid out

    std::vector<int> baseIndices(currentWarp.n_levels, 0);

    for(int l = currentWarp.n_levels-2; l >= 0; l--)
    {
        baseIndices[l] = baseIndices[l+1]+6*((int)regNodes[l].size());
    }

    for(const int& i: baseIndices) std::cout << i << ", ";
    std::cout << std::endl;

    fillRegularization(A_reg, b_reg, currentWarp, totalNodes, baseIndices);

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
            nodeNeighboursType neighbours = volume->getVoxelNeighbours(p, n);
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

    double det = determinant(A_reg);
    std::cout << "A_reg det:" << det  << std::endl;

    std::cout << "Solving " << 6*totalNodes << std::endl;
    Mat_<float> nodeTwists(6*totalNodes, 1, 0.f);
    bool result;

#if defined(HAVE_EIGEN)
    std::cout << "starting eigen-insertion..." << std::endl;
    Eigen::SparseMatrix<float> sA_reg(6*totalNodes, 6*totalNodes);
    //TODO: batch insertion/offload here and everywhere
    for (int y = 0; y < 6*totalNodes; y++)
    {
        for (int x = 0; x < 6*totalNodes; x++)
        {
            float v = A_reg(y, x);
            //TODO: add real check
            if(abs(v) > 0.01f)
            {
                sA_reg.insert(y, x) = v;
            }
        }
    }
    sA_reg.makeCompressed();
    Eigen::VectorXf sb_reg(6*totalNodes);
    for (int y = 0; y < 6*totalNodes; y++)
    {
        sb_reg(y) = b_reg(y);
    }
    //Eigen::SimplicialCholesky<Eigen::SparseMatrix<float>> solver;
    Eigen::SparseQR<Eigen::SparseMatrix<float>, Eigen::NaturalOrdering<int>> solver;
    std::cout << "starting eigen-compute..." << std::endl;
    solver.compute(sA_reg);
    if(solver.info() != Eigen::Success)
    {
        std::cout << "failed to eigen-decompose" << std::endl;
        result = false;
    }
    else
    {
        std::cout << "starting eigen-solve..." << std::endl;
        Eigen::VectorXf sx = solver.solve(sb_reg);
        if(solver.info() != Eigen::Success)
        {
            std::cout << "failed to eigen-solve" << std::endl;
            result = false;
        }
        else
        {
            for (int y = 0; y < 6*totalNodes; y++)
            {
                nodeTwists(y) = sx(y);
            }
            result = true;
        }
    }
#else
    std::cout << "no eigen" << std::endl;
    result = solve(A_reg, b_reg, nodeTwists, DECOMP_SVD);
#endif

    std::cout << "Done " << result << std::endl;

    if(!result)
        return false;

    for(int i = 0; i < (int)warpNodes.size(); i++)
    {
        int idx = baseIndices[0]+6*i;
        Vec3f r(nodeTwists(idx), nodeTwists(idx+1), nodeTwists(idx+2));
        Vec3f t(nodeTwists(idx+3), nodeTwists(idx+4), nodeTwists(idx+5));
        Affine3f tinc(r, t);
        warpNodes[i]->transform = warpNodes[i]->transform * tinc;
    }

    std::cout << "Nan count: " << "/" << warpNodes.size() << "\n" << std::endl;

    return true;
}

cv::Ptr<NonRigidICP> makeNonRigidICP(const cv::kinfu::Intr _intrinsics, const cv::Ptr<TSDFVolume>& _volume,
                                     int _iterations)
{
    return makePtr<ICPImpl>(_intrinsics, _volume, _iterations);
}

} // namespace dynafu
} // namespace cv
