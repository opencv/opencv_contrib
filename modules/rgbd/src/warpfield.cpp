#include "precomp.hpp"
#include "warpfield.hpp"

namespace cv {
namespace dynafu {

bool PtCmp(cv::Point3f a, cv::Point3f b)
{
    return (a.x < b.x) ||
            ((a.x >= b.x) && (a.y < b.y)) ||
            ((a.x >= b.x) && (a.y >= b.y) && (a.z < b.z));
}

Mat getNodesPos(const std::vector<Ptr<WarpNode> >& nv)
{
    Mat nodePos((int)nv.size(), 3, CV_32F);
    for(int i = 0; i < (int)nv.size(); i++)
    {
        nodePos.at<float>(i, 0) = nv[i]->pos.x;
        nodePos.at<float>(i, 1) = nv[i]->pos.y;
        nodePos.at<float>(i, 2) = nv[i]->pos.z;
    }
    return nodePos;
}

/*
1. Find a set of new points that are not covered by existing nodes yet
2. Covers them by a new set of nodes
3. Adds these nodes to a set of nodes
4. Rebuilds an index
*/
void WarpField::updateNodesFromPoints(InputArray inputPoints)
{
    Mat points_matrix(inputPoints.size().height, 3, CV_32F);
    if(inputPoints.channels() == 1)
    {
        points_matrix = inputPoints.getMat().colRange(0, 3);
    }
    else
    {
        points_matrix = inputPoints.getMat().reshape(1).colRange(0, 3).clone();
    }

    //TODO URGENT: check this
    //cvflann::LinearIndexParams params;
    cvflann::AutotunedIndexParams params;
    
    //TODO URGENT: this takes too long, what to do?
    Index searchIndex(points_matrix, params);

    AutoBuffer<bool> validIndex;
    removeSupported(searchIndex, validIndex);

    std::vector<Ptr<WarpNode> > newNodes;
    if((int)nodes.size() > k)
    {
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes, nodeIndex);
    }
    else
    {
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes);
    }

    initTransforms(newNodes);
    nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());
    nodesPos = getNodesPos(nodes);
    //re-build index
    nodeIndex = new Index(nodesPos,
                          /* cvflann::LinearIndexParams() */
                          cvflann::AutotunedIndexParams()
                          );

    constructRegGraph();
}


void WarpField::removeSupported(Index& ind,
                                AutoBuffer<bool>& validInd)
{
    validInd.allocate(ind.size());
    std::fill_n(validInd.data(), ind.size(), true);

    for(WarpNode* n: nodes)
    {
        std::vector<float> query = {n->pos.x, n->pos.y, n->pos.z};

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dists_vec(maxNeighbours);

        ind.radiusSearch(query, indices_vec, dists_vec, n->radius, cvflann::SearchParams());

        for(auto i: indices_vec)
        {
            validInd[i] = false;
        }
    }
}

/*
Covers given (valid) points `pmat` and their search index `ind` by nodes with radius `res`.
Returns a set of nodes that cover the points.
*/
//TODO: fix its undefined order of coverage
std::vector<Ptr<WarpNode> > WarpField::subsampleIndex(Mat& pmat,
                                                      Index& ind,
                                                      AutoBuffer<bool>& validIndex, float res,
                                                      Ptr<Index> knnIndex)
{
    CV_TRACE_FUNCTION();

    std::vector<Ptr<WarpNode> > temp_nodes;

    for(int i = 0; i < (int)validIndex.size(); i++)
    {
        if(!validIndex[i])
        {
            continue;
        }

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dist_vec(maxNeighbours);

        int amt = ind.radiusSearch(pmat.row(i), indices_vec, dist_vec, res, cvflann::SearchParams());
        // set of results should be sorted, crop the results
        indices_vec.resize(amt);
        dist_vec.resize(amt);

        Ptr<WarpNode> wn = new WarpNode;
        Point3f centre(0, 0, 0);
        float len = 0;
        for(int index: indices_vec)
        {
            if(validIndex[index])
            {
                centre += Point3f(pmat.at<float>(index, 0),
                                  pmat.at<float>(index, 1),
                                  pmat.at<float>(index, 2));
                len++;
            }
        }

        centre /= len;
        wn->pos = centre;

        for(auto index: indices_vec)
        {
            validIndex[index] = false;
        }

        std::vector<int> knn_indices(k+1, 0);
        std::vector<float> knn_dists(k+1, 0);

        std::vector<float> query = {wn->pos.x, wn->pos.y, wn->pos.z};

        if(knnIndex != nullptr)
        {
            knnIndex->knnSearch(query, knn_indices, knn_dists, (k+1), cvflann::SearchParams());
            wn->radius = *std::max_element(knn_dists.begin(), knn_dists.end());
        }
        else
        {
            wn->radius = res;
        }

        wn->transform = UnitDualQuaternion();

        temp_nodes.push_back(wn);
    }

    return temp_nodes;
}


/*
Sets each node's transform to a DQB-interpolated transform of other nodes in the node's center.
*/
void WarpField::initTransforms(std::vector<Ptr<WarpNode> > nv)
{
    if(nodesPos.size().height == 0)
    {
        return;
    }

    for(auto &nodePtr: nv)
    {
        Point3f pos = nodePtr->pos;
        NodeNeighboursType neighbours = findNeighbours(pos);
       
        DualQuaternion dqsum = warpForVertex(pos, neighbours);

        UnitDualQuaternion pose = dqsum.normalized();
        // Here we prepare pose for (possible) centering
        pose = disableCentering ? pose : pose.centered(-pos);
        nodePtr->transform = pose;
    }
}


/*
 At each level l of nodes:
 * subsample them -> nodes[l+1]
 *
*/
void WarpField::constructRegGraph()
{
    CV_TRACE_FUNCTION();

    regGraphNodes.clear();

    if(n_levels == 1) // First level (Nwarp) is already stored in nodes
    {
        return;
    }

    float effResolution = baseRes*resGrowthRate;
    std::vector<Ptr<WarpNode> > curNodes = nodes;
    Mat curNodeMatrix = getNodesPos(curNodes);

    Ptr<Index> curNodeIndex(new Index(curNodeMatrix,
                                      /* cvflann::LinearIndexParams() */
                                      cvflann::AutotunedIndexParams()
                                      )
                            );

    for(int l = 0; l < (n_levels-1); l++)
    {
        AutoBuffer<bool> nodeValidity;
        nodeValidity.allocate(curNodeIndex->size());

        std::fill_n(nodeValidity.data(), curNodeIndex->size(), true);
        std::vector<Ptr<WarpNode> > coarseNodes = subsampleIndex(curNodeMatrix, *curNodeIndex, nodeValidity,
                                                                 effResolution);

        initTransforms(coarseNodes);

        Mat coarseNodeMatrix = getNodesPos(coarseNodes);

        Ptr<Index> coarseNodeIndex(new Index(coarseNodeMatrix,
                                             /* cvflann::LinearIndexParams() */
                                             cvflann::AutotunedIndexParams()
                                            )
                                  );

        hierarchy[l] = std::vector<NodeNeighboursType>(curNodes.size());
        for(int i = 0; i < (int)curNodes.size(); i++)
        {
            std::vector<int> children_indices(k);
            std::vector<float> children_dists(k);

            std::vector<float> query = {curNodeMatrix.at<float>(i, 0),
                                        curNodeMatrix.at<float>(i, 1),
                                        curNodeMatrix.at<float>(i, 2)};

            coarseNodeIndex->knnSearch(query, children_indices, children_dists, k,
                                       cvflann::SearchParams());
            hierarchy[l][i].fill((size_t)-1);
            std::copy(children_indices.begin(), children_indices.end(), hierarchy[l][i].begin());
        }

        regGraphNodes.push_back(coarseNodes);
        curNodes = coarseNodes;
        curNodeMatrix = coarseNodeMatrix;
        curNodeIndex = coarseNodeIndex;
        effResolution *= resGrowthRate;
    }
}


// Calculate DQB transform at point p and apply it to p
// If p is normal, do not apply translation
Point3f WarpField::applyWarp(const Point3f p, const NodeNeighboursType neighbours, bool normal) const
{
    CV_TRACE_FUNCTION();

    Affine3f rt = warpForVertex(p, neighbours).getRt();

    if (normal)
    {
        return rt.rotation() * p;
    }
    else
    {
        return rt * p;
    }
}


Point3f WarpField::applyWarp(const Point3f p, const NodeWeightsType weights, const NodeNeighboursType neighbours, bool normal) const
{
    CV_TRACE_FUNCTION();

    throw "useless function";

    Affine3f rt = warpForKnns(neighbours, weights).getRt();

    if (normal)
    {
        return rt.rotation() * p;
    }
    else
    {
        return rt * p;
    }
}


DualQuaternion WarpField::warpForKnns(const NodeNeighboursType neighbours, const NodeWeightsType weights) const
{
    CV_TRACE_FUNCTION();

    DualQuaternion dqsum;
    float wsum = 0; int nValid = 0;
    for (int i = 0; i < DYNAFU_MAX_NEIGHBOURS; i++)
    {
        int ixn = neighbours[i];
        if (ixn >= 0)
        {
            Ptr<WarpNode> node = nodes[ixn];

            // center(x) := (1+e*1/2*c)*x*(1-e*1/2*c)
            UnitDualQuaternion dqi = disableCentering ? node->transform : node->centeredRt();

            float w = weights[i];

            dqsum += w * dqi.dq();
            wsum += w;
            nValid++;
        }
        else
            break;
    }

    dqsum += dampedDQ(nValid, wsum, damping);
    return dqsum;
}


DualQuaternion WarpField::warpForVertex(const Point3f vertex, NodeNeighboursType neighbours) const
{
    NodeWeightsType weights { };
    int n = 0;
    for (int i = 0; i < DYNAFU_MAX_NEIGHBOURS; i++)
    {
        int ixn = neighbours[i];
        if (ixn >= 0)
        {
            weights[i] = nodes[ixn]->weight(vertex);
            n++;
        }
        else
            break;
    }
    // to prevent access to incorrect indices
    for (int i = n; i < DYNAFU_MAX_NEIGHBOURS; i++)
    {
        neighbours[i] = -1;
    }

    return warpForKnns(neighbours, weights);
}


} // namepsace dynafu
} // namespace cv
