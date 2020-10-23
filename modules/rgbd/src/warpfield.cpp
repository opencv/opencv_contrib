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

    cvflann::LinearIndexParams params;
    flann::GenericIndex<flann::L2_Simple <float> > searchIndex(points_matrix, params);

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
    nodeIndex = new flann::GenericIndex<flann::L2_Simple<float> >(nodesPos,
                                                                  cvflann::LinearIndexParams());

    constructRegGraph();
}


void WarpField::removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind,
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
                                                      flann::GenericIndex<flann::L2_Simple<float> >& ind,
                                                      AutoBuffer<bool>& validIndex, float res,
                                                      Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex)
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

        ind.radiusSearch(pmat.row(i), indices_vec, dist_vec, res, cvflann::SearchParams());

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

    for(auto nodePtr: nv)
    {
        std::vector<int> knnIndices(k);
        std::vector<float> knnDists(k);

        std::vector<float> query = {nodePtr->pos.x, nodePtr->pos.y, nodePtr->pos.z};

        nodeIndex->knnSearch(query, knnIndices, knnDists, k, cvflann::SearchParams());

        std::vector<float> weights(knnIndices.size());
        std::vector<UnitDualQuaternion> transforms(knnIndices.size());

        size_t i = 0;
        for(int idx: knnIndices)
        {
            weights[i] = nodes[idx]->weight(nodePtr->pos);
            transforms[i++] = nodes[idx]->transform;
        }

        UnitDualQuaternion pose = DQB(weights, transforms);
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

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > curNodeIndex(
        new flann::GenericIndex<flann::L2_Simple<float> >(curNodeMatrix,
                                                          cvflann::LinearIndexParams()));

    for(int l = 0; l < (n_levels-1); l++)
    {
        AutoBuffer<bool> nodeValidity;
        nodeValidity.allocate(curNodeIndex->size());

        std::fill_n(nodeValidity.data(), curNodeIndex->size(), true);
        std::vector<Ptr<WarpNode> > coarseNodes = subsampleIndex(curNodeMatrix, *curNodeIndex, nodeValidity,
                                                                 effResolution);

        initTransforms(coarseNodes);

        Mat coarseNodeMatrix = getNodesPos(coarseNodes);

        Ptr<flann::GenericIndex<flann::L2_Simple<float> > > coarseNodeIndex(
            new flann::GenericIndex<flann::L2_Simple<float> >(coarseNodeMatrix,
                                                              cvflann::LinearIndexParams()));

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


/*
Calculate DQB transform at point p and apply it to p
Normal calculation is done the same way but translation is not applied
*/
Point3f WarpField::applyWarp(Point3f p, const NodeNeighboursType neighbours, int n, bool normal) const
{
    CV_TRACE_FUNCTION();

    // DQB:

    if(!n)
        return p;

    std::vector<float> weights(n);
    std::vector<UnitDualQuaternion> transforms(n);
    float totalWeightSquare = 0.f;
    for(int i = 0; i < n; i++)
    {
        Ptr<WarpNode> neigh = nodes[neighbours[i]];
        transforms[i] = neigh->centeredRt();
        float w = neigh->weight(p);
        weights[i]= w;
        totalWeightSquare = w*w;
    }
    if(abs(totalWeightSquare) > 0.001f)
    {
        Affine3f rt = DQB(weights, transforms).getRt();
        if(normal)
        {
            Affine3f r(rt.rotation());
            return r*p;
        }
        else
        {
            return rt*p;
        }
    }
    else
    {
        return p;
    }
}

} // namepsace dynafu
} // namespace cv
