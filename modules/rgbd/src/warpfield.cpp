#include "precomp.hpp"
#include "warpfield.hpp"

namespace cv {
namespace dynafu {

WarpField::WarpField(int _maxNeighbours, int K, int levels, float baseResolution, float resolutionGrowth):
k(K), n_levels(levels),
nodes(), maxNeighbours(_maxNeighbours), // good amount for dense kinfu pointclouds
baseRes(baseResolution),
resGrowthRate(resolutionGrowth),
regGraphNodes(n_levels-1),
heirarchy(n_levels-1),
nodeIndex(nullptr)
{
    CV_Assert(k <= DYNAFU_MAX_NEIGHBOURS);
}

NodeVectorType const& WarpField::getNodes() const
{
    return nodes;
}

std::vector<NodeVectorType> const& WarpField::getGraphNodes() const
{
    return regGraphNodes;
}

heirarchyType const& WarpField::getRegGraph() const
{
    return heirarchy;
}

bool PtCmp(cv::Point3f a, cv::Point3f b)
{
    return (a.x < b.x) ||
            ((a.x >= b.x) && (a.y < b.y)) ||
            ((a.x >= b.x) && (a.y >= b.y) && (a.z < b.z));
}

Ptr<flann::GenericIndex<flann::L2_Simple<float> > > WarpField::getNodeIndex() const
{
    return nodeIndex;
}

Mat getNodesPos(NodeVectorType nv)
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

    NodeVectorType newNodes;
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

NodeVectorType WarpField::subsampleIndex(Mat& pmat,
                                         flann::GenericIndex<flann::L2_Simple<float> >& ind,
                                         AutoBuffer<bool>& validIndex, float res,
                                         Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex)
{
    CV_TRACE_FUNCTION();

    NodeVectorType temp_nodes;

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

        wn->transform = Affine3f::Identity();

        temp_nodes.push_back(wn);
    }

    return temp_nodes;
}

void WarpField::initTransforms(NodeVectorType nv)
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

        nodeIndex->knnSearch(query ,knnIndices, knnDists, k, cvflann::SearchParams());

        std::vector<float> weights(knnIndices.size());
        std::vector<Affine3f> transforms(knnIndices.size());

        size_t i = 0;
        for(int idx: knnIndices)
        {
            weights[i] = nodes[idx]->weight(nodePtr->pos);
            transforms[i++] = nodes[idx]->transform;
        }

        Affine3f pose = DQB(weights, transforms);
        // linearly interpolate translations
        Vec3f translation(0,0,0);
        float totalWeight = 0;
        for(i = 0; i < transforms.size(); i++)
        {
            translation += weights[i]*transforms[i].translation();
            totalWeight += weights[i];
        }

        if(totalWeight < 1e-5) translation = Vec3f(0, 0, 0);
        else translation /= totalWeight;
        nodePtr->transform = Affine3f(pose.rotation(), translation);
    }
}

void WarpField::constructRegGraph()
{
    CV_TRACE_FUNCTION();

    regGraphNodes.clear();

    if(n_levels == 1) // First level (Nwarp) is already stored in nodes
    {
        return;
    }

    float effResolution = baseRes*resGrowthRate;
    NodeVectorType curNodes = nodes;
    Mat curNodeMatrix = getNodesPos(curNodes);

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > curNodeIndex(
        new flann::GenericIndex<flann::L2_Simple<float> >(curNodeMatrix,
                                                          cvflann::LinearIndexParams()));

    for(int l = 0; l < (n_levels-1); l++)
    {
        AutoBuffer<bool> nodeValidity;
        nodeValidity.allocate(curNodeIndex->size());

        std::fill_n(nodeValidity.data(), curNodeIndex->size(), true);
        NodeVectorType coarseNodes = subsampleIndex(curNodeMatrix, *curNodeIndex, nodeValidity,
                                                    effResolution);

        initTransforms(coarseNodes);

        Mat coarseNodeMatrix = getNodesPos(coarseNodes);

        Ptr<flann::GenericIndex<flann::L2_Simple<float> > > coarseNodeIndex(
            new flann::GenericIndex<flann::L2_Simple<float> >(coarseNodeMatrix,
                                                              cvflann::LinearIndexParams()));

        heirarchy[l] = std::vector<nodeNeighboursType>(curNodes.size());
        for(int i = 0; i < (int)curNodes.size(); i++)
        {
            std::vector<int> children_indices(k);
            std::vector<float> children_dists(k);

            std::vector<float> query = {curNodeMatrix.at<float>(i, 0),
                                        curNodeMatrix.at<float>(i, 1),
                                        curNodeMatrix.at<float>(i, 2)};

            coarseNodeIndex->knnSearch(query, children_indices, children_dists, k,
                                       cvflann::SearchParams());
            heirarchy[l][i].fill(-1);
            std::copy(children_indices.begin(), children_indices.end(), heirarchy[l][i].begin());
        }

        regGraphNodes.push_back(coarseNodes);
        curNodes = coarseNodes;
        curNodeMatrix = coarseNodeMatrix;
        curNodeIndex = coarseNodeIndex;
        effResolution *= resGrowthRate;
    }

}

Point3f WarpField::applyWarp(Point3f p, const nodeNeighboursType neighbours, int n, bool normal) const
{
    CV_TRACE_FUNCTION();

    if(n == 0)
    {
        return p;
    }

    float totalWeight = 0;
    Point3f WarpedPt(0,0,0);

    for(int i = 0; i < n; i++)
    {
        Ptr<WarpNode> neigh = nodes[neighbours[i]];
        float w = neigh->weight(p);
        if(w < 0.01)
        {
            continue;
        }

        Matx33f R = neigh->transform.rotation();
        Point3f newPt(0, 0, 0);

        if(normal)
        {
            newPt = R * p;
        }
        else
        {
            newPt = R * (p - neigh->pos) + neigh->pos;
            Vec3f T = neigh->transform.translation();
            newPt.x += T[0];
            newPt.y += T[1];
            newPt.z += T[2];
        }

        WarpedPt += newPt * w;
        totalWeight += w;

    }
    WarpedPt /= totalWeight;

    if(totalWeight == 0)
    {
        return p;
    }
    else
    {
        return WarpedPt;
    }

}

void WarpField::setAllRT(Affine3f warpRT)
{
    for(auto n: nodes)
    {
        n->transform = warpRT;
    }
}

} // namepsace dynafu
} // namespace cv
