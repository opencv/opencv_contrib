#include "precomp.hpp"
#include "warpfield.hpp"

namespace cv {
namespace dynafu {

WarpField::WarpField(int _maxNeighbours, int K, int levels, float baseResolution, float resolutionGrowth): 
k(K), nodes(), maxNeighbours(_maxNeighbours), // good amount for dense kinfu pointclouds
n_levels(levels), baseRes(baseResolution), 
resGrowthRate(resolutionGrowth),
regGraphNodes(std::vector<NodeVectorType>(n_levels)),
nodeIndex(nullptr)
{
}

NodeVectorType WarpField::getNodes() const
{
    return nodes;
}

std::vector<NodeVectorType> WarpField::getGraphNodes() const
{
    return regGraphNodes;
}

bool PtCmp(cv::Point3f a, cv::Point3f b) {
    return (a.x < b.x) || ((a.x >= b.x) && (a.y < b.y)) || ((a.x >= b.x) && (a.y >= b.y) && (a.z < b.z));
}

Ptr<flann::GenericIndex<flann::L2_Simple<float> > > WarpField::getNodeIndex() const
{
    return nodeIndex; 
}

Mat getNodesPos(NodeVectorType nv) {
    Mat nodePos(nv.size(), 3, CV_32F);
    for(size_t i = 0; i < nv.size(); i++) {
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
        points_matrix = inputPoints.getMat().colRange(0, 3);
    else
        points_matrix = inputPoints.getMat().reshape(1).colRange(0, 3).clone();

    cvflann::LinearIndexParams params;
    flann::GenericIndex<flann::L2_Simple <float> > searchIndex(points_matrix, params);

    std::vector<bool> validIndex;
    removeSupported(searchIndex, validIndex);

    Mat nodePosMatrix = getNodesPos(nodes);

    NodeVectorType newNodes;
    if((int)nodes.size() > k) {
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes, nodeIndex);
    } else
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes);

    initTransforms(newNodes);
    nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());
    //re-build index
    nodeIndex = new flann::GenericIndex<flann::L2_Simple<float> >(getNodesPos(nodes), cvflann::LinearIndexParams());

    constructRegGraph();
}


void WarpField::removeSupported(flann::GenericIndex<flann::L2_Simple<float> >& ind, std::vector<bool>& validInd)
{
    std::vector<bool> validIndex(ind.size(), true);

    for(WarpNode* n: nodes)
    {
        std::vector<float> query = {n->pos.x, n->pos.y, n->pos.z};

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dists_vec(maxNeighbours);

        ind.radiusSearch(query, indices_vec, dists_vec, n->radius, cvflann::SearchParams());
        
        for(auto i: indices_vec)
            validIndex[i] = false;

    }

    validInd = validIndex;

}

NodeVectorType WarpField::subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind,
    std::vector<bool>& validIndex, float res, Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex)
{
    CV_TRACE_FUNCTION();

    NodeVectorType temp_nodes;

    for(size_t i = 0; i < validIndex.size(); i++)
    {
        if(!validIndex[i])
            continue;

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dist_vec(maxNeighbours);
        
        ind.radiusSearch(pmat.row(i), indices_vec, dist_vec, res, cvflann::SearchParams());
        

        Ptr<WarpNode> wn = new WarpNode;
        Point3f centre(0, 0, 0);
        float len = 0;
        for(int index: indices_vec)
            if(validIndex[index])
            {
                centre += Point3f(pmat.at<float>(index, 0), pmat.at<float>(index, 1), pmat.at<float>(index, 2));
                len++;
            }

        centre /= len;
        wn->pos = centre;

        for(auto index: indices_vec)
            validIndex[index] = false;

        std::vector<int> knn_indices(k+1, 0);
        std::vector<float> knn_dists(k+1, 0);

        std::vector<float> query = {wn->pos.x, wn->pos.y, wn->pos.z};

        if(knnIndex != nullptr)
        {
            knnIndex->knnSearch(query, knn_indices, knn_dists, (k+1), cvflann::SearchParams());
            wn->radius = knn_dists.back();
        } else wn->radius = res;

        wn->transform = Affine3f::Identity();

        temp_nodes.push_back(wn);
    }     
    
    return temp_nodes;
}

void WarpField::initTransforms(NodeVectorType nv)
{
    Mat nodePos = getNodesPos(nodes);
    if(nodePos.size().height == 0) return;
    

    for(auto nodePtr: nv)
    {
        std::vector<int> knnIndices(k);
        std::vector<float> knnDists(k);

        std::vector<float> query = {nodePtr->pos.x, nodePtr->pos.y, nodePtr->pos.z};

        nodeIndex->knnSearch(query ,knnIndices, knnDists, k, cvflann::SearchParams());

        float totalWeight = 0;
        Mat partialR(3,3,CV_32F);
        Vec3f partialT = Vec3f::all(0);
    
        for(int idx: knnIndices) {
            float w = nodes[idx]->weight(nodePtr->pos);

            totalWeight += w;
            partialR += nodes[idx]->transform.rotation() * w;
            partialT += nodes[idx]->transform.translation() * w;
        }

        partialR /= totalWeight;
        partialT /= totalWeight;

        nodePtr->transform = Affine3f(partialR, partialT);
    }
}

void WarpField::constructRegGraph()
{
    CV_TRACE_FUNCTION();

    regGraphNodes.clear();

    if(n_levels == 1) // First level (Nwarp) is already stored in nodes
        return;

    float effResolution = baseRes*resGrowthRate;
    NodeVectorType curNodes = nodes;
    Mat curNodeMatrix = getNodesPos(curNodes);

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > curNodeIndex(
        new flann::GenericIndex<flann::L2_Simple<float> >(curNodeMatrix, cvflann::LinearIndexParams()));
    
    for(int l = 0; l < (n_levels-1); l++)
    {
        std::vector<bool> nodeValidity(curNodeIndex->size(), true);
        NodeVectorType coarseNodes = subsampleIndex(curNodeMatrix, *curNodeIndex, nodeValidity, effResolution);

        Mat coarseNodeMatrix = getNodesPos(coarseNodes);

        Ptr<flann::GenericIndex<flann::L2_Simple<float> > > coarseNodeIndex(
            new flann::GenericIndex<flann::L2_Simple<float> >(coarseNodeMatrix, cvflann::LinearIndexParams()));

        for(size_t i = 0; i < curNodes.size(); i++)
        {
            std::vector<int> children_indices(k);
            std::vector<float> children_dists(k);
            
            std::vector<float> query = {curNodeMatrix.at<float>(i, 0), curNodeMatrix.at<float>(i, 1), curNodeMatrix.at<float>(i, 2)};
            coarseNodeIndex->knnSearch(query, children_indices, children_dists, k, cvflann::SearchParams());

            curNodes[i]->children.clear();
            for(auto index: children_indices)
                curNodes[i]->children.push_back(coarseNodes[index]);
        }

        regGraphNodes.push_back(coarseNodes);
        curNodes = coarseNodes;
        curNodeMatrix = coarseNodeMatrix;
        curNodeIndex = coarseNodeIndex;
        effResolution *= resGrowthRate;
    }

}

Point3f WarpField::applyWarp(Point3f p, WarpNode neighbours[], int n) const
{
    CV_TRACE_FUNCTION();
    
    if(n == 0) return p;

    float totalWeight = 0;
    Point3f WarpedPt(0,0,0);

    for(int i = 0; i < n; i++)
    {
        float w = neighbours[i].weight(p);
        if(w < 0.01) continue;

        Matx33f R = neighbours[i].transform.rotation();
        Vec3f T = neighbours[i].transform.translation();

        Point3f newPt = R * (p - neighbours[i].pos) + neighbours[i].pos;
        newPt.x += T[0];
        newPt.y += T[1];
        newPt.z += T[2];
        
        WarpedPt += newPt * w;
        totalWeight += w;

    }
    WarpedPt /= totalWeight;

    if(totalWeight == 0) 
        return p;
    else
        return WarpedPt;

}

void WarpField::setAllRT(Affine3f rt) {
    for(auto n: nodes)
        n->transform = rt;
}

} // namepsace dynafu
} // namespace cv
