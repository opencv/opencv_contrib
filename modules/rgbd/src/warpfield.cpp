#include "precomp.hpp"
#include "kinfu_frame.hpp" // for ptype

namespace cv {
namespace dynafu {

WarpField::WarpField(int _maxNeighbours, int K, int levels, float baseResolution, float resolutionGrowth): 
nodes(), maxNeighbours(_maxNeighbours), // good amount for dense kinfu pointclouds
k(K), n_levels(levels), 
baseRes(baseResolution), resGrowthRate(resolutionGrowth),
regGraphNodes(std::vector<NodesLevelType>(n_levels))
{
}

std::vector<Ptr<WarpNode> > WarpField::getNodes()
{
    return nodes;
}

std::vector<NodesLevelType> WarpField::getGraphNodes()
{
    return regGraphNodes;
}

void WarpField::updateNodesFromPoints(InputArray _points)
{
    // Build an index of points
    Mat m = _points.getMat();
    Mat points_matrix(m.size().height, 3, CV_32F);
    points_matrix = m.colRange(0, 3);

    cvflann::KDTreeSingleIndexParams params;
    flann::GenericIndex<flann::L2_Simple <float> > searchIndex(points_matrix, params);

    std::vector<bool> validIndex;
    removeSupported(searchIndex, validIndex);

    Mat nodePosMatrix(nodes.size(), 3, CV_32F);
    
    for(auto n: nodes)
    {
        Mat row = (Mat_<float>(1, 3) << n->pos.x, n->pos.y, n->pos.z);
        nodePosMatrix.push_back(row);
    }
    std::vector<Ptr<WarpNode> > newNodes;
    if((int)nodes.size() > k) {
        Ptr<flann::GenericIndex<flann::L2_Simple<float> > > nodeIndexPtr = new flann::GenericIndex<flann::L2_Simple<float> >(nodePosMatrix, params);
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes, nodeIndexPtr);
    } else
        newNodes = subsampleIndex(points_matrix, searchIndex, validIndex, baseRes);

    nodes.insert(nodes.end(), newNodes.begin(), newNodes.end());
    
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

std::vector<Ptr<WarpNode> > WarpField::subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind,
    std::vector<bool>& validIndex, float res, Ptr<flann::GenericIndex<flann::L2_Simple<float> > > knnIndex)
{
    std::vector<Ptr<WarpNode> > temp_nodes;

    for(size_t i = 0; i < validIndex.size(); i++)
    {
        if(!validIndex[i])
            continue;

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dist_vec(maxNeighbours);
        
        ind.radiusSearch(pmat.row(i), indices_vec, dist_vec, res, cvflann::SearchParams());
        
        for(auto index: indices_vec)
            validIndex[index] = false;

        Ptr<WarpNode> wn = new WarpNode;
        wn->pos = Point3f(pmat.at<float>(i, 0), pmat.at<float>(i, 1), pmat.at<float>(i, 2));

        std::vector<int> indices_vec2(k+1, 0);
        std::vector<float> dist_vec2(k+1, 0);

        std::vector<float> query = {wn->pos.x, wn->pos.y, wn->pos.z};

        if(knnIndex != nullptr)
        {
            knnIndex->knnSearch(query, indices_vec2, dist_vec2, (k+1), cvflann::SearchParams());
            wn->radius = dist_vec2.back();
        } else wn->radius = res;

        temp_nodes.push_back(wn);
    }     
    
    return temp_nodes;
}

void WarpField::constructRegGraph()
{
    regGraphNodes.clear();

    if(n_levels == 1) // First level (Nwarp) is already stored in nodes
        return;

    float effResolution = baseRes*resGrowthRate;
    std::vector<Ptr<WarpNode> > curNodes = nodes;
    Mat curNodeMatrix(curNodes.size(), 3, CV_32F);
    for(Ptr<WarpNode> n: curNodes)
    {
        Mat row = (Mat_<float>(1, 3) << n->pos.x, n->pos.y, n->pos.z);
        curNodeMatrix.push_back(row);
    }

    Ptr<flann::GenericIndex<flann::L2_Simple<float> > > curNodeIndex(
        new flann::GenericIndex<flann::L2_Simple<float> >(curNodeMatrix, cvflann::KDTreeSingleIndexParams()));
    
    for(int l = 0; l < (n_levels-1); l++)
    {
        std::vector<bool> nodeValidity(curNodeIndex->size(), true);
        std::vector<Ptr<WarpNode> > coarseNodes = subsampleIndex(curNodeMatrix, *curNodeIndex, nodeValidity, effResolution);

        Mat coarseNodeMatrix(coarseNodes.size(), 3, CV_32F);
        for(size_t i = 0; i < coarseNodes.size(); i++)
        {
            coarseNodeMatrix.at<float>(i, 0) = coarseNodes[i]->pos.x;
            coarseNodeMatrix.at<float>(i, 1) = coarseNodes[i]->pos.y;
            coarseNodeMatrix.at<float>(i, 2) = coarseNodes[i]->pos.z;
        }

        Ptr<flann::GenericIndex<flann::L2_Simple<float> > > coarseNodeIndex(
            new flann::GenericIndex<flann::L2_Simple<float> >(coarseNodeMatrix, cvflann::KDTreeSingleIndexParams()));

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

} // namepsace dynafu
} // namespace cv
