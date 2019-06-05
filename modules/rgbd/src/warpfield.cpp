#include "precomp.hpp"
#include "kinfu_frame.hpp" // for ptype

namespace cv {
namespace dynafu {

WarpField::WarpField(): nodes(), maxNeighbours(1000000) // good amount for dense kinfu pointclouds
{
}

std::vector<Ptr<WarpNode> > WarpField::getNodes()
{
    return nodes;
}

void WarpField::updateNodesFromPoints(InputArray _points, float resolution)
{
    // Build an index of points
    Mat m = _points.getMat();
    Mat points_matrix(m.size().height, 3, CV_32F);
    points_matrix = m.colRange(0, 3);

    cvflann::KDTreeSingleIndexParams params;
    flann::GenericIndex<flann::L2_Simple <float> > searchIndex(points_matrix, params);

    std::vector<bool> validIndex;
    removeSupported(searchIndex, validIndex);

    subsampleIndex(points_matrix, searchIndex, validIndex, resolution);
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

void WarpField::subsampleIndex(Mat& pmat, flann::GenericIndex<flann::L2_Simple<float> >& ind, std::vector<bool>& validIndex, float res)
{
    for(size_t i = 0; i < validIndex.size(); i++)
    {
        if(!validIndex[i])
            continue;

        std::vector<int> indices_vec(maxNeighbours);
        std::vector<float> dist_vec(maxNeighbours);
        
        int neighbours = ind.radiusSearch(pmat.row(i), indices_vec, dist_vec, res, cvflann::SearchParams());

        appendNodeFromCluster(res, Point3f(pmat.at<float>(i, 0), pmat.at<float>(i, 1), pmat.at<float>(i, 2)));
        for(int j = 0; j < neighbours; j++)
            validIndex[indices_vec[j]] = false;
    }      
}

void WarpField::appendNodeFromCluster(float res, Point3f p)
{
    Ptr<WarpNode> wn = new WarpNode;

    wn->pos = p;
    wn->radius = res;
    nodes.push_back(wn);
}

} // namepsace dynafu
} // namespace cv
