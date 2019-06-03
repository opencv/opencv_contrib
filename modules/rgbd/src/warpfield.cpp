#include "precomp.hpp"
#include "kinfu_frame.hpp" // for ptype

namespace cv {
namespace dynafu {

WarpField::WarpField(): nodes() {

}

void WarpField::updateNodesFromPoints(InputArray _points, float resolution) {
    // Build an index of points
    Mat m = _points.getMat();

    std::vector<float> points_vec; 
    int w = m.size().width;
    for(int i = 0; i < m.size().height; i++) {
        kinfu::ptype p = m.at<kinfu::ptype>(i*w);
        points_vec.push_back(p[0]);
        points_vec.push_back(p[1]);
        points_vec.push_back(p[2]);
    } 

    ::flann::Matrix<float> points_matrix(&points_vec[0], m.size().height, 3);

    ::flann::KDTreeIndex<::flann::L2_Simple <float> > searchIndex(points_matrix);
    searchIndex.buildIndex();

    std::vector<bool> validIndex;

    removeSupported(searchIndex, validIndex);

    subsampleIndex(searchIndex, validIndex, resolution);

}


void WarpField::removeSupported(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& validInd) {
    
    std::vector<bool> validIndex(ind.size(), true);

    for(WarpNode* n: nodes) {
        float point_array[] = {n->pos.x, n->pos.y, n->pos.y};
        ::flann::Matrix<float> query(&point_array[0], 1, 3);

        std::vector< std::vector<int> > indices_vec;
        std::vector<std::vector<float> > dists_vec;

        ind.radiusSearch(query, indices_vec, dists_vec, n->radius, ::flann::SearchParams());
        
        for(auto vec: indices_vec) {
            for(auto i: vec) {
                ind.removePoint(i);
                validIndex[i] = false;
            }
        }

    }

    validInd = validIndex;
}

void WarpField::subsampleIndex(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<bool>& validIndex, const float res) {
    for(size_t i = 0; i < ind.size(); i++) {
        if(!validIndex[i])
            continue;

        float* pt = ind.getPoint(i);
        ::flann::Matrix<float> query(pt, 1, 3);

        std::vector<std::vector<int> > indices_vec;
        std::vector<std::vector<float> > dist_vec;

        ind.radiusSearch(query, indices_vec, dist_vec, res, ::flann::SearchParams());

        appendNodeFromCluster(ind, indices_vec[0], res);
        
    }
}

void WarpField::appendNodeFromCluster(::flann::KDTreeIndex<::flann::L2_Simple<float> >& ind, std::vector<int> indices, float res) {
    Ptr<WarpNode> wn = new WarpNode;

    float avg_x=0, avg_y=0, avg_z=0;
    for(int index: indices) {
        float* pt = ind.getPoint(index);
        avg_x += pt[0];
        avg_y += pt[1];
        avg_z += pt[2];
    }

    avg_x /= (float)ind.size();
    avg_y /= (float)ind.size();
    avg_z /= (float)ind.size();

    wn->pos = Point3f(avg_x, avg_y, avg_z);
    wn->radius = res;
    nodes.push_back(wn);
}

} // namepsace dynafu
} // namespace cv