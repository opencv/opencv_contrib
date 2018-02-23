// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "merge.hpp"

using namespace std;

namespace cv { namespace hfs {

Ptr<RegionSet> egb_merge(int num_vertices, int num_edges,
    vector<Edge> &edges, float c, vector<int> size)
{
    sort(edges.begin(), edges.end());

    Ptr<RegionSet> regions(new RegionSet(num_vertices, size));

    vector<float> threshold(num_vertices);
    for (int i = 0; i < num_vertices; i++)
        threshold[i] = c;


    for (int i = 0; i < num_edges; i++) {
        Edge *pedge = &edges[i];

        int a = regions->find(pedge->a);
        int b = regions->find(pedge->b);
        if (a != b) {
            if ((pedge->w <= threshold[a]) &&
                (pedge->w <= threshold[b])) {
                regions->join(a, b);
                a = regions->find(a);
                threshold[a] = pedge->w + c / regions->mergedSize(a);
            }
        }
    }
    return regions;
}

}}
