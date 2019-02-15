// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_EGB_SEGMENT_HPP_
#define _OPENCV_EGB_SEGMENT_HPP_


#include <vector>
#include <opencv2/core.hpp>


namespace cv { namespace hfs {

struct Region
{
    int rank;
    int p;
    int mergedSize;
    int numPix;
};

class Edge
{
public:
    float w;
    int a, b;

    bool operator<(const Edge &other) const
    {
        return this->w < other.w;
    }
};

class RegionSet
{
public:
    RegionSet(int elements, std::vector<int> size)
    {
        elts = std::vector<Region>(elements);
        num = elements;
        for (int i = 0; i < elements; i++)
        {
            elts[i].rank = 0;
            elts[i].mergedSize = 1;
            elts[i].numPix = size[i];
            elts[i].p = i;
        }
    }

    ~RegionSet() {}

    int find(int x)
    {
        int y = x;
        while (y != elts[y].p)
            y = elts[y].p;
        elts[x].p = y;
        return y;
    }

    void join(int x, int y)
    {
        if (elts[x].rank > elts[y].rank)
        {
            elts[y].p = x;
            elts[x].mergedSize += elts[y].mergedSize;
            elts[x].numPix += elts[y].numPix;
        }
        else
        {
            elts[x].p = y;
            elts[y].mergedSize += elts[x].mergedSize;
            elts[y].numPix += elts[x].numPix;
            if (elts[x].rank == elts[y].rank)
                elts[y].rank++;
        }
        num--;
    }

    int mergedSize(int x) const { return elts[x].mergedSize; }
    int numPix(int x) const { return elts[x].numPix; }

    int num_sets() const { return num; }

private:
    std::vector<Region> elts;
    int num;
};

Ptr<RegionSet> egb_merge(int num_vertices, int num_edges,
    std::vector<Edge>& edges, float c, std::vector<int> size);

}}

#endif
