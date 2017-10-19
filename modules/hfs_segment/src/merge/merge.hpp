/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                              License Agreement
//                    For Open Source Computer Vision Library
//                           (3 - clause BSD License)
//
// Copyright(C) 2000 - 2016, Intel Corporation, all rights reserved.
// Copyright(C) 2009 - 2011, Willow Garage Inc., all rights reserved.
// Copyright(C) 2009 - 2016, NVIDIA Corporation, all rights reserved.
// Copyright(C) 2010 - 2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright(C) 2015 - 2016, OpenCV Foundation, all rights reserved.
// Copyright(C) 2015 - 2016, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
//      * Redistributions of source code must retain the above copyright notice,
//        this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright notice,
//        this list of conditions and the following disclaimer in the documentation
//        and / or other materials provided with the distribution.
//
//      * Neither the names of the copyright holders nor the names of the contributors
//        may be used to endorse or promote products derived from this software
//        without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/



#ifndef _OPENCV_EGB_SEGMENT_HPP_
#define _OPENCV_EGB_SEGMENT_HPP_
#ifdef __cplusplus


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

    bool operator<(const Edge &other)
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
#endif
