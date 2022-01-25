/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __PHOTOMONTAGE_HPP__
#define __PHOTOMONTAGE_HPP__

#include <vector>
#include <stack>
#include <limits>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <time.h>
#include <functional>

#include "norm2.hpp"
#include "blending.hpp"

namespace gcoptimization
{

#include "gcgraph.hpp"


typedef float TWeight;
typedef  int  labelTp;


#define GCInfinity 10*1000*1000
#define eps 0.02


template <typename Tp> static int min_idx(std::vector <Tp> vec)
{
    return int( std::min_element(vec.begin(), vec.end()) - vec.begin() );
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename Tp> class Photomontage
{
private:
    const std::vector <std::vector <Tp> > &pointSeq;   // points for stitching
    const std::vector <std::vector <uchar> > &maskSeq; // corresponding masks

    const std::vector <std::vector <int> > &linkIdx;   // vector of neighbors for pointSeq

    std::vector <std::vector <labelTp> > labelings;    // vector of labelings
    std::vector <TWeight>  distances;                  // vector of max-flow costs for different labeling

    std::vector <labelTp> &labelSeq;                   // current best labeling

    TWeight singleExpansion(const int alpha);          // single neighbor computing

    class ParallelExpansion : public cv::ParallelLoopBody
    {
    public:
        Photomontage <Tp> *main;

        ParallelExpansion(Photomontage <Tp> *_main) : main(_main){}
        ~ParallelExpansion(){};

        void operator () (const cv::Range &range) const CV_OVERRIDE
        {
            for (int i = range.start; i <= range.end - 1; ++i)
                main->distances[i] = main->singleExpansion(i);
        }
    } parallelExpansion;

    void operator =(const Photomontage <Tp>&) const {};

protected:
    virtual TWeight dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2);
    virtual void setWeights(GCGraph <TWeight> &graph,
        const int idx1, const int idx2, const int l1, const int l2, const int lx);

public:
    void gradientDescent(); // gradient descent in alpha-expansion topology

    Photomontage(const std::vector <std::vector <Tp> > &pointSeq,
                 const std::vector <std::vector <uchar> > &maskSeq,
                 const std::vector <std::vector <int> > &linkIdx,
                       std::vector <labelTp> &labelSeq);
    virtual ~Photomontage(){};
};

template <typename Tp> inline TWeight Photomontage <Tp>::
dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2)
{
    return norm2(l1p1, l2p1) + norm2(l1p2, l2p2);
}

template <typename Tp> void Photomontage <Tp>::
setWeights(GCGraph <TWeight> &graph, const int idx1, const int idx2,
    const int l1, const int l2, const int lx)
{
    if ((size_t)idx1 >= pointSeq.size() || (size_t)idx2 >= pointSeq.size()
        || (size_t)l1 >= pointSeq[idx1].size() || (size_t)l1 >= pointSeq[idx2].size()
        || (size_t)l2 >= pointSeq[idx1].size() || (size_t)l2 >= pointSeq[idx2].size()
        || (size_t)lx >= pointSeq[idx1].size() || (size_t)lx >= pointSeq[idx2].size())
        return;

    if (l1 == l2)
    {
        /** Link from A to B **/
        TWeight weightAB = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][lx], pointSeq[idx2][lx] );
        graph.addEdges( idx1, idx2, weightAB, weightAB );
    }
    else
    {
        int X = graph.addVtx();

        /** Link from X to sink **/
        TWeight weightXS = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][l2], pointSeq[idx2][l2] );
        graph.addTermWeights( X, 0, weightXS );

        /** Link from A to X **/
        TWeight weightAX = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][lx], pointSeq[idx2][lx] );
        graph.addEdges( idx1, X, weightAX, weightAX );

        /** Link from X to B **/
        TWeight weightXB = dist( pointSeq[idx1][lx], pointSeq[idx1][lx],
                                 pointSeq[idx1][l2], pointSeq[idx1][l2] );
        graph.addEdges( X, idx2, weightXB, weightXB );
    }
}

template <typename Tp> TWeight Photomontage <Tp>::
singleExpansion(const int alpha)
{
    GCGraph <TWeight> graph( 3*int(pointSeq.size()), 4*int(pointSeq.size()) );

    /** Terminal links **/
    for (size_t i = 0; i < maskSeq.size(); ++i)
        graph.addTermWeights( graph.addVtx(),
            maskSeq[i][alpha] ? TWeight(0) : TWeight(GCInfinity), 0 );

    /** Neighbor links **/
    for (size_t i = 0; i < pointSeq.size(); ++i)
        for (size_t j = 0; j < linkIdx[i].size(); ++j)
            if ( linkIdx[i][j] != -1)
                setWeights( graph, int(i), linkIdx[i][j],
                    labelSeq[i], labelSeq[linkIdx[i][j]], alpha );

    /** Max-flow computation **/
    TWeight result = graph.maxFlow();

    /** Writing results **/
    for (size_t i = 0; i < pointSeq.size(); ++i)
        labelings[i][alpha] = graph.inSourceSegment(int(i)) ? labelSeq[i] : alpha;

    return result;
}

template <typename Tp> void Photomontage <Tp>::
gradientDescent()
{
    TWeight optValue = std::numeric_limits<TWeight>::max();

    for (int num = -1; /**/; num = -1)
    {
        int range = int( pointSeq[0].size() );
        parallel_for_( cv::Range(0, range), parallelExpansion );

        int minIndex = min_idx(distances);
        TWeight minValue = distances[minIndex];

        if (minValue < (1.00 - eps)*optValue)
            optValue = distances[num = minIndex];

        if (num == -1)
            break;

        for (size_t i = 0; i < labelSeq.size(); ++i)
            labelSeq[i] = labelings[i][num];
    }
}

template <typename Tp> Photomontage <Tp>::
Photomontage( const std::vector <std::vector <Tp> > &_pointSeq,
            const std::vector <std::vector <uchar> > &_maskSeq,
              const std::vector <std::vector <int> > &_linkIdx,
                              std::vector <labelTp> &_labelSeq )
  :
    pointSeq(_pointSeq), maskSeq(_maskSeq), linkIdx(_linkIdx),
    distances(pointSeq[0].size()), labelSeq(_labelSeq), parallelExpansion(this)
{
    size_t lsize = pointSeq[0].size();
    labelings.assign( pointSeq.size(),
      std::vector <labelTp>( lsize ) );
}

}

template <typename Tp> static inline
void photomontage( const std::vector <std::vector <Tp> > &pointSeq,
                 const std::vector <std::vector <uchar> > &maskSeq,
                   const std::vector <std::vector <int> > &linkIdx,
                   std::vector <gcoptimization::labelTp> &labelSeq )
{
    gcoptimization::Photomontage <Tp>(pointSeq, maskSeq,
        linkIdx, labelSeq).gradientDescent();
}

#endif /* __PHOTOMONTAGE_HPP__ */
