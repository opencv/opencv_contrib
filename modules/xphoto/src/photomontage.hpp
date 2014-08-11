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


#include "norm2.hpp"
#include "gcgraph.hpp"

#define GCInfinity 10*1000*1000*1000.0

template <typename Tp> static int min_idx(std::vector <Tp> vec)
{
    return int( std::min_element(vec.begin(), vec.end()) - vec.begin() );
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename Tp> class Photomontage
{
private:
    const std::vector <cv::Mat> &images; // vector of images for different labels
    const std::vector <cv::Mat>  &masks; // vector of definition domains for each image

    std::vector <cv::Mat> labelings; // vector of labelings for different expansions
    std::vector <double>  distances; // vector of max-flow costs for different labeling

    const int height;
    const int width;
    const int type;
    const int channels;
    const int lsize;

    cv::Mat x_i; // current best labeling

    double singleExpansion(const int alpha); // single neighbor computing
    void gradientDescent(); // gradient descent in alpha-expansion topology

    class ParallelExpansion : public cv::ParallelLoopBody
    {
    public:
        Photomontage <Tp> *main;

        ParallelExpansion(Photomontage <Tp> *_main) : main(_main){}
        ~ParallelExpansion(){};

        void operator () (const cv::Range &range) const
        {
            for (int i = range.start; i <= range.end - 1; ++i)
                main->distances[i] = main->singleExpansion(i);
        }
    };

    void operator =(const Photomontage <Tp>&) const {};

protected:
    virtual double dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2);
    virtual void setWeights(GCGraph <double> &graph, const cv::Point &pA, const cv::Point &pB, const int lA, const int lB, const int lX);

public:
    Photomontage(const std::vector <cv::Mat> &images, const std::vector <cv::Mat> &masks);
    virtual ~Photomontage(){};

    void assignLabeling(cv::Mat &img);
    void assignResImage(cv::Mat &img);
};

template <typename Tp> inline double Photomontage <Tp>::
dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2)
{
    return norm2(l1p1, l2p1) + norm2(l1p2, l2p2);
}

template <typename Tp> void Photomontage <Tp>::
setWeights(GCGraph <double> &graph, const cv::Point &pA, const cv::Point &pB, const int lA, const int lB, const int lX)
{
    if (lA == lB)
    {
        /** Link from A to B **/
        double weightAB = dist( images[lA].template at<Tp>(pA),
                                images[lA].template at<Tp>(pB),
                                images[lX].template at<Tp>(pA),
                                images[lX].template at<Tp>(pB) );
        graph.addEdges( int(pA.y*width + pA.x), int(pB.y*width + pB.x), weightAB, weightAB);
    }
    else
    {
        int X = graph.addVtx();

        /** Link from X to sink **/
        double weightXS = dist( images[lA].template at<Tp>(pA),
                                images[lA].template at<Tp>(pB),
                                images[lB].template at<Tp>(pA),
                                images[lB].template at<Tp>(pB) );
        graph.addTermWeights(X, 0, weightXS);

        /** Link from A to X **/
        double weightAX = dist( images[lA].template at<Tp>(pA),
                                images[lA].template at<Tp>(pB),
                                images[lX].template at<Tp>(pA),
                                images[lX].template at<Tp>(pB) );
        graph.addEdges( int(pA.y*width + pA.x), X, weightAX, weightAX);

        /** Link from X to B **/
        double weightXB = dist( images[lX].template at<Tp>(pA),
                                images[lX].template at<Tp>(pB),
                                images[lB].template at<Tp>(pA),
                                images[lB].template at<Tp>(pB) );
        graph.addEdges(X, int(pB.y*width + pB.x), weightXB, weightXB);
    }
}

template <typename Tp> double Photomontage <Tp>::
singleExpansion(const int alpha)
{
    int actualEdges = (height - 1)*width + height*(width - 1);
    GCGraph <double> graph(actualEdges + height*width, 2*actualEdges);

    /** Terminal links **/
    for (int i = 0; i < height; ++i)
    {
        const uchar *maskAlphaRow = masks[alpha].template ptr <uchar>(i);
        const int *labelRow = (const int *) x_i.template ptr <int>(i);

        for (int j = 0; j < width; ++j)
            graph.addTermWeights( graph.addVtx(),
                                  maskAlphaRow[j] ? 0 : GCInfinity,
             masks[ labelRow[j] ].template at<uchar>(i, j) ? 0 : GCInfinity );
    }

    /** Neighbor links **/
    for (int i = 0; i < height - 1; ++i)
    {
        const int *currentRow = (const int *) x_i.template ptr <int>(i);
        const int *nextRow = (const int *) x_i.template ptr <int>(i + 1);

        for (int j = 0; j < width - 1; ++j)
        {
            setWeights( graph, cv::Point(i, j), cv::Point(i, j + 1), currentRow[j], currentRow[j + 1], alpha );
            setWeights( graph, cv::Point(i, j), cv::Point(i + 1, j), currentRow[j],     nextRow[j],    alpha );
        }

        setWeights( graph, cv::Point(i, width - 1), cv::Point(i + 1, width - 1),
                    currentRow[width - 1], nextRow[width - 1], alpha );
    }

    const int *currentRow = (const int *) x_i.template ptr <int>(height - 1);
    for (int i = 0; i < width - 1; ++i)
        setWeights( graph, cv::Point(height - 1, i), cv::Point(height - 1, i + 1),
                    currentRow[i], currentRow[i + 1], alpha );

    /** Max-flow computation **/
    double result = graph.maxFlow();

    /** Writing results **/
    labelings[alpha].create( height, width, CV_32SC1 );
    for (int i = 0; i < height; ++i)
    {
        const int *inRow = (const int *) x_i.template ptr <int>(i);
        int *outRow = (int *) labelings[alpha].template ptr <int>(i);

        for (int j = 0; j < width; ++j)
            outRow[j] = graph.inSourceSegment(i*width + j) ? inRow[j] : alpha;
    }

    return result;
}

template <typename Tp> void Photomontage <Tp>::
gradientDescent()
{
    double optValue = std::numeric_limits<double>::max();

    for (int num = -1; /**/; num = -1)
    {
        parallel_for_( cv::Range(0, lsize),
            ParallelExpansion(this) );

        int minIndex = min_idx(distances);
        double minValue = distances[minIndex];

        if (minValue < 0.98*optValue)
            optValue = distances[num = minIndex];

        if (num == -1)
            break;
        labelings[num].copyTo(x_i);
    }
}

template <typename Tp> void Photomontage <Tp>::
assignLabeling(cv::Mat &img)
{
    x_i.setTo(0);
    gradientDescent();
    x_i.copyTo(img);
}

template <typename Tp> void Photomontage <Tp>::
assignResImage(cv::Mat &img)
{
    cv::Mat optimalLabeling;
    assignLabeling(optimalLabeling);

    img.create( height, width, type );

    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
        {
            cv::Mat M = images[optimalLabeling.template at<int>(i, j)];
            img.template at<Tp>(i, j) = M.template at<Tp>(i, j);
        }
}

template <typename Tp> Photomontage <Tp>::
Photomontage(const std::vector <cv::Mat> &_images, const std::vector <cv::Mat> &_masks)
  :
    images(_images), masks(_masks), labelings(images.size()), distances(images.size()),
    height(int(images[0].rows)), width(int(images[0].cols)), type(images[0].type()),
    channels(images[0].channels()), lsize(int(images.size())), x_i(height, width, CV_32SC1){}


#endif /* __PHOTOMONTAGE_HPP__ */
