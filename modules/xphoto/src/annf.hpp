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

#ifndef __ANNF_HPP__
#define __ANNF_HPP__

#include "norm2.hpp"
#include "whs.hpp"

/************************* KDTree class *************************/

template <typename ForwardIterator> void
generate_seq(ForwardIterator it, int first, int last)
{
    for (int i = first; i < last; ++i, ++it)
        *it = i;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

template <typename Tp, int cn> class KDTree
{
private:
    class KDTreeComparator
    {
        const KDTree <Tp, cn> *main; // main class
        int dimIdx; // dimension to compare

    public:
        bool operator () (const int &x, const int &y) const
        {
            cv::Vec <Tp, cn> u = main->data[main->idx[x]];
            cv::Vec <Tp, cn> v = main->data[main->idx[y]];

            return  u[dimIdx] < v[dimIdx];
        }

        KDTreeComparator(const KDTree <Tp, cn> *_main, int _dimIdx)
            : main(_main), dimIdx(_dimIdx) {}
    };

    const int leafNumber;

    int getMaxSpreadN(const int left, const int right) const;
    void operator =(const KDTree <Tp, cn> &) const {};

public:
    std::vector <cv::Vec <Tp, cn> > data;
    std::vector <int> idx;
    std::vector <cv::Point2i> nodes;

    KDTree(const cv::Mat &data, const int leafNumber = 8);
    ~KDTree(){};
};

template <typename Tp, int cn> int KDTree <Tp, cn>::
getMaxSpreadN(const int _left, const int _right) const
{
    cv::Vec<Tp, cn> maxValue = data[ idx[_left] ],
                    minValue = data[ idx[_left] ];
    for (int i = _left + 1; i < _right; i += cn)
        for (int j = 0; j < cn; ++j)
        {
            minValue[j] = std::min( minValue[j], data[idx[i]][j] );
            maxValue[j] = std::max( maxValue[j], data[idx[i]][j] );
        }
    cv::Vec<Tp, cn> spread = maxValue - minValue;

    Tp *begIt = &spread[0];
    return int(std::max_element(begIt, begIt + cn) - begIt);
}

template <typename Tp, int cn> KDTree <Tp, cn>::
KDTree(const cv::Mat &img, const int _leafNumber)
    : leafNumber(_leafNumber)
///////////////////////////////////////////////////
{
    for (int i = 0; i < img.rows; ++i)
        for (int j = 0; j < img.cols; ++j)
            data.push_back(img.template at<cv::Vec <Tp, cn> >(i, j));

    generate_seq( std::back_inserter(idx), 0, int(data.size()) );
    fill_n( std::back_inserter(nodes), int(data.size()), cv::Point2i(0, 0) );

    std::stack <int> left, right;
    left.push( 0 );
    right.push( int(idx.size()) );

    while ( !left.empty() )
    {
        int _left = left.top(); left.pop();
        int _right = right.top(); right.pop();

        if ( _right - _left <= leafNumber)
        {
            for (int i = _left; i < _right; ++i)
            {
                nodes[idx[i]].x = _left;
                nodes[idx[i]].y = _right;
            }

            continue;
        }

        std::vector <int>::iterator begIt = idx.begin();
        int nth = _left + (_right - _left)/2;
        std::nth_element(/**/ begIt + _left,
            begIt + nth, begIt + _right,
            KDTreeComparator( this,
                getMaxSpreadN(_left, _right) ) /**/);

        left.push(_left); right.push(nth + 1);
        left.push(nth + 1); right.push(_right);
    }
}

/************************** ANNF search **************************/

template <typename Tp, int cn>
static void updateDist(const KDTree <Tp, cn> &kdTree, const cv::Point2i &I, const int height,
                       const int width, const int &currentIdx, int &bestIdx, double &dist)
{
    for (int k = I.x; k < I.y; ++k)
    {
        int newIdx = kdTree.idx[k];

        if (newIdx%width == width  - 1)
            continue;

        if (newIdx/width == height - 1)
            continue;

        int dx = currentIdx%width - newIdx%width;
        int dy = currentIdx/width - newIdx/width;

        if (abs(dx) + abs(dy) < 32)
            continue;

        double ndist = norm2(kdTree.data[newIdx],
                        kdTree.data[currentIdx]);
        if (ndist < dist)
        {
            dist = ndist;
            bestIdx = newIdx;
        }
    }
}

static void getANNF(const cv::Mat &img, std::vector <cv::Matx33f> &transforms,
                    const int nTransform, const int psize)
{
    /** Walsh-Hadamard Transformation **/

    std::vector <cv::Mat> channels;
    cv::split(img, channels);

    const int np[] = {16, 4, 4};
    for (int i = 0; i < img.channels(); ++i)
        getWHSeries(channels[i], channels[i], np[i], psize);

    cv::Mat whs; // Walsh-Hadamard series
    cv::merge(channels, whs);

    KDTree <float, 24> kdTree(whs);
    std::vector <int> annf( whs.total(), 0 );

    /** Propagation-assisted kd-tree search **/

    for (int i = 0; i < whs.rows; ++i)
        for (int j = 0; j < whs.cols; ++j)
        {
            double dist = std::numeric_limits <double>::max();
            int current = i*whs.cols + j;

            cv::Point2i I = kdTree.nodes[i*whs.cols + j];
            updateDist(kdTree, I, whs.rows, whs.cols, current, annf[i*whs.cols + j], dist);

            if (i != 0)
            {
                int idx = annf[(i - 1)*whs.cols + j] + whs.cols;
                cv::Point2i I = kdTree.nodes[idx];
                updateDist(kdTree, I, whs.rows, whs.cols, current, annf[i*whs.cols + j], dist);
            }

            if (j != 0)
            {
                int idx = annf[i*whs.cols + (j - 1)] + 1;
                cv::Point2i I = kdTree.nodes[idx];
                updateDist(kdTree, I, whs.rows, whs.cols, current, annf[i*whs.cols + j], dist);
            }
        }

    /** Local maxima extraction **/

    cv::Mat_<double> annfHist(2*whs.rows, 2*whs.cols, 0.0),
                    _annfHist(2*whs.rows, 2*whs.cols, 0.0);
    for (size_t i = 0; i < annf.size(); ++i)
        ++annfHist( (annf[i] - int(i))/whs.cols + whs.rows,
                    (annf[i] - int(i))%whs.cols + whs.cols);

    cv::GaussianBlur( annfHist, annfHist,
        cv::Size(9, 9), 1.41, 0.0, cv::BORDER_CONSTANT);
    cv::dilate(annfHist, _annfHist,
        cv::Matx<uchar, 9, 9>::ones());

    std::vector < std::pair<double, int> > amount;
    std::vector <cv::Point2i> shiftM;

    for (int i = 0, t = 0; i < annfHist.rows; ++i)
    {
        double  *pAnnfHist =  annfHist.template ptr<double>(i);
        double *_pAnnfHist = _annfHist.template ptr<double>(i);

        for (int j = 0; j < annfHist.cols; ++j)
            if ( pAnnfHist[j] != 0 && pAnnfHist[j] == _pAnnfHist[j] )
            {
                amount.push_back( std::make_pair(pAnnfHist[j], t++) );
                shiftM.push_back(cv::Point2i(j - whs.cols,
                                             i - whs.rows));
            }
    }

    std::partial_sort( amount.begin(), amount.begin() + nTransform,
        amount.end(), std::greater< std::pair<double, int> >() );

    transforms.resize(nTransform);
    for (int i = 0; i < nTransform; ++i)
    {
        int idx = amount[i].second;
        transforms[i] = cv::Matx33f(1, 0, float(shiftM[idx].x),
                                    0, 1, float(shiftM[idx].y),
                                    0, 0,          1          );
    }
}

#endif /* __ANNF_HPP__ */
