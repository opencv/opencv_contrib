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
 // Copyright (C) 2015, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
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


#include "precomp.hpp"

namespace cv {
namespace phase_unwrapping {
class CV_EXPORTS_W HistogramPhaseUnwrapping_Impl : public HistogramPhaseUnwrapping
{
public:
    // Constructor
    explicit HistogramPhaseUnwrapping_Impl( const HistogramPhaseUnwrapping::Params &parameters =
                                            HistogramPhaseUnwrapping::Params() );
    // Destructor
    virtual ~HistogramPhaseUnwrapping_Impl() CV_OVERRIDE {};

    // Unwrap phase map
    void unwrapPhaseMap( InputArray wrappedPhaseMap, OutputArray unwrappedPhaseMap,
                         InputArray shadowMask = noArray() ) CV_OVERRIDE;
    // Get reliability map computed from the wrapped phase map
    void getInverseReliabilityMap( OutputArray reliabilityMap ) CV_OVERRIDE;

private:
    // Class describing a pixel
    class Pixel
    {
    private:
        // Value from the wrapped phase map
        float phaseValue;
        // Id of a pixel. Computed from its position in the Mat
        int idx;
        // Pixel is valid if it's not in a shadow region
        bool valid;
        // "Quality" parameter. See reference paper
        float inverseReliability;
        // Number of 2pi  that needs to be added to the pixel to unwrap the phase map
        int increment;
        // Number of pixels that are in the same group as the current pixel
        int nbrOfPixelsInGroup;
        // Group id. At first, group id is the same value as idx
        int groupId;
        // Pixel is alone in its group
        bool singlePixelGroup;
    public:
        Pixel();
        Pixel( float pV, int id, bool v, float iR, int inc );
        float getPhaseValue();
        int getIndex();
        bool getValidity();
        float getInverseReliability();
        int getIncrement();
        int getNbrOfPixelsInGroup();
        int getGroupId();
        bool getSinglePixelGroup();
        void setIncrement( int inc );
        // When a pixel which is not in a single group is added to a new group, we need to keep the previous increment and add "inc" to it.
        void changeIncrement( int inc );
        void setNbrOfPixelsInGroup( int nbr );
        void setGroupId( int gId );
        void setSinglePixelGroup( bool s );
    };
    // Class describing an Edge as presented in the reference paper
    class Edge
    {
    private:
        // Id of the first pixel that forms the edge
        int pixOneId;
        // Id of the second pixel that forms the edge
        int pixTwoId;
        // Number of 2pi that needs to be added to the second pixel to remove discontinuities
        int increment;
    public:
        Edge();
        Edge( int p1, int p2, int inc );
        int getPixOneId();
        int getPixTwoId();
        int getIncrement();
    };
    // Class describing a bin from the histogram
    class HistogramBin
    {
    private:
        float start;
        float end;
        std::vector<Edge> edges;
    public:
        HistogramBin();
        HistogramBin( float s, float e );
        void addEdge( Edge e );
        std::vector<Edge> getEdges();
    };
    // Class describing the histogram. Bins before "thresh" are smaller than the one after "thresh" value
    class Histogram
    {
    private:
        std::vector<HistogramBin> bins;
        float thresh;
        float smallWidth;
        float largeWidth;
        int nbrOfSmallBins;
        int nbrOfLargeBins;
        int nbrOfBins;
    public:
        Histogram();
        void createBins( float t, int nbrOfBinsBeforeThresh, int nbrOfBinsAfterThresh );
        void addBin( HistogramBin b );
        void addEdgeInBin( Edge e, int binIndex);
        float getThresh();
        float getSmallWidth();
        float getLargeWidth();
        int getNbrOfBins();
        std::vector<Edge> getEdgesFromBin( int binIndex );
    };
    // Params for phase unwrapping
    Params params;
    // Pixels from the wrapped phase map
    std::vector<Pixel> pixels;
    // Histogram used to unwrap
    Histogram histogram;
    // Compute pixel reliability.
    void computePixelsReliability( InputArray wrappedPhaseMap, InputArray shadowMask = noArray() );
    // Compute edges reliability and sort them in the histogram
    void computeEdgesReliabilityAndCreateHistogram();
    // Methods that is used in the previous one
    void createAndSortEdge( int idx1, int idx2 );
    // Unwrap the phase map thanks to the histogram
    void unwrapHistogram();
    // add right number of 2*pi to the pixels
    void addIncrement( OutputArray unwrappedPhaseMap );
    // Gamma function from the paper
    float wrap( float a, float b );
    // Similar to the previous one but returns the number of 2pi that needs to be added
    int findInc( float a, float b );
};
// Default parameters
HistogramPhaseUnwrapping::Params::Params(){
    width = 800;
    height = 600;
    histThresh = static_cast<float>(3 * CV_PI * CV_PI);
    nbrOfSmallBins = 10;
    nbrOfLargeBins = 5;
}
HistogramPhaseUnwrapping_Impl::HistogramPhaseUnwrapping_Impl(
                            const HistogramPhaseUnwrapping::Params &parameters ) : params(parameters)
{

}

HistogramPhaseUnwrapping_Impl::Pixel::Pixel()
{

}
// Constructor
HistogramPhaseUnwrapping_Impl::Pixel::Pixel( float pV, int id, bool v, float iR, int inc )
{
    phaseValue = pV;
    idx = id;
    valid = v;
    inverseReliability = iR;
    increment = inc;
    nbrOfPixelsInGroup = 1;
    groupId = id;
    singlePixelGroup = true;
}

float HistogramPhaseUnwrapping_Impl::Pixel::getPhaseValue()
{
    return phaseValue;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getIndex()
{
    return idx;
}

bool HistogramPhaseUnwrapping_Impl::Pixel::getValidity()
{
    return valid;
}

float HistogramPhaseUnwrapping_Impl::Pixel::getInverseReliability()
{
    return inverseReliability;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getIncrement()
{
    return increment;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getNbrOfPixelsInGroup()
{
    return nbrOfPixelsInGroup;
}

int HistogramPhaseUnwrapping_Impl::Pixel::getGroupId()
{
    return groupId;
}

bool HistogramPhaseUnwrapping_Impl::Pixel::getSinglePixelGroup()
{
    return singlePixelGroup;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setIncrement( int inc )
{
    increment = inc;
}
/* When a pixel of a non-single group is added to an other non-single group, we need to add a new
increment to the one that was there previously and that was already removing some wraps.
*/
void HistogramPhaseUnwrapping_Impl::Pixel::changeIncrement( int inc )
{
    increment += inc;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setNbrOfPixelsInGroup( int nbr )
{
    nbrOfPixelsInGroup = nbr;
}
void HistogramPhaseUnwrapping_Impl::Pixel::setGroupId( int gId )
{
    groupId = gId;
}

void HistogramPhaseUnwrapping_Impl::Pixel::setSinglePixelGroup( bool s )
{
    singlePixelGroup = s;
}

HistogramPhaseUnwrapping_Impl::Edge::Edge()
{

}
// Constructor
HistogramPhaseUnwrapping_Impl::Edge::Edge( int p1, int p2, int inc )
{
    pixOneId = p1;
    pixTwoId = p2;
    increment = inc;
}

int HistogramPhaseUnwrapping_Impl::Edge::getPixOneId()
{
    return pixOneId;
}

int HistogramPhaseUnwrapping_Impl::Edge::getPixTwoId()
{
    return pixTwoId;
}

int HistogramPhaseUnwrapping_Impl::Edge::getIncrement()
{
    return increment;
}

HistogramPhaseUnwrapping_Impl::HistogramBin::HistogramBin()
{

}

HistogramPhaseUnwrapping_Impl::HistogramBin::HistogramBin( float s, float e )
{
    start = s;
    end = e;
}

void HistogramPhaseUnwrapping_Impl::HistogramBin::addEdge( Edge e )
{
    edges.push_back(e);
}
std::vector<HistogramPhaseUnwrapping_Impl::Edge> HistogramPhaseUnwrapping_Impl::HistogramBin::getEdges()
{
    return edges;
}
HistogramPhaseUnwrapping_Impl::Histogram::Histogram()
{

}
/*
 * create histogram bins. Bins size is not uniform, as in the reference paper
 *
 */
void HistogramPhaseUnwrapping_Impl::Histogram::createBins( float t, int nbrOfBinsBeforeThresh,
                                                           int nbrOfBinsAfterThresh )
{
    thresh = t;

    nbrOfSmallBins = nbrOfBinsBeforeThresh;
    nbrOfLargeBins = nbrOfBinsAfterThresh;
    nbrOfBins = nbrOfBinsBeforeThresh + nbrOfBinsAfterThresh;

    smallWidth = thresh / nbrOfSmallBins;
    largeWidth = static_cast<float>(32 * CV_PI * CV_PI - thresh) / static_cast<float>(nbrOfLargeBins);

    for( int i = 0; i < nbrOfSmallBins; ++i )
    {
        addBin(HistogramBin(i * smallWidth, ( i + 1 ) * smallWidth));
    }
    for( int i = 0; i < nbrOfLargeBins; ++i )
    {
        addBin(HistogramBin(thresh + i * largeWidth, thresh + ( i + 1 ) * largeWidth));
    }
}
// Add a bin b to the histogram
void HistogramPhaseUnwrapping_Impl::Histogram::addBin( HistogramBin b )
{
    bins.push_back(b);
}
// Add edge E in bin binIndex
void HistogramPhaseUnwrapping_Impl::Histogram::addEdgeInBin( Edge e, int binIndex )
{
    bins[binIndex].addEdge(e);
}
float HistogramPhaseUnwrapping_Impl::Histogram::getThresh()
{
    return thresh;
}

float HistogramPhaseUnwrapping_Impl::Histogram::getSmallWidth()
{
    return smallWidth;
}

float HistogramPhaseUnwrapping_Impl::Histogram::getLargeWidth()
{
    return largeWidth;
}

int HistogramPhaseUnwrapping_Impl::Histogram::getNbrOfBins()
{
    return nbrOfBins;
}

std::vector<HistogramPhaseUnwrapping_Impl::Edge> HistogramPhaseUnwrapping_Impl::
                                                 Histogram::getEdgesFromBin( int binIndex )
{
    std::vector<HistogramPhaseUnwrapping_Impl::Edge> temp;
    temp = bins[binIndex].getEdges();
    return temp;
}
/* Method in which reliabilities are computed and edges are sorted in the histogram.
Increments are computed for each pixels.
 */
void HistogramPhaseUnwrapping_Impl::unwrapPhaseMap( InputArray wrappedPhaseMap,
                                                    OutputArray unwrappedPhaseMap,
                                                    InputArray shadowMask )
{
    Mat &wPhaseMap = *(Mat*) wrappedPhaseMap.getObj();
    Mat mask;
    int rows = params.height;
    int cols = params.width;
    if( shadowMask.empty() )
    {
        mask.create(rows, cols, CV_8UC1);
        mask = Scalar::all(255);
    }
    else
    {
        Mat &temp = *(Mat*) shadowMask.getObj();
        temp.copyTo(mask);
    }

    computePixelsReliability(wPhaseMap, mask);
    computeEdgesReliabilityAndCreateHistogram();

    unwrapHistogram();
    addIncrement(unwrappedPhaseMap);
}

//compute pixels reliabilities according to "A novel algorithm based on histogram processing of reliability for two-dimensional phase unwrapping"

void HistogramPhaseUnwrapping_Impl::computePixelsReliability( InputArray wrappedPhaseMap,
                                                              InputArray shadowMask )
{
    int rows = params.height;
    int cols = params.width;

    Mat &wPhaseMap = *(Mat*) wrappedPhaseMap.getObj();
    Mat &mask = *(Mat*) shadowMask.getObj();

    int idx; //idx is used to store pixel position (idx = i*cols + j)
    bool valid;//tells if a pixel is in the valid mask region

    // H, V, D1, D2 are from the paper
    float H, V, D1, D2, D;
    /* used to store neighbours coordinates
     * ul = upper left, um = upper middle, ur = upper right
     * ml = middle left, mr = middle right
     * ll = lower left, lm = lower middle, lr = lower right
     */
    Point ul, um, ur, ml, mr, ll, lm, lr;

    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            if( mask.at<uchar>( i, j ) != 0 ) //if pixel is in a valid region
            {
                if( i == 0 || i == rows - 1 || j == 0 || j == cols - 1 )
                {
                    idx = i * cols + j;
                    valid = true;
                    Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                            static_cast<float>(16 * CV_PI * CV_PI), 0);
                    pixels.push_back(p);
                }
                else
                {
                    ul = Point(j-1, i-1);
                    um = Point(j, i-1);
                    ur = Point(j+1, i-1);
                    ml = Point(j-1, i);
                    mr = Point(j+1, i);
                    ll = Point(j-1, i+1);
                    lm = Point(j, i+1);
                    lr = Point(j+1, i+1);

                    Mat neighbourhood = mask( Rect( j-1, i-1, 3, 3 ) );
                    Scalar meanValue = mean(neighbourhood);

                    /* if mean value is different from 255, it means that one of the neighbouring
                     * pixel is not valid -> pixel (i,j) is considered as being on the border.
                     */
                    if( meanValue[0] != 255 )
                    {
                        idx = i * cols + j;
                        valid = true;
                        Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                                static_cast<float>(16 * CV_PI * CV_PI), 0);
                        pixels.push_back(p);
                    }
                    else
                    {
                        H = wrap(wPhaseMap.at<float>(ml.y, ml.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(mr.y, mr.x));
                        V = wrap(wPhaseMap.at<float>(um.y, um.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(lm.y, lm.x));
                        D1 = wrap(wPhaseMap.at<float>(ul.y, ul.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(lr.y, lr.x));
                        D2 = wrap(wPhaseMap.at<float>(ur.y, ur.x), wPhaseMap.at<float>(i, j))
                            - wrap(wPhaseMap.at<float>(i, j), wPhaseMap.at<float>(ll.y, ll.x));
                        D = H * H + V * V + D1 * D1 + D2 * D2;

                        idx = i * cols + j;
                        valid = true;
                        Pixel p(wPhaseMap.at<float>(i, j), idx, valid, D, 0);
                        pixels.push_back(p);
                    }
                }
            }
            else // pixel is not in a valid region. It's inverse reliability is set to the maximum
            {
                idx = i * cols + j;
                valid = false;
                Pixel p(wPhaseMap.at<float>(i, j), idx, valid,
                        static_cast<float>(16 * CV_PI * CV_PI), 0);
                pixels.push_back(p);
            }
        }
    }
}
/* Edges are created from the vector of pixels. We loop on the vector and create the edges
 * that link the current pixel to his right neighbour (first edge) and the one that is under it (second edge)
 */
void HistogramPhaseUnwrapping_Impl::computeEdgesReliabilityAndCreateHistogram()
{
    int row;
    int col;
    histogram.createBins(params.histThresh, params.nbrOfSmallBins, params.nbrOfLargeBins);
    int nbrOfPixels = static_cast<int>(pixels.size());
    /* Edges are built by considering a pixel and it's right-neighbour and lower-neighbour.
     We discard non-valid pixels here.
     */
    for( int i = 0; i < nbrOfPixels; ++i )
    {
        if( pixels[i].getValidity() )
        {
            row = pixels[i].getIndex() / params.width;
            col = pixels[i].getIndex() % params.width;

            if( row != params.height - 1 && col != params.width -1 )
            {
                int idxRight, idxDown;
                idxRight = row * params.width + col + 1; // Pixel to the right
                idxDown = ( row + 1 ) * params.width + col; // Pixel under pixel i.
                createAndSortEdge(i, idxRight);
                createAndSortEdge(i, idxDown);
            }
            else if( row != params.height - 1 && col == params.width - 1 )
            {
                int idxDown = ( row + 1 ) * params.width + col;
                createAndSortEdge(i, idxDown);
            }
            else if( row == params.height - 1 && col != params.width - 1 )
            {
                int idxRight = row * params.width + col + 1;
                createAndSortEdge(i, idxRight);
            }
        }
    }
}
/*used along the previous method to sort edges in the histogram*/
void HistogramPhaseUnwrapping_Impl::createAndSortEdge( int idx1, int idx2 )
{
    if( pixels[idx2].getValidity() )
    {
        float edgeReliability = pixels[idx1].getInverseReliability() +
                                pixels[idx2].getInverseReliability();
        int inc = findInc(pixels[idx2].getPhaseValue(), pixels[idx1].getPhaseValue());
        Edge e(idx1, idx2, inc);

        if( edgeReliability < histogram.getThresh() )
        {
            int binIndex = static_cast<int> (ceil(edgeReliability / histogram.getSmallWidth()) - 1);
            if( binIndex == -1 )
            {
                binIndex = 0;
            }
            histogram.addEdgeInBin(e, binIndex);
        }
        else
        {
            int binIndex = params.nbrOfSmallBins +
                           static_cast<int> (ceil((edgeReliability - histogram.getThresh()) /
                                 histogram.getLargeWidth()) - 1);
            histogram.addEdgeInBin(e, binIndex);
        }
    }
}

void HistogramPhaseUnwrapping_Impl::unwrapHistogram()
{
    int nbrOfPixels = static_cast<int>(pixels.size());
    int nbrOfBins = histogram.getNbrOfBins();
    /* This vector is used to keep track of the number of pixels in each group and avoid useless group.
       For example, if lastPixelAddedToGroup[10] is equal to 5, it means that pixel "5" was the last one
       to be added to group 10. So, pixel "5" is the only one that has the correct value for parameter
       "numberOfPixelsInGroup" in order to avoid a loop on all the pixels to update this number*/
    std::vector<int> lastPixelAddedToGroup(nbrOfPixels, 0);
    for( int i = 0; i < nbrOfBins; ++i )
    {
        std::vector<Edge> currentEdges = histogram.getEdgesFromBin(i);
        int nbrOfEdgesInBin = static_cast<int>(currentEdges.size());

        for( int j = 0; j < nbrOfEdgesInBin; ++j )
        {

            int pOneId = currentEdges[j].getPixOneId();
            int pTwoId = currentEdges[j].getPixTwoId();
            // Both pixels are in a single group.
            if( pixels[pOneId].getSinglePixelGroup() && pixels[pTwoId].getSinglePixelGroup() )
            {
                float invRel1 = pixels[pOneId].getInverseReliability();
                float invRel2 = pixels[pTwoId].getInverseReliability();
                // Quality of pixel 2 is better than that of pixel 1 -> pixel 1 is added to group 2
                if( invRel1 > invRel2 )
                {
                    int newGroupId = pixels[pTwoId].getGroupId();
                    int newInc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement();
                    pixels[pOneId].setGroupId(newGroupId);
                    pixels[pOneId].setIncrement(newInc);
                    lastPixelAddedToGroup[newGroupId] = pOneId; // Pixel 1 is the last one to be added to group 2
                }
                else
                {
                    int newGroupId = pixels[pOneId].getGroupId();
                    int newInc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement();
                    pixels[pTwoId].setGroupId(newGroupId);
                    pixels[pTwoId].setIncrement(newInc);
                    lastPixelAddedToGroup[newGroupId] = pTwoId;
                }
                pixels[pOneId].setNbrOfPixelsInGroup(2);
                pixels[pTwoId].setNbrOfPixelsInGroup(2);
                pixels[pOneId].setSinglePixelGroup(false);
                pixels[pTwoId].setSinglePixelGroup(false);
            }
            //p1 is in a single group, p2 is not -> p1 added to p2
            else if( pixels[pOneId].getSinglePixelGroup() && !pixels[pTwoId].getSinglePixelGroup() )
            {
                int newGroupId = pixels[pTwoId].getGroupId();
                int lastPix = lastPixelAddedToGroup[newGroupId];
                int newNbrOfPixelsInGroup = pixels[lastPix].getNbrOfPixelsInGroup() + 1;
                int newInc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement();

                pixels[pOneId].setGroupId(newGroupId);
                pixels[pOneId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pTwoId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pOneId].setIncrement(newInc);
                pixels[pOneId].setSinglePixelGroup(false);

                lastPixelAddedToGroup[newGroupId] = pOneId;
            }
            //p2 is in a single group, p1 is not -> p2 added to p1
            else if( !pixels[pOneId].getSinglePixelGroup() && pixels[pTwoId].getSinglePixelGroup() )
            {
                int newGroupId = pixels[pOneId].getGroupId();
                int lastPix = lastPixelAddedToGroup[newGroupId];
                int newNbrOfPixelsInGroup = pixels[lastPix].getNbrOfPixelsInGroup() + 1;
                int newInc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement();

                pixels[pTwoId].setGroupId(newGroupId);
                pixels[pTwoId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pOneId].setNbrOfPixelsInGroup(newNbrOfPixelsInGroup);
                pixels[pTwoId].setIncrement(newInc);
                pixels[pTwoId].setSinglePixelGroup(false);

                lastPixelAddedToGroup[newGroupId] = pTwoId;
            }
            //p1 and p2 are in two different groups
            else if( pixels[pOneId].getGroupId() != pixels[pTwoId].getGroupId() )
            {
                int pOneGroupId = pixels[pOneId].getGroupId();
                int pTwoGroupId = pixels[pTwoId].getGroupId();

                float invRel1 = pixels[pOneId].getInverseReliability();
                float invRel2 = pixels[pTwoId].getInverseReliability();

                int lastAddedToGroupOne = lastPixelAddedToGroup[pOneGroupId];
                int lastAddedToGroupTwo = lastPixelAddedToGroup[pTwoGroupId];

                int nbrOfPixelsInGroupOne = pixels[lastAddedToGroupOne].getNbrOfPixelsInGroup();
                int nbrOfPixelsInGroupTwo = pixels[lastAddedToGroupTwo].getNbrOfPixelsInGroup();

                int totalNbrOfPixels = nbrOfPixelsInGroupOne + nbrOfPixelsInGroupTwo;

                if( nbrOfPixelsInGroupOne < nbrOfPixelsInGroupTwo ||
                   (nbrOfPixelsInGroupOne == nbrOfPixelsInGroupTwo && invRel1 >= invRel2) ) //group p1 added to group p2
                {
                    pixels[pTwoId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    pixels[pOneId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    int inc = pixels[pTwoId].getIncrement() + currentEdges[j].getIncrement() -
                                 pixels[pOneId].getIncrement();
                    lastPixelAddedToGroup[pTwoGroupId] = pOneId;

                    for( int k = 0; k < nbrOfPixels; ++k )
                    {
                        if( pixels[k].getGroupId() == pOneGroupId )
                        {
                            pixels[k].setGroupId(pTwoGroupId);
                            pixels[k].changeIncrement(inc);
                        }
                    }
                }
                else if( nbrOfPixelsInGroupOne > nbrOfPixelsInGroupTwo ||
                        (nbrOfPixelsInGroupOne == nbrOfPixelsInGroupTwo && invRel2 > invRel1) ) //group p2 added to group p1
                {
                    int oldGroupId = pTwoGroupId;
                    pixels[pOneId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    pixels[pTwoId].setNbrOfPixelsInGroup(totalNbrOfPixels);
                    int inc = pixels[pOneId].getIncrement() - currentEdges[j].getIncrement() -
                              pixels[pTwoId].getIncrement();
                    lastPixelAddedToGroup[pOneGroupId] = pTwoId;

                    for( int k = 0; k < nbrOfPixels; ++k )
                    {
                        if( pixels[k].getGroupId() == oldGroupId )
                        {
                            pixels[k].setGroupId(pOneGroupId);
                            pixels[k].changeIncrement(inc);
                        }
                    }
                }
            }
        }
    }
}
void HistogramPhaseUnwrapping_Impl::addIncrement( OutputArray unwrappedPhaseMap )
{
    Mat &uPhaseMap = *(Mat*) unwrappedPhaseMap.getObj();
    int rows = params.height;
    int cols = params.width;
    if( uPhaseMap.empty() )
        uPhaseMap.create(rows, cols, CV_32FC1);
    int nbrOfPixels = static_cast<int>(pixels.size());
    for( int i = 0; i < nbrOfPixels; ++i )
    {
        int row = pixels[i].getIndex() / params.width;
        int col = pixels[i].getIndex() % params.width;

        if( pixels[i].getValidity() )
        {
            uPhaseMap.at<float>(row, col) = pixels[i].getPhaseValue() +
                                            static_cast<float>(2 * CV_PI * pixels[i].getIncrement());
        }
    }
}
float HistogramPhaseUnwrapping_Impl::wrap( float a, float b )
{
    float result;
    float difference = a - b;
    float pi = static_cast<float>(CV_PI);
    if( difference > pi )
        result = ( difference - 2 * pi );
    else if( difference < -pi )
        result = ( difference + 2 * pi );
    else
        result = difference;
    return result;
}

int HistogramPhaseUnwrapping_Impl::findInc( float a, float b )
{
    float difference;
    int wrapValue;
    difference = b - a;
    float pi = static_cast<float>(CV_PI);
    if( difference > pi )
        wrapValue = -1;
    else if( difference < -pi )
        wrapValue = 1;
    else
        wrapValue = 0;
    return wrapValue;
}

//create a Mat that shows pixel inverse reliabilities
void HistogramPhaseUnwrapping_Impl::getInverseReliabilityMap( OutputArray inverseReliabilityMap )
{
    int rows = params.height;
    int cols = params.width;
    Mat &reliabilityMap_ = *(Mat*) inverseReliabilityMap.getObj();
    if( reliabilityMap_.empty() )
        reliabilityMap_.create(rows, cols, CV_32FC1);
    for( int i = 0; i < rows; ++i )
    {
        for( int j = 0; j < cols; ++j )
        {
            int idx = i * cols + j;
            reliabilityMap_.at<float>(i, j) = pixels[idx].getInverseReliability();
        }
    }
}

Ptr<HistogramPhaseUnwrapping> HistogramPhaseUnwrapping::create( const HistogramPhaseUnwrapping::Params
                                                                &params )
{
    return makePtr<HistogramPhaseUnwrapping_Impl>(params);
}

}
}
