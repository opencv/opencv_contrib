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

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>

#include "precomp.hpp"

#include "advanced_types.hpp"

/********************* Helper functions *********************/

/*!
 * Lightweight wrapper over cv::resize
 *
 * \param src : source image to resize
 * \param dst : destination image size
 * \return resized image
 */
static cv::Mat imresize(const cv::Mat &src, const cv::Size &nSize)
{
    cv::Mat dst;
    if (nSize.width < src.size().width
    &&  nSize.height < src.size().height)
        cv::resize(src, dst, nSize, 0.0, 0.0, cv::INTER_AREA);
    else
        cv::resize(src, dst, nSize, 0.0, 0.0, cv::INTER_LINEAR);

    return dst;
}

/*!
 * The function filters src with triangle filter with radius equal rad
 *
 * \param src : source image to filter
 * \param rad : radius of filtering kernel
 * \return filtering result
 */
static cv::Mat imsmooth(const cv::Mat &src, const int rad)
{
    if (rad == 0)
        return src;
    else
    {
        const float p = 12.0f/rad/(rad + 2) - 2;
        cv::Mat dst;

        if (rad <= 1)
        {
            CV_INIT_VECTOR(kernelXY, float, {1/(p + 2), p/(p + 2), 1/(p + 2)});
            cv::sepFilter2D(src, dst, -1, kernelXY, kernelXY);
        }
        else
        {
            float nrml = CV_SQR(rad + 1.0f);

            std::vector <float> kernelXY(2*rad + 1);
            for (int i = 0; i <= rad; ++i)
            {
                kernelXY[2*rad - i] = (i + 1) / nrml;
                kernelXY[i] = (i + 1) / nrml;
            }
            sepFilter2D(src, dst, -1, kernelXY, kernelXY);
        }

        return dst;
    }
}

/*!
 *  The function implements rgb to luv conversion in a way similar
 *  to UCSD computer vision toolbox
 *
 * \param src : source image (RGB, float, in [0;1]) to convert
 * \return converted image in luv colorspace
 */
static cv::Mat rgb2luv(const cv::Mat &src)
{
    cv::Mat dst(src.size(), src.type());

    const float a  = CV_CUBE(29.0f)/27;
    const float y0 = 8.0f/a;

    const float mX[] = {0.430574f, 0.341550f, 0.178325f};
    const float mY[] = {0.222015f, 0.706655f, 0.071330f};
    const float mZ[] = {0.020183f, 0.129553f, 0.939180f};

    const float maxi= 1.0f/270;
    const float minu=  -88*maxi;
    const float minv= -134*maxi;

    const float un = 0.197833f;
    const float vn = 0.468331f;

    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    std::vector <float> lTable(1024);
    for (int i = 0; i < 1024; ++i)
    {
        float y = i/1024.0f;
        float l = y > y0 ? 116*powf(y, 1.0f/3.0f) - 16 : y*a;

        lTable[i] = l*maxi;
    }
    for (int i = 0; i < 40; ++i)
        lTable.push_back(*--lTable.end());

    const int nchannels = 3;

    for (int i = 0; i < src.rows; ++i)
    {
        const float *pSrc = src.ptr<float>(i);
        float *pDst = dst.ptr<float>(i);

        for (int j = 0; j < src.cols*nchannels; j += nchannels)
        {
            const float rgb[] = {pSrc[j + 0], pSrc[j + 1], pSrc[j + 2]};

            const float xyz[] = {mX[0]*rgb[0] + mX[1]*rgb[1] + mX[2]*rgb[2],
                                 mY[0]*rgb[0] + mY[1]*rgb[1] + mY[2]*rgb[2],
                                 mZ[0]*rgb[0] + mZ[1]*rgb[1] + mZ[2]*rgb[2]};
            const float nz = 1.0f / float(xyz[0] + 15*xyz[1] + 3*xyz[2] + 1e-35);

            const float l = pDst[j] = lTable[cvFloor(1024*xyz[1])];

            pDst[j + 1] = l * (13*4*xyz[0]*nz - 13*un) - minu;;
            pDst[j + 2] = l * (13*9*xyz[1]*nz - 13*vn) - minv;
        }
    }

    return dst;
}

/*!
 * The function computes gradient magnitude and weighted (with magnitude)
 * orientation histogram. Magnitude is additionally normalized
 * by dividing on imsmooth(M, gnrmRad) + 0.01;
 *
 * \param src : source image
 * \param magnitude : gradient magnitude
 * \param histogram : gradient orientation nBins-channels histogram
 * \param nBins : number of gradient orientations
 * \param pSize : factor to downscale histogram
 * \param gnrmRad : radius for magnitude normalization
 */
static void gradientHist(const cv::Mat &src, cv::Mat &magnitude, cv::Mat &histogram,
                         const int nBins, const int pSize, const int gnrmRad)
{
    cv::Mat phase, Dx, Dy;

    magnitude.create( src.size(), cv::DataType<float>::type );
    phase.create( src.size(), cv::DataType<float>::type );
    histogram.create( cv::Size( cvCeil(src.size().width/float(pSize)),
                                cvCeil(src.size().height/float(pSize)) ),
        CV_MAKETYPE(cv::DataType<float>::type, nBins) );

    histogram.setTo(0);

    cv::Sobel( src, Dx, cv::DataType<float>::type,
        1, 0, 1, 1.0, 0.0, cv::BORDER_REFLECT );
    cv::Sobel( src, Dy, cv::DataType<float>::type,
        0, 1, 1, 1.0, 0.0, cv::BORDER_REFLECT );

    int nchannels = src.channels();

    for (int i = 0; i < src.rows; ++i)
    {
        const float *pDx = Dx.ptr<float>(i);
        const float *pDy = Dy.ptr<float>(i);

        float *pMagnitude = magnitude.ptr<float>(i);
        float *pPhase = phase.ptr<float>(i);

        for (int j = 0; j < src.cols*nchannels; j += nchannels)
        {
            float fMagn = float(-1e-5), fdx = 0, fdy = 0;
            for (int k = 0; k < nchannels; ++k)
            {
                float cMagn = CV_SQR( pDx[j + k] ) + CV_SQR( pDy[j + k] );
                if (cMagn > fMagn)
                {
                    fMagn = cMagn;
                    fdx = pDx[j + k];
                    fdy = pDy[j + k];
                }
            }

            pMagnitude[j/nchannels] = sqrtf(fMagn);

            float angle = cv::fastAtan2(fdy, fdx) / 180.0f - 1.0f * (fdy < 0);
            if (std::fabs(fdx) + std::fabs(fdy) < 1e-5)
                angle = 0.5f;
            pPhase[j/nchannels] = angle;
        }
    }

    magnitude /= imsmooth( magnitude, gnrmRad )
        + 0.01*cv::Mat::ones( magnitude.size(), magnitude.type() );

    int pHistSize = histogram.cols*histogram.channels() - 1;
    for (int i = 0; i < phase.rows; ++i)
    {
        const float *pPhase = phase.ptr<float>(i);
        const float *pMagn  = magnitude.ptr<float>(i);

        float *pHist = histogram.ptr<float>(i/pSize);

        for (int j = 0; j < phase.cols; ++j)
        {
            int index  = cvRound((j/pSize + pPhase[j])*nBins);
            index = std::max(0, std::min(index, pHistSize));
            pHist[index] += pMagn[j] / CV_SQR(pSize);
        }
    }
}

/********************* RFFeatureGetter class *********************/

namespace cv
{
namespace ximgproc
{

class RFFeatureGetterImpl : public RFFeatureGetter
{
public:
    /*!
     * Default constructor
     */
    RFFeatureGetterImpl() : name("RFFeatureGetter"){}

    /*!
     * The method extracts features from img and store them to features.
     * Extracted features are appropriate for StructuredEdgeDetection::predictEdges.
     *
     * \param src : source image (RGB, float, in [0;1]) to extract features
     * \param features : destination feature image
     *
     * \param gnrmRad : __rf.options.gradientNormalizationRadius
     * \param gsmthRad : __rf.options.gradientSmoothingRadius
     * \param shrink : __rf.options.shrinkNumber
     * \param outNum : __rf.options.numberOfOutputChannels
     * \param gradNum : __rf.options.numberOfGradientOrientations
     */
    virtual void getFeatures(const Mat &src, Mat &features, const int gnrmRad, const int gsmthRad,
                             const int shrink, const int outNum, const int gradNum) const
    {
        cv::Mat luvImg = rgb2luv(src);

        std::vector <cv::Mat> featureArray;

        cv::Size nSize = src.size() / float(shrink);
        split( imresize(luvImg, nSize), featureArray );

        CV_INIT_VECTOR(scales, float, {1.0f, 0.5f});

        for (size_t i = 0; i < scales.size(); ++i)
        {
            int pSize = std::max( 1, int(shrink*scales[i]) );

            cv::Mat magnitude, histogram;
            gradientHist(/**/ imsmooth(imresize(luvImg, scales[i]*src.size()), gsmthRad),
                magnitude, histogram, gradNum, pSize, gnrmRad /**/);

            featureArray.push_back(/**/ imresize( magnitude, nSize ).clone() /**/);
            featureArray.push_back(/**/ imresize( histogram, nSize ).clone() /**/);
        }

        // Mixing
        int resType = CV_MAKETYPE(cv::DataType<float>::type, outNum);
        features.create(nSize, resType);

        std::vector <int> fromTo;
        for (int i = 0; i < 2*outNum; ++i)
            fromTo.push_back(i/2);

        mixChannels(featureArray, features, fromTo);
    }

protected:
    /*! algorithm name */
    String name;
};

Ptr<RFFeatureGetter> createRFFeatureGetter()
{
        return makePtr<RFFeatureGetterImpl>();
}

}
}

/********************* StructuredEdgeDetection class *********************/

namespace cv
{
namespace ximgproc
{

class StructuredEdgeDetectionImpl : public StructuredEdgeDetection
{
public:
    /*!
     * This constructor loads __rf model from filename
     *
     * \param filename : name of the file where the model is stored
     */
    StructuredEdgeDetectionImpl(const cv::String &filename,
        Ptr<const RFFeatureGetter> _howToGetFeatures)
        : name("StructuredEdgeDetection"),
          howToGetFeatures( (!_howToGetFeatures.empty())
                          ? _howToGetFeatures
                          : createRFFeatureGetter().staticCast<const RFFeatureGetter>() )
    {
        cv::FileStorage modelFile(filename, FileStorage::READ);
        CV_Assert( modelFile.isOpened() );

        __rf.options.stride
            = modelFile["options"]["stride"];
        __rf.options.shrinkNumber
            = modelFile["options"]["shrinkNumber"];
        __rf.options.patchSize
            = modelFile["options"]["patchSize"];
        __rf.options.patchInnerSize
            = modelFile["options"]["patchInnerSize"];

        __rf.options.numberOfGradientOrientations
            = modelFile["options"]["numberOfGradientOrientations"];
        __rf.options.gradientSmoothingRadius
            = modelFile["options"]["gradientSmoothingRadius"];
        __rf.options.regFeatureSmoothingRadius
            = modelFile["options"]["regFeatureSmoothingRadius"];
        __rf.options.ssFeatureSmoothingRadius
            = modelFile["options"]["ssFeatureSmoothingRadius"];
        __rf.options.gradientNormalizationRadius
            = modelFile["options"]["gradientNormalizationRadius"];

        __rf.options.selfsimilarityGridSize
            = modelFile["options"]["selfsimilarityGridSize"];

        __rf.options.numberOfTrees
            = modelFile["options"]["numberOfTrees"];
        __rf.options.numberOfTreesToEvaluate
            = modelFile["options"]["numberOfTreesToEvaluate"];

        __rf.options.numberOfOutputChannels =
            2*(__rf.options.numberOfGradientOrientations + 1) + 3;
        //--------------------------------------------

        cv::FileNode childs = modelFile["childs"];
        cv::FileNode featureIds = modelFile["featureIds"];

        std::vector <int> currentTree;

        for(cv::FileNodeIterator it = childs.begin();
            it != childs.end(); ++it)
        {
            (*it) >> currentTree;
            std::copy(currentTree.begin(), currentTree.end(),
                std::back_inserter(__rf.childs));
        }

        for(cv::FileNodeIterator it = featureIds.begin();
            it != featureIds.end(); ++it)
        {
            (*it) >> currentTree;
            std::copy(currentTree.begin(), currentTree.end(),
                std::back_inserter(__rf.featureIds));
        }

        cv::FileNode thresholds = modelFile["thresholds"];
        std::vector <float> fcurrentTree;

        for(cv::FileNodeIterator it = thresholds.begin();
            it != thresholds.end(); ++it)
        {
            (*it) >> fcurrentTree;
            std::copy(fcurrentTree.begin(), fcurrentTree.end(),
                std::back_inserter(__rf.thresholds));
        }

        cv::FileNode edgeBoundaries = modelFile["edgeBoundaries"];
        cv::FileNode edgeBins = modelFile["edgeBins"];

        for(cv::FileNodeIterator it = edgeBoundaries.begin();
            it != edgeBoundaries.end(); ++it)
        {
            (*it) >> currentTree;
            std::copy(currentTree.begin(), currentTree.end(),
                std::back_inserter(__rf.edgeBoundaries));
        }

        for(cv::FileNodeIterator it = edgeBins.begin();
            it != edgeBins.end(); ++it)
        {
            (*it) >> currentTree;
            std::copy(currentTree.begin(), currentTree.end(),
                std::back_inserter(__rf.edgeBins));
        }

        __rf.numberOfTreeNodes = int( __rf.childs.size() ) / __rf.options.numberOfTrees;
    }

    /*!
     * The function detects edges in src and draw them to dst
     *
     * \param src : source image (RGB, float, in [0;1]) to detect edges
     * \param dst : destination image (grayscale, float, in [0;1])
     *              where edges are drawn
     */
    void detectEdges(const cv::Mat &src, cv::Mat &dst) const
    {
        CV_Assert( src.type() == CV_32FC3 );

        dst.create( src.size(), cv::DataType<float>::type );

        int padding = ( __rf.options.patchSize
            - __rf.options.patchInnerSize )/2;

        cv::Mat nSrc;
        copyMakeBorder( src, nSrc, padding, padding,
            padding, padding, BORDER_REFLECT );

        NChannelsMat features;
        createRFFeatureGetter()->getFeatures( nSrc, features,
            __rf.options.gradientNormalizationRadius,
            __rf.options.gradientSmoothingRadius,
            __rf.options.shrinkNumber,
            __rf.options.numberOfOutputChannels,
            __rf.options.numberOfGradientOrientations );
        predictEdges( features, dst );
    }

protected:
    /*!
     * Private method used by process method. The function
     * predict edges in n-channel feature image and store them to dst.
     *
     * \param features : source image (n-channels, float) to detect edges
     * \param dst : destination image (grayscale, float, in [0;1]) where edges are drawn
     */
    void predictEdges(const NChannelsMat &features, cv::Mat &dst) const
    {
        int shrink = __rf.options.shrinkNumber;
        int rfs = __rf.options.regFeatureSmoothingRadius;
        int sfs = __rf.options.ssFeatureSmoothingRadius;

        int nTreesEval = __rf.options.numberOfTreesToEvaluate;
        int nTrees = __rf.options.numberOfTrees;
        int nTreesNodes = __rf.numberOfTreeNodes;

        const int nchannels = features.channels();
        int pSize  = __rf.options.patchSize;

        int nFeatures = CV_SQR(pSize/shrink)*nchannels;
        int outNum = __rf.options.numberOfOutputChannels;

        int stride = __rf.options.stride;
        int ipSize = __rf.options.patchInnerSize;
        int gridSize = __rf.options.selfsimilarityGridSize;

        const int height = cvCeil( double(features.rows*shrink - pSize) / stride );
        const int width  = cvCeil( double(features.cols*shrink - pSize) / stride );
        // image size in patches with overlapping

        //-------------------------------------------------------------------------

        NChannelsMat regFeatures = imsmooth(features, cvRound(rfs / float(shrink)));
        NChannelsMat  ssFeatures = imsmooth(features, cvRound(sfs / float(shrink)));

        NChannelsMat indexes(height, width, CV_MAKETYPE(DataType<int>::type, nTreesEval));

        std::vector <int> offsetI(/**/ CV_SQR(pSize/shrink)*nchannels, 0);
        for (int i = 0; i < CV_SQR(pSize/shrink)*nchannels; ++i)
        {
            int z = i / CV_SQR(pSize/shrink);
            int y = ( i % CV_SQR(pSize/shrink) )/(pSize/shrink);
            int x = ( i % CV_SQR(pSize/shrink) )%(pSize/shrink);

            offsetI[i] = x*features.cols*nchannels + y*nchannels + z;
        }
        // lookup table for mapping linear index to offsets

        std::vector <int> offsetE(/**/ CV_SQR(ipSize)*outNum, 0);
        for (int i = 0; i < CV_SQR(ipSize)*outNum; ++i)
        {
            int z = i / CV_SQR(ipSize);
            int y = ( i % CV_SQR(ipSize) )/ipSize;
            int x = ( i % CV_SQR(ipSize) )%ipSize;

            offsetE[i] = x*dst.cols*outNum + y*outNum + z;
        }
        // lookup table for mapping linear index to offsets

        std::vector <int> offsetX( CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)/2 * nchannels, 0);
        std::vector <int> offsetY( CV_SQR(gridSize)*(CV_SQR(gridSize) - 1)/2 * nchannels, 0);

        int hc = cvRound( (pSize/shrink) / (2.0*gridSize) );
        // half of cell
        std::vector <int> gridPositions;
        for(int i = 0; i < gridSize; i++)
            gridPositions.push_back( int( (i+1)*(pSize/shrink + 2*hc - 1)/(gridSize + 1.0) - hc + 0.5f ) );

        for (int i = 0, n = 0; i < CV_SQR(gridSize)*nchannels; ++i)
            for (int j = (i%CV_SQR(gridSize)) + 1; j < CV_SQR(gridSize); ++j, ++n)
            {
                int z = i / CV_SQR(gridSize);

                int x1 = gridPositions[i%CV_SQR(gridSize)%gridSize];
                int y1 = gridPositions[i%CV_SQR(gridSize)/gridSize];

                int x2 = gridPositions[j%gridSize];
                int y2 = gridPositions[j/gridSize];

                offsetX[n] = x1*features.cols*nchannels + y1*nchannels + z;
                offsetY[n] = x2*features.cols*nchannels + y2*nchannels + z;
            }
            // lookup tables for mapping linear index to offset pairs
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < height; ++i)
        {
            float *regFeaturesPtr = regFeatures.ptr<float>(i*stride/shrink);
            float  *ssFeaturesPtr = ssFeatures.ptr<float>(i*stride/shrink);

            int *indexPtr = indexes.ptr<int>(i);

            for (int j = 0, k = 0; j < width; ++k, j += !(k %= nTreesEval))
                // for j,k in [0;width)x[0;nTreesEval)
            {
                int baseNode = ( ((i + j)%(2*nTreesEval) + k)%nTrees )*nTreesNodes;
                int currentNode = baseNode;
                // select root node of the tree to evaluate

                int offset = (j*stride/shrink)*nchannels;
                while ( __rf.childs[currentNode] != 0 )
                {
                    int currentId = __rf.featureIds[currentNode];
                    float currentFeature;

                    if (currentId >= nFeatures)
                    {
                        int xIndex = offsetX[currentId - nFeatures];
                        float A = ssFeaturesPtr[offset + xIndex];

                        int yIndex = offsetY[currentId - nFeatures];
                        float B = ssFeaturesPtr[offset + yIndex];

                        currentFeature = A - B;
                    }
                    else
                        currentFeature = regFeaturesPtr[offset + offsetI[currentId]];

                    // compare feature to threshold and move left or right accordingly
                    if (currentFeature < __rf.thresholds[currentNode])
                        currentNode = baseNode + __rf.childs[currentNode] - 1;
                    else
                        currentNode = baseNode + __rf.childs[currentNode];
                }

                indexPtr[j*nTreesEval + k] = currentNode;
            }
        }

        NChannelsMat dstM(dst.size(),
            CV_MAKETYPE(DataType<float>::type, outNum));
        dstM.setTo(0);

        float step = 2.0f * CV_SQR(stride) / CV_SQR(ipSize) / nTreesEval;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int i = 0; i < height; ++i)
        {
            int *pIndex = indexes.ptr<int>(i);
            float *pDst = dstM.ptr<float>(i*stride);

                for (int j = 0, k = 0; j < width; ++k, j += !(k %= nTreesEval))
                {// for j,k in [0;width)x[0;nTreesEval)

                    int currentNode = pIndex[j*nTreesEval + k];

                    int start  = __rf.edgeBoundaries[currentNode];
                    int finish = __rf.edgeBoundaries[currentNode + 1];

                    if (start == finish)
                        continue;

                    int offset = j*stride*outNum;
                    for (int p = start; p < finish; ++p)
                        pDst[offset + offsetE[__rf.edgeBins[p]]] += step;
                }
        }

        cv::reduce( dstM.reshape(1, int( dstM.total() ) ), dstM, 2, CV_REDUCE_SUM);
        imsmooth( dstM.reshape(1, dst.rows), 1 ).copyTo(dst);
    }

/********************* Members *********************/
protected:
    /*! algorithm name */
    String name;

    /*! optional feature getter (getFeatures method) */
    Ptr<const RFFeatureGetter> howToGetFeatures;

    /*! random forest used to detect edges */
    struct RandomForest
    {
        /*! random forest options, e.g. number of trees */
        struct RandomForestOptions
        {
            // model params

            int numberOfOutputChannels; /*!< number of edge orientation bins for output */

            int patchSize;              /*!< width of image patches */
            int patchInnerSize;         /*!< width of predicted part inside patch*/

            // feature params

            int regFeatureSmoothingRadius;    /*!< radius for smoothing of regular features
                                               *   (using convolution with triangle filter) */

            int ssFeatureSmoothingRadius;     /*!< radius for smoothing of additional features
                                               *   (using convolution with triangle filter) */

            int shrinkNumber;                 /*!< amount to shrink channels */

            int numberOfGradientOrientations; /*!< number of orientations per gradient scale */

            int gradientSmoothingRadius;      /*!< radius for smoothing of gradients
                                               *   (using convolution with triangle filter) */

            int gradientNormalizationRadius;  /*!< gradient normalization radius */
            int selfsimilarityGridSize;       /*!< number of self similarity cells */

            // detection params
            int numberOfTrees;            /*!< number of trees in forest to train */
            int numberOfTreesToEvaluate;  /*!< number of trees to evaluate per location */

            int stride;                   /*!< stride at which to compute edges */

        } options;

        int numberOfTreeNodes;

        std::vector <int> featureIds;     /*!< feature coordinate thresholded at k-th node */
        std::vector <float> thresholds;   /*!< threshold applied to featureIds[k] at k-th node */
        std::vector <int> childs;         /*!< k --> child[k] - 1, child[k] */

        std::vector <int> edgeBoundaries; /*!< ... */
        std::vector <int> edgeBins;       /*!< ... */
    } __rf;
};

Ptr<StructuredEdgeDetection> createStructuredEdgeDetection(const String &model,
    Ptr<const RFFeatureGetter> howToGetFeatures)
{
        return makePtr<StructuredEdgeDetectionImpl>(model, howToGetFeatures);
}

}
}
