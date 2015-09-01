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
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include <opencv2/highgui.hpp>

namespace cv
{
namespace optflow
{

typedef int T_state;					// T_state is the type for state
typedef float T_input;			// T_input is the data type of the input image
#ifdef INTMESSAGE
    typedef int T_message;
#else
    typedef double T_message;
#endif

//------------------------------------------------------------------------------------------------
//	function to allocate buffer for the messages
//------------------------------------------------------------------------------------------------
template <typename T1, typename T2>
size_t allocateBuffer(T1*& pBuffer, size_t area, size_t factor, T2*& ptrData, const int* pWinSize)
{
    pBuffer = new T1[area*factor];
    size_t totalElements = 0;
    for (ptrdiff_t i = 0; i < area; i++)
    {
        totalElements += pWinSize[i] * 2 + 1;
        for (ptrdiff_t j = 0; j<factor; j++)
            pBuffer[i*factor + j].allocate(pWinSize[i] * 2 + 1);
    }
    totalElements *= factor;
    ptrData = new T2[totalElements];
    memset(ptrData, 0, sizeof(T2)*totalElements);

    T2* ptrDynamic = ptrData;
    size_t total = 0;
    for (ptrdiff_t i = 0; i<area*factor; i++)
    {
        pBuffer[i].data() = ptrDynamic;
        ptrDynamic += pBuffer[i].nElements();
        total += pBuffer[i].nElements();
    }
    return total;
}

template<typename T1, typename T2>
size_t allocateBuffer(T1*& pBuffer, T2*& ptrData, size_t area, const int* pWinSize1, const int* pWinSize2)
{
    pBuffer = new T1[area];
    size_t totalElements = 0;
    for (ptrdiff_t i = 0; i<area; i++)
    {
        totalElements += (pWinSize1[i] * 2 + 1)*(pWinSize2[i] * 2 + 1);
        pBuffer[i].allocate(pWinSize1[i] * 2 + 1, pWinSize2[i] * 2 + 1);
    }
    ptrData = new T2[totalElements];
    memset(ptrData, 0, sizeof(T2)*totalElements);

    T2* ptrDynamic = ptrData;
    size_t total = 0;
    for (ptrdiff_t i = 0; i < area; i++)
    {
        pBuffer[i].data() = ptrDynamic;
        ptrDynamic += pBuffer[i].nElements();
        total += pBuffer[i].nElements();
    }
    return total;
}

template <typename T>
void release1DBuffer(T* pBuffer)
{
    if (pBuffer != NULL)
        delete[]pBuffer;
    pBuffer = NULL;
}

template <typename T>
void release2DBuffer(T** pBuffer, size_t nElements)
{
    for (size_t i = 0; i<nElements; i++)
        delete[](pBuffer[i]);
    delete[]pBuffer;
    pBuffer = NULL;
}

template <typename InputType>
BPFlow<InputType>::BPFlow()
{
    IsDataTermTruncated = false;
    IsTRW = false;
    CTRW = (double)1 / 2;
    //CTRW=0.55;
    Width = Height = Area = 0;
    pIm1 = pIm2 = NULL;
    for (int i = 0; i<2; i++)
    {
        pOffset[i] = NULL;
        pWinSize[i] = NULL;
    }
    pDataTerm = NULL;
    ptrDataTerm = NULL;
    for (int i = 0; i<2; i++)
    {
        pRangeTerm[i] = pSpatialMessage[i] = pDualMessage[i] = pBelief[i] = NULL;
        ptrRangeTerm[i] = ptrSpatialMessage[i] = ptrDualMessage[i] = ptrBelief[i] = NULL;
    }
    pX = NULL;
    nNeighbors = 4;

    // parameters
    d = 40 * 255;
    levels = 4;
    alpha = 2 * 255;
    gamma = 0.005 * 255;
    iterations = 60;
    topiterations = 100;
    wsize = 5;
    topwsize = 20;
    hierarchy = 4;//
}

template <typename InputType>
BPFlow<InputType>::~BPFlow()
{
    release();
}

template <typename InputType>
void BPFlow<InputType>::release()
{
    //Release1DBuffer(pIm1);
    //_Release1DBuffer(pIm2);

    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pOffset[i]); // release the buffer of the offset
        release1DBuffer(pWinSize[i]); // release the buffer of the size
    }

    release1DBuffer(pDataTerm);
    release1DBuffer(ptrDataTerm);
    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pRangeTerm[i]);
        release1DBuffer(ptrRangeTerm[i]);
        release1DBuffer(pSpatialMessage[i]);
        release1DBuffer(ptrSpatialMessage[i]);
        release1DBuffer(pDualMessage[i]);
        release1DBuffer(ptrDualMessage[i]);
        release1DBuffer(pBelief[i]);
        release1DBuffer(ptrBelief[i]);
    }
    release1DBuffer(pX);
}

template <typename InputType>
void BPFlow<InputType>::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow)
{
    // Get images
    Mat I0 = _I0.getMat();
    Mat I1 = _I1.getMat();
    CV_Assert(I0.size() == I1.size());
    CV_Assert(I0.type() == I1.type());
    CV_Assert(I0.channels() == 1);
    // TODO: currently only grayscale - data term could be computed in color version as well...

    // Create flow
    _flow.create(I0.size(), CV_32FC2);
    Mat W = _flow.getMat(); // if any data present - will be discarded
    Mat vx = Mat(Height, Width, CV_32F, 0.0f);
    Mat vy = Mat(Height, Width, CV_32F, 0.0f);

    calcOneLevel(I0, I1, W, topwsize, vx, vy);

    // Output flow
    W.copyTo(_flow);
}

template <typename InputType>
void BPFlow<InputType>::collectGarbage()
{

}

//
//CV_INIT_ALGORITHM(OpticalFlowDeepFlow, "DenseOpticalFlow.DeepFlow",
//        obj.info()->addParam(obj, "sigma", obj.sigma, false, 0, 0, "Gaussian blur parameter");
//        obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Smoothness assumption weight");
//        obj.info()->addParam(obj, "delta", obj.delta, false, 0, 0, "Color constancy weight");
//        obj.info()->addParam(obj, "gamma", obj.gamma, false, 0, 0, "Gradient constancy weight");
//        obj.info()->addParam(obj, "omega", obj.omega, false, 0, 0, "Relaxation factor in SOR");
//        obj.info()->addParam(obj, "minSize", obj.minSize, false, 0, 0, "Min. image size in the pyramid");
//        obj.info()->addParam(obj, "fixedPointIterations", obj.fixedPointIterations, false, 0, 0, "Fixed point iterations");
//        obj.info()->addParam(obj, "sorIterations", obj.sorIterations, false, 0, 0, "SOR iterations");
//        obj.info()->addParam(obj, "downscaleFactor", obj.downscaleFactor, false, 0, 0,"Downscale factor"))

template <typename InputType>
std::vector<Mat> BPFlow<InputType>::buildPyramid(const Mat& src, int _levels)
{
    std::vector<Mat> pyramid;
    pyramid.push_back(src);
    Mat tmp;
    int w, h;

    // For each pyramid level starting from level 1
    for (size_t i = 1; i < _levels; ++i)
    {
        // Apply gaussian filter
        GaussianBlur(pyramid[i - 1], tmp, Size(5, 5), 0.67, 0, BORDER_REPLICATE);

        // Resize image using bicubic interpolation
        //w = (int)ceil(pyramid[i - 1].cols / 2.0f);
        //h = (int)ceil(pyramid[i - 1].rows / 2.0f);
        w = pyramid[i - 1].cols / 2 + (pyramid[i - 1].cols % 2 == 1);
        h = pyramid[i - 1].rows / 2 + (pyramid[i - 1].rows % 2 == 1);
        resize(tmp, pyramid[i], Size(w, h), 0, 0, INTER_CUBIC);
    }

    return pyramid;
}

template <typename InputType>
void BPFlow<InputType>::calcOneLevel(const Mat I0, const Mat I1, Mat W,
    int winsize, const cv::Mat& offsetX, const cv::Mat& offsetY)
{
    setHomogeneousMRF(wsize);
    setOffset(offsetX, offsetY);

    Mat_<int> winSizeX(W.rows, W.cols, winsize);
    Mat_<int> winSizeY(W.rows, W.cols, winsize);
    setWinSize(winSizeX, winSizeY);

    computeDataTerm();
    computeRangeTerm(gamma);
    messagePassing(iterations, 2);
    computeVelocity(W);
}

template <typename InputType>
void BPFlow<InputType>::computeDataTerm()
{
    // allocate the buffer for data term
    nTotalMatches = allocateBuffer<PixelBuffer2D<T_message>, T_message>(pDataTerm, ptrDataTerm, Area, pWinSize[0], pWinSize[1]);

    T_message HistMin, HistMax;
    double HistInterval;
    double* pHistogramBuffer;
    int nBins = 20000;
    int total = 0; // total is the total number of plausible matches, used to normalize the histogram
    pHistogramBuffer = new double[nBins];
    memset(pHistogramBuffer, 0, sizeof(double)*nBins);
    HistMin = 32767;
    HistMax = 0;
    //--------------------------------------------------------------------------------------------------
    // step 1. the first sweep to compute the data term for the visible matches
    //--------------------------------------------------------------------------------------------------
    for (ptrdiff_t i = 0; i<Height; i++)			// index over y
        for (ptrdiff_t j = 0; j<Width; j++)		// index over x
        {
            size_t index = i*Width + j;
            int XWinLength = pWinSize[0][index] * 2 + 1;
            // loop over a local window
            for (ptrdiff_t k = -pWinSize[1][index]; k <= pWinSize[1][index]; k++)  // index over y
                for (ptrdiff_t l = -pWinSize[0][index]; l <= pWinSize[0][index]; l++)  // index over x
                {
                    ptrdiff_t x = j + pOffset[0][index] + l;
                    ptrdiff_t y = i + pOffset[1][index] + k;

                    // if the point is outside the image boundary then continue
                    if (!isInsideImage(x, y))
                        continue;
                    ptrdiff_t index2 = y*Width2 + x;
                    T_message foo = 0;
                    for (int n = 0; n < nChannels; n++)
                        //foo += abs(pIm1[index*nChannels + n] - pIm2[index2*nChannels + n]); // L1 norm
                        foo += abs(((double*)im_s.data)[index*nChannels + n] - ((double*)im_d.data)[index*nChannels + n]);
                    //#ifdef INTMESSAGE
                    //						foo+=abs(pIm1[index*nChannels+n]-pIm2[index2*nChannels+n]); // L1 norm
                    //#else
                    //						foo+=fabs(pIm1[index*nChannels+n]-pIm2[index2*nChannels+n]); // L1 norm
                    //#endif


                    pDataTerm[index][(k + pWinSize[1][index])*XWinLength + l + pWinSize[0][index]] = foo;
                    HistMin = __min(HistMin, foo);
                    HistMax = __max(HistMax, foo);
                    total++;
                }
        }
    // compute the histogram info
    HistInterval = (double)(HistMax - HistMin) / nBins;
    //HistInterval/=21;

    //--------------------------------------------------------------------------------------------------
    // step 2. get the histogram of the matching
    //--------------------------------------------------------------------------------------------------
    for (ptrdiff_t i = 0; i<Height; i++)			// index over y
        for (ptrdiff_t j = 0; j<Width; j++)		// index over x
        {
            size_t index = i*Width + j;
            int XWinLength = pWinSize[0][index] * 2 + 1;
            // loop over a local window
            for (ptrdiff_t k = -pWinSize[1][index]; k <= pWinSize[1][index]; k++)  // index over y
                for (ptrdiff_t l = -pWinSize[0][index]; l <= pWinSize[0][index]; l++)  // index over x
                {
                    ptrdiff_t x = j + pOffset[0][index] + l;
                    ptrdiff_t y = i + pOffset[1][index] + k;

                    // if the point is outside the image boundary then continue
                    if (!isInsideImage(x, y))
                        continue;
                    int foo = __min(pDataTerm[index][(k + pWinSize[1][index])*XWinLength + l + pWinSize[0][index]] / HistInterval, nBins - 1);
                    pHistogramBuffer[foo]++;
                }
        }
    for (size_t i = 0; i<nBins; i++) // normalize the histogram
        pHistogramBuffer[i] /= total;

    T_message DefaultMatchingScore;
    double Prob = 0;
    for (size_t i = 0; i<nBins; i++)
    {
        Prob += pHistogramBuffer[i];
        if (Prob >= 0.5)//(double)Area/nTotalMatches) // find the matching score
        {
            DefaultMatchingScore = __max(i, 1)*HistInterval + HistMin;
            break;
        }
    }
    //DefaultMatchingScore=__min(100*DefaultMatchingScore,HistMax/10);
    if (IsDisplay)
#ifdef INTMESSAGE
        printf("Min: %d, Default: %d, Max: %d\n", HistMin, DefaultMatchingScore, HistMax);
#else
        printf("Min: %f, Default: %f, Max: %f\n", HistMin, DefaultMatchingScore, HistMax);
#endif

    //DefaultMatchingScore=0.1;
    //--------------------------------------------------------------------------------------------------
    // step 3. assigning the default matching score to the outside matches
    //--------------------------------------------------------------------------------------------------
    for (ptrdiff_t i = 0; i<Height; i++)			// index over y
        for (ptrdiff_t j = 0; j<Width; j++)		// index over x
        {
            size_t index = i*Width + j;
            int XWinLength = pWinSize[0][index] * 2 + 1;
            // loop over a local window
            for (ptrdiff_t k = -pWinSize[1][index]; k <= pWinSize[1][index]; k++)  // index over y
                for (ptrdiff_t l = -pWinSize[0][index]; l <= pWinSize[0][index]; l++)  // index over x
                {
                    ptrdiff_t x = j + pOffset[0][index] + l;
                    ptrdiff_t y = i + pOffset[1][index] + k;

                    int _ptr = (k + pWinSize[1][index])*XWinLength + l + pWinSize[0][index];
                    // if the point is outside the image boundary then continue
                    if (!isInsideImage(x, y))
                        pDataTerm[index][_ptr] = DefaultMatchingScore;
                    else if (IsDataTermTruncated) // put truncaitons to the data term
                        pDataTerm[index][_ptr] = __min(pDataTerm[index][_ptr], DefaultMatchingScore);
                }
        }
    delete pHistogramBuffer;
}

template <typename InputType>
void BPFlow<InputType>::computeRangeTerm(double _gamma)
{
    level_gamma = _gamma;
    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pRangeTerm[i]);
        release1DBuffer(ptrRangeTerm[i]);
        allocateBuffer(pRangeTerm[i], Area, 1, ptrRangeTerm[i], pWinSize[i]);
    }
    for (ptrdiff_t offset = 0; offset<Area; offset++)
    {
        for (ptrdiff_t plane = 0; plane<2; plane++)
        {
            int winsize = pWinSize[plane][offset];
            for (ptrdiff_t j = -winsize; j <= winsize; j++)
                pRangeTerm[plane][offset].data()[j + winsize] = level_gamma*fabs((double)j + pOffset[plane][offset]);
        }
    }
}

template <typename InputType>
void BPFlow<InputType>::allocateMessage()
{
    // delete the buffers for the messages
    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pSpatialMessage[i]);
        release1DBuffer(ptrSpatialMessage[i]);
        release1DBuffer(pDualMessage[i]);
        release1DBuffer(ptrDualMessage[i]);
        release1DBuffer(pBelief[i]);
        release1DBuffer(ptrBelief[i]);
    }
    // allocate the buffers for the messages
    for (int i = 0; i<2; i++)
    {
        nTotalSpatialElements[i] = allocateBuffer(pSpatialMessage[i], Area, nNeighbors, ptrSpatialMessage[i], pWinSize[i]);
        nTotalDualElements[i] = allocateBuffer(pDualMessage[i], Area, 1, ptrDualMessage[i], pWinSize[i]);
        nTotalBelifElements[i] = allocateBuffer(pBelief[i], Area, 1, ptrBelief[i], pWinSize[i]);
    }
}

template <typename InputType>
double BPFlow<InputType>::messagePassing(int _iterations, int _hierarchy)
{
    allocateMessage();
    
    if (_hierarchy > 0)
    {
        BPFlow bp;
        generateCoarserLevel(bp);
        bp.messagePassing(20, hierarchy - 1);
        bp.propagateFinerLevel(*this);
    }
    
    if (pX != NULL)
        release1DBuffer(pX);
    pX = new int[Area * 2];
    double energy;
    for (int count = 0; count<_iterations; count++)
    {
        //Bipartite(count);
        bp_s(count);
        //TRW_S(count);

        //FindOptimalSolutionSequential();
        computeBelief();
        findOptimalSolution();

        energy = getEnergy();

        /*
        if (IsDisplay)
            printf("No. %d energy: %f...\n", count, energy);
        if (pEnergyList != NULL)
            pEnergyList[count] = energy
        */
    }
    return energy;
}

template <typename InputType>
bool BPFlow<InputType>::isInsideImage(ptrdiff_t x, ptrdiff_t y)
{
    return (x >= 0 && x < Width2 && y >= 0 && y < Height2);
}

//void bipartite(int count);

template <typename InputType>
void BPFlow<InputType>::bp_s(int count)
{
    int k = count % 2;
    if (count % 4 < 2) // forward update
    {
        for (int i = 0; i<Height; i++)
            for (int j = 0; j<Width; j++)
            {
                updateSpatialMessage(j, i, k, 0);
                updateSpatialMessage(j, i, k, 2);
                if (count % 8<4)
                    updateDualMessage(j, i, k);
            }
    }     
    else // backward update
    {
        for (int i = Height - 1; i >= 0; i--)
            for (int j = Width - 1; j >= 0; j--)
            {
                updateSpatialMessage(j, i, k, 1);
                updateSpatialMessage(j, i, k, 3);
                if (count % 8<4)
                    updateDualMessage(j, i, k);
            }
    } 
}

template <typename InputType>
void BPFlow<InputType>::trw_s(int count)
{
    int k = count % 2;
    if (k == 0) // forward update
    {
        for (int i = 0; i<Height; i++)
            for (int j = 0; j<Width; j++)
            {
                for (int l = 0; l<2; l++)
                {
                    updateDualMessage(j, i, l);
                    updateSpatialMessage(j, i, l, 0);
                    updateSpatialMessage(j, i, l, 2);
                }
            }
    }
    else // backward update
    {
        for (int i = Height - 1; i >= 0; i--)
            for (int j = Width - 1; j >= 0; j--)
            {
                for (int l = 0; l<2; l++)
                {
                    updateDualMessage(j, i, l);
                    updateSpatialMessage(j, i, l, 1);
                    updateSpatialMessage(j, i, l, 3);
                }
            }
    }
       
}

template<class T>
void add2Message(T* message, const T* other, int nstates)
{
    for (size_t i = 0; i<nstates; i++)
        message[i] += other[i];
}

template<class T>
void add2Message(T* message, const T* other, int nstates, double Coeff)
{
    for (size_t i = 0; i<nstates; i++)
        message[i] += other[i] * Coeff;
}

template <class T>
T min(int NumData, T* pData)
{
    int i;
    T result = pData[0];
    for (i = 1; i<NumData; i++)
        result = __min(result, pData[i]);
    return result;
}

template <class T>
T min(int NumData, T* pData1, T* pData2)
{
    int i;
    T result = pData1[0] + pData2[0];
    for (i = 1; i<NumData; i++)
        result = __min(result, pData1[i] + pData2[i]);
    return result;
}

template <class T>
T max(int NumData, T* pData)
{
    int i;
    T result = pData[0];
    for (i = 1; i<NumData; i++)
        result = __max(result, pData[i]);
    return result;
}

//------------------------------------------------------------------------------------------------
//  update the message from (x0,y0,plane) to the neighbors on the same plane
//    the encoding of the direction
//               2  |
//                   v
//    0 ------> <------- 1
//                   ^
//                3 |
//------------------------------------------------------------------------------------------------
template <typename InputType>
void BPFlow<InputType>::updateSpatialMessage(int x, int y, int plane, int direction)
{
    // eliminate impossible messages
    if (direction == 0 && x == Width - 1)
        return;
    if (direction == 1 && x == 0)
        return;
    if (direction == 2 && y == Height - 1)
        return;
    if (direction == 3 && y == 0)
        return;

    int offset = y*Width + x;
    int nStates = pWinSize[plane][offset] * 2 + 1;



    T_message* message_org;
    message_org = new T_message[nStates];

    int x1 = x, y1 = y; // get the destination
    switch (direction){
    case 0:
        x1++;
        ls = ((double*)im_s.data)[offset * 2 + plane];
        ld = ((double*)im_d.data)[offset * 2 + plane];
        break;
    case 1:
        x1--;
        ls = ((double*)im_s.data)[(offset - 1) * 2 + plane];
        ld = ((double*)im_d.data)[(offset - 1) * 2 + plane];
        break;
    case 2:
        y1++;
        ls = ((double*)im_s.data)[offset * 2 + plane];
        ld = ((double*)im_d.data)[offset * 2 + plane];
        break;
    case 3:
        y1--;
        ls = ((double*)im_s.data)[(offset - Width) * 2 + plane];
        ld = ((double*)im_d.data)[(offset - Width) * 2 + plane];
        break;
    }
    //s=m_s;
    //d=m_d;
    int offset1 = y1*Width + x1;
    int nStates1 = pWinSize[plane][offset1] * 2 + 1; // get the number of states for the destination node
    int wsize = pWinSize[plane][offset];
    int wsize1 = pWinSize[plane][offset1];

    T_message*& message = pSpatialMessage[plane][offset1*nNeighbors + direction].data();

    // initialize the message from the dual plane
    if (!IsTRW)
        memcpy(message_org, pDualMessage[plane][offset].data(), sizeof(T_message)*nStates);
    else
    {
        memset(message_org, 0, sizeof(T_message)*nStates);
        add2Message(message_org, pDualMessage[plane][offset].data(), nStates, CTRW);
    }

    // add the range term
    if (!IsTRW)
        add2Message(message_org, pRangeTerm[plane][offset].data(), nStates);
    else
        add2Message(message_org, pRangeTerm[plane][offset].data(), nStates, CTRW);

    // add spatial messages
    if (!IsTRW)
    {
        if (x>0 && direction != 1) // add left to right
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors].data(), nStates);
        if (x<Width - 1 && direction != 0) // add right to left 
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates);
        if (y>0 && direction != 3) // add top down
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates);
        if (y<Height - 1 && direction != 2) // add bottom up
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 3].data(), nStates);
    }
    else
    {
        if (x>0) // add left to right
            if (direction == 1)
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors].data(), nStates, CTRW - 1);
            else
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors].data(), nStates, CTRW);
        if (x<Width - 1) // add right to left 
            if (direction == 0)
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates, CTRW - 1);
            else
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates, CTRW);
        if (y>0) // add top down
            if (direction == 3)
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates, CTRW - 1);
            else
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates, CTRW);
        if (y<Height - 1) // add bottom up
            if (direction == 2)
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 3].data(), nStates, CTRW - 1);
            else
                add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 3].data(), nStates, CTRW);
    }
    // use distance transform function to impose smoothness compatibility
    T_message Min = min(nStates, message_org) + d;
    for (ptrdiff_t l = 1; l<nStates; l++)
        message_org[l] = __min(message_org[l], message_org[l - 1] + ls);
    for (ptrdiff_t l = nStates - 2; l >= 0; l--)
        message_org[l] = __min(message_org[l], message_org[l + 1] + ls);


    // transform the compatibility 
    int shift = pOffset[plane][offset1] - pOffset[plane][offset];
    if (abs(shift)>wsize + wsize1) // the shift is too big that there is no overlap
    {
        if (offset>0)
            for (ptrdiff_t l = 0; l<nStates1; l++)
                message[l] = l*ls;
        else
            for (ptrdiff_t l = 0; l<nStates1; l++)
                message[l] = -l*ls;
    }
    else
    {
        int start = __max(-wsize, shift - wsize1);
        int end = __min(wsize, shift + wsize1);
        for (ptrdiff_t i = start; i <= end; i++)
            message[i - shift + wsize1] = message_org[i + wsize];
        if (start - shift + wsize1>0)
            for (ptrdiff_t i = start - shift + wsize1 - 1; i >= 0; i--)
                message[i] = message[i + 1] + ls;
        if (end - shift + wsize1<nStates1)
            for (ptrdiff_t i = end - shift + wsize1 + 1; i<nStates1; i++)
                message[i] = message[i - 1] + ls;
    }

    // put back the threshold
    for (ptrdiff_t l = 0; l<nStates1; l++)
        message[l] = __min(message[l], Min);

    // normalize the message by subtracting the minimum value
    Min = min(nStates1, message);
    for (ptrdiff_t l = 0; l<nStates1; l++)
        message[l] -= Min;

    delete message_org;
}

//------------------------------------------------------------------------------------------------
// update dual message passing from one plane to the other
//------------------------------------------------------------------------------------------------
template <typename InputType>
void BPFlow<InputType>::updateDualMessage(int x, int y, int plane)
{
    int offset = y*Width + x;
    int offset1 = offset;
    int wsize = pWinSize[plane][offset];
    int nStates = wsize * 2 + 1;
    int wsize1 = pWinSize[1 - plane][offset];
    int nStates1 = wsize1 * 2 + 1;

    ls = ((double*)im_s.data)[offset * 2 + plane];
    ld = ((double*)im_d.data)[offset * 2 + plane];
    //s=m_s;
    //d=m_d;

    T_message* message_org;
    message_org = new T_message[nStates];
    memset(message_org, 0, sizeof(T_message)*nStates);

    // add the range term
    if (!IsTRW)
        add2Message(message_org, pRangeTerm[plane][offset].data(), nStates);
    else
        add2Message(message_org, pRangeTerm[plane][offset].data(), nStates, CTRW);

    // add spatial messages
    if (x>0)  //add left to right
    {
        if (!IsTRW)
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors].data(), nStates);
        else
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors].data(), nStates, CTRW);
    }
    if (x<Width - 1) // add right to left
    {
        if (!IsTRW)
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates);
        else
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates, CTRW);
    }
    if (y>0) // add top down
    {
        if (!IsTRW)
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates);
        else
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates, CTRW);
    }
    if (y<Height - 1) // add bottom up
    {
        if (!IsTRW)
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 3].data(), nStates);
        else
            add2Message(message_org, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates, CTRW);
    }

    if (IsTRW)
        add2Message(message_org, pDualMessage[plane][offset1].data(), nStates, CTRW - 1);

    T_message*& message = pDualMessage[1 - plane][offset1].data();

    T_message Min;
    // use the data term
    if (plane == 0) // from vx plane to vy plane
        for (size_t l = 0; l<nStates1; l++)
            message[l] = min(nStates, pDataTerm[offset].data() + l*nStates, message_org);
    else					// from vy plane to vx plane
        for (size_t l = 0; l<nStates1; l++)
        {
            Min = message_org[0] + pDataTerm[offset].data()[l];
            for (size_t h = 0; h<nStates; h++)
                Min = __min(Min, message_org[h] + pDataTerm[offset].data()[h*nStates1 + l]);
            message[l] = Min;
        }

    // normalize the message
    Min = min(nStates1, message);
    for (size_t l = 0; l<nStates; l++)
        message[l] -= Min;

    delete message_org;
}

template <typename InputType>
void BPFlow<InputType>::computeBelief()
{
    for (size_t plane = 0; plane<2; plane++)
    {
        memset(ptrBelief[plane], 0, sizeof(T_message)*nTotalBelifElements[plane]);
        for (size_t i = 0; i<Height; i++)
            for (size_t j = 0; j<Width; j++)
            {
                size_t offset = i*Width + j;
                T_message* belief = pBelief[plane][offset].data();
                int nStates = pWinSize[plane][offset] * 2 + 1;
                // add range term
                add2Message(belief, pRangeTerm[plane][offset].data(), nStates);
                // add message from the dual layer
                add2Message(belief, pDualMessage[plane][offset].data(), nStates);
                if (j>0)
                    add2Message(belief, pSpatialMessage[plane][offset*nNeighbors].data(), nStates);
                if (j<Width - 1)
                    add2Message(belief, pSpatialMessage[plane][offset*nNeighbors + 1].data(), nStates);
                if (i>0)
                    add2Message(belief, pSpatialMessage[plane][offset*nNeighbors + 2].data(), nStates);
                if (i<Height - 1)
                    add2Message(belief, pSpatialMessage[plane][offset*nNeighbors + 3].data(), nStates);
            }
    }
}

template <typename InputType>
void BPFlow<InputType>::findOptimalSolution()
{
    for (size_t plane = 0; plane<2; plane++)
        for (size_t i = 0; i<Area; i++)
        {
            int nStates = pWinSize[plane][i] * 2 + 1;
            double Min;
            int index = 0;
            T_message* belief = pBelief[plane][i].data();
            Min = belief[0];
            for (int l = 1; l<nStates; l++)
                if (Min>belief[l])
                {
                    Min = belief[l];
                    index = l;
                }
            pX[i * 2 + plane] = index;
        }
}

template <typename InputType>
void BPFlow<InputType>::computeVelocity(Mat& flow)
{
    //mFlow.allocate(Width, Height, 2);
    float* flow_data = (float*)flow.data;
    for (int i = 0; i<Area; i++)
    {
        flow_data[i * 2] = pX[i * 2] + pOffset[0][i] - pWinSize[0][i];
        flow_data[i * 2 + 1] = pX[i * 2 + 1] + pOffset[1][i] - pWinSize[1][i];
    }
}

template <typename InputType>
double BPFlow<InputType>::getEnergy()
{
    double energy = 0;
    for (size_t i = 0; i<Height; i++)
        for (size_t j = 0; j<Width; j++)
        {
            size_t offset = i*Width + j;
            for (size_t k = 0; k<2; k++)
            {
                if (j<Width - 1)
                {
                    ls = ((double*)im_s.data)[offset * 2 + k];
                    ld = ((double*)im_d.data)[offset * 2 + k];
                    //s=m_s;
                    //d=m_d;
                    energy += __min((double)abs(pX[offset * 2 + k] - pWinSize[k][offset] + pOffset[k][offset] - pX[(offset + 1) * 2 + k] + pWinSize[k][offset + 1] - pOffset[k][offset + 1])*ls, ld);
                }
                if (i<Height - 1)
                {
                    ls = ((double*)im_s.data)[offset * 2 + k];
                    ld = ((double*)im_d.data)[offset * 2 + k];
                    //s=m_s;
                    //d=m_d;
                    energy += __min((double)abs(pX[offset * 2 + k] - pWinSize[k][offset] + pOffset[k][offset] - pX[(offset + Width) * 2 + k] + pWinSize[k][offset + Width] - pOffset[k][offset + Width])*ls, ld);
                }
            }
            int vx = pX[offset * 2];
            int vy = pX[offset * 2 + 1];
            int nStates = pWinSize[0][offset] * 2 + 1;
            energy += pDataTerm[offset].data()[vy*nStates + vx];
            for (size_t k = 0; k<2; k++)
                energy += pRangeTerm[k][offset].data()[pX[offset * 2 + k]];
        }
    return energy;
}

//------------------------------------------------------------------------------------------------
// function to set the homogeneous MRF parameters
// There is no offset, and the window size is identical for each pixel (winSize)
//------------------------------------------------------------------------------------------------
template <typename InputType>
void BPFlow<InputType>::setHomogeneousMRF(int winSize)
{
    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pOffset[i]); // release the buffer of the offset
        release1DBuffer(pWinSize[i]); // release the buffer of the size
        pOffset[i] = new T_state[Area];
        memset(pOffset[i], 0, sizeof(T_state)*Area);

        pWinSize[i] = new T_state[Area];
        for (size_t j = 0; j<Area; j++)
            pWinSize[i][j] = winSize;//+CStochastic::UniformSampling(3)-1;
    }
   
    // add some disturbance
    for (int i = 0; i<2; i++)
        for (int j = 0; j<Area; j++)
            pOffset[i][j] = rng.uniform(0, 5) - 2;
}

template <typename InputType>
void BPFlow<InputType>::setOffset(const Mat& offsetX, const Mat& offsetY)
{
    if (offsetX.empty() || offsetY.empty()) return;

    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pOffset[i]);
        pOffset[i] = new T_state[Area];
    }

    int* offsetX_data = (int*)offsetX.data;
    int* offsetY_data = (int*)offsetY.data;
    for (size_t j = 0; j<Area; j++)
    {
        pOffset[0][j] = offsetX_data[j];
        pOffset[1][j] = offsetY_data[j];
    }
}

template <typename InputType>
void BPFlow<InputType>::setWinSize(const Mat& winSizeX, const Mat& winSizeY)
{
    for (int i = 0; i<2; i++)
    {
        release1DBuffer(pWinSize[i]);
        pWinSize[i] = new T_state[Area];
    }

    int* winSizeX_data = (int*)winSizeX.data;
    int* winSizeY_data = (int*)winSizeY.data;
    for (size_t j = 0; j < Area; j++)
    {
        pWinSize[0][j] = winSizeX_data[j];
        pWinSize[1][j] = winSizeY_data[j];
    }
}

//------------------------------------------------------------------------
// multi-grid belief propagation
//------------------------------------------------------------------------

template<typename T>
void reduceImage(T* pDstData, int width, int height, const T *pSrcData)
{
    int DstWidth = width / 2;
    if (width % 2 == 1)
        DstWidth++;
    int DstHeight = height / 2;
    if (height % 2 == 1)
        DstHeight++;
    memset(pDstData, 0, sizeof(T)*DstWidth*DstHeight);
    int sum = 0;
    for (int i = 0; i<DstHeight; i++)
        for (int j = 0; j<DstWidth; j++)
        {
            int offset = i*DstWidth + j;
            sum = 0;
            for (int ii = 0; ii<2; ii++)
                for (int jj = 0; jj<2; jj++)
                {
                    int x = j * 2 + jj;
                    int y = i * 2 + ii;
                    if (y<height && x<width)
                    {
                        pDstData[offset] += pSrcData[y*width + x];
                        sum++;
                    }
                }
            pDstData[offset] /= sum;
        }
}

template <typename InputType>
void BPFlow<InputType>::generateCoarserLevel(BPFlow<InputType>& bp)
{
    //------------------------------------------------------------------------------------------------
    // set the dimensions and parameters
    //------------------------------------------------------------------------------------------------
    bp.Width = Width / 2;
    if (Width % 2 == 1)
        bp.Width++;

    bp.Height = Height / 2;
    if (Height % 2 == 1)
        bp.Height++;

    bp.Area = bp.Width*bp.Height;
    bp.ls = ls;
    bp.ld = ld;

    // Resize image
    //DImage foo;
    //Im_s.smoothing(foo);
    //foo.imresize(bp.Im_s, bp.Width, bp.Height);
    //Im_d.smoothing(foo);
    //foo.imresize(bp.Im_d, bp.Width, bp.Height);
    std::vector<Mat> pyrd_s = buildPyramid(im_s, 2);
    std::vector<Mat> pyrd_d = buildPyramid(im_d, 2);
    bp.im_s = pyrd_s[1];
    bp.im_d = pyrd_d[1];

    bp.IsDisplay = IsDisplay;
    bp.nNeighbors = nNeighbors;

    //------------------------------------------------------------------------------------------------
    // allocate buffers
    //------------------------------------------------------------------------------------------------
    for (int i = 0; i<2; i++)
    {
        bp.pOffset[i] = new int[bp.Area];
        bp.pWinSize[i] = new int[bp.Area];
        reduceImage(bp.pOffset[i], Width, Height, pOffset[i]);
        reduceImage(bp.pWinSize[i], Width, Height, pWinSize[i]);
    }
    //------------------------------------------------------------------------------------------------
    // generate data term
    //------------------------------------------------------------------------------------------------
    bp.nTotalMatches = allocateBuffer(bp.pDataTerm, bp.ptrDataTerm, Area, bp.pWinSize[0], bp.pWinSize[1]);
    for (int i = 0; i<bp.Height; i++)
        for (int j = 0; j<bp.Width; j++)
        {
            int offset = i*bp.Width + j;
            for (int ii = 0; ii<2; ii++)
                for (int jj = 0; jj<2; jj++)
                {
                    int y = i * 2 + ii;
                    int x = j * 2 + jj;
                    if (y<Height && x<Width)
                    {
                        int nStates = (bp.pWinSize[0][offset] * 2 + 1)*(bp.pWinSize[1][offset] * 2 + 1);
                        for (int k = 0; k<nStates; k++)
                            bp.pDataTerm[offset].data()[k] += pDataTerm[y*Width + x].data()[k];
                    }
                }
        }
    //------------------------------------------------------------------------------------------------
    // generate range term
    //------------------------------------------------------------------------------------------------
    bp.computeRangeTerm(gamma / 2);
}

template <typename InputType>
void BPFlow<InputType>::propagateFinerLevel(BPFlow<InputType>& bp)
{
    for (int i = 0; i<bp.Height; i++)
        for (int j = 0; j<bp.Width; j++)
        {
            int y = i / 2;
            int x = j / 2;
            int nStates1 = pWinSize[0][y*Width + x] * 2 + 1;
            int nStates2 = pWinSize[1][y*Width + x] * 2 + 1;
            for (int k = 0; k<2; k++)
            {
                memcpy(bp.pDualMessage[k][i*bp.Width + j].data(), pDualMessage[k][y*Width + x].data(), sizeof(T_message)*(pWinSize[k][y*Width + x] * 2 + 1));
                for (int l = 0; l<nNeighbors; l++)
                    memcpy(bp.pSpatialMessage[k][(i*bp.Width + j)*nNeighbors + l].data(), pSpatialMessage[k][(y*Width + x)*nNeighbors + l].data(), sizeof(T_message)*(pWinSize[k][y*Width + x] * 2 + 1));
            }
        }
}



}//optflow
}//cv
