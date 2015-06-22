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

//-----------------------------------------------------------------------------------
// class for 1D pixel buffer
//-----------------------------------------------------------------------------------
template <class T>
class PixelBuffer1D
{
private:
    short int nDim;
    T* pData;
public:
    PixelBuffer1D(void)
    {
        nDim = 0;
        pData = NULL;
    }
    PixelBuffer1D(int ndims)
    {
        allocate(ndims);
    }
    void allocate(int ndims)
    {
        nDim = ndims;
    }
    ~PixelBuffer1D()
    {
        nDim = 0;
        pData = NULL;
    }
    inline const T operator [](int index) const
    {
        return pData[index];
    }
    inline T& operator [](int index)
    {
        return pData[index];
    }
    T*& data(){ return pData; };
    const T* data() const{ return pData; };
    int nElements() const{ return nDim; };
};

//-----------------------------------------------------------------------------------
// class for 2D pixel buffer
//-----------------------------------------------------------------------------------
template <class T>
class PixelBuffer2D
{
private:
    short int nDimX, nDimY;
    T* pData;
public:
    PixelBuffer2D(void)
    {
        nDimX = nDimY = 0;
        pData = NULL;
    }
    PixelBuffer2D(int ndimx, int ndimy)
    {
        allocate(ndimx, ndimy);
    }
    void allocate(int ndimx, int ndimy)
    {
        nDimX = ndimx;
        nDimY = ndimy;
        pData = NULL;
    }
    ~PixelBuffer2D()
    {
        nDimX = nDimY = 0;
        pData = NULL;
    }
    inline const T operator [](int index) const
    {
        return pData[index];
    }
    inline T& operator [](int index)
    {
        return pData[index];
    }
    T*& data(){ return pData; };
    const T* data() const{ return pData; };
    int nElements()const{ return nDimX*nDimY; };
};

//------------------------------------------------------------------------------------------------
//	function to allocate buffer for the messages
//------------------------------------------------------------------------------------------------
template <class T1, class T2>
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

template<class T1, class T2>
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

template <class T>
void release1DBuffer(T* pBuffer)
{
    if (pBuffer != NULL)
        delete[]pBuffer;
    pBuffer = NULL;
}

template <class T>
void release2DBuffer(T** pBuffer, size_t nElements)
{
    for (size_t i = 0; i<nElements; i++)
        delete[](pBuffer[i]);
    delete[]pBuffer;
    pBuffer = NULL;
}

class OpticalFlowSiftFlow : public DenseOpticalFlow
{
public:
    OpticalFlowSiftFlow();

    void calc(InputArray I0, InputArray I1, InputOutputArray flow);
    void collectGarbage();

protected:
	float sigma;        //!< Gaussian smoothing parameter

    // Parameters
    double d;			//!< Smoothness term threshold
    int levels;		//!< The number of pyramid levels
    double alpha;		//!< Smoothness term weight
    double gamma;		//!< Range term weight
    int iterations;	    //!< The number of iterations for each inner pyramid level
    int topiterations;	//!< The number of iterations for the top pyramid level
    int wsize;			//!< The window size for each inner pyramid level
    int topwsize;		//!< The window size for the top pyramid level

protected:
    bool IsDisplay;
    bool IsDataTermTruncated;
    bool IsTRW;
    double CTRW;

    size_t Height, Width, Area, nChannels;
    size_t Height2, Width2;

    T_input *pIm1, *pIm2;   // the two images for matching

    int *pOffset[2]; // the predicted flow 
    int *pWinSize[2];// the dimension of the matching size
    size_t nTotalMatches;

    // the buffers for belief propagation
    PixelBuffer2D<T_message>* pDataTerm;					// the data term
    PixelBuffer1D<T_message>* pRangeTerm[2];               // the range term
    PixelBuffer1D<T_message>* pSpatialMessage[2];      // the spatial message
    PixelBuffer1D<T_message>* pDualMessage[2];            // the dual message between two layers
    PixelBuffer1D<T_message>* pBelief[2];							// the belief

    T_message *ptrDataTerm;
    T_message *ptrRangeTerm[2];
    T_message* ptrSpatialMessage[2];
    T_message *ptrDualMessage[2];
    T_message* ptrBelief[2];

    size_t nTotalSpatialElements[2];
    size_t nTotalDualElements[2];
    size_t nTotalBelifElements[2];

    int *pX; // the final states

    int nNeighbors;
    double ls, ld, level_gamma; // the parameters of regularization

    Mat im_s, im_d;

private:
    void calcOneLevel(const Mat I0, const Mat I1, Mat W);
    void computeDataTerm();
    void computeRangeTerm(double _gamma);
    void allocateMessage();
    double messagePassing(int _iterations, int _hierarchy);
    bool isInsideImage(ptrdiff_t x, ptrdiff_t y);
    void bipartite(int count);
    void bp_s(int count);
    void trw_s(int count);
    void updateSpatialMessage(int x, int y, int plane, int direction);
    void updateDualMessage(int x, int y, int plane);
    void computeBelief();
    void findOptimalSolution();
    void computeVelocity(Mat& flow);
    double getEnergy();
    std::vector<Mat> buildPyramid(const Mat& src);
};

OpticalFlowSiftFlow::OpticalFlowSiftFlow()
{
    
}

void OpticalFlowSiftFlow::calc(InputArray _I0, InputArray _I1, InputOutputArray _flow)
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

    calcOneLevel(I0, I1, W);

    // Output flow
    W.copyTo(_flow);
}

void OpticalFlowSiftFlow::collectGarbage()
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


Ptr<DenseOpticalFlow> createOptFlow_SiftFlow()
{
    return makePtr<OpticalFlowSiftFlow>();
}

std::vector<Mat> OpticalFlowSiftFlow::buildPyramid(const Mat& src)
{
    std::vector<Mat> pyramid;
    pyramid.push_back(src);
    Mat tmp;
    int w, h;

    // For each pyramid level starting from level 1
    for (size_t i = 1; i < levels; ++i)
    {
        // Apply gaussian filter
        GaussianBlur(pyramid[i - 1], tmp, Size(5, 5), 0.67, 0, BORDER_REPLICATE);

        // Resize image using bicubic interpolation
        w = (int)ceil(pyramid[i - 1].cols / 2.0f);
        h = (int)ceil(pyramid[i - 1].rows / 2.0f);
        resize(tmp, pyramid[i], Size(w, h), 0, 0, INTER_CUBIC);
    }

    return pyramid;
}

void OpticalFlowSiftFlow::calcOneLevel(const Mat I0, const Mat I1, Mat W)
{
    computeDataTerm();
    computeRangeTerm(gamma);
    messagePassing(iterations, 2);
    computeVelocity(W);
}

void OpticalFlowSiftFlow::computeDataTerm()
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
                    for (int n = 0; n<nChannels; n++)
                        foo += abs(pIm1[index*nChannels + n] - pIm2[index2*nChannels + n]); // L1 norm
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

void OpticalFlowSiftFlow::computeRangeTerm(double _gamma)
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

void OpticalFlowSiftFlow::allocateMessage()
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

double OpticalFlowSiftFlow::messagePassing(int _iterations, int _hierarchy)
{
    allocateMessage();
    /*
    if (_hierarchy > 0)
    {
        BPFlow bp;
        generateCoarserLevel(bp);
        bp.MessagePassing(20, nHierarchy - 1);
        bp.propagateFinerLevel(*this);
    }
    */
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

bool OpticalFlowSiftFlow::isInsideImage(ptrdiff_t x, ptrdiff_t y)
{
    return (x >= 0 && x < Width2 && y >= 0 && y < Height2);
}

void bipartite(int count);

void OpticalFlowSiftFlow::bp_s(int count)
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

void OpticalFlowSiftFlow::trw_s(int count)
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
void OpticalFlowSiftFlow::updateSpatialMessage(int x, int y, int plane, int direction)
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
void OpticalFlowSiftFlow::updateDualMessage(int x, int y, int plane)
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

void OpticalFlowSiftFlow::computeBelief()
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

void OpticalFlowSiftFlow::findOptimalSolution()
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

void OpticalFlowSiftFlow::computeVelocity(Mat& flow)
{
    //mFlow.allocate(Width, Height, 2);
    float* flow_data = (float*)flow.data;
    for (int i = 0; i<Area; i++)
    {
        flow_data[i * 2] = pX[i * 2] + pOffset[0][i] - pWinSize[0][i];
        flow_data[i * 2 + 1] = pX[i * 2 + 1] + pOffset[1][i] - pWinSize[1][i];
    }
}

double OpticalFlowSiftFlow::getEnergy()
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

}//optflow
}//cv
