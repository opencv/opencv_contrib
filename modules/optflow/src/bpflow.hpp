/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


License Agreement
For Open Source Computer Vision Library
(3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_BPFLOW_HPP__
#define __OPENCV_BPFLOW_HPP__

#include "precomp.hpp"
#include <opencv2/highgui.hpp>

namespace cv
{
    namespace optflow
    {

        //typedef int T_state;					// T_state is the type for state
        //typedef float T_input;			// T_input is the data type of the input image
#ifdef INTMESSAGE
        typedef int T_message;
#else
        typedef double T_message;
#endif

        //-----------------------------------------------------------------------------------
        // class for 1D pixel buffer
        //-----------------------------------------------------------------------------------
        template <typename T>
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
        template <typename T>
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

        template <typename InputType>
        class BPFlow : public DenseOpticalFlow
        {
        public:
            BPFlow();
            ~BPFlow();

            void release();

            void calc(InputArray I0, InputArray I1, InputOutputArray flow);
            void collectGarbage();

        protected:
            float sigma;        //!< Gaussian smoothing parameter

            // Parameters
            double d;			//!< Smoothness term threshold
            int levels;		    //!< The number of pyramid levels
            double alpha;		//!< Smoothness term weight
            double gamma;		//!< Range term weight
            int iterations;	    //!< The number of iterations for each inner pyramid level
            int topiterations;	//!< The number of iterations for the top pyramid level
            int wsize;			//!< The window size for each inner pyramid level
            int topwsize;		//!< The window size for the top pyramid level
            //int hierarchy;      //!< The number of levels for the multi-grid bp

        protected:
            bool IsDisplay;
            bool IsDataTermTruncated;
            bool IsTRW;
            double CTRW;

            size_t Height, Width, Area, nChannels;
            size_t Height2, Width2;

            InputType *pIm1, *pIm2;   // the two images for matching

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
            int level_hierarchy;

            Mat im_s, im_d;

            RNG rng;    //!< Random number generator

        private:
            void calcOneLevel(const Mat I0, const Mat I1, Mat W,
                int _iterations, int winsize, int hierarchy,
                const cv::Mat& offsetX, const cv::Mat& offsetY);
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
            void setHomogeneousMRF(int winSize);
            void setOffset(const Mat& offsetX, const Mat& offsetY);
            void setWinSize(const Mat& winSizeX, const Mat& winSizeY);
            std::vector<Mat> buildPyramid(const Mat& src, int _levels);
            void downOffset(cv::Mat& offset, int width, int height);

            //------------------------------------------------------------------------
            // multi-grid belief propagation
            //------------------------------------------------------------------------

            void generateCoarserLevel(BPFlow& bp);
            void propagateFinerLevel(BPFlow& bp);
        };

    } // optflow
}

#include "bpflow.inl.hpp"

#endif  // __OPENCV_BPFLOW_HPP__