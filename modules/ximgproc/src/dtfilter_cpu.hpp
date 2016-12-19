/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *  
 *  
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *  
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *  
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *  
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *  
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

#ifndef __OPENCV_DTFILTER_HPP__
#define __OPENCV_DTFILTER_HPP__
#include "precomp.hpp"

#ifdef _MSC_VER
#pragma warning(disable: 4512)
#pragma warning(disable: 4127)
#endif

#define CV_GET_NUM_THREAD_WORKS_PROPERLY
#undef CV_GET_NUM_THREAD_WORKS_PROPERLY

namespace cv
{
namespace ximgproc
{

class DTFilterCPU : public DTFilter
{
public: /*Non-template methods*/

    static Ptr<DTFilterCPU> create(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    static Ptr<DTFilterCPU> createRF(InputArray adistHor, InputArray adistVert, double sigmaSpatial, double sigmaColor, int numIters = 3);

    void filter(InputArray src, OutputArray dst, int dDepth = -1);

    void setSingleFilterCall(bool value);

public: /*Template methods*/

    /*Use this static methods instead of constructor*/
    template<typename GuideVec>
    static DTFilterCPU* create_p_(const Mat& guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    template<typename GuideVec>
    static DTFilterCPU create_(const Mat& guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    template<typename GuideVec>
    void init_(Mat& guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    template<typename SrcVec>
    void filter_(const Mat& src, Mat& dst, int dDepth = -1);

protected: /*Typedefs declarations*/

    typedef float                   IDistType;
    typedef Vec<IDistType, 1>       IDistVec;

    typedef float                   DistType;
    typedef Vec<DistType, 1>        DistVec;

    typedef float                   WorkType;

public: /*Members declarations*/

    int h, w, mode;
    float sigmaSpatial, sigmaColor;

    bool singleFilterCall;
    int numFilterCalls;

    Mat idistHor, idistVert;
    Mat distHor, distVert;

    Mat a0distHor, a0distVert;
    Mat adistHor, adistVert;
    int numIters;

protected: /*Functions declarations*/

    DTFilterCPU() : mode(-1), singleFilterCall(false), numFilterCalls(0) {}

    void init(InputArray guide, double sigmaSpatial, double sigmaColor, int mode = DTF_NC, int numIters = 3);

    void release();

    template<typename GuideVec>
    inline IDistType getTransformedDistance(const GuideVec &l, const GuideVec &r)
    {
        return (IDistType)(1.0f + sigmaSpatial / sigmaColor * norm1<IDistType>(l, r));
    }

    inline double getIterSigmaH(int iterNum)
    {
        return sigmaSpatial * std::pow(2.0, numIters - iterNum) / sqrt(std::pow(4.0, numIters) - 1);
    }

    inline IDistType getIterRadius(int iterNum)
    {
        return (IDistType)(3.0*getIterSigmaH(iterNum));
    }

    inline float getIterAlpha(int iterNum)
    {
        return (float)std::exp(-std::sqrt(2.0 / 3.0) / getIterSigmaH(iterNum));
    }

protected: /*Wrappers for parallelization*/

    template <typename WorkVec>
    struct FilterNC_horPass : public ParallelLoopBody
    {
        Mat &src, &idist, &dst;
        float radius;

        FilterNC_horPass(Mat& src_, Mat& idist_, Mat& dst_);
        void operator() (const Range& range) const;
    };

    template <typename WorkVec>
    struct FilterIC_horPass : public ParallelLoopBody
    {
        Mat &src, &idist, &dist, &dst, isrcBuf;
        float radius;

        FilterIC_horPass(Mat& src_, Mat& idist_, Mat& dist_, Mat& dst_);
        void operator() (const Range& range) const;
    };

    template <typename WorkVec>
    struct FilterRF_horPass : public ParallelLoopBody
    {
        Mat &res, &alphaD;
        int iteration;

        FilterRF_horPass(Mat& res_, Mat& alphaD_, int iteration_);
        void operator() (const Range& range) const;
        Range getRange() const { return Range(0, res.rows); }
    };

    template <typename WorkVec>
    struct FilterRF_vertPass : public ParallelLoopBody
    {
        Mat &res, &alphaD;
        int iteration;

        FilterRF_vertPass(Mat& res_, Mat& alphaD_, int iteration_);
        void operator() (const Range& range) const;
        #ifdef CV_GET_NUM_THREAD_WORKS_PROPERLY
        Range getRange() const { return Range(0, cv::getNumThreads()); }
        #else
        Range getRange() const { return Range(0, res.cols); }
        #endif
    };

    template <typename GuideVec>
    struct ComputeIDTHor_ParBody: public ParallelLoopBody
    {
        DTFilterCPU &dtf;
        Mat &guide, &dst;

        ComputeIDTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_, Mat& dst_);
        void operator() (const Range& range) const;
        Range getRange() { return Range(0, guide.rows); }
    };

    template <typename GuideVec>
    struct ComputeDTandIDTHor_ParBody : public ParallelLoopBody
    {
        DTFilterCPU &dtf;
        Mat &guide, &dist, &idist;
        IDistType maxRadius;

        ComputeDTandIDTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_, Mat& dist_, Mat& idist_);
        void operator() (const Range& range) const;
        Range getRange() { return Range(0, guide.rows); }
    };

    template <typename GuideVec>
    struct ComputeA0DTHor_ParBody : public ParallelLoopBody
    {
        DTFilterCPU &dtf;
        Mat &guide;
        float lna;

        ComputeA0DTHor_ParBody(DTFilterCPU& dtf_, Mat& guide_);
        void operator() (const Range& range) const;
        Range getRange() { return Range(0, guide.rows); }
        ~ComputeA0DTHor_ParBody();
    };

    template <typename GuideVec>
    struct ComputeA0DTVert_ParBody : public ParallelLoopBody
    {
        DTFilterCPU &dtf;
        Mat &guide;
        float lna;

        ComputeA0DTVert_ParBody(DTFilterCPU& dtf_, Mat& guide_);
        void operator() (const Range& range) const;
        Range getRange() const { return Range(0, guide.rows - 1); }
        ~ComputeA0DTVert_ParBody();
    };

protected: /*Auxiliary implementation functions*/

    static Range getWorkRangeByThread(const Range& itemsRange, const Range& rangeThread, int maxThreads = 0);
    static Range getWorkRangeByThread(int items, const Range& rangeThread, int maxThreads = 0);

    template<typename SrcVec>
    static void prepareSrcImg_IC(const Mat& src, Mat& inner, Mat& outer);

    static Mat getWExtendedMat(int h, int w, int type, int brdleft = 0, int brdRight = 0, int cacheAlign = 0);

    template<typename SrcVec, typename SrcWorkVec>
    static void integrateSparseRow(const SrcVec *src, const float *dist, SrcWorkVec *dst, int cols);

    template<typename SrcVec, typename SrcWorkVec>
    static void integrateRow(const SrcVec *src, SrcWorkVec *dst, int cols);

    inline static int getLeftBound(IDistType *idist, int pos, IDistType searchValue)
    {
        while (idist[pos] < searchValue)
            pos++;
        return pos;
    }

    inline static int getRightBound(IDistType *idist, int pos, IDistType searchValue)
    {
        while (idist[pos + 1] < searchValue)
            pos++;
        return pos;
    }

    template <typename T, typename T1, typename T2, int n>
    inline static T norm1(const cv::Vec<T1, n>& v1, const cv::Vec<T2, n>& v2)
    {
        T sum = (T) 0;
        for (int i = 0; i < n; i++)
            sum += std::abs( (T)v1[i] - (T)v2[i] );
        return sum;
    }
};

/*One-line template wrappers for DT call*/

template<typename GuideVec, typename SrcVec>
void domainTransformFilter( const Mat_<GuideVec>& guide,
                            const Mat_<SrcVec>& source,
                            Mat& dst,
                            double sigmaSpatial, double sigmaColor,
                            int mode = DTF_NC, int numPasses = 3
                          );

template<typename GuideVec, typename SrcVec>
void domainTransformFilter( const Mat& guide,
                            const Mat& source,
                            Mat& dst,
                            double sigmaSpatial, double sigmaColor,
                            int mode = DTF_NC, int numPasses = 3
                          );
}
}

#include "dtfilter_cpu.inl.hpp"

#endif