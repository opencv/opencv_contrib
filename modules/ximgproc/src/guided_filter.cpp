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
 *  *Redistributions of source code must retain the above copyright notice,
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

#include "precomp.hpp"
#include "edgeaware_filters_common.hpp"
#include <vector>

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace cv
{
namespace ximgproc
{

using std::vector;
using namespace cv::ximgproc::intrinsics;

template <typename T>
struct SymArray2D
{
    vector<T> vec;
    int sz;

    SymArray2D()
    {
        sz = 0;
    }

    void create(int sz_)
    {
        CV_DbgAssert(sz_ > 0);
        sz = sz_;
        vec.resize(total());
    }

    inline T& operator()(int i, int j)
    {
        CV_DbgAssert(i >= 0 && i < sz && j >= 0 && j < sz);
        if (i < j) std::swap(i, j);
        return vec[i*(i+1)/2 + j];
    }

    inline T& operator()(int i)
    {
        return vec[i];
    }

    int total() const
    {
        return sz*(sz + 1)/2;
    }

    void release()
    {
        vec.clear();
        sz = 0;
    }
};


template <typename XMat>
static void splitFirstNChannels(InputArrayOfArrays src, vector<XMat>& dst, int maxDstCn)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if ( (src.isMat() || src.isUMat()) && src.channels() == maxDstCn )
    {
        split(src, dst);
    }
    else
    {
        Size sz;
        int depth, totalCnNum;
        
        checkSameSizeAndDepth(src, sz, depth);
        totalCnNum = std::min(maxDstCn, getTotalNumberOfChannels(src));
        
        dst.resize(totalCnNum);
        vector<int> fromTo(2*totalCnNum);
        for (int i = 0; i < totalCnNum; i++)
        {
            fromTo[i*2 + 0] = i;
            fromTo[i*2 + 1] = i;

            dst[i].create(sz, CV_MAKE_TYPE(depth, 1));
        }

        mixChannels(src, dst, fromTo);
    }
}

class GuidedFilterImpl : public GuidedFilter
{
public:
    
    static Ptr<GuidedFilterImpl> create(InputArray guide, int radius, double eps);

    void filter(InputArray src, OutputArray dst, int dDepth = -1);

protected:

    int radius;
    double eps;
    int h, w;

    vector<Mat> guideCn;
    vector<Mat> guideCnMean;

    SymArray2D<Mat> covarsInv;

    int gCnNum;

protected:

    GuidedFilterImpl() {}
    
    void init(InputArray guide, int radius, double eps);

    void computeCovGuide(SymArray2D<Mat>& covars);

    void computeCovGuideAndSrc(vector<Mat>& srcCn, vector<Mat>& srcCnMean, vector<vector<Mat> >& cov);

    void getWalkPattern(int eid, int &cn1, int &cn2);

    inline void meanFilter(Mat& src, Mat& dst)
    {
        boxFilter(src, dst, CV_32F, Size(2 * radius + 1, 2 * radius + 1), cv::Point(-1, -1), true, BORDER_REFLECT);
    }

    inline void convertToWorkType(Mat& src, Mat& dst)
    {
        src.convertTo(dst, CV_32F);
    }

private: /*Routines to parallelize boxFilter and convertTo*/
    
    typedef void (GuidedFilterImpl::*TransformFunc)(Mat& src, Mat& dst);

    struct GFTransform_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        mutable vector<Mat*> src;
        mutable vector<Mat*> dst;
        TransformFunc func;

        GFTransform_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_);
        GFTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_);

        void operator () (const Range& range) const;

        Range getRange() const
        {
            return Range(0, (int)src.size());
        }
    };

    template<typename V>
    void parConvertToWorkType(V &src, V &dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::convertToWorkType);
        parallel_for_(pb.getRange(), pb);
    }

    template<typename V>
    void parMeanFilter(V &src, V &dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::meanFilter);
        parallel_for_(pb.getRange(), pb);
    }

private: /*Parallel body classes*/

    inline void runParBody(const ParallelLoopBody& pb)
    {
        parallel_for_(Range(0, h), pb);
    }

    struct MulChannelsGuide_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        SymArray2D<Mat> &covars;

        MulChannelsGuide_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const Range& range) const;
    };

    struct ComputeCovGuideFromChannelsMul_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        SymArray2D<Mat> &covars;

        ComputeCovGuideFromChannelsMul_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const Range& range) const;
    };

    struct ComputeCovGuideInv_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        SymArray2D<Mat> &covars;

        ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_);

        void operator () (const Range& range) const;
    };


    struct MulChannelsGuideAndSrc_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        vector<vector<Mat> > &cov;
        vector<Mat> &srcCn;

        MulChannelsGuideAndSrc_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcCn_, vector<vector<Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCn(srcCn_) {}

        void operator () (const Range& range) const;
    };

    struct ComputeCovFromSrcChannelsMul_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        vector<vector<Mat> > &cov;
        vector<Mat> &srcCnMean;

        ComputeCovFromSrcChannelsMul_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcCnMean_, vector<vector<Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCnMean(srcCnMean_) {}

        void operator () (const Range& range) const;
    };

    struct ComputeAlpha_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        vector<vector<Mat> > &alpha;
        vector<vector<Mat> > &covSrc;

        ComputeAlpha_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<vector<Mat> >& covSrc_)
            : gf(gf_), alpha(alpha_), covSrc(covSrc_) {}

        void operator () (const Range& range) const;
    };

    struct ComputeBeta_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        vector<vector<Mat> > &alpha;
        vector<Mat> &srcCnMean;
        vector<Mat> &beta;

        ComputeBeta_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<Mat>& srcCnMean_, vector<Mat>& beta_)
            : gf(gf_), alpha(alpha_), srcCnMean(srcCnMean_), beta(beta_) {}

        void operator () (const Range& range) const;
    };

    struct ApplyTransform_ParBody : public ParallelLoopBody
    {
        GuidedFilterImpl &gf;
        vector<vector<Mat> > &alpha;
        vector<Mat> &beta;

        ApplyTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& alpha_, vector<Mat>& beta_)
            : gf(gf_),  alpha(alpha_), beta(beta_) {}

        void operator () (const Range& range) const;
    };
};

void GuidedFilterImpl::MulChannelsGuide_ParBody::operator()(const Range& range) const
{
    int total = covars.total();

    for (int i = range.start; i < range.end; i++)
    {
        int c1, c2;
        float *cov, *guide1, *guide2;

        for (int k = 0; k < total; k++)
        {
            gf.getWalkPattern(k, c1, c2);

            guide1 = gf.guideCn[c1].ptr<float>(i);
            guide2 = gf.guideCn[c2].ptr<float>(i);
            cov = covars(c1, c2).ptr<float>(i);

            mul(cov, guide1, guide2, gf.w);
        }
    }
}

void GuidedFilterImpl::ComputeCovGuideFromChannelsMul_ParBody::operator()(const Range& range) const
{
    int total = covars.total();
    float diagSummand = (float)(gf.eps);

    for (int i = range.start; i < range.end; i++)
    {
        int c1, c2;
        float *cov, *guide1, *guide2;

        for (int k = 0; k < total; k++)
        {
            gf.getWalkPattern(k, c1, c2);

            guide1 = gf.guideCnMean[c1].ptr<float>(i);
            guide2 = gf.guideCnMean[c2].ptr<float>(i);
            cov = covars(c1, c2).ptr<float>(i);

            if (c1 != c2)
            {
                sub_mul(cov, guide1, guide2, gf.w);
            }
            else
            {
                sub_mad(cov, guide1, guide2, -diagSummand, gf.w);
            }
        }
    }
}

GuidedFilterImpl::ComputeCovGuideInv_ParBody::ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<Mat>& covars_) 
    : gf(gf_), covars(covars_)
{
    gf.covarsInv.create(gf.gCnNum);

    if (gf.gCnNum == 3)
    {
        for (int k = 0; k < 2; k++)
            for (int l = 0; l < 3; l++)
                gf.covarsInv(k, l).create(gf.h, gf.w, CV_32FC1);

        ////trick to avoid memory allocation
        gf.covarsInv(2, 0).create(gf.h, gf.w, CV_32FC1);
        gf.covarsInv(2, 1) = covars(2, 1);
        gf.covarsInv(2, 2) = covars(2, 2);

        return;
    }

    if (gf.gCnNum == 2)
    {
        gf.covarsInv(0, 0) = covars(1, 1);
        gf.covarsInv(0, 1) = covars(0, 1);
        gf.covarsInv(1, 1) = covars(0, 0);
        return;
    }

    if (gf.gCnNum == 1)
    {
        gf.covarsInv(0, 0) = covars(0, 0);
        return;
    }
}

void GuidedFilterImpl::ComputeCovGuideInv_ParBody::operator()(const Range& range) const
{
    if (gf.gCnNum == 3)
    {
        vector<float> covarsDet(gf.w);
        float *det = &covarsDet[0];

        for (int i = range.start; i < range.end; i++)
        {
            for (int k = 0; k < 3; k++)
                for (int l = 0; l <= k; l++)
                {
                    float *dst = gf.covarsInv(k, l).ptr<float>(i);

                    float *a00 = covars((k + 1) % 3, (l + 1) % 3).ptr<float>(i);
                    float *a01 = covars((k + 1) % 3, (l + 2) % 3).ptr<float>(i);
                    float *a10 = covars((k + 2) % 3, (l + 1) % 3).ptr<float>(i);
                    float *a11 = covars((k + 2) % 3, (l + 2) % 3).ptr<float>(i);

                    det_2x2(dst, a00, a01, a10, a11, gf.w);
                }

            for (int k = 0; k < 3; k++)
            {
                register float *a = covars(k, 0).ptr<float>(i);
                register float *ac = gf.covarsInv(k, 0).ptr<float>(i);

                if (k == 0)
                    mul(det, a, ac, gf.w);
                else
                    add_mul(det, a, ac, gf.w);
            }

            for (int k = 0; k < gf.covarsInv.total(); k += 1)
            {
                div_1x(gf.covarsInv(k).ptr<float>(i), det, gf.w);
            }
        }
        return;
    }

    if (gf.gCnNum == 2)
    {
        for (int i = range.start; i < range.end; i++)
        {
            float *a00 = gf.covarsInv(0, 0).ptr<float>(i);
            float *a10 = gf.covarsInv(1, 0).ptr<float>(i);
            float *a11 = gf.covarsInv(1, 1).ptr<float>(i);

            div_det_2x2(a00, a10, a11, gf.w);
        }
        return;
    }

    if (gf.gCnNum == 1)
    {
        //divide(1.0, covars(0, 0)(range, Range::all()), gf.covarsInv(0, 0)(range, Range::all()));
        //return;

        for (int i = range.start; i < range.end; i++)
        {
            float *res = covars(0, 0).ptr<float>(i);
            inv_self(res, gf.w);
        }
        return;
    }
}

void GuidedFilterImpl::MulChannelsGuideAndSrc_ParBody::operator()(const Range& range) const
{
    int srcCnNum = (int)srcCn.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            int step    = (si % 2) * 2 - 1;
            int start   = (si % 2) ? 0 : gf.gCnNum - 1;
            int end     = (si % 2) ? gf.gCnNum : -1;

            float *srcLine = srcCn[si].ptr<float>(i);

            for (int gi = start; gi != end; gi += step)
            {
                float *guideLine = gf.guideCn[gi].ptr<float>(i);
                float *dstLine = cov[si][gi].ptr<float>(i);

                mul(dstLine, srcLine, guideLine, gf.w);
            }
        }
    }
}

void GuidedFilterImpl::ComputeCovFromSrcChannelsMul_ParBody::operator()(const Range& range) const
{
    int srcCnNum = (int)srcCnMean.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            int step    = (si % 2) * 2 - 1;
            int start   = (si % 2) ? 0 : gf.gCnNum - 1;
            int end     = (si % 2) ? gf.gCnNum : -1;

            register float *srcMeanLine = srcCnMean[si].ptr<float>(i);

            for (int gi = start; gi != end; gi += step)
            {
                float *guideMeanLine = gf.guideCnMean[gi].ptr<float>(i);
                float *covLine = cov[si][gi].ptr<float>(i);

                sub_mul(covLine, srcMeanLine, guideMeanLine, gf.w);
            }
        }
    }
}

void GuidedFilterImpl::ComputeAlpha_ParBody::operator()(const Range& range) const
{
    int srcCnNum = (int)covSrc.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            for (int gi = 0; gi < gf.gCnNum; gi++)
            {
                float *y, *A, *dstAlpha;

                dstAlpha = alpha[si][gi].ptr<float>(i);
                for (int k = 0; k < gf.gCnNum; k++)
                {
                    y = covSrc[si][k].ptr<float>(i);
                    A = gf.covarsInv(gi, k).ptr<float>(i);

                    if (k == 0)
                    {
                        mul(dstAlpha, A, y, gf.w);
                    }
                    else
                    {
                        add_mul(dstAlpha, A, y, gf.w);
                    }
                }
            }
        }
    }
}

void GuidedFilterImpl::ComputeBeta_ParBody::operator()(const Range& range) const
{
    int srcCnNum = (int)srcCnMean.size();
    CV_DbgAssert(&srcCnMean == &beta);

    for (int i = range.start; i < range.end; i++)
    {
        float *_g[4];
        for (int gi = 0; gi < gf.gCnNum; gi++)
            _g[gi] = gf.guideCnMean[gi].ptr<float>(i);

        float *betaDst, *g, *a;
        for (int si = 0; si < srcCnNum; si++)
        {
            betaDst = beta[si].ptr<float>(i);
            for (int gi = 0; gi < gf.gCnNum; gi++)
            {
                a = alpha[si][gi].ptr<float>(i);
                g = _g[gi];

                sub_mul(betaDst, a, g, gf.w);
            }
        }
    }
}

void GuidedFilterImpl::ApplyTransform_ParBody::operator()(const Range& range) const
{
    int srcCnNum = (int)alpha.size();

    for (int i = range.start; i < range.end; i++)
    {
        float *_g[4];
        for (int gi = 0; gi < gf.gCnNum; gi++)
            _g[gi] = gf.guideCn[gi].ptr<float>(i);

        float *betaDst, *g, *a;
        for (int si = 0; si < srcCnNum; si++)
        {
            betaDst = beta[si].ptr<float>(i);
            for (int gi = 0; gi < gf.gCnNum; gi++)
            {
                a = alpha[si][gi].ptr<float>(i);
                g = _g[gi];

                add_mul(betaDst, a, g, gf.w);
            }
        }
    }
}

GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, vector<Mat>& srcv, vector<Mat>& dstv, TransformFunc func_)
    : gf(gf_), func(func_)
{
    CV_DbgAssert(srcv.size() == dstv.size());
    src.resize(srcv.size());
    dst.resize(srcv.size());

    for (int i = 0; i < (int)srcv.size(); i++)
    {
        src[i] = &srcv[i];
        dst[i] = &dstv[i];
    }
}

GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, vector<vector<Mat> >& srcvv, vector<vector<Mat> >& dstvv, TransformFunc func_)
    : gf(gf_), func(func_)
{
    CV_DbgAssert(srcvv.size() == dstvv.size());
    int n = (int)srcvv.size();
    int total = 0;

    for (int i = 0; i < n; i++)
    {
        CV_DbgAssert(srcvv[i].size() == dstvv[i].size());
        total += (int)srcvv[i].size();
    }

    src.resize(total);
    dst.resize(total);

    int k = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)srcvv[i].size(); j++)
        {
            src[k] = &srcvv[i][j];
            dst[k] = &dstvv[i][j];
            k++;
        }
    }
}

void GuidedFilterImpl::GFTransform_ParBody::operator()(const Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        (gf.*func)(*src[i], *dst[i]);
    }
}

void GuidedFilterImpl::getWalkPattern(int eid, int &cn1, int &cn2)
{
    static int wdata[] = {
        0, -1, -1, -1, -1, -1,
        0, -1, -1, -1, -1, -1,

        0,  0,  1, -1, -1, -1,
        0,  1,  1, -1, -1, -1,

        0,  0,  0,  2,  1,  1, 
        0,  1,  2,  2,  2,  1,
    };

    cn1 = wdata[6 * 2 * (gCnNum-1) + eid];
    cn2 = wdata[6 * 2 * (gCnNum-1) + 6 + eid];
}

Ptr<GuidedFilterImpl> GuidedFilterImpl::create(InputArray guide, int radius, double eps)
{
    GuidedFilterImpl *gf = new GuidedFilterImpl();
    gf->init(guide, radius, eps);
    return Ptr<GuidedFilterImpl>(gf);
}

void GuidedFilterImpl::init(InputArray guide, int radius_, double eps_)
{
    CV_Assert( !guide.empty() && radius_ >= 0 && eps_ >= 0 );
    CV_Assert( (guide.depth() == CV_32F || guide.depth() == CV_8U || guide.depth() == CV_16U) && (guide.channels() <= 3) );

    radius = radius_;
    eps = eps_;

    splitFirstNChannels(guide, guideCn, 3);
    gCnNum = (int)guideCn.size();
    h = guideCn[0].rows;
    w = guideCn[0].cols;

    guideCnMean.resize(gCnNum);
    parConvertToWorkType(guideCn, guideCn);
    parMeanFilter(guideCn, guideCnMean);
    
    SymArray2D<Mat> covars;
    computeCovGuide(covars);
    runParBody(ComputeCovGuideInv_ParBody(*this, covars));
    covars.release();
}

void GuidedFilterImpl::computeCovGuide(SymArray2D<Mat>& covars)
{
    covars.create(gCnNum);
    for (int i = 0; i < covars.total(); i++)
        covars(i).create(h, w, CV_32FC1);

    runParBody(MulChannelsGuide_ParBody(*this, covars));

    parMeanFilter(covars.vec, covars.vec);

    runParBody(ComputeCovGuideFromChannelsMul_ParBody(*this, covars));
}

void GuidedFilterImpl::filter(InputArray src, OutputArray dst, int dDepth /*= -1*/)
{
    CV_Assert( !src.empty() && (src.depth() == CV_32F || src.depth() == CV_8U) );
    if (src.rows() != h || src.cols() != w)
    {
        CV_Error(Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
        return;
    }

    if (dDepth == -1) dDepth = src.depth();
    int srcCnNum = src.channels();

    vector<Mat> srcCn(srcCnNum);
    vector<Mat>& srcCnMean = srcCn;
    split(src, srcCn);

    if (src.depth() != CV_32F)
    {
        parConvertToWorkType(srcCn, srcCn);
    }

    vector<vector<Mat> > covSrcGuide(srcCnNum);
    computeCovGuideAndSrc(srcCn, srcCnMean, covSrcGuide);

    vector<vector<Mat> > alpha(srcCnNum);
    for (int si = 0; si < srcCnNum; si++)
    {
        alpha[si].resize(gCnNum);
        for (int gi = 0; gi < gCnNum; gi++)
            alpha[si][gi].create(h, w, CV_32FC1);
    }
    runParBody(ComputeAlpha_ParBody(*this, alpha, covSrcGuide));
    covSrcGuide.clear();

    vector<Mat>& beta = srcCnMean;
    runParBody(ComputeBeta_ParBody(*this, alpha, srcCnMean, beta));

    parMeanFilter(beta, beta);
    parMeanFilter(alpha, alpha);

    runParBody(ApplyTransform_ParBody(*this, alpha, beta));
    if (dDepth != CV_32F)
    {
        for (int i = 0; i < srcCnNum; i++)
            beta[i].convertTo(beta[i], dDepth);
    }
    merge(beta, dst);
}

void GuidedFilterImpl::computeCovGuideAndSrc(vector<Mat>& srcCn, vector<Mat>& srcCnMean, vector<vector<Mat> >& cov)
{
    int srcCnNum = (int)srcCn.size();

    cov.resize(srcCnNum);
    for (int i = 0; i < srcCnNum; i++)
    {
        cov[i].resize(gCnNum);
        for (int j = 0; j < gCnNum; j++)
            cov[i][j].create(h, w, CV_32FC1);
    }

    runParBody(MulChannelsGuideAndSrc_ParBody(*this, srcCn, cov));

    parMeanFilter(srcCn, srcCnMean);
    parMeanFilter(cov, cov);

    runParBody(ComputeCovFromSrcChannelsMul_ParBody(*this, srcCnMean, cov));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

CV_EXPORTS_W
Ptr<GuidedFilter> createGuidedFilter(InputArray guide, int radius, double eps)
{
    return Ptr<GuidedFilter>(GuidedFilterImpl::create(guide, radius, eps));
}

CV_EXPORTS_W
void guidedFilter(InputArray guide, InputArray src, OutputArray dst, int radius, double eps, int dDepth)
{
    Ptr<GuidedFilter> gf = createGuidedFilter(guide, radius, eps);
    gf->filter(src, dst, dDepth);
}

}
}