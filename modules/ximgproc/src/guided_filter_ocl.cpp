#include "precomp.hpp"
#include <vector>
#include <iostream>
using std::vector;

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace cv
{

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

class GuidedFilterImplOCL : public GuidedFilter
{
    Size sz;

    int radius;
    double eps;

    SymArray2D<UMat> covGuide;
    SymArray2D<UMat> covGuideInv;

    int guideCnNum;
    vector<UMat> guideCn;
    vector<UMat> guideCnMean;

    inline void meanFilter(UMat& src, UMat& dst);

public:

    GuidedFilterImplOCL(InputArray guide, int radius, double eps);

    void filter(InputArray src, OutputArray dst, int dDepth);

    void computeCovGuide();

    void computeCovGuideInv();

    void computeCovSrcGuide(vector<UMat>& srcCn, vector<UMat>& srcCnMean, vector<vector<UMat> >& covSrcGuide);

    void computeAlpha(vector<vector<UMat> >& covSrcGuide, vector<vector<UMat> >& alpha);

    void computeBeta(vector<vector<UMat> >&alpha, vector<UMat>& srcCnMean, vector<UMat>& beta);

    void applyTransform(vector<vector<UMat> >& alpha, vector<UMat>& beta);

};

void GuidedFilterImplOCL::meanFilter(UMat& src, UMat& dst)
{
    boxFilter(src, dst, CV_32F, Size(2 * radius + 1, 2 * radius + 1), Point(-1, -1), true, BORDER_REFLECT);
}

GuidedFilterImplOCL::GuidedFilterImplOCL(InputArray guide, int radius_, double eps_)
{
    CV_Assert(!guide.empty() && guide.channels() <= 3);
    CV_Assert(radius_ >= 0 && eps_ >= 0);

    radius = radius_;
    eps = eps_;

    guideCnNum = guide.channels();

    guideCn.resize(guideCnNum);
    guideCnMean.resize(guideCnNum);
    if (guide.depth() == CV_32F)
    {
        split(guide, guideCn);
    }
    else
    {
        UMat buf;
        guide.getUMat().convertTo(buf, CV_32F);
        split(buf, guideCn);
    }

    for (int i = 0; i < guideCnNum; i++)
    {
        meanFilter(guideCn[i], guideCnMean[i]);
    }

    computeCovGuide();
    computeCovGuideInv();
}

void GuidedFilterImplOCL::computeCovGuide()
{
    covGuide.create(guideCnNum);

    UMat buf;
    for (int i = 0; i < guideCnNum; i++)
        for (int j = 0; j <= i; j++)
        {
            multiply(guideCn[i], guideCn[j], covGuide(i, j));
            meanFilter(covGuide(i, j), covGuide(i, j));
            
            multiply(guideCnMean[i], guideCnMean[j], buf);
            subtract(covGuide(i, j), buf, covGuide(i, j));

            //add regulariztion term
            if (i == j)
                covGuide(i, j).convertTo(covGuide(i, j), covGuide(i, j).depth(), 1.0, eps);
        }
}

void GuidedFilterImplOCL::computeCovGuideInv()
{
    covGuideInv.create(guideCnNum);

    if (guideCnNum == 3)
    {
        //for (int l = 0; l < 3; l++)
        //    covGuideInv(2, l) = covGuide(2, l);

        UMat det, tmp1, tmp2;
        for (int l = 0; l < 3; l++)
        {
            int l1 = (l + 1) % 3;
            int l2 = (l + 2) % 3;
            multiply(covGuide(1, l1), covGuide(2, l2), tmp1);
            multiply(covGuide(1, l2), covGuide(2, l1), tmp2);
            if (l == 0)
            {
                subtract(tmp1, tmp2, det);
            }
            else
            {
                add(det, tmp1, det);
                subtract(det, tmp2, det);
            }
        }
        
        for (int k = 0; k < 3; k++)
            for (int l = 0; l <= k; l++)
            {
                int k1 = (k + 1) % 3, l1 = (l + 1) % 3;
                int k2 = (k + 2) % 3, l2 = (l + 2) % 3;

                multiply(covGuide(k1, l1), covGuide(k2, l2), tmp1);
                multiply(covGuide(k1, l2), covGuide(k2, l1), tmp2);
                subtract(tmp1, tmp2, covGuideInv(k, l));
                divide(covGuideInv(k, l), det, covGuideInv(k, l));
            }
    }
    else if (guideCnNum == 2)
    {
        covGuideInv(0, 0) = covGuide(1, 1);
        covGuideInv(1, 1) = covGuide(0, 0);
        covGuideInv(0, 1) = covGuide(0, 1);

        UMat det, tmp;
        multiply(covGuide(0, 0), covGuide(1, 1), det);
        multiply(covGuide(0, 1), covGuide(1, 0), tmp);
        subtract(det, tmp, det);
        tmp.release();

        divide(covGuideInv(0, 0), det, covGuideInv(0, 0));
        divide(covGuideInv(1, 1), det, covGuideInv(1, 1));
        divide(covGuideInv(0, 1), det, covGuideInv(0, 1), -1);
    }
    else
    {
        covGuideInv(0, 0) = covGuide(0, 0);
        divide(1.0, covGuide(0, 0), covGuideInv(0, 0));
    }

    covGuide.release();
}

void GuidedFilterImplOCL::filter(InputArray src, OutputArray dst, int dDepth)
{
    if (dDepth == -1) dDepth = src.depth();
    int srcCnNum = src.channels();
    vector<UMat> srcCn(srcCnNum);
    vector<UMat> srcCnMean(srcCnNum);

    if (src.depth() != CV_32F)
    {
        UMat tmp;
        src.getUMat().convertTo(tmp, CV_32F);
        split(tmp, srcCn);
    }
    else
    {
        split(src, srcCn);
    }

    for (int i = 0; i < srcCnNum; i++)
        meanFilter(srcCn[i], srcCnMean[i]);
    
    vector<vector<UMat> > covSrcGuide(srcCnNum);
    computeCovSrcGuide(srcCn, srcCnMean, covSrcGuide);

    vector<vector<UMat> > alpha(srcCnNum);
    vector<UMat>& beta = srcCnMean;

    computeAlpha(covSrcGuide, alpha);
    computeBeta(alpha, srcCnMean, beta);

    //if (true)
    //{
    //    for (int i = 0; i < srcCnNum; i++)
    //    {
    //        Mat alphaViz;
    //        merge(alpha[i], alphaViz);
    //        imwrite("res/alpha" + format("%d", i) + ".png", alphaViz * 100);
    //    }
    //    Mat betaViz;
    //    merge(beta, betaViz);
    //    imwrite("res/beta.png", betaViz + Scalar::all(127));
    //}

    for (int i = 0; i < srcCnNum; i++)
        for (int j = 0; j < guideCnNum; j++)
            meanFilter(alpha[i][j], alpha[i][j]);

    applyTransform(alpha, beta);
    
    if (dDepth != CV_32F)
    {
        for (int i = 0; i < srcCnNum; i++)
            beta[i].convertTo(beta[i], dDepth);
    }
    merge(beta, dst);
}

void GuidedFilterImplOCL::computeCovSrcGuide(vector<UMat>& srcCn, vector<UMat>& srcCnMean, vector<vector<UMat> >& covSrcGuide)
{
    int srcCnNum = (int)srcCn.size();
    covSrcGuide.resize(srcCnNum);

    UMat buf;
    for (int i = 0; i < srcCnNum; i++)
    {
        covSrcGuide[i].resize(guideCnNum);
        for (int j = 0; j < guideCnNum; j++)
        {
            UMat& cov = covSrcGuide[i][j];
            multiply(srcCn[i], guideCn[i], cov);
            meanFilter(cov, cov);
            multiply(srcCnMean[i], guideCnMean[j], buf);
            subtract(cov, buf, cov);
        }
    }
}

void GuidedFilterImplOCL::computeAlpha(vector<vector<UMat> >& covSrcGuide, vector<vector<UMat> >& alpha)
{
    int srcCnNum = (int)covSrcGuide.size();
    alpha.resize(srcCnNum);
    
    UMat buf;
    for (int i = 0; i < srcCnNum; i++)
    {
        alpha[i].resize(guideCnNum);

        for (int k = 0; k < guideCnNum; k++)
        {
            multiply(covGuideInv(k, 0), covSrcGuide[i][0], alpha[i][k]);
            for (int l = 1; l < guideCnNum; l++)
            {
                multiply(covGuideInv(k, l), covSrcGuide[i][l], buf);
                add(buf, alpha[i][k], alpha[i][k]);
            }
        }
    }
}

void GuidedFilterImplOCL::computeBeta(vector<vector<UMat> >& alpha, vector<UMat>& srcCnMean, vector<UMat>& beta)
{
    int srcCnNum = (int)srcCnMean.size();
    CV_Assert(&beta == &srcCnMean);

    UMat buf;
    for (int i = 0; i < srcCnNum; i++)
    {
        multiply(alpha[i][0], guideCnMean[0], beta[i]);
        for (int j = 1; j < guideCnNum; j++)
        {
            multiply(alpha[i][j], guideCnMean[j], buf);
            subtract(beta[i], buf, beta[i]);
        }
        meanFilter(beta[i], beta[i]);
    }
}

void GuidedFilterImplOCL::applyTransform(vector<vector<UMat> >& alpha, vector<UMat>& beta)
{
    int srcCnNum = (int)beta.size();

    UMat buf;
    for (int i = 0; i < srcCnNum; i++)
    {
        for (int j = 0; j < guideCnNum; j++)
        {
            multiply(guideCn[j], alpha[i][j], buf);
            add(beta[i], buf, beta[i]);
        }
    }
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


}