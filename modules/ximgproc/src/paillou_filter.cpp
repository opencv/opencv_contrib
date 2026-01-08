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
#include "precomp.hpp"
#include <math.h>
#include <vector>
#include <iostream>

/*
If you use this code please cite this @cite paillou1997detecting
Detecting step edges in noisy SAR images: a new linear operator  IEEE Transactions on Geoscience and Remote Sensing  (Volume:35 ,  Issue: 1 )  1997
Equation xx pyyy mean equation xx at page yyy in previous paper
*/

namespace cv {
namespace ximgproc {

template<typename T> static void
VerticalIIRFilter(Mat &img, Mat &dst, const Range &range, double a,double w)
{
    float *f2;
    int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
    Mat matYp(1, tailleSequence, CV_64FC1), matYm(1, tailleSequence, CV_64FC1);
    double *yp = matYp.ptr<double>(0), *ym = matYm.ptr<double>(0);
    int rows = img.rows, cols = img.cols;

    // Equation 12 p193
    double                b1 = -2 * exp(-a)*cosh(w);
    double                a1 = 2 * exp(-a)*cosh(w) - exp(-2 * a) - 1;
    double                b2 = exp(-2 * a);
    for (int j = range.start; j < range.end; j++)
    {
        // Equation 26 p194
        T *c1 = (T *)img.ptr(0) + j;
        f2 = dst.ptr<float>(0) + j;
        double border = *c1;
        yp[0] = *c1;
        c1 += cols;
        yp[1] = *c1 - b1*yp[0] - b2*border;
        c1 += cols;
        for (int i = 2; i < rows; i++, c1 += cols)
            yp[i] = *c1 - b1*yp[i - 1] - b2*yp[i - 2];
        // Equation 27 p194
        c1 = (T *)img.ptr(rows - 1) + j;
        border = *c1;
        ym[rows - 1] = *c1;
        c1 -= cols;
        ym[rows - 2] = *c1 - b1*ym[rows - 1];
        c1 -= cols;
        for (int i = rows - 3; i >= 0; i--, c1 -= cols)
            ym[i] = *c1 - b1*ym[i + 1] - b2*ym[i + 2];
        // Equation 25 p193
        for (int i = 0; i < rows; i++, f2 += cols)
            *f2 = (float)(a1*(ym[i] - yp[i]));
    }
}

template<typename T> static void
HorizontalIIRFilter(Mat &img, Mat &dst, const Range &range, double a, double w)
{
    int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
    Mat matYp(1, tailleSequence, CV_64FC1), matYm(1, tailleSequence, CV_64FC1);
    double *yp = matYp.ptr<double>(0), *ym = matYm.ptr<double>(0);
    int cols = img.cols;
    float *f2;

    // Equation 12 p193
    double                b1 = -2 * exp(-a)*cosh(w);
    double                a1 = 2 * exp(-a)*cosh(w) - exp(-2 * a) - 1;
    double                b2 = exp(-2 * a);

    for (int i = range.start; i<range.end; i++)
    {
        // Equation 26 p194
        T *c1 = img.ptr<T>(i);
        double border = *c1;
        yp[0] = *c1;
        c1++;
        yp[1] = *c1 - b1*yp[0] - b2*border;
        c1++;
        for (int j = 2; j<cols; j++, c1++)
            yp[j] = *c1 - b1*yp[j - 1] - b2*yp[j - 2];
        // Equation 27 p194
        c1 = img.ptr<T>(i) + cols - 1;
        border = *c1;
        ym[cols - 1] = *c1;
        c1--;
        ym[cols - 2] = *c1 - b1*ym[cols - 1];
        c1--;
        for (int j = cols - 3; j >= 0; j--, c1--)
            ym[j] = *c1 - b1*ym[j + 1] - b2*ym[j + 2];
        // Equation 25 p193
        f2 = dst.ptr<float>(i);
        for (int j = 0; j<cols; j++, f2++)
            *f2 = (float)(a1*(ym[j] - yp[j]));
    }
}

class ParallelGradientPaillouYCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double a;
    double w;
    bool verbose;
public:
    ParallelGradientPaillouYCols(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        dst(d),
        a(aa),
        w(ww),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_8UC1 || type == CV_8SC1 || type == CV_16SC1 || type == CV_16UC1 || type == CV_32FC1, "Wrong input type for GradientPaillouY");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientPaillouYCols");
    }
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;

        switch(img.depth()){
        case CV_8U:
            VerticalIIRFilter<uchar>(img, dst, range, a, w);
            break;
        case CV_8S:
            VerticalIIRFilter<char>(img, dst, range, a, w);
            break;
        case CV_16S :
            VerticalIIRFilter<short>(img, dst, range, a, w);
            break;
        case CV_16U:
            VerticalIIRFilter<short>(img, dst, range, a, w);
            break;
        case CV_32F:
            VerticalIIRFilter<float>(img, dst, range, a, w);
            break;
        default :
            return ;
            }
    };
    ParallelGradientPaillouYCols& operator=(const ParallelGradientPaillouYCols &) {
         return *this;
    };
};


class ParallelGradientPaillouYRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double a;
    double w;
    bool verbose;

public:
    ParallelGradientPaillouYRows(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        dst(d),
        a(aa),
        w(ww),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong input type for GradientPaillouYRows");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientPaillouYRows");
    }
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *iy,*iy0;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        Mat matIym(1,tailleSequence,CV_64FC1),  matIyp(1,tailleSequence,CV_64FC1);
        double *iym=matIym.ptr<double>(0), *iyp=matIyp.ptr<double>(0);
        int cols=img.cols;

        // Equation 13 p193
        double                d=(1-2*exp(-a)*cosh(w)+exp(-2*a))/(2*a*exp(-a)*sinh(w)+w*(1-exp(-2*a)));
        double                c1=a*d;
        double                c2=w*d;
        // Equation 12 p193
        double                b1=-2*exp(-a)*cosh(w);
        double                b2=exp(-2*a);
        // Equation 14 p193
        double                a0p=c2;
        double                a1p=(c1*sinh(w)-c2*cosh(w))*exp(-a);
        double                a1m=a1p-c2*b1;
        double                a2m=-c2*b2;

        for (int i=range.start;i<range.end;i++)
            {
            iy0 = img.ptr<float>(i);
            int j=0;
            iyp[0] = a0p*iy0[0] ;
            iyp[1] = a0p*iy0[1] + a1p*iy0[0] - b1*iyp[0];
            iy0 += 2;
            for (j=2;j<cols;j++,iy0++)
                iyp[j] = a0p*iy0[0] + a1p*iy0[-1] - b1*iyp[j-1] - b2*iyp[j-2];
            iy0 = img.ptr<float>(i)+cols-1;
            iym[cols-1] = 0;
            iy0--;
            iym[cols-2] = a1m*iy0[1]  - b1*iym[cols-1];
            iy0--;
            for (j=cols-3;j>=0;j--,iy0--)
                iym[j] = a1m*iy0[1] + a2m*iy0[2] - b1*iym[j+1] - b2*iym[j+2];
            iy = dst.ptr<float>(i);
            for (j=0;j<cols;j++,iy++)
                *iy = (float)(iym[j]+iyp[j]);
            }

    };
    ParallelGradientPaillouYRows& operator=(const ParallelGradientPaillouYRows &) {
         return *this;
    };
};


class ParallelGradientPaillouXCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double a;
    double w;
    bool verbose;

public:
    ParallelGradientPaillouXCols(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        dst(d),
        a(aa),
        w(ww),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong input type for GradientPaillouXCols");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientPaillouXCols");
    }
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
        float *iy, *iy0;
        int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
        Mat matIym(1,tailleSequence,CV_64FC1),  matIyp(1,tailleSequence,CV_64FC1);
        double *iym=matIym.ptr<double>(0), *iyp=matIyp.ptr<double>(0);
        int rows = img.rows,cols=img.cols;

        // Equation 13 p193
        double                d = (1 - 2 * exp(-a)*cosh(w) + exp(-2 * a)) / (2 * a*exp(-a)*sinh(w) + w*(1 - exp(-2 * a)));
        double                c1 = a*d;
        double                c2 = w*d;
        // Equation 12 p193
        double                b1 = -2 * exp(-a)*cosh(w);
        double                b2 = exp(-2 * a);
        // Equation 14 p193
        double                a0p = c2;
        double                a1p = (c1*sinh(w) - c2*cosh(w))*exp(-a);
        double                a1m = a1p - c2*b1;
        double                a2m = -c2*b2;

        for (int j = range.start; j<range.end; j++)
        {
            iy0 = img.ptr<float>(0)+j;
            iyp[0] = a0p*iy0[0];
            iy0 +=cols;
            iyp[1] = a0p*iy0[0] + a1p*iy0[-cols] - b1*iyp[0];
            iy0 +=cols;
            for (int i = 2; i<rows; i++, iy0+=cols)
                iyp[i] = a0p*iy0[0] + a1p*iy0[-cols] - b1*iyp[i - 1] - b2*iyp[i - 2];
            iy0 = img.ptr<float>(rows-1) + j;
            iym[rows - 1] = 0;
            iy0 -=cols;
            iym[rows - 2] = a1m*iy0[cols] - b1*iym[rows-1];
            iy0-=cols;
            for (int i = rows - 3; i >= 0; i--, iy0-=cols)
                iym[i] = a1m*iy0[cols] + a2m*iy0[2*cols] - b1*iym[i + 1] - b2*iym[i + 2];
            iy = dst.ptr<float>(0)+j;
            for (int i = 0; i<rows; i++, iy+=cols)
                *iy = (float)(iym[i] + iyp[i]);
        }

    };
    ParallelGradientPaillouXCols& operator=(const ParallelGradientPaillouXCols &) {
         return *this;
    };
};


class ParallelGradientPaillouXRows: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &im1;
    double a;
    double w;
    bool verbose;

public:
    ParallelGradientPaillouXRows(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        im1(d),
        a(aa),
        w(ww),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_8UC1 || type == CV_8SC1 || type == CV_16SC1 || type == CV_16UC1 || type == CV_32FC1, "Wrong input type for GradientPaillouXRows");
        type = im1.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientPaillouXRows");
    }
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;

        // Equation 12 p193
        switch(img.depth()){
        case CV_8U :
            HorizontalIIRFilter<uchar>(img,im1,range,a,w);
            break;
        case CV_8S :
            HorizontalIIRFilter<char>(img, im1, range, a, w);
            break;
        case CV_16S :
            HorizontalIIRFilter<short>(img, im1, range, a, w);
            break;
        case CV_16U:
            HorizontalIIRFilter<ushort>(img, im1, range, a, w);
            break;
        case CV_32F:
            HorizontalIIRFilter<float>(img, im1, range, a, w);
            break;
        default :
            return ;
            }
    };
    ParallelGradientPaillouXRows& operator=(const ParallelGradientPaillouXRows &) {
         return *this;
    };
};

void GradientPaillouY(InputArray _op, OutputArray _dst, double alpha, double omega)
{
    std::vector<Mat> planSrc;
    split(_op,planSrc);
    std::vector<Mat> planTmp;
    std::vector<Mat> planDst;
    for (int i = 0; i <static_cast<int>(planSrc.size()); i++)
    {
		planTmp.push_back(Mat(_op.size(), CV_32FC1));
		planDst.push_back(Mat(_op.size(), CV_32FC1));
        CV_Assert(planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous());
        ParallelGradientPaillouYCols x(planSrc[i],planTmp[i],alpha,omega);
        parallel_for_(Range(0, planSrc[i].cols), x,getNumThreads());
        ParallelGradientPaillouYRows xr(planTmp[i],planDst[i],alpha,omega);
        parallel_for_(Range(0, planTmp[i].rows), xr,getNumThreads());
    }
    merge(planDst,_dst);
}

void GradientPaillouX(InputArray _op, OutputArray _dst, double alpha, double omega)
{
    std::vector<Mat> planSrc;
    split(_op,planSrc);
    std::vector<Mat> planTmp;
    std::vector<Mat> planDst;
    for (int i = 0; i <static_cast<int>(planSrc.size()); i++)
    {
		planTmp.push_back(Mat(_op.size(), CV_32FC1));
		planDst.push_back(Mat(_op.size(), CV_32FC1));
        CV_Assert(planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous());
        ParallelGradientPaillouXRows x(planSrc[i],planTmp[i],alpha,omega);
        parallel_for_(Range(0, planSrc[i].rows), x,getNumThreads());
        ParallelGradientPaillouXCols xr(planTmp[i],planDst[i],alpha,omega);
        parallel_for_(Range(0, planTmp[i].cols), xr,getNumThreads());
    }
    merge(planDst,_dst);
}
}
}
