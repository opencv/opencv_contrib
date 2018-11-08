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
If you use this code please cite this @cite deriche1987using
Using Canny's criteria to derive a recursively implemented optimal edge detector  International journal of computer vision  (Volume:1 ,  Issue: 2 )  1987
*/

namespace cv {
namespace ximgproc {
template<typename T> static void
VerticalIIRFilter(Mat &img,Mat &dst,const Range &r,double alpha,double omega)
{
    float                *f2;
    int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
    Mat matG1(1, tailleSequence, CV_64FC1), matG2(1, tailleSequence, CV_64FC1);
    double *g1 = matG1.ptr<double>(0), *g2 = (double*)matG2.ptr<double>(0);
    double a2, a3;
    double b1, b2;
    int rows = img.rows, cols = img.cols;
    double c = (1 - 2 * exp(-alpha)*cos(omega) + exp(-2 * alpha)) / (exp(-alpha)*sin(omega));

    double a = -c * exp(-alpha)*sin(omega);
    a2 = 1;// kp*exp(-alpha);
    a3 = 1;//-kp*exp(-alpha);
    b1 = -2 * exp(-alpha)*cos(omega);
    b2 = exp(-2 * alpha);

    for (int j = r.start; j<r.end; j++)
    {
        // Causal vertical  IIR filter
        T *c1 = img.ptr<T>(0);
        f2 = dst.ptr<float>(0);
        f2 += j;
        c1 += j;
        int i = 0;
        g1[i] = a2* *c1;
        i++;
        c1 += cols;
        g1[i] =a2 * c1[-cols] - (b1)* g1[i - 1];
        i++;
        c1 += cols;
        for (i = 2; i<rows; i++, c1 += cols)
            g1[i] = a2 * c1[-cols] - b1*g1[i - 1] - b2 *g1[i - 2];
        // Anticausal vertical IIR filter
        c1 = img.ptr<T>(0);
        c1 += (rows - 1)*cols + j;
        i = rows - 1;
        g2[i] = a3 * *c1;
        i--;
        c1 -= cols;
        g2[i] = a3* c1[cols]  + (b1)*g2[i + 1];
        i--;
        c1 -= cols;
        for (i = rows - 3; i >= 0; i--, c1 -= cols)
            g2[i] = a3*c1[cols] -
            b1*g2[i + 1] - b2*g2[i + 2];
        for (i = 0; i<rows; i++, f2 += cols)
            *f2 = static_cast<float>(a*(g1[i] - g2[i]));
    }
}

template<typename T> static void
HorizontalIIRFilter(Mat &img, Mat &dst, const Range &r, double alpha, double omega)
{
    float *f1;
    int rows = img.rows, cols = img.cols;
    int tailleSequence = (rows>cols) ? rows : cols;
    Mat matG1(1, tailleSequence, CV_64FC1), matG2(1, tailleSequence, CV_64FC1);
    double *g1 = (double*)matG1.ptr(0), *g2 = (double*)matG2.ptr(0);
    double a,a2, a3;
    double b1, b2;
    double c = (1 - 2 * exp(-alpha)*cos(omega) + exp(-2 * alpha)) / (exp(-alpha)*sin(omega));

    a = -c*exp(-alpha)*sin(omega);
    a2 = 1;// kp*exp(-alpha);
    a3 = 1;//-kp*exp(-alpha);
    b1 = -2 * exp(-alpha)*cos(omega);
    b2 = exp(-2 * alpha);

    for (int i = r.start; i<r.end; i++)
    {
        f1 = dst.ptr<float>(i);
        T *c1 = img.ptr<T>(i);
        int j = 0;
        g1[j] = a2* *c1;
        j++;
        c1++;
        g1[j] = a2*c1[j - 1] - (b1)* g1[j - 1];
        j++;
        c1++;
        for (j = 2; j<cols; j++, c1++)
            g1[j] = a2 * c1[-1] - b1*g1[j - 1] - b2*g1[j - 2];
        c1 = img.ptr<T>(0);
        c1 += i*cols + cols - 1;
        j = cols - 1;
        g2[j] = a3* *c1;
        j--;
        c1--;
        g2[j] = a3 * c1[1] - b1 * g2[j + 1];
        j--;
        c1--;
        for (j = cols - 3; j >= 0; j--, c1--)
            g2[j] = a3*c1[1] - b1*g2[j + 1] - b2*g2[j + 2];
        for (j = 0; j<cols; j++, f1++)
            *f1 = static_cast<float>(a*(g1[j] - g2[j]));
    }
}

class ParallelGradientDericheYCols : public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alpha;
    double omega;
    bool verbose;


public:
    ParallelGradientDericheYCols(Mat &imgSrc, Mat &d, double ald,double o) :
        img(imgSrc),
        dst(d),
        alpha(ald),
        omega(o),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_8UC1 || type == CV_8SC1 || type == CV_16SC1 || type == CV_16UC1 || type == CV_32FC1, "Wrong input type for GradientDericheYCols");
        type = dst.depth();
        CV_CheckType(type, type==CV_32FC1, "Wrong output type for GradientDericheYCols");
    }
    void Verbose(bool b) { verbose = b; }
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;

        switch (img.depth()) {
        case CV_8U:
            VerticalIIRFilter<uchar>(img,dst,range, alpha,omega);
        break;
        case CV_8S:
            VerticalIIRFilter<char>(img, dst, range, alpha, omega);
        break;
        case CV_16U:
            VerticalIIRFilter<ushort>(img, dst, range, alpha, omega);
            break;
        case CV_16S:
            VerticalIIRFilter<short>(img, dst, range, alpha, omega);
            break;
        case CV_32F:
            VerticalIIRFilter<float>(img, dst, range, alpha, omega);
            break;
        default:
            return;
        }
    };
    ParallelGradientDericheYCols& operator=(const ParallelGradientDericheYCols &) {
        return *this;
    };
};


class ParallelGradientDericheYRows : public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alpha;
    double omega;
    bool verbose;

public:
    ParallelGradientDericheYRows(Mat& imgSrc, Mat &d, double ald,double o) :
        img(imgSrc),
        dst(d),
        alpha(ald),
        omega(o),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong input type for GradientDericheYRows");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientDericheYRows");
    }
    void Verbose(bool b) { verbose = b; }
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
        float *f1, *f2;
        int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
        Mat matG1(1,tailleSequence,CV_64FC1), matG2(1,tailleSequence,CV_64FC1);
        double *g1 = matG1.ptr<double>(0), *g2 = matG2.ptr<double>(0);
        int cols = img.cols;


        double a2po2 = (alpha*alpha + omega * omega);
        double k = (1 - 2 * exp(-alpha)*cos(omega) + exp(-2 * alpha))*a2po2;
        k = k / (2 * alpha*exp(-alpha)*sin(omega) + omega - omega * exp(-2 * alpha));
        double c1 = k * alpha / a2po2;
        double c2 = k * omega / a2po2;
        double a0 = c2;
        double a1 = (-c2 * cos(omega) + c1 * sin(omega))*exp(-alpha);
        double b1 = -2 * exp(-alpha)*cos(omega);
        double b2 = exp(-2 * alpha);
        double a2 = a1 - c2 * b1, a3 = -c2 * b2;

        for (int i = range.start; i<range.end; i++)
        {
            f2 = dst.ptr<float>(i);
            f1 = img.ptr<float>(i);
            int j = 0;
            g1[j] = (a0 + a1)* *f1;
            j++;
            f1++;
            g1[j] = a0 * f1[0] + a1*f1[j - 1] - b1* g1[j - 1];
            j++;
            f1++;
            for (j = 2; j<cols; j++, f1++)
                g1[j] = a0 * f1[0] + a1 * f1[-1] - b1*g1[j - 1] - b2*g1[j - 2];
            f1 = ((float*)img.ptr(0));
            f1 += i*cols + cols - 1;
            j = cols - 1;
            g2[j] = (a2 + a3)* *f1;
            j--;
            f1--;
            g2[j] = (a2 + a3) * f1[1] - b2* g2[j + 1];
            j--;
            f1--;
            for (j = cols - 3; j >= 0; j--, f1--)
                g2[j] = a2*f1[1] + a3*f1[2] - b1*g2[j + 1] - b2*g2[j + 2];
            for (j = 0; j<cols; j++, f2++)
                *f2 = (float)(g1[j] + g2[j]);
        }

    };
    ParallelGradientDericheYRows& operator=(const ParallelGradientDericheYRows &) {
        return *this;
    };
};


class ParallelGradientDericheXCols : public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alpha;
    double omega;
    bool verbose;

public:
    ParallelGradientDericheXCols(Mat& imgSrc, Mat &d, double alm,double o) :
        img(imgSrc),
        dst(d),
        alpha(alm),
        omega(o),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong input type for GradientDericheXCols");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientDericheXCols");
    }
    void Verbose(bool b) { verbose = b; }
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
        float                *f1, *f2;
        int rows = img.rows, cols = img.cols;

        int tailleSequence = (rows>cols) ? rows : cols;
        Mat matG1(1,tailleSequence,CV_64FC1), matG2(1,tailleSequence,CV_64FC1);
        double *g1 = (double*)matG1.ptr(0), *g2 = (double*)matG2.ptr(0);
        double a2po2 = (alpha*alpha + omega * omega);
        double k = (1 - 2 * exp(-alpha)*cos(omega) + exp(-2 * alpha))*a2po2;
        k = k / (2 * alpha*exp(-alpha)*sin(omega) + omega - omega * exp(-2 * alpha));
        double c1 = k * alpha / a2po2;
        double c2 = k * omega / a2po2;
        double a0 = c2;
        double a1 = (-c2 * cos(omega) + c1 * sin(omega))*exp(-alpha);
        double b1 = -2 * exp(-alpha)*cos(omega);
        double b2=exp(-2*alpha);
        double a2=a1-c2*b1, a3=-c2*b2;


        for (int j = range.start; j<range.end; j++)
        {
            f1 = img.ptr<float>(0);
            f1 += j;
            int i = 0;
            g1[i] = (a0 + a1)* *f1;
            i++;
            f1 += cols;
            g1[i] = a0 * *f1 + a1 * f1[-cols] - (b1)* g1[i - 1];
            i++;
            f1 += cols;
            for (i = 2; i<rows; i++, f1 += cols)
                g1[i] = a0 * *f1 + a1 * f1[-cols] - b1*g1[i - 1] - b2 *g1[i - 2];
            f1 = img.ptr<float>(0);
            f1 += (rows - 1)*cols + j;
            i = rows - 1;
            g2[i] = (a2 + a3)* *f1;
            i--;
            f1 -= cols;
            g2[i] = (a2 + a3)* f1[cols] - b2*g2[i + 1];
            i--;
            f1 -= cols;
            for (i = rows - 3; i >= 0; i--, f1 -= cols)
                g2[i] = a2*f1[cols] + a3* f1[2 * cols] -
                b1*g2[i + 1] - b2*g2[i + 2];
            for (i = 0; i<rows; i++, f2 += cols)
            {
                f2 = (dst.ptr<float>(i)) + (j*img.channels());
                *f2 = (float)(g1[i] + g2[i]);
            }
        }
    };
    ParallelGradientDericheXCols& operator=(const ParallelGradientDericheXCols &) {
        return *this;
    };
};


class ParallelGradientDericheXRows : public ParallelLoopBody
{
private:
    Mat &img;
    Mat &dst;
    double alpha;
    double omega;
    bool verbose;

public:
    ParallelGradientDericheXRows(Mat& imgSrc, Mat &d, double ald, double o) :
        img(imgSrc),
        dst(d),
        alpha(ald),
        omega(o),
        verbose(false)
    {
        int type = img.depth();
        CV_CheckType(type, type == CV_8UC1 || type == CV_8SC1 || type == CV_16SC1 || type == CV_16UC1 || type == CV_32FC1, "Wrong input type for GradientDericheXRows");
        type = dst.depth();
        CV_CheckType(type, type == CV_32FC1, "Wrong output type for GradientDericheXRows");
    }
    void Verbose(bool b) { verbose = b; }
    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        if (verbose)
            std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;

        switch (img.depth()) {
        case CV_8U:
            HorizontalIIRFilter<uchar>(img,dst,range, alpha,omega);
            break;
        case CV_8S:
            HorizontalIIRFilter<char>(img, dst, range, alpha, omega);
            break;
        case CV_16U:
            HorizontalIIRFilter<ushort>(img, dst, range, alpha, omega);
            break;
        case CV_16S:
            HorizontalIIRFilter<short>(img, dst, range, alpha, omega);
            break;
        case CV_32F:
            HorizontalIIRFilter<float>(img, dst, range, alpha, omega);
            break;
        default:
            return;
        }
    };
    ParallelGradientDericheXRows& operator=(const ParallelGradientDericheXRows &) {
        return *this;
    };
};

void GradientDericheY(InputArray _op, OutputArray _dst,double alphaDerive, double omega)
{
    std::vector<Mat> planSrc;
    split(_op, planSrc);

    std::vector<Mat> planTmp;
    std::vector<Mat> planDst;
    for (size_t i = 0; i < planSrc.size(); i++)
    {
        planTmp.push_back(Mat(_op.size(), CV_32FC1));
        planDst.push_back(Mat(_op.size(), CV_32FC1));
        CV_Assert(planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous());
        ParallelGradientDericheYCols x(planSrc[i], planTmp[i], alphaDerive,omega);
        parallel_for_(Range(0, planSrc[i].cols), x, getNumThreads());
        ParallelGradientDericheYRows xr(planTmp[i], planDst[i], alphaDerive, omega);
        parallel_for_(Range(0, planTmp[i].rows), xr, getNumThreads());
    }
    merge(planDst, _dst);
}

void GradientDericheX(InputArray _op, OutputArray _dst, double alpha, double omega)
{
    std::vector<Mat> planSrc;
    split(_op, planSrc);
    std::vector<Mat> planTmp;
    std::vector<Mat> planDst;
    for (size_t i = 0; i < planSrc.size(); i++)
    {
        planTmp.push_back(Mat(_op.size(), CV_32FC1));
        planDst.push_back(Mat(_op.size(), CV_32FC1));
        CV_Assert(planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous());

        ParallelGradientDericheXRows x(planSrc[i], planTmp[i], alpha, omega);
        parallel_for_(Range(0, planSrc[i].rows), x, getNumThreads());
        ParallelGradientDericheXCols xr(planTmp[i], planDst[i], alpha, omega);
        parallel_for_(Range(0, planTmp[i].cols), xr, getNumThreads());
    }
    merge(planDst, _dst);
}

} //end of cv::ximgproc
} //end of cv
