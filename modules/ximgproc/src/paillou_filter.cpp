#include "precomp.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <vector>
#include <iostream>

namespace cv {
namespace ximgproc {

/*
If you use this code please cite this @cite paillou1997detecting
Detecting step edges in noisy SAR images: a new linear operator  IEEE Transactions on Geoscience and Remote Sensing  (Volume:35 ,  Issue: 1 )  1997
*/


class ParallelGradientPaillouYCols: public ParallelLoopBody
{
private:
    Mat &img;
    Mat &im1;
    double a;
    double w;
    bool verbose;
public:
    ParallelGradientPaillouYCols(Mat& imgSrc, Mat &d,double aa,double ww):
        img(imgSrc),
        im1(d),
        a(aa),
        w(ww),
        verbose(false)
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;

        float                *f2;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *yp=new double[tailleSequence],*ym=new double[tailleSequence];
        int rows=img.rows,cols=img.cols;

        // Formule 12 p193
        double				b1=-2*exp(-a)*cosh(w);
        double				a1=2*exp(-a)*cosh(w)-exp(-2*a)-1;
        double				b2=exp(-2*a);

        switch(img.depth()){
        case CV_8U :
        {
            for (int j=range.start;j<range.end;j++)
            {
                // Formule 26 p194
                unsigned char *c1 = (unsigned char*)img.ptr(0)+j;// c1 pointeur sur x(i-1)
                f2 = ((float*)im1.ptr(0))+j;
                double border=*c1;
				yp[0] = *c1 ;
				c1+=cols;
                yp[1] = *c1 - b1*yp[0]-b2*border;
                c1+=cols;
                for (int i=2;i<rows;i++,c1+=cols)
                    yp[i] = *c1-b1*yp[i-1]-b2*yp[i-2];
                // Formule 27 p194
                c1 = (unsigned char*)img.ptr(rows-1)+j;
                border=*c1;
				ym[rows - 1] = *c1;
				c1 -= cols;
				ym[rows-2] =*c1 - b1*ym[rows-1];
                c1 -= cols;
                for (int i=rows-3;i>=0;i--,c1-=cols)
                    ym[i]=*c1-b1*ym[i+1]-b2*ym[i+2];
                // Formule 25 p193
                for (int i=0;i<rows;i++,f2+=cols)
                    *f2 = (float)(a1*(ym[i]-yp[i]));
            }
        }
            break;
        case CV_16S :
		{
			for (int j = range.start; j<range.end; j++)
			{
				// Formule 26 p194
				short *c1 = (short*)img.ptr(0) + j;// c1 pointeur sur x(i-1)
				f2 = ((float*)im1.ptr(0)) + j;
				double border = *c1;
				yp[0] = *c1;
				c1 += cols;
				yp[1] = *c1 - b1*yp[0] - b2*border;
				c1 += cols;
				for (int i = 2; i<rows; i++, c1 += cols)
					yp[i] = *c1 - b1*yp[i - 1] - b2*yp[i - 2];
				// Formule 27 p194
				c1 = (short*)img.ptr(rows - 1) + j;
				border = *c1;
				ym[rows - 1] = *c1;
				c1 -= cols;
				ym[rows - 2] = *c1 - b1*ym[rows - 1];
				c1 -= cols;
				for (int i = rows - 3; i >= 0; i--, c1 -= cols)
					ym[i] = *c1 - b1*ym[i + 1] - b2*ym[i + 2];
				// Formule 25 p193
				for (int i = 0; i<rows; i++, f2 += cols)
					*f2 = (float)(a1*(ym[i] - yp[i]));
			}
		}
		break;
		case CV_16U :
        {
			for (int j = range.start; j<range.end; j++)
			{
				// Formule 26 p194
				unsigned short *c1 = (unsigned short*)img.ptr(0) + j;// c1 pointeur sur x(i-1)
				f2 = ((float*)im1.ptr(0)) + j;
				double border = *c1;
				yp[0] = *c1;
				c1 += cols;
				yp[1] = *c1 - b1*yp[0] - b2*border;
				c1 += cols;
				for (int i = 2; i<rows; i++, c1 += cols)
					yp[i] = *c1 - b1*yp[i - 1] - b2*yp[i - 2];
				// Formule 27 p194
				c1 = (unsigned short*)img.ptr(rows - 1) + j;
				border = *c1;
				ym[rows - 1] = *c1;
				c1 -= cols;
				ym[rows - 2] = *c1 - b1*ym[rows - 1];
				c1 -= cols;
				for (int i = rows - 3; i >= 0; i--, c1 -= cols)
					ym[i] = *c1 - b1*ym[i + 1] - b2*ym[i + 2];
				// Formule 25 p193
				for (int i = 0; i<rows; i++, f2 += cols)
					*f2 = (float)(a1*(ym[i] - yp[i]));
			}
		}
            break;
        case CV_32S :
        case CV_32F :
        case CV_64F :
        default :
            delete []ym;
            delete []yp;
            return ;
            }
        delete []ym;
        delete []yp;
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
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
        float *iy,*iy0;
        int tailleSequence=(img.rows>img.cols)?img.rows:img.cols;
        double *iym=new double[tailleSequence],*iyp=new double[tailleSequence];
        int cols=img.cols;

        // Formule 13
        double				d=(1-2*exp(-a)*cosh(w)+exp(-2*a))/(2*a*exp(-a)*sinh(w)+w*(1-exp(-2*a)));
        double				c1=a*d;
        double				c2=w*d;
        // formule 12
        double				b1=-2*exp(-a)*cosh(w);
        double				b2=exp(-2*a);
        // Formule 14
        double				a0p=c2;
        double				a1p=(c1*sinh(w)-c2*cosh(w))*exp(-a);
        double				a1m=a1p-c2*b1;
        double				a2m=-c2*b2;

        for (int i=range.start;i<range.end;i++)
            {
            iy0 = ((float*)img.ptr(i));
            int j=0;
            iyp[0] = a0p*iy0[0] ;
            iyp[1] = a0p*iy0[1] + a1p*iy0[0] - b1*iyp[0];
			iy0 += 2;
            for (j=2;j<cols;j++,iy0++)
                iyp[j] = a0p*iy0[0] + a1p*iy0[-1] - b1*iyp[j-1] - b2*iyp[j-2];
            iy0 = ((float*)img.ptr(i))+cols-1;
            iym[cols-1] = 0;
            iy0--;
            iym[cols-2] = a1m*iy0[1]  - b1*iym[cols-1];
			iy0--;
            for (j=cols-3;j>=0;j--,iy0--)
                iym[j] = a1m*iy0[1] + a2m*iy0[2] - b1*iym[j+1] - b2*iym[j+2];
            iy = ((float*)dst.ptr(i));
            for (j=0;j<cols;j++,iy++)
                *iy = (float)(iym[j]+iyp[j]);
            }
        delete []iym;
        delete []iyp;

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
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
		if (verbose)
			std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
		float *iy, *iy0;
		int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
		double *iym = new double[tailleSequence], *iyp = new double[tailleSequence];
		int rows = img.rows,cols=img.cols;

		// Formule 13
		double				d = (1 - 2 * exp(-a)*cosh(w) + exp(-2 * a)) / (2 * a*exp(-a)*sinh(w) + w*(1 - exp(-2 * a)));
		double				c1 = a*d;
		double				c2 = w*d;
		// formule 12
		double				b1 = -2 * exp(-a)*cosh(w);
		double				b2 = exp(-2 * a);
		// Formule 14
		double				a0p = c2;
		double				a1p = (c1*sinh(w) - c2*cosh(w))*exp(-a);
		double				a1m = a1p - c2*b1;
		double				a2m = -c2*b2;

		for (int j = range.start; j<range.end; j++)
		{
			iy0 = ((float*)img.ptr(0)+j);
			iyp[0] = a0p*iy0[0];
			iy0 +=cols;
			iyp[1] = a0p*iy0[0] + a1p*iy0[-cols] - b1*iyp[0];
			iy0 +=cols;
			for (int i = 2; i<rows; i++, iy0+=cols)
				iyp[i] = a0p*iy0[0] + a1p*iy0[-cols] - b1*iyp[i - 1] - b2*iyp[i - 2];
			iy0 = ((float*)img.ptr(rows-1)) + j;
			iym[rows - 1] = 0;
			iy0 -=cols;
			iym[rows - 2] = a1m*iy0[cols] - b1*iym[rows-1];
			iy0-=cols;
			for (int i = rows - 3; i >= 0; i--, iy0-=cols)
				iym[i] = a1m*iy0[cols] + a2m*iy0[2*cols] - b1*iym[i + 1] - b2*iym[i + 2];
			iy = ((float*)dst.ptr(0)+j);
			for (int i = 0; i<rows; i++, iy+=cols)
				*iy = (float)(iym[i] + iyp[i]);
		}
		delete[]iym;
		delete[]iyp;

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
    {}
    void Verbose(bool b){verbose=b;}
    virtual void operator()(const Range& range) const
    {
        if (verbose)
            std::cout << getThreadNum()<<"# :Start from row " << range.start << " to "  << range.end-1<<" ("<<range.end-range.start<<" loops)" << std::endl;
		float                *f2;
		int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
		double *yp = new double[tailleSequence], *ym = new double[tailleSequence];
		int cols = img.cols;

		// Formule 12 p193
		double				b1 = -2 * exp(-a)*cosh(w);
		double				a1 = 2 * exp(-a)*cosh(w) - exp(-2 * a) - 1;
		double				b2 = exp(-2 * a);

        switch(img.depth()){
        case CV_8U :
        case CV_8S :
        {
			for (int i = range.start; i<range.end; i++)
			{
				// Formule 26 p194
				unsigned char *c1 = (unsigned char*)img.ptr(i);// c1 pointeur sur x(i-1)
				double border = *c1;
				yp[0] = *c1;
				c1++;
				yp[1] = *c1 - b1*yp[0] - b2*border;
				c1++;
				for (int j = 2; j<cols; j++, c1++)
					yp[j] = *c1 - b1*yp[j - 1] - b2*yp[j - 2];
				// Formule 27 p194
				c1 = (unsigned char*)img.ptr(i)+cols-1;
				border = *c1;
				ym[cols - 1] = *c1;
				c1--;
				ym[cols - 2] = *c1 - b1*ym[cols - 1];
				c1--;
				for (int j = cols - 3; j >= 0; j--, c1--)
					ym[j] = *c1 - b1*ym[j + 1] - b2*ym[j + 2];
				// Formule 25 p193
				f2 = ((float*)im1.ptr(i));
				for (int j = 0; j<cols; j++, f2 ++)
					*f2 = (float)(a1*(ym[j] - yp[j]));
			}
		}
            break;
        case CV_16S :
        case CV_16U :
            {
				for (int i = range.start; i<range.end; i++)
				{
					// Formule 26 p194
					unsigned short *c1 = (unsigned short*)img.ptr(i);// c1 pointeur sur x(i-1)
					f2 = ((float*)im1.ptr(i));
					double border = *c1;
					yp[0] = *c1;
					c1++;
					yp[1] = *c1 - b1*yp[0] - b2*border;
					c1++;
					for (int j = 2; j<cols; j++, c1++)
						yp[j] = *c1 - b1*yp[j - 1] - b2*yp[j - 2];
					// Formule 27 p194
					c1 = (unsigned short*)img.ptr(i) + cols - 1;
					border = *c1;
					ym[cols - 1] = *c1;
					c1--;
					ym[cols - 2] = *c1 - b1*ym[cols - 1];
					c1--;
					for (int j = cols - 3; j >= 0; j--, c1--)
						ym[j] = *c1 - b1*ym[j + 1] - b2*ym[j + 2];
					// Formule 25 p193
					for (int j = 0; j<cols; j++, f2++)
						*f2 = (float)(a1*(ym[i] - yp[i]));
				}
		}
            break;
        default :
            return ;
            }
        delete []ym;
        delete []yp;
    };
    ParallelGradientPaillouXRows& operator=(const ParallelGradientPaillouXRows &) {
         return *this;
    };
};

UMat GradientPaillouY(UMat op, double a,double w)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientPaillouYCols x(planSrc[i],planTmp[i],a,w);
            parallel_for_(Range(0,opSrc.cols), x,getNumThreads());
            ParallelGradientPaillouYRows xr(planTmp[i],planDst[i],a,w);
            parallel_for_(Range(0,opSrc.rows), xr,getNumThreads());

        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}

UMat GradientPaillouX(UMat op, double a,double w)
{
    Mat tmp(op.size(),CV_32FC(op.channels()));
    UMat imDst(op.rows,op.cols,CV_32FC(op.channels()));
    cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
    cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
    std::vector<Mat> planSrc;
    split(opSrc,planSrc);
    std::vector<Mat> planTmp;
    split(tmp,planTmp);
    std::vector<Mat> planDst;
    split(dst,planDst);
    for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
    {
        if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
        {
            ParallelGradientPaillouXRows x(planSrc[i],planTmp[i],a,w);
            parallel_for_(Range(0,opSrc.rows), x,getNumThreads());
            ParallelGradientPaillouXCols xr(planTmp[i],planDst[i],a,w);
            parallel_for_(Range(0,opSrc.cols), xr,getNumThreads());
        }
        else
            std::cout << "PB";
    }
    merge(planDst,imDst);
    return imDst;
}
}
}
