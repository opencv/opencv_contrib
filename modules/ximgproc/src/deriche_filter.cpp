#include "precomp.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>
#include <vector>
#include <iostream>

namespace cv {
namespace ximgproc {

/*
If you use this code please cite this @cite deriche1987using
Using Canny's Criteria to Derive a Recursively Implemented Optimal Edge Detector International Journal of Computer Vision,167-187 (1987)
*/

	class ParallelGradientDericheYCols : public ParallelLoopBody
	{
	private:
		Mat &img;
		Mat &im1;
		double alphaDerive;
		bool verbose;
	public:
		ParallelGradientDericheYCols(Mat& imgSrc, Mat &d, double ald) :
			img(imgSrc),
			im1(d),
			alphaDerive(ald),
			verbose(false)
		{}
		void Verbose(bool b) { verbose = b; }
		virtual void operator()(const Range& range) const
		{
			if (verbose)
				std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;

			float                *f2;
			int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
			double *g1 = new double[tailleSequence], *g2 = new double[tailleSequence];
			double    kp = pow(1 - exp(-alphaDerive), 2.0) / exp(-alphaDerive);
			double a1, a2, a3, a4;
			double b1, b2;
			int rows = img.rows, cols = img.cols;

			kp = pow(1 - exp(-alphaDerive), 2.0) / exp(-alphaDerive);
			a1 = 0;
			a2 = kp*exp(-alphaDerive), a3 = -kp*exp(-alphaDerive);
			a4 = 0;
			b1 = 2 * exp(-alphaDerive);
			b2 = -exp(-2 * alphaDerive);

			switch (img.depth()) {
			case CV_8U:
			{
				unsigned char *c1;
				for (int j = range.start; j<range.end; j++)
				{
					// Causal vertical  IIR filter
					c1 = (unsigned char*)img.ptr(0);
					f2 = (float*)im1.ptr(0);
					f2 += j;
					c1 += j;
					int i = 0;
					g1[i] = (a1 + a2)* *c1;
					i++;
					c1 += cols;
					g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1)* g1[i - 1];
					i++;
					c1 += cols;
					for (i = 2; i<rows; i++, c1 += cols)
						g1[i] = a1 * *c1 + a2 * c1[-cols] + b1*g1[i - 1] + b2 *g1[i - 2];
					// Anticausal vertical IIR filter
					c1 = (unsigned char*)img.ptr(0);
					c1 += (rows - 1)*cols + j;
					i = rows - 1;
					g2[i] = (a3 + a4)* *c1;
					i--;
					c1 -= cols;
					g2[i] = a3* c1[cols] + a4 * c1[cols] + (b1)*g2[i + 1];
					i--;
					c1 -= cols;
					for (i = rows - 3; i >= 0; i--, c1 -= cols)
						g2[i] = a3*c1[cols] + a4* c1[2 * cols] +
						b1*g2[i + 1] + b2*g2[i + 2];
					for (i = 0; i<rows; i++, f2 += cols)
						*f2 = (float)(g1[i] + g2[i]);
				}
			}
			break;
			case CV_16S:
			case CV_16U:
			{
				unsigned short *c1;
				for (int j = range.start; j<range.end; j++)
				{
					c1 = ((unsigned short*)img.ptr(0));
					f2 = ((float*)im1.ptr(0));
					f2 += j;
					c1 += j;
					int i = 0;
					g1[i] = (a1 + a2)* *c1;
					i++;
					c1 += cols;
					g1[i] = a1 * *c1 + a2 * c1[-cols] + (b1)* g1[i - 1];
					i++;
					c1 += cols;
					for (i = 2; i<rows; i++, c1 += cols)
						g1[i] = a1 * *c1 + a2 * c1[-cols] + b1*g1[i - 1] + b2 *g1[i - 2];
					// Anticausal vertical IIR filter
					c1 = ((unsigned short*)img.ptr(0));
					c1 += (rows - 1)*cols + j;
					i = rows - 1;
					g2[i] = (a3 + a4)* *c1;
					i--;
					c1 -= cols;
					g2[i] = (a3 + a4)* c1[cols] + (b1)*g2[i + 1];
					i--;
					c1 -= cols;
					for (i = rows - 3; i >= 0; i--, c1 -= cols)
						g2[i] = a3*c1[cols] + a4* c1[2 * cols] + b1*g2[i + 1] + b2*g2[i + 2];
					c1 = ((unsigned short*)img.ptr(0)) + j;
					for (i = 0; i<rows; i++, f2 += cols, c1 += cols)
						*f2 = 0 * *c1 + (float)(g1[i] + g2[i]);
				}
			}
			break;
			case CV_32S:
				break;
			case CV_32F:
				break;
			case CV_64F:
				break;
			default:
				delete[]g1;
				delete[]g2;
				return;
			}
			delete[]g1;
			delete[]g2;
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
		double alphaMoyenne;
		bool verbose;

	public:
		ParallelGradientDericheYRows(Mat& imgSrc, Mat &d, double alm) :
			img(imgSrc),
			dst(d),
			alphaMoyenne(alm),
			verbose(false)
		{}
		void Verbose(bool b) { verbose = b; }
		virtual void operator()(const Range& range) const
		{
			if (verbose)
				std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
			float *f1, *f2;
			int tailleSequence = (img.rows>img.cols) ? img.rows : img.cols;
			double *g1 = new double[tailleSequence], *g2 = new double[tailleSequence];
			double k, a5, a6, a7, a8;
			double b3, b4;
			int cols = img.cols;

			k = pow(1 - exp(-alphaMoyenne), 2.0) / (1 + 2 * alphaMoyenne*exp(-alphaMoyenne) - exp(-2 * alphaMoyenne));
			a5 = k;
			a6 = k*exp(-alphaMoyenne)*(alphaMoyenne - 1);
			a7 = k*exp(-alphaMoyenne)*(alphaMoyenne + 1);
			a8 = -k*exp(-2 * alphaMoyenne);
			b3 = 2 * exp(-alphaMoyenne);
			b4 = -exp(-2 * alphaMoyenne);

			for (int i = range.start; i<range.end; i++)
			{
				f2 = ((float*)dst.ptr(i));
				f1 = ((float*)img.ptr(i));
				int j = 0;
				g1[j] = (a5 + a6)* *f1;
				j++;
				f1++;
				g1[j] = a5 * f1[0] + a6*f1[j - 1] + (b3)* g1[j - 1];
				j++;
				f1++;
				for (j = 2; j<cols; j++, f1++)
					g1[j] = a5 * f1[0] + a6 * f1[-1] + b3*g1[j - 1] + b4*g1[j - 2];
				f1 = ((float*)img.ptr(0));
				f1 += i*cols + cols - 1;
				j = cols - 1;
				g2[j] = (a7 + a8)* *f1;
				j--;
				f1--;
				g2[j] = (a7 + a8) * f1[1] + (b3)* g2[j + 1];
				j--;
				f1--;
				for (j = cols - 3; j >= 0; j--, f1--)
					g2[j] = a7*f1[1] + a8*f1[2] + b3*g2[j + 1] + b4*g2[j + 2];
				for (j = 0; j<cols; j++, f2++)
					*f2 = (float)(g1[j] + g2[j]);
			}
			delete[]g1;
			delete[]g2;

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
		double alphaMoyenne;
		bool verbose;

	public:
		ParallelGradientDericheXCols(Mat& imgSrc, Mat &d, double alm) :
			img(imgSrc),
			dst(d),
			alphaMoyenne(alm),
			verbose(false)
		{}
		void Verbose(bool b) { verbose = b; }
		virtual void operator()(const Range& range) const
		{

			if (verbose)
				std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
			float                *f1, *f2;
			int rows = img.rows, cols = img.cols;

			int tailleSequence = (rows>cols) ? rows : cols;
			double *g1 = new double[tailleSequence], *g2 = new double[tailleSequence];
			double k, a5, a6, a7, a8 = 0;
			double b3, b4;

			k = pow(1 - exp(-alphaMoyenne), 2.0) / (1 + 2 * alphaMoyenne*exp(-alphaMoyenne) - exp(-2 * alphaMoyenne));
			a5 = k, a6 = k*exp(-alphaMoyenne)*(alphaMoyenne - 1);
			a7 = k*exp(-alphaMoyenne)*(alphaMoyenne + 1), a8 = -k*exp(-2 * alphaMoyenne);
			b3 = 2 * exp(-alphaMoyenne);
			b4 = -exp(-2 * alphaMoyenne);

			for (int j = range.start; j<range.end; j++)
			{
				f1 = (float*)img.ptr(0);
				f1 += j;
				int i = 0;
				g1[i] = (a5 + a6)* *f1;
				i++;
				f1 += cols;
				g1[i] = a5 * *f1 + a6 * f1[-cols] + (b3)* g1[i - 1];
				i++;
				f1 += cols;
				for (i = 2; i<rows; i++, f1 += cols)
					g1[i] = a5 * *f1 + a6 * f1[-cols] + b3*g1[i - 1] + b4 *g1[i - 2];
				f1 = (float*)img.ptr(0);
				f1 += (rows - 1)*cols + j;
				i = rows - 1;
				g2[i] = (a7 + a8)* *f1;
				i--;
				f1 -= cols;
				g2[i] = (a7 + a8)* f1[cols] + (b3)*g2[i + 1];
				i--;
				f1 -= cols;
				for (i = rows - 3; i >= 0; i--, f1 -= cols)
					g2[i] = a7*f1[cols] + a8* f1[2 * cols] +
					b3*g2[i + 1] + b4*g2[i + 2];
				for (i = 0; i<rows; i++, f2 += cols)
				{
					f2 = ((float*)dst.ptr(i)) + (j*img.channels());
					*f2 = (float)(g1[i] + g2[i]);
				}
			}
			delete[]g1;
			delete[]g2;
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
		double alphaDerive;
		bool verbose;

	public:
		ParallelGradientDericheXRows(Mat& imgSrc, Mat &d, double ald) :
			img(imgSrc),
			dst(d),
			alphaDerive(ald),
			verbose(false)
		{}
		void Verbose(bool b) { verbose = b; }
		virtual void operator()(const Range& range) const
		{
			if (verbose)
				std::cout << getThreadNum() << "# :Start from row " << range.start << " to " << range.end - 1 << " (" << range.end - range.start << " loops)" << std::endl;
			float *f1;
			int rows = img.rows, cols = img.cols;
			int tailleSequence = (rows>cols) ? rows : cols;
			double *g1 = new double[tailleSequence], *g2 = new double[tailleSequence];
			double kp;;
			double a1, a2, a3, a4;
			double b1, b2;

			kp = pow(1 - exp(-alphaDerive), 2.0) / exp(-alphaDerive);
			a1 = 0;
			a2 = kp*exp(-alphaDerive);
			a3 = -kp*exp(-alphaDerive);
			a4 = 0;
			b1 = 2 * exp(-alphaDerive);
			b2 = -exp(-2 * alphaDerive);

			switch (img.depth()) {
			case CV_8U:
			case CV_8S:
			{
				unsigned char *c1;
				for (int i = range.start; i<range.end; i++)
				{
					f1 = (float*)dst.ptr(i);
					c1 = (unsigned char*)img.ptr(i);
					int j = 0;
					g1[j] = (a1 + a2)* *c1;
					j++;
					c1++;
					g1[j] = a1 * c1[0] + a2*c1[j - 1] + (b1)* g1[j - 1];
					j++;
					c1++;
					for (j = 2; j<cols; j++, c1++)
						g1[j] = a1 * c1[0] + a2 * c1[-1] + b1*g1[j - 1] + b2*g1[j - 2];
					c1 = (unsigned char*)img.ptr(0);
					c1 += i*cols + cols - 1;
					j = cols - 1;
					g2[j] = (a3 + a4)* *c1;
					j--;
					g2[j] = (a3 + a4) * c1[1] + b1 * g2[j + 1];
					j--;
					c1--;
					for (j = cols - 3; j >= 0; j--, c1--)
						g2[j] = a3*c1[1] + a4*c1[2] + b1*g2[j + 1] + b2*g2[j + 2];
					for (j = 0; j<cols; j++, f1++)
						*f1 = (float)(g1[j] + g2[j]);
				}
			}
			break;
			case CV_16S:
			case CV_16U:
			{
				unsigned short *c1;
				f1 = ((float*)dst.ptr(0));
				for (int i = range.start; i<range.end; i++)
				{
					c1 = ((unsigned short*)img.ptr(0));
					c1 += i*cols;
					int j = 0;
					g1[j] = (a1 + a2)* *c1;
					j++;
					c1++;
					g1[j] = a1 * c1[0] + a2*c1[j - 1] + (b1)* g1[j - 1];
					j++;
					c1++;
					for (j = 2; j<cols; j++, c1++)
						g1[j] = a1 * c1[0] + a2 * c1[-1] + b1*g1[j - 1] + b2*g1[j - 2];
					c1 = ((unsigned short*)img.ptr(0));
					c1 += i*cols + cols - 1;
					j = cols - 1;
					g2[j] = (a3 + a4)* *c1;
					j--;
					c1--;
					g2[j] = (a3 + a4) * c1[1] + (b1)* g2[j + 1];
					j--;
					c1--;
					for (j = cols - 3; j >= 0; j--, c1--)
						g2[j] = a3*c1[1] + a4*c1[2] + b1*g2[j + 1] + b2*g2[j + 2];
					for (j = 0; j<cols; j++, f1++)
						*f1 = (float)(g1[j] + g2[j]);
				}
			}
			break;
			default:
				return;
			}
			delete[]g1;
			delete[]g2;
		};
		ParallelGradientDericheXRows& operator=(const ParallelGradientDericheXRows &) {
			return *this;
		};
	};

	UMat GradientDericheY(UMat op, double alphaDerive, double alphaMean)
	{
		Mat tmp(op.size(), CV_32FC(op.channels()));
		UMat imDst(op.rows, op.cols, CV_32FC(op.channels()));
		cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
		cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
		std::vector<Mat> planSrc;
		split(opSrc, planSrc);
		std::vector<Mat> planTmp;
		split(tmp, planTmp);
		std::vector<Mat> planDst;
		split(dst, planDst);
		for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
		{
			if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
			{
				ParallelGradientDericheYCols x(planSrc[i], planTmp[i], alphaDerive);
				parallel_for_(Range(0, opSrc.cols), x, getNumThreads());
				ParallelGradientDericheYRows xr(planTmp[i], planDst[i], alphaMean);
				parallel_for_(Range(0, opSrc.rows), xr, getNumThreads());

			}
			else
				std::cout << "PB";
		}
		merge(planDst, imDst);
		return imDst;
	}

	UMat GradientDericheX(UMat op, double alphaDerive, double alphaMean)
	{
		Mat tmp(op.size(), CV_32FC(op.channels()));
		UMat imDst(op.rows, op.cols, CV_32FC(op.channels()));
		cv::Mat opSrc = op.getMat(cv::ACCESS_RW);
		cv::Mat dst = imDst.getMat(cv::ACCESS_RW);
		std::vector<Mat> planSrc;
		split(opSrc, planSrc);
		std::vector<Mat> planTmp;
		split(tmp, planTmp);
		std::vector<Mat> planDst;
		split(dst, planDst);
		for (int i = 0; i < static_cast<int>(planSrc.size()); i++)
		{
			if (planSrc[i].isContinuous() && planTmp[i].isContinuous() && planDst[i].isContinuous())
			{
				ParallelGradientDericheXRows x(planSrc[i], planTmp[i], alphaDerive);
				parallel_for_(Range(0, opSrc.rows), x, getNumThreads());
				ParallelGradientDericheXCols xr(planTmp[i], planDst[i], alphaMean);
				parallel_for_(Range(0, opSrc.cols), xr, getNumThreads());
			}
			else
				std::cout << "PB";
		}
		merge(planDst, imDst);
		return imDst;
	}

}
}
