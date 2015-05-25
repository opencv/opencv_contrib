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
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <math.h>

namespace cv
{
namespace ximgproc
{
	void SetROI(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,
				Mat& disp,Mat& src,Mat& dst,Rect ROI)
	{
		Mat disp_full_size = disparity_map.getMat();
		Mat src_full_size = left_view.getMat();
		CV_Assert( (disp_full_size.depth() == CV_16S) && (disp_full_size.channels() == 1) );
		CV_Assert( (src_full_size.depth()  == CV_8U)  && (src_full_size.channels()  <= 4) );
		disp = Mat(disp_full_size,ROI);
		src  = Mat(src_full_size ,ROI);
		filtered_disparity_map.create(disp_full_size.size(), disp_full_size.type());
		Mat& dst_full_size = filtered_disparity_map.getMatRef();
		dst = Mat(dst_full_size,ROI);
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////

	class DisparityDTFilter : public DisparityMapFilter
	{
	protected:
		double sigmaSpatial, sigmaColor;
		DisparityDTFilter() {}
		void init(double _sigmaSpatial, double _sigmaColor)
		{
			sigmaColor = _sigmaColor;
			sigmaSpatial = _sigmaSpatial;
		}
	public:
		double getsigmaSpatial() {return sigmaSpatial;}
		void setsigmaSpatial(double _sigmaSpatial) {sigmaSpatial = _sigmaSpatial;}
		double getsigmaColor() {return sigmaColor;}
		void setsigmaColor(double _sigmaColor) {sigmaColor = _sigmaColor;}
		static Ptr<DisparityDTFilter> create()
		{
			DisparityDTFilter *dtf = new DisparityDTFilter();
			dtf->init(25.0,20.0); 
			return Ptr<DisparityDTFilter>(dtf);
		}
		void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
		{
			Mat disp,src,dst;
			SetROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);

			Mat disp_32F,dst_32F;
			disp.convertTo(disp_32F,CV_32F);
			dtFilter(src,disp_32F,dst_32F,sigmaSpatial,sigmaColor);
			dst_32F.convertTo(dst,CV_16S);
		}
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////

	class DisparityGuidedFilter : public DisparityMapFilter
	{
	protected:
		int radius; 
		double eps;
		DisparityGuidedFilter() {}
		void init(int _radius, double _eps)
		{
			radius = _radius;
			eps = _eps;
		}
	public:
		double geteps() {return eps;}
		void seteps(double _eps) {eps = _eps;}
		int getradius() {return radius;}
		void setradius(int _radius) {radius = _radius;}
		static Ptr<DisparityGuidedFilter> create()
		{
			DisparityGuidedFilter *gf = new DisparityGuidedFilter();
			gf->init(20,1.0);
			return Ptr<DisparityGuidedFilter>(gf);
		}
		void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
		{
			Mat disp,src,dst;
			SetROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);

			Mat disp_32F,dst_32F;
			disp.convertTo(disp_32F,CV_32F);
			guidedFilter(src,disp_32F,dst_32F,radius,eps);
			dst_32F.convertTo(dst,CV_16S);
		}
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////

	class DisparityWLSFilter : public DisparityMapFilter
	{
	protected:
		double sigmaColor,lambda;
		int num_iter;
		double *weights_LUT;
		double *Ahor,  *Bhor,  *Chor, 
			   *Avert, *Bvert, *Cvert;
		double *interD, *interE, *cur_res;
		DisparityWLSFilter() {}
		void InitLUT()
		{
			int num_levels = 3*256*256;
			weights_LUT = new double[num_levels];
			for(int i=0;i<num_levels;i++)
				weights_LUT[i] = exp(-sqrt((double)i)/sigmaColor);
		}
		void init(double _sigmaColor, double _lambda,int _num_iter)
		{
			sigmaColor = _sigmaColor;
			lambda = _lambda;
			num_iter = _num_iter;
			InitLUT();
		}
		void BuildCoefMatrices(Mat& guide)
		{
			//assuming 3-channel guide (for now)
			int w = guide.cols;
			int h = guide.rows;
			double hor_weight;
			const unsigned char *row,*row_prev,*row_next;
			for(int i=0;i<h;i++)
			{
				//compute horizontal coefs:
				row = guide.ptr(i);
				Ahor[i*w] = 0;
				hor_weight = weights_LUT[ (row[0]-row[3])*(row[0]-row[3])+
										  (row[1]-row[4])*(row[1]-row[4])+
										  (row[2]-row[5])*(row[2]-row[5]) ];
				Chor[i*w] = -lambda*hor_weight;
				Bhor[i*w] = 1 - Ahor[i*w] - Chor[i*w];
				row+=3;
				for(int j=1;j<w-1;j++)
				{
					Ahor[i*w+j] = -lambda*hor_weight;
					hor_weight = weights_LUT[ (row[0]-row[3])*(row[0]-row[3])+
											  (row[1]-row[4])*(row[1]-row[4])+
											  (row[2]-row[5])*(row[2]-row[5]) ];
					Chor[i*w+j] = -lambda*hor_weight;
					Bhor[i*w+j] = 1 - Ahor[i*w+j] - Chor[i*w+j];
					row+=3;
				}
				Ahor[i*w+w-1] = -lambda*hor_weight;
				Chor[i*w+w-1] = 0;
				Bhor[i*w+w-1] = 1 - Ahor[i*w+w-1] - Chor[i*w+w-1];

				//compute vertical coefs:
				row = guide.ptr(i);
				if(i==0)
				{
					row_next = guide.ptr(i+1);
					for(int j=0;j<w;j++)
					{
						Avert[i*w+j] = 0;
						Cvert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_next[0])*(row[0]-row_next[0])+
															(row[1]-row_next[1])*(row[1]-row_next[1])+
															(row[2]-row_next[2])*(row[2]-row_next[2]) ];
						Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
						row+=3;
						row_next+=3;
					}
				}
				else if(i==h-1)
				{
					row_prev = guide.ptr(i-1);
					for(int j=0;j<w;j++)
					{
						Avert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_prev[0])*(row[0]-row_prev[0])+
														   (row[1]-row_prev[1])*(row[1]-row_prev[1])+
														   (row[2]-row_prev[2])*(row[2]-row_prev[2]) ];
						Cvert[i*w+j] = 0;
						Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
						row+=3;
						row_prev+=3;
					}
				}
				else
				{
					row_prev = guide.ptr(i-1);
					row_next = guide.ptr(i+1);
					for(int j=0;j<w;j++)
					{
						Avert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_prev[0])*(row[0]-row_prev[0])+
														   (row[1]-row_prev[1])*(row[1]-row_prev[1])+
														   (row[2]-row_prev[2])*(row[2]-row_prev[2]) ];
						Cvert[i*w+j] = -lambda*weights_LUT[ (row[0]-row_next[0])*(row[0]-row_next[0])+
															(row[1]-row_next[1])*(row[1]-row_next[1])+
															(row[2]-row_next[2])*(row[2]-row_next[2]) ];
						Bvert[i*w+j] = 1 - Avert[i*w+j] - Cvert[i*w+j];
						row+=3;
						row_prev+=3;
						row_next+=3;
					}
				}
			}
		}
		void HorizontalPass(double* cur,int w,int h)
		{
			double denom;
			for(int i=0;i<h;i++)
			{
				//forward pass:
				interD[i*w] = Chor[i*w]/Bhor[i*w];
				interE[i*w] = cur[i*w] /Bhor[i*w];
				for(int j=1;j<w;j++)
				{
					denom = Bhor[i*w+j]-interD[i*w+j-1]*Ahor[i*w+j];
					interD[i*w+j] = Chor[i*w+j]/denom;
					interE[i*w+j] = (cur[i*w+j]-interE[i*w+j-1]*Ahor[i*w+j])/denom;
				}

				//backward pass:
				cur[i*w+w-1] = interE[i*w+w-1];
				for(int j=w-2;j>=0;j--)
					cur[i*w+j] = interE[i*w+j]-interD[i*w+j]*cur[i*w+j+1];
			}
		}
		void VerticalPass(double* cur,int w,int h)
		{
			double denom;
			//forward pass:
			for(int j=0;j<w;j++)
			{
				interD[j] = Cvert[j]/Bvert[j];
				interE[j] = cur[j]/Bvert[j];
			}
			for(int i=1;i<h;i++)
			{
				for(int j=0;j<w;j++)
				{
					denom = Bvert[i*w+j]-interD[(i-1)*w+j]*Avert[i*w+j];
					interD[i*w+j] = Cvert[i*w+j]/denom;
					interE[i*w+j] = (cur[i*w+j]-interE[(i-1)*w+j]*Avert[i*w+j])/denom;
				}
			}
			//backward pass:
			for(int j=0;j<w;j++)
				cur[(h-1)*w+j] = interE[(h-1)*w+j];
			for(int i=h-2;i>=0;i--)
				for(int j=w-1;j>=0;j--)
					cur[i*w+j] = interE[i*w+j]-interD[i*w+j]*cur[(i+1)*w+j];
		}
	public:
		static Ptr<DisparityWLSFilter> create()
		{
			DisparityWLSFilter *wls = new DisparityWLSFilter();
			wls->init(2.0,200.0,3);
			return Ptr<DisparityWLSFilter>(wls);
		}
		void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
		{
			Mat disp,src,dst;
			SetROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);
			int w = disp.cols;
			int h = disp.rows;
			double* memory_chunk = new double[9*w*h];
			Ahor   = memory_chunk+0;     Bhor   = memory_chunk+w*h;   Chor    = memory_chunk+2*w*h;
			Avert  = memory_chunk+3*w*h; Bvert  = memory_chunk+4*w*h; Cvert   = memory_chunk+5*w*h;
			interD = memory_chunk+6*w*h; interE = memory_chunk+7*w*h; cur_res = memory_chunk+8*w*h;

			BuildCoefMatrices(src);

			short* row;
			for(int i=0;i<h;i++)
			{
				row = (short*)disp.ptr(i);
				for(int j=0;j<w;j++)
				{
					cur_res[i*w+j] = (double)(*row);
					row++;
				}
			}

			for(int n=0;n<num_iter;n++)
			{
				HorizontalPass(cur_res,w,h);
				VerticalPass(cur_res,w,h);
			}

			for(int i=0;i<h;i++)
			{
				row = (short*)dst.ptr(i);
				for(int j=0;j<w;j++)
				{
					*row = saturate_cast<short>(cur_res[i*w+j]);
					row++;
				}
			}

			delete[] memory_chunk;
		}
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	
	CV_EXPORTS_W
	Ptr<DisparityMapFilter> createDisparityMapFilter(int mode)
	{
		Ptr<DisparityMapFilter> res;
		switch(mode)
		{
		case DTF:
			res = Ptr<DisparityMapFilter>(DisparityDTFilter::create());
			break;
		case GF:
			res = Ptr<DisparityMapFilter>(DisparityGuidedFilter::create());
			break;
		case WLS:
			res = Ptr<DisparityMapFilter>(DisparityWLSFilter::create());
			break;
		default:
			CV_Error(Error::StsBadArg, "Unsupported disparity map filter type");
		}
		return res;
	}
}
}
