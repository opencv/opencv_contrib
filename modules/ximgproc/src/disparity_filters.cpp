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
		void setradius(double _radius) {radius = _radius;}
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
	public:
		static Ptr<DisparityWLSFilter> create(Size2i src_size);
		void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map);
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
		default:
			CV_Error(Error::StsBadArg, "Unsupported disparity map filter type");
		}
		return res;
	}
}
}
