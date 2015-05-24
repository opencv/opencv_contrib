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

namespace cv {
namespace ximgproc {

void setROI(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,
            Mat& disp,Mat& src,Mat& dst,Rect ROI);

class DisparityDTFilterImpl : public DisparityDTFilter
{
protected:
    double sigmaSpatial,sigmaColor;
    void init(double _sigmaSpatial, double _sigmaColor)
    {
        sigmaColor = _sigmaColor;
        sigmaSpatial = _sigmaSpatial;
    }
public:
    double getSigmaSpatial() {return sigmaSpatial;}
    void setSigmaSpatial(double _sigmaSpatial) {sigmaSpatial = _sigmaSpatial;}
    double getSigmaColor() {return sigmaColor;}
    void setSigmaColor(double _sigmaColor) {sigmaColor = _sigmaColor;}

    static Ptr<DisparityDTFilterImpl> create()
    {
        DisparityDTFilterImpl *dtf = new DisparityDTFilterImpl();
        dtf->init(25.0,60.0); 
        return Ptr<DisparityDTFilterImpl>(dtf);
    }

    void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
    {
        Mat disp,src,dst;
        setROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);

        Mat disp_32F,dst_32F;
        disp.convertTo(disp_32F,CV_32F);
        dtFilter(src,disp_32F,dst_32F,sigmaSpatial,sigmaColor);
        dst_32F.convertTo(dst,CV_16S);
    }
};

CV_EXPORTS_W
Ptr<DisparityDTFilter> createDisparityDTFilter()
{
    return Ptr<DisparityDTFilter>(DisparityDTFilterImpl::create());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
    
class DisparityGuidedFilterImpl : public DisparityGuidedFilter
{
protected:
    int radius;
    double eps;
    void init(int _radius, double _eps)
    {
        radius = _radius;
        eps = _eps;
    }
public:
    double getEps() {return eps;}
    void setEps(double _eps) {eps = _eps;}
    int getRadius() {return radius;}
    void setRadius(int _radius) {radius = _radius;}

    static Ptr<DisparityGuidedFilterImpl> create()
    {
        DisparityGuidedFilterImpl *gf = new DisparityGuidedFilterImpl();
        gf->init(20,100.0);
        return Ptr<DisparityGuidedFilterImpl>(gf);
    }

    void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
    {
        Mat disp,src,dst;
        setROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);

        Mat disp_32F,dst_32F;
        disp.convertTo(disp_32F,CV_32F);
        guidedFilter(src,disp_32F,dst_32F,radius,eps);
        dst_32F.convertTo(dst,CV_16S);
    }
};

CV_EXPORTS_W
Ptr<DisparityGuidedFilter> createDisparityGuidedFilter()
{
    return Ptr<DisparityGuidedFilter>(DisparityGuidedFilterImpl::create());
}
    
//////////////////////////////////////////////////////////////////////////////////////////////////////

class DisparityWLSFilterImpl : public DisparityWLSFilter
{
protected:
    double lambda,sigma_color;
    void init(double _lambda, double _sigma_color)
    {
        lambda = _lambda;
        sigma_color = _sigma_color;
    }
public:
    double getLambda() {return lambda;}
    void setLambda(double _lambda) {lambda = _lambda;}
    double getSigmaColor() {return sigma_color;}
    void setSigmaColor(double _sigma_color) {sigma_color = _sigma_color;}

    static Ptr<DisparityWLSFilterImpl> create()
    {
        DisparityWLSFilterImpl *wls = new DisparityWLSFilterImpl();
        wls->init(500.0,2.0);
        return Ptr<DisparityWLSFilterImpl>(wls);
    }

    void filter(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,Rect ROI)
    {
        Mat disp,src,dst;
        setROI(disparity_map,left_view,filtered_disparity_map,disp,src,dst,ROI);
        weightedLeastSquaresFilter(src,disp,dst,lambda,sigma_color);
    }
};

CV_EXPORTS_W
Ptr<DisparityWLSFilter> createDisparityWLSFilter()
{
    return Ptr<DisparityWLSFilter>(DisparityWLSFilterImpl::create());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void setROI(InputArray disparity_map, InputArray left_view, OutputArray filtered_disparity_map,
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

}
}
