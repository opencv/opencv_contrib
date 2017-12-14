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

#include "perf_precomp.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

namespace cvtest
{

using std::tr1::tuple;
using std::tr1::get;
using namespace perf;
using namespace testing;
using namespace cv;
using namespace cv::ximgproc;

void MakeArtificialExample(RNG rng, Mat& dst_left_view, Mat& dst_left_disparity_map, Mat& dst_right_disparity_map, Rect& dst_ROI);

CV_ENUM(GuideTypes, CV_8UC3);
CV_ENUM(SrcTypes, CV_16S);
typedef tuple<GuideTypes, SrcTypes, Size, bool, bool> DisparityWLSParams;

typedef TestBaseWithParam<DisparityWLSParams> DisparityWLSFilterPerfTest;

PERF_TEST_P( DisparityWLSFilterPerfTest, perf, Combine(GuideTypes::all(), SrcTypes::all(), Values(sz720p), Values(true,false), Values(true,false)) )
{
    RNG rng(0);

    DisparityWLSParams params = GetParam();
    int guideType        = get<0>(params);
    int srcType          = get<1>(params);
    Size sz              = get<2>(params);
    bool use_conf        = get<3>(params);
    bool use_downscale   = get<4>(params);

    Mat guide(sz, guideType);
    Mat disp_left(sz, srcType);
    Mat disp_right(sz, srcType);
    Mat dst(sz, srcType);
    Rect ROI;

    MakeArtificialExample(rng,guide,disp_left,disp_right,ROI);
    if(use_downscale)
    {
        resize(disp_left,disp_left,Size(),0.5,0.5, INTER_LINEAR_EXACT);
        disp_left/=2;
        resize(disp_right,disp_right,Size(),0.5,0.5, INTER_LINEAR_EXACT);
        disp_right/=2;
        ROI = Rect(ROI.x/2,ROI.y/2,ROI.width/2,ROI.height/2);
    }

    cv::setNumThreads(cv::getNumberOfCPUs());
    TEST_CYCLE_N(10)
    {
        Ptr<DisparityWLSFilter> wls_filter = createDisparityWLSFilterGeneric(use_conf);
        wls_filter->filter(disp_left,guide,dst,disp_right,ROI);
    }

    SANITY_CHECK_NOTHING();
}

void MakeArtificialExample(RNG rng, Mat& dst_left_view, Mat& dst_left_disparity_map, Mat& dst_right_disparity_map, Rect& dst_ROI)
{
    int w = dst_left_view.cols;
    int h = dst_left_view.rows;

    //params:
    unsigned char bg_level = (unsigned char)rng.uniform(0.0,255.0);
    unsigned char fg_level = (unsigned char)rng.uniform(0.0,255.0);
    int rect_width  = (int)rng.uniform(w/16,w/2);
    int rect_height = (int)rng.uniform(h/16,h/2);
    int rect_disparity = (int)(0.15*w); //typical maximum disparity value
    double sigma = 6.0;

    int rect_x_offset = (w-rect_width) /2;
    int rect_y_offset = (h-rect_height)/2;

    if(dst_left_view.channels()==3)
        dst_left_view = Scalar(Vec3b(bg_level,bg_level,bg_level));
    else
        dst_left_view = Scalar(bg_level);
    dst_left_disparity_map  = Scalar(0);
    dst_right_disparity_map = Scalar(0);
    Mat dst_left_view_rect =          Mat(dst_left_view,         Rect(rect_x_offset,rect_y_offset,rect_width,rect_height));
    Mat dst_left_disparity_map_rect = Mat(dst_left_disparity_map,Rect(rect_x_offset,rect_y_offset,rect_width,rect_height));
    if(dst_left_view.channels()==3)
        dst_left_view_rect = Scalar(Vec3b(fg_level,fg_level,fg_level));
    else
        dst_left_view_rect = Scalar(fg_level);
    dst_left_disparity_map_rect = Scalar(16*rect_disparity);

    rect_x_offset-=rect_disparity;
    Mat dst_right_disparity_map_rect = Mat(dst_right_disparity_map,Rect(rect_x_offset,rect_y_offset,rect_width,rect_height));
    dst_right_disparity_map_rect = Scalar(-16*rect_disparity);

    //add some gaussian noise:
    unsigned char *l;
    short *ldisp, *rdisp;
    for(int i=0;i<h;i++)
    {
        l = dst_left_view.ptr(i);
        ldisp = (short*)dst_left_disparity_map.ptr(i);
        rdisp = (short*)dst_right_disparity_map.ptr(i);

        if(dst_left_view.channels()==3)
        {
            for(int j=0;j<w;j++)
            {
                l[0] = saturate_cast<unsigned char>(l[0] + rng.gaussian(sigma));
                l[1] = saturate_cast<unsigned char>(l[1] + rng.gaussian(sigma));
                l[2] = saturate_cast<unsigned char>(l[2] + rng.gaussian(sigma));
                l+=3;
                ldisp[0] = saturate_cast<short>(ldisp[0] + rng.gaussian(sigma));
                ldisp++;
                rdisp[0] = saturate_cast<short>(rdisp[0] + rng.gaussian(sigma));
                rdisp++;
            }
        }
        else
        {
            for(int j=0;j<w;j++)
            {
                l[0] = saturate_cast<unsigned char>(l[0] + rng.gaussian(sigma));
                l++;
                ldisp[0] = saturate_cast<short>(ldisp[0] + rng.gaussian(sigma));
                ldisp++;
                rdisp[0] = saturate_cast<short>(rdisp[0] + rng.gaussian(sigma));
                rdisp++;
            }
        }
    }

    dst_ROI = Rect(rect_disparity,0,w-rect_disparity,h);
}

}
