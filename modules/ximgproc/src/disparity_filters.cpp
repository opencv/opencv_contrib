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
#include "opencv2/ximgproc/disparity_filter.hpp"
#include "opencv2/imgcodecs.hpp"
#include <math.h>
#include <vector>

namespace cv {
namespace ximgproc {

using std::vector;
#define EPS 1e-43f

class DisparityWLSFilterImpl : public DisparityWLSFilter
{
protected:
    int left_offset, right_offset, top_offset, bottom_offset;
    Rect valid_disp_ROI;
    Rect right_view_valid_disp_ROI;
    int min_disp;
    bool use_confidence;
    Mat confidence_map;

    double lambda,sigma_color;
    int LRC_thresh,depth_discontinuity_radius;
    float depth_discontinuity_roll_off_factor;
    float resize_factor;
    int num_stripes;

    void init(double _lambda, double _sigma_color, bool _use_confidence, int l_offs, int r_offs, int t_offs, int b_offs, int _min_disp);
    void computeDepthDiscontinuityMaps(Mat& left_disp, Mat& right_disp, Mat& left_dst, Mat& right_dst);
    void computeConfidenceMap(InputArray left_disp, InputArray right_disp);

protected:
    struct ComputeDiscontinuityAwareLRC_ParBody : public ParallelLoopBody
    {
        DisparityWLSFilterImpl* wls;
        Mat *left_disp, *right_disp;
        Mat *left_disc, *right_disc, *dst;
        Rect left_ROI, right_ROI;
        int nstripes, stripe_sz;

        ComputeDiscontinuityAwareLRC_ParBody(DisparityWLSFilterImpl& _wls, Mat& _left_disp, Mat& _right_disp, Mat& _left_disc, Mat& _right_disc, Mat& _dst, Rect _left_ROI, Rect _right_ROI, int _nstripes);
        void operator () (const Range& range) const CV_OVERRIDE;
    };

    struct ComputeDepthDisc_ParBody : public ParallelLoopBody
    {
        DisparityWLSFilterImpl* wls;
        Mat *disp,*disp_squares,*dst;
        int nstripes, stripe_sz;

        ComputeDepthDisc_ParBody(DisparityWLSFilterImpl& _wls, Mat& _disp, Mat& _disp_squares, Mat& _dst, int _nstripes);
        void operator () (const Range& range) const CV_OVERRIDE;
    };

    typedef void (DisparityWLSFilterImpl::*MatOp)(Mat& src, Mat& dst);

    struct ParallelMatOp_ParBody : public ParallelLoopBody
    {
        DisparityWLSFilterImpl* wls;
        vector<MatOp> ops;
        vector<Mat*> src;
        vector<Mat*> dst;

        ParallelMatOp_ParBody(DisparityWLSFilterImpl& _wls, vector<MatOp> _ops, vector<Mat*>& _src, vector<Mat*>& _dst);
        void operator () (const Range& range) const CV_OVERRIDE;
    };

    void boxFilterOp(Mat& src,Mat& dst)
    {
        int rad = depth_discontinuity_radius;
        boxFilter(src,dst,CV_32F,Size(2*rad+1,2*rad+1),Point(-1,-1));
    }

    void sqrBoxFilterOp(Mat& src,Mat& dst)
    {
        int rad = depth_discontinuity_radius;
        sqrBoxFilter(src,dst,CV_32F,Size(2*rad+1,2*rad+1),Point(-1,-1));
    }

    void copyToOp(Mat& src,Mat& dst)
    {
        src.copyTo(dst);
    }

public:
    static Ptr<DisparityWLSFilterImpl> create(bool _use_confidence, int l_offs, int r_offs, int t_offs, int b_offs, int min_disp);
    void filter(InputArray disparity_map_left, InputArray left_view, OutputArray filtered_disparity_map, InputArray disparity_map_right, Rect ROI, InputArray) CV_OVERRIDE;
    void filter_(InputArray disparity_map_left, InputArray left_view, OutputArray filtered_disparity_map, InputArray disparity_map_right, Rect ROI);

    double getLambda() CV_OVERRIDE { return lambda; }
    void setLambda(double _lambda) CV_OVERRIDE { lambda = _lambda; }

    double getSigmaColor() CV_OVERRIDE { return sigma_color; }
    void setSigmaColor(double _sigma_color) CV_OVERRIDE { sigma_color = _sigma_color; }

    int getLRCthresh() CV_OVERRIDE { return LRC_thresh; }
    void setLRCthresh(int _LRC_thresh) CV_OVERRIDE { LRC_thresh = _LRC_thresh; }

    int getDepthDiscontinuityRadius() CV_OVERRIDE { return depth_discontinuity_radius; }
    void setDepthDiscontinuityRadius(int _disc_radius) CV_OVERRIDE { depth_discontinuity_radius = _disc_radius; }

    Mat getConfidenceMap() CV_OVERRIDE { return confidence_map; }
    Rect getROI() CV_OVERRIDE { return valid_disp_ROI; }
};

void DisparityWLSFilterImpl::init(double _lambda, double _sigma_color, bool _use_confidence,  int l_offs, int r_offs, int t_offs, int b_offs, int _min_disp)
{
    left_offset = l_offs; right_offset  = r_offs;
    top_offset  = t_offs; bottom_offset = b_offs;
    min_disp = _min_disp;
    valid_disp_ROI = Rect();
    right_view_valid_disp_ROI = Rect();
    min_disp=0;
    lambda = _lambda;
    sigma_color = _sigma_color;
    use_confidence = _use_confidence;
    confidence_map = Mat();
    LRC_thresh = 24;
    depth_discontinuity_radius = 5;
    depth_discontinuity_roll_off_factor = 0.001f;
    resize_factor = 1.0;
    num_stripes = getNumThreads();
}

void DisparityWLSFilterImpl::computeDepthDiscontinuityMaps(Mat& left_disp, Mat& right_disp, Mat& left_dst, Mat& right_dst)
{
    Mat left_disp_ROI (left_disp, valid_disp_ROI);
    Mat right_disp_ROI(right_disp,right_view_valid_disp_ROI);
    Mat ldisp,rdisp,ldisp_squared,rdisp_squared;

    {
        vector<Mat*> _src; _src.push_back(&left_disp_ROI);_src.push_back(&right_disp_ROI);
                           _src.push_back(&left_disp_ROI);_src.push_back(&right_disp_ROI);
        vector<Mat*> _dst; _dst.push_back(&ldisp);        _dst.push_back(&rdisp);
                           _dst.push_back(&ldisp_squared);_dst.push_back(&rdisp_squared);
        vector<MatOp> _ops; _ops.push_back(&DisparityWLSFilterImpl::copyToOp); _ops.push_back(&DisparityWLSFilterImpl::copyToOp);
                            _ops.push_back(&DisparityWLSFilterImpl::copyToOp); _ops.push_back(&DisparityWLSFilterImpl::copyToOp);
        parallel_for_(Range(0,4),ParallelMatOp_ParBody(*this,_ops,_src,_dst));
    }

    {
        vector<Mat*> _src; _src.push_back(&ldisp);        _src.push_back(&rdisp);
                           _src.push_back(&ldisp_squared);_src.push_back(&rdisp_squared);
        vector<Mat*> _dst; _dst.push_back(&ldisp);        _dst.push_back(&rdisp);
                           _dst.push_back(&ldisp_squared);_dst.push_back(&rdisp_squared);
        vector<MatOp> _ops; _ops.push_back(&DisparityWLSFilterImpl::boxFilterOp);   _ops.push_back(&DisparityWLSFilterImpl::boxFilterOp);
                            _ops.push_back(&DisparityWLSFilterImpl::sqrBoxFilterOp);_ops.push_back(&DisparityWLSFilterImpl::sqrBoxFilterOp);
        parallel_for_(Range(0,4),ParallelMatOp_ParBody(*this,_ops,_src,_dst));
    }

    left_dst  = Mat::zeros(left_disp.rows,left_disp.cols,CV_32F);
    right_dst = Mat::zeros(right_disp.rows,right_disp.cols,CV_32F);
    Mat left_dst_ROI (left_dst,valid_disp_ROI);
    Mat right_dst_ROI(right_dst,right_view_valid_disp_ROI);

    parallel_for_(Range(0,num_stripes),ComputeDepthDisc_ParBody(*this,ldisp,ldisp_squared,left_dst_ROI ,num_stripes));
    parallel_for_(Range(0,num_stripes),ComputeDepthDisc_ParBody(*this,rdisp,rdisp_squared,right_dst_ROI,num_stripes));
}


void DisparityWLSFilterImpl::computeConfidenceMap(InputArray left_disp, InputArray right_disp)
{
    Mat ldisp = left_disp.getMat();
    Mat rdisp = right_disp.getMat();
    Mat depth_discontinuity_map_left,depth_discontinuity_map_right;
    right_view_valid_disp_ROI = Rect(ldisp.cols-(valid_disp_ROI.x+valid_disp_ROI.width),valid_disp_ROI.y,
                                     valid_disp_ROI.width,valid_disp_ROI.height);
    computeDepthDiscontinuityMaps(ldisp,rdisp,depth_discontinuity_map_left,depth_discontinuity_map_right);

    confidence_map = depth_discontinuity_map_left;

    parallel_for_(Range(0,num_stripes),ComputeDiscontinuityAwareLRC_ParBody(*this,ldisp,rdisp, depth_discontinuity_map_left,depth_discontinuity_map_right,confidence_map,valid_disp_ROI,right_view_valid_disp_ROI,num_stripes));
    confidence_map = 255.0f*confidence_map;
}

Ptr<DisparityWLSFilterImpl> DisparityWLSFilterImpl::create(bool _use_confidence, int l_offs=0, int r_offs=0, int t_offs=0, int b_offs=0, int min_disp=0)
{
    DisparityWLSFilterImpl *wls = new DisparityWLSFilterImpl();
    wls->init(8000.0,1.0,_use_confidence,l_offs, r_offs, t_offs, b_offs, min_disp);
    return Ptr<DisparityWLSFilterImpl>(wls);
}

void DisparityWLSFilterImpl::filter(InputArray disparity_map_left, InputArray left_view, OutputArray filtered_disparity_map, InputArray disparity_map_right, Rect ROI, InputArray)
{
    CV_Assert(!disparity_map_left.empty() && (disparity_map_left.channels() == 1));
    CV_Assert(!left_view.empty() && (left_view.depth() == CV_8U) && (left_view.channels() == 3 || left_view.channels() == 1));
    Mat left, right, filt_disp;

    if (disparity_map_left.depth() != CV_32F)
    {
        disparity_map_left.getMat().convertTo(left, CV_32F);
    }
    else
    {
        left = disparity_map_left.getMat();
        filt_disp = filtered_disparity_map.getMat();
    }

    if (!disparity_map_right.empty() && use_confidence)
    {
        if(disparity_map_right.depth() != CV_32F)
            disparity_map_right.getMat().convertTo(right, CV_32F);
        else
            right = disparity_map_right.getMat();
    }

    filter_(left, left_view, filt_disp, right, ROI);
    if (disparity_map_left.depth() != CV_32F){
        filt_disp.convertTo(filtered_disparity_map, disparity_map_left.depth());
    } else {
        filt_disp.copyTo(filtered_disparity_map);
    }
}

void DisparityWLSFilterImpl::filter_(InputArray disparity_map_left, InputArray left_view, OutputArray filtered_disparity_map, InputArray disparity_map_right, Rect ROI)
{
    CV_Assert( !disparity_map_left.empty()
               && ( disparity_map_left.depth() == CV_32F)
               && (disparity_map_left.channels() == 1) );
    CV_Assert( !left_view.empty()
               && (left_view.depth() == CV_8U)
               && (left_view.channels() == 3 || left_view.channels() == 1) );

    Mat disp,src,dst;
    if(disparity_map_left.size()!=left_view.size())
        resize_factor = disparity_map_left.cols()/(float)left_view.cols();
    else
        resize_factor = 1.0;
    if(!ROI.empty()) /* user provided a ROI */
        valid_disp_ROI = ROI;
    else
        valid_disp_ROI = Rect(left_offset,top_offset,
                              disparity_map_left.cols()-left_offset-right_offset,
                              disparity_map_left.rows()-top_offset-bottom_offset);

    if(!use_confidence)
    {
        Mat disp_full_size = disparity_map_left.getMat();
        Mat src_full_size = left_view.getMat();
        if(disp_full_size.size!=src_full_size.size)
        {
            float x_ratio = src_full_size.cols/(float)disp_full_size.cols;
            float y_ratio = src_full_size.rows/(float)disp_full_size.rows;
            resize(disp_full_size,disp_full_size,src_full_size.size());
            disp_full_size = disp_full_size*x_ratio;
            ROI = Rect((int)(valid_disp_ROI.x*x_ratio),    (int)(valid_disp_ROI.y*y_ratio),
                       (int)(valid_disp_ROI.width*x_ratio),(int)(valid_disp_ROI.height*y_ratio));
        }
        else
            ROI = valid_disp_ROI;

        disp = Mat(disp_full_size,ROI);
        src  = Mat(src_full_size ,ROI);
        filtered_disparity_map.create(disp_full_size.size(), disp_full_size.type());
        Mat& dst_full_size = filtered_disparity_map.getMatRef();
        dst_full_size = Scalar(16*(min_disp-1));
        dst = Mat(dst_full_size,ROI);
        Mat filtered_disp;
        fastGlobalSmootherFilter(src,disp,filtered_disp,lambda,sigma_color);
        filtered_disp.copyTo(dst);
    }
    else
    {
        CV_Assert( !disparity_map_right.empty()
                   && (disparity_map_right.depth() == CV_32F)
                   && (disparity_map_right.channels() == 1) );
        CV_Assert( (disparity_map_left.cols() == disparity_map_right.cols()) );
        CV_Assert( (disparity_map_left.rows() == disparity_map_right.rows()) );
        computeConfidenceMap(disparity_map_left,disparity_map_right);
        Mat disp_full_size = disparity_map_left.getMat();
        Mat src_full_size = left_view.getMat();
        if(disp_full_size.size!=src_full_size.size){
            float x_ratio = src_full_size.cols/(float)disp_full_size.cols;
            float y_ratio = src_full_size.rows/(float)disp_full_size.rows;
            resize(disp_full_size,disp_full_size,src_full_size.size());
            disp_full_size = disp_full_size*x_ratio;
            resize(confidence_map,confidence_map,src_full_size.size());
            ROI = Rect((int)(valid_disp_ROI.x*x_ratio),    (int)(valid_disp_ROI.y*y_ratio),
                       (int)(valid_disp_ROI.width*x_ratio),(int)(valid_disp_ROI.height*y_ratio));
        } else {
            ROI = valid_disp_ROI;
        }
        disp = Mat(disp_full_size,ROI);
        src  = Mat(src_full_size ,ROI);
        filtered_disparity_map.create(disp_full_size.size(), disp_full_size.type());
        Mat& dst_full_size = filtered_disparity_map.getMatRef();
        dst_full_size = Scalar(16*(min_disp-1));
        dst = Mat(dst_full_size,ROI);
        Mat conf(confidence_map,ROI);

        Mat disp_mul_conf;
        disp_mul_conf = conf.mul(disp);
        Mat conf_filtered;
        Ptr<FastGlobalSmootherFilter> wls = createFastGlobalSmootherFilter(src,lambda,sigma_color);
        wls->filter(disp_mul_conf,disp_mul_conf);
        wls->filter(conf,conf_filtered);
        dst = disp_mul_conf.mul(1/(conf_filtered+EPS));
    }
}

DisparityWLSFilterImpl::ComputeDiscontinuityAwareLRC_ParBody::ComputeDiscontinuityAwareLRC_ParBody(DisparityWLSFilterImpl& _wls, Mat& _left_disp, Mat& _right_disp, Mat& _left_disc, Mat& _right_disc, Mat& _dst, Rect _left_ROI, Rect _right_ROI, int _nstripes):
wls(&_wls),left_disp(&_left_disp),right_disp(&_right_disp),left_disc(&_left_disc),right_disc(&_right_disc),dst(&_dst),left_ROI(_left_ROI),right_ROI(_right_ROI),nstripes(_nstripes)
{
    stripe_sz = (int)ceil(left_disp->rows/(double)nstripes);
}

void DisparityWLSFilterImpl::ComputeDiscontinuityAwareLRC_ParBody::operator() (const Range& range) const
{
    float* row_left;
    float* row_left_conf;
    float* row_right;
    float* row_right_conf;
    float* row_dst;
    int right_idx;
    int h = left_disp->rows;

    int start = std::min(range.start * stripe_sz, h);
    int end   = std::min(range.end   * stripe_sz, h);
    int thresh = (int)(wls->resize_factor*wls->LRC_thresh);
    for(int i=start;i<end;i++)
    {
        row_left  = (float*)left_disp->ptr(i);
        row_left_conf  = (float*)left_disc->ptr(i);
        row_right = (float*)right_disp->ptr(i);
        row_right_conf  = (float*)right_disc->ptr(i);
        row_dst   = (float*)dst->ptr(i);
        int j_start = left_ROI.x;
        int j_end = left_ROI.x+left_ROI.width;
        int right_end = right_ROI.x+right_ROI.width;
        for(int j=j_start;j<j_end;j++)
        {
            right_idx = j-(((int)row_left[j])>>4);
            if( right_idx>=right_ROI.x && right_idx<right_end)
            {
                if(abs(row_left[j] + row_right[right_idx])< thresh)
                    row_dst[j] = min(row_left_conf[j],row_right_conf[right_idx]);
                else
                    row_dst[j] = 0.0f;
            }
        }
    }
}

DisparityWLSFilterImpl::ComputeDepthDisc_ParBody::ComputeDepthDisc_ParBody(DisparityWLSFilterImpl& _wls, Mat& _disp, Mat& _disp_squares, Mat& _dst, int _nstripes):
wls(&_wls),disp(&_disp),disp_squares(&_disp_squares),dst(&_dst),nstripes(_nstripes)
{
    stripe_sz = (int)ceil(disp->rows/(double)nstripes);
}

void DisparityWLSFilterImpl::ComputeDepthDisc_ParBody::operator() (const Range& range) const
{
    float* row_disp;
    float* row_disp_squares;
    float* row_dst;
    float variance;
    int h = disp->rows;
    int w = disp->cols;
    int start = std::min(range.start * stripe_sz, h);
    int end   = std::min(range.end   * stripe_sz, h);
    float roll_off = wls->depth_discontinuity_roll_off_factor/(wls->resize_factor*wls->resize_factor);

    for(int i=start;i<end;i++)
    {
        row_disp = (float*)disp->ptr(i);
        row_disp_squares = (float*)disp_squares->ptr(i);
        row_dst = (float*)dst->ptr(i);

        for(int j=0;j<w;j++)
        {
            variance = row_disp_squares[j] - (row_disp[j])*(row_disp[j]);
            row_dst[j] = max(1.0f - roll_off*variance,0.0f);
        }
    }
}

DisparityWLSFilterImpl::ParallelMatOp_ParBody::ParallelMatOp_ParBody(DisparityWLSFilterImpl& _wls, vector<MatOp> _ops, vector<Mat*>& _src, vector<Mat*>& _dst):
wls(&_wls),ops(_ops),src(_src),dst(_dst)
{}

void DisparityWLSFilterImpl::ParallelMatOp_ParBody::operator() (const Range& range) const
{
    for(int i=range.start;i<range.end;i++)
        (wls->*ops[i])(*src[i],*dst[i]);
}

Ptr<DisparityWLSFilter> createDisparityWLSFilter(Ptr<StereoMatcher> matcher_left)
{
    Ptr<DisparityWLSFilter> wls;
    matcher_left->setDisp12MaxDiff(1000000);
    matcher_left->setSpeckleWindowSize(0);

    int min_disp = matcher_left->getMinDisparity();
    int num_disp = matcher_left->getNumDisparities();
    int wsize    = matcher_left->getBlockSize();
    int wsize2   = wsize/2;

    if(Ptr<StereoBM> bm = matcher_left.dynamicCast<StereoBM>())
    {
        bm->setTextureThreshold(0);
        bm->setUniquenessRatio(0);
        wls = DisparityWLSFilterImpl::create(true,max(0,min_disp+num_disp)+wsize2,max(0,-min_disp)+wsize2,wsize2,wsize2,min_disp);
        wls->setDepthDiscontinuityRadius((int)ceil(0.33*wsize));
    }
    else if(Ptr<StereoSGBM> sgbm = matcher_left.dynamicCast<StereoSGBM>())
    {
        sgbm->setUniquenessRatio(0);
        wls = DisparityWLSFilterImpl::create(true,max(0,min_disp+num_disp),max(0,-min_disp),0,0,min_disp);
        wls->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));
    }
    else
        CV_Error(Error::StsBadArg, "DisparityWLSFilter natively supports only StereoBM and StereoSGBM");

    return wls;
}

Ptr<StereoMatcher> createRightMatcher(Ptr<StereoMatcher> matcher_left)
{
    int min_disp = matcher_left->getMinDisparity();
    int num_disp = matcher_left->getNumDisparities();
    int wsize    = matcher_left->getBlockSize();
    if(Ptr<StereoBM> bm = matcher_left.dynamicCast<StereoBM>())
    {
        Ptr<StereoBM> right_bm = StereoBM::create(num_disp,wsize);
        right_bm->setMinDisparity(-(min_disp+num_disp)+1);
        right_bm->setTextureThreshold(0);
        right_bm->setUniquenessRatio(0);
        right_bm->setDisp12MaxDiff(1000000);
        right_bm->setSpeckleWindowSize(0);
        return right_bm;
    }
    else if(Ptr<StereoSGBM> sgbm = matcher_left.dynamicCast<StereoSGBM>())
    {
        Ptr<StereoSGBM> right_sgbm = StereoSGBM::create(-(min_disp+num_disp)+1,num_disp,wsize);
        right_sgbm->setUniquenessRatio(0);
        right_sgbm->setP1(sgbm->getP1());
        right_sgbm->setP2(sgbm->getP2());
        right_sgbm->setMode(sgbm->getMode());
        right_sgbm->setPreFilterCap(sgbm->getPreFilterCap());
        right_sgbm->setDisp12MaxDiff(1000000);
        right_sgbm->setSpeckleWindowSize(0);
        return right_sgbm;
    }
    else
    {
        CV_Error(Error::StsBadArg, "createRightMatcher supports only StereoBM and StereoSGBM");
    }
}

Ptr<DisparityWLSFilter> createDisparityWLSFilterGeneric(bool use_confidence)
{
    return Ptr<DisparityWLSFilter>(DisparityWLSFilterImpl::create(use_confidence));
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

#define UNKNOWN_DISPARITY 16320

int readGT(String src_path,OutputArray dst)
{
    Mat src = imread(src_path,IMREAD_UNCHANGED);
    dst.create(src.rows,src.cols,CV_16S);
    Mat& dstMat = dst.getMatRef();

    if(!src.empty() && src.channels()==3 && src.depth()==CV_8U)
    {
        //MPI-Sintel format:
        for(int i=0;i<src.rows;i++)
            for(int j=0;j<src.cols;j++)
            {
                Vec3b bgrPixel = src.at<Vec3b>(i, j);
                dstMat.at<short>(i,j) = 64*bgrPixel.val[2]+bgrPixel.val[1]/4; //16-multiplied disparity
            }
        return 0;
    }
    else if(!src.empty() && src.channels()==1 && src.depth()==CV_8U)
    {
        //Middlebury format:
        for(int i=0;i<src.rows;i++)
            for(int j=0;j<src.cols;j++)
            {
                short src_val = src.at<unsigned char>(i, j);
                if(src_val==0)
                    dstMat.at<short>(i,j) = UNKNOWN_DISPARITY;
                else
                    dstMat.at<short>(i,j) = 16*src_val; //16-multiplied disparity
            }
         return 0;
    }
    else
        return 1;
}

double computeMSE(InputArray GT, InputArray src, Rect ROI)
{
    CV_Assert( !GT.empty()  && (GT.depth()  == CV_16S || GT.depth()  == CV_32F) && (GT.channels()  == 1) );
    CV_Assert( !src.empty() && (src.depth() == CV_16S || src.depth()  == CV_32F) && (src.channels() == 1) );
    CV_Assert( src.rows() == GT.rows() && src.cols() == GT.cols() );
    Mat GT_ROI (GT.getMat(), ROI);
    Mat src_ROI(src.getMat(),ROI);

    Mat tmp, dtmp, gt_mask = (GT_ROI == UNKNOWN_DISPARITY);
    absdiff(GT_ROI, src_ROI, tmp); tmp.setTo(0, gt_mask); multiply(tmp, tmp, tmp);
    tmp.convertTo(dtmp, CV_64FC1);
    return sum(dtmp)[0] / ((gt_mask.total() - countNonZero(gt_mask))*256);
}

double computeBadPixelPercent(InputArray GT, InputArray src, Rect ROI, int thresh)
{
    CV_Assert( !GT.empty()  && (GT.depth()  == CV_16S || GT.depth()  == CV_32F) && (GT.channels()  == 1) );
    CV_Assert( !src.empty() && (src.depth() == CV_16S || src.depth()  == CV_32F) && (src.channels() == 1) );
    CV_Assert( src.rows() == GT.rows() && src.cols() == GT.cols() );
    Mat GT_ROI (GT.getMat(), ROI);
    Mat src_ROI(src.getMat(),ROI);

    Mat tmp, gt_mask = (GT_ROI == UNKNOWN_DISPARITY);
    absdiff(GT_ROI, src_ROI, tmp); tmp.setTo(0, gt_mask);
    cv::threshold(tmp, tmp, thresh - 1, 1, THRESH_BINARY);
    return (100.0 * countNonZero(tmp)) / (gt_mask.total() - countNonZero(gt_mask));
}

void getDisparityVis(InputArray src,OutputArray dst,double scale)
{
    CV_Assert( !src.empty() && (src.depth() == CV_16S || src.depth()  == CV_32F) && (src.channels() == 1) );
    Mat srcMat = src.getMat();
    dst.create(srcMat.rows,srcMat.cols,CV_8UC1);
    Mat& dstMat = dst.getMatRef();

    srcMat.convertTo(dstMat, CV_8UC1, scale / 16.0);
    dstMat &= (srcMat != UNKNOWN_DISPARITY);
}

}
}
