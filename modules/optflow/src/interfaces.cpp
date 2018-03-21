/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 // Copyright (C) 2009, Willow Garage Inc., all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#include "precomp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"

namespace cv
{
namespace optflow
{
class OpticalFlowSimpleFlow : public DenseOpticalFlow
{
public:
    OpticalFlowSimpleFlow();
    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;

protected:
    int layers;
    int averaging_radius;
    int max_flow;
    double sigma_dist;
    double sigma_color;
    int postprocess_window;
    double sigma_dist_fix;
    double sigma_color_fix;
    double occ_thr;
    int upscale_averaging_radius;
    double upscale_sigma_dist;
    double upscale_sigma_color;
    double speed_up_thr;
};

OpticalFlowSimpleFlow::OpticalFlowSimpleFlow()
{
    // values from the example app
    layers = 3;
    averaging_radius = 2;
    max_flow = 4;
    // values from the default function parameters
    sigma_dist = 4.1;
    sigma_color = 25.5;
    postprocess_window = 18;
    sigma_dist_fix = 55.0;
    sigma_color_fix = 25.5;
    occ_thr = 0.35;
    upscale_averaging_radius = 18;
    upscale_sigma_dist = 55.0;
    upscale_sigma_color = 25.5;
    speed_up_thr = 10;
}

void OpticalFlowSimpleFlow::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    optflow::calcOpticalFlowSF(I0, I1, flow, layers, averaging_radius, max_flow, sigma_dist, sigma_color,
                      postprocess_window, sigma_dist_fix, sigma_color_fix, occ_thr,
                      upscale_averaging_radius, upscale_sigma_dist, upscale_sigma_color, speed_up_thr);
}
void OpticalFlowSimpleFlow::collectGarbage()
{

}
//CV_INIT_ALGORITHM(OpticalFlowSimpleFlow, "DenseOpticalFlow.SimpleFlow"),
//        obj.info()->addParam(obj, "layers", obj.layers);
//        obj.info()->addParam(obj, "averaging_radius", obj.averaging_radius);
//        obj.info()->addParam(obj, "max_flow", obj.max_flow);
//        obj.info()->addParam(obj, "sigma_dist", obj.sigma_dist);
//        obj.info()->addParam(obj, "sigma_color", obj.sigma_color);
//        obj.info()->addParam(obj, "postprocess_window", obj.postprocess_window);
//        obj.info()->addParam(obj, "sigma_dist_fix", obj.sigma_dist_fix);
//        obj.info()->addParam(obj, "sigma_color_fix", obj.sigma_color_fix);
//        obj.info()->addParam(obj, "occ_thr", obj.occ_thr);
//        obj.info()->addParam(obj, "upscale_averaging_radius", obj.upscale_averaging_radius);
//        obj.info()->addParam(obj, "upscale_sigma_dist", obj.upscale_sigma_dist);
//        obj.info()->addParam(obj, "upscale_sigma_color", obj.upscale_sigma_color);
//        obj.info()->addParam(obj, "speed_up_thr", obj.speed_up_thr))

Ptr<DenseOpticalFlow> createOptFlow_SimpleFlow()
{
    return makePtr<OpticalFlowSimpleFlow>();
}

class OpticalFlowFarneback : public DenseOpticalFlow
{
public:
    OpticalFlowFarneback();
    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;
protected:
    int numLevels;
    double pyrScale;
    bool fastPyramids;
    int winSize;
    int numIters;
    int polyN;
    double polySigma;
    int flags;
};

OpticalFlowFarneback::OpticalFlowFarneback()
{
    // values copied from the FarnebackOpticalFlow class
    numLevels = 5;
    pyrScale = 0.5;
    fastPyramids = false;
    winSize = 13;
    numIters = 10;
    polyN = 5;
    polySigma = 1.1;
    flags = 0;
}

void OpticalFlowFarneback::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    calcOpticalFlowFarneback(I0, I1, flow, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
}
void OpticalFlowFarneback::collectGarbage()
{
}

//CV_INIT_ALGORITHM(OpticalFlowFarneback, "DenseOpticalFlow.Farneback",
//        obj.info()->addParam(obj, "numLevels", obj.numLevels);
//        obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
//        obj.info()->addParam(obj, "fastPyramids", obj.fastPyramids);
//        obj.info()->addParam(obj, "winSize", obj.winSize);
//        obj.info()->addParam(obj, "numIters", obj.numIters);
//        obj.info()->addParam(obj, "polyN", obj.polyN);
//        obj.info()->addParam(obj, "polySigma", obj.polySigma);
//        obj.info()->addParam(obj, "flags", obj.flags))




Ptr<DenseOpticalFlow> createOptFlow_Farneback()
{
    return makePtr<OpticalFlowFarneback>();
}

class OpticalFlowSparseToDense : public DenseOpticalFlow
{
public:
    OpticalFlowSparseToDense(int _grid_step, int _k, float _sigma, bool _use_post_proc, float _fgs_lambda, float _fgs_sigma);
    void calc(InputArray I0, InputArray I1, InputOutputArray flow) CV_OVERRIDE;
    void collectGarbage() CV_OVERRIDE;
protected:
    int grid_step;
    int k;
    float sigma;
    bool use_post_proc;
    float fgs_lambda;
    float fgs_sigma;
};

OpticalFlowSparseToDense::OpticalFlowSparseToDense(int _grid_step, int _k, float _sigma, bool _use_post_proc, float _fgs_lambda, float _fgs_sigma)
{
    grid_step     = _grid_step;
    k             = _k;
    sigma         = _sigma;
    use_post_proc = _use_post_proc;
    fgs_lambda    = _fgs_lambda;
    fgs_sigma     = _fgs_sigma;
}

void OpticalFlowSparseToDense::calc(InputArray I0, InputArray I1, InputOutputArray flow)
{
    calcOpticalFlowSparseToDense(I0,I1,flow,grid_step,k,sigma,use_post_proc,fgs_lambda,fgs_sigma);
}

void OpticalFlowSparseToDense::collectGarbage() {}

Ptr<DenseOpticalFlow> createOptFlow_SparseToDense()
{
    return makePtr<OpticalFlowSparseToDense>(8,128,0.05f,true,500.0f,1.5f);
}
}
}
