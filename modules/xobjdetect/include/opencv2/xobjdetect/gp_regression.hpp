// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/******************************************************************************
 *
 * This software is the core implementation of the fine-grained search (FGS) method
 * based on the Gaussian process regression with latent variables.
 *
 * Details of the method can be found in the follow paper
 *   Yuting Zhang, Kihyuk Sohn, Ruben Villegas, Gang Pan, Honglak Lee
 *   ``Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction''
 *   IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Boston, MA, 2015.
 *
 * The following people contributed to the code:
 *   Yuting Zhang   <zyt.cse@gmail.com>
 *   Kihyuk Sohn    <kihyuks@umich.edu>
 *   Ruben Villegas <rubville@umich.edu>
 *   Xinchen Yan    <xcyan@umich.edu>
 *   Gang Pan       <gpan@zju.edu.cn>
 *   Honglak Lee    <honglak@eecs.umich.edu>
 *
*******************************************************************************/

#ifndef INCLUDE_OPENCV2_FGS_GP_REGRESSION_HPP_
#define INCLUDE_OPENCV2_FGS_GP_REGRESSION_HPP_

#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "opencv2/core.hpp"

namespace cv2 {

// argmax z on joint probability of the existing boxes
// argmax yNp1 on conditional probability

//    kNp1: N x 1; KN: N x N; psiNp1: 4 x 1

class fgs_gp_box_reg {
protected:
    /**
     * variables
     */
     struct GPmodel_dfn {
        double m0;
        cv::Mat diagSqrtLambda; // 4 x 1
        double normCov;
        double noiseSigma2;
        cv::Mat idxbScaleEnabled; // 4 x 1
    } GPmodel_;
    //cv2::fgs_base_detector& detector_;

    /**
     * private/protected functions
     */
    cv::Mat sgp_kNp1_forward(const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    cv::Mat sgp_kNp1_backward(const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    double sgp_posterior_mu_forward(const cv::Mat &, const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    cv::Mat sgp_posterior_mu_backward(const cv::Mat &, const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    double sgp_posterior_s2_forward(const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    cv::Mat sgp_posterior_s2_backward(const cv::Mat &, const cv::Mat &, const GPmodel_dfn &);
    double sgp_ei_forward(const double &, const double &, const double &);
    std::vector<double> sgp_ei_backward(const double &, const double &, const double &);
    cv::Mat sgp_KN(const cv::Mat &, const GPmodel_dfn &);
    cv::Mat sgp_cov_forward(const GPmodel_dfn &, const double &, const cv::Mat &);
    cv::Mat sgp_cov_backward(const GPmodel_dfn &, const double &, const cv::Mat &);
    double sgp_neg_acquisition_ei_forward(const GPmodel_dfn &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const double &, const cv::Mat &);
    cv::Mat sgp_neg_acquisition_ei_backward(const GPmodel_dfn &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const double &, const cv::Mat &);
    double sgp_negloglik_forward(const GPmodel_dfn &, const double &, const cv::Mat &, const cv::Mat &);
    double sgp_negloglik_backward(const GPmodel_dfn &, const double &, const cv::Mat &, const cv::Mat &);
    double sgp_negloglik_givenC_forward(const GPmodel_dfn &, const cv::Mat &, const cv::Mat &);
    cv::Mat sgp_negloglik_givenC_backward(const GPmodel_dfn &, const cv::Mat &, const cv::Mat &);

    //utils;
    //
    double normpdf(const double &);
    double normcdf(const double &);
    double erf(const double &);
    double erfc(const double &);
    cv::Mat pdist(const cv::Mat &); // n x m --> n(n-1)/2 x 1
    cv::Mat squareform(const cv::Mat &); //n(n-1)/2 x 1 --> n x n
    cv::Mat rect2cvmat(cv::Rect &);
    cv::Rect cvmat2rect(cv::Mat &);
    double findMinZ(const GPmodel_dfn &, const double &, const cv::Mat &, const cv::Mat &);
    cv::Mat findMinPsiNp1(const GPmodel_dfn &, const cv::Mat &, const cv::Mat &, const cv::Mat &, const double &, const cv::Mat &);

public:
    typedef cv2::fgs_gp_box_reg BoxReg;

    void Load( const std::string& model_path );
    // function for generate new proposals
    // input: current existing boxes:  std::vector< cv::Rect2d >
    // output: list of box cv::Rect2d
    cv::Rect ProposeBox( std::vector< std::pair<cv::Rect,double> > known_boxes );
};

}

#endif /* INCLUDE_OPENCV2_FGS_GP_REGRESSION_HPP_ */
