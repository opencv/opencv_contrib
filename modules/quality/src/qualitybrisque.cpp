// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
Copyright (c) 2011 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy,
modify, and distribute this code (the source files) and its documentation for
any purpose, provided that the copyright notice in its entirety appear in all copies of this code, and the
original source of this code, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu)
and Center for Perceptual Systems (CPS, http://www.cps.utexas.edu) at the University of Texas at Austin (UT Austin,
http://www.utexas.edu), is acknowledged in any publication that reports research using this code. The research
is to be cited in the bibliography as:

1) A. Mittal, A. K. Moorthy and A. C. Bovik, "BRISQUE Software Release",
URL: http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip, 2011

2) A. Mittal, A. K. Moorthy and A. C. Bovik, "No Reference Image Quality Assessment in the Spatial Domain"
submitted

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL,
OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS
AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS,
AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
*/

/* Original Paper: @cite Mittal2 and Original Implementation: @cite Mittal2_software */
#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/quality/qualitybrisque.hpp"
#include "opencv2/quality/quality_utils.hpp"

namespace
{
    using namespace cv;
    using namespace cv::quality;

    // type of mat we're working with internally
    //  Win32+UMat:  performance is 15-20X worse than Mat
    //  Win32+UMat+OCL:  performance is 200-300X worse than Mat, plus accuracy errors
    //  Linux+UMat: 15X worse performance than Linux+Mat
    using brisque_mat_type = cv::Mat;

    // brisque intermediate calculation type
    //  Linux+Mat:  CV_64F is 3X slower than CV_32F
    //  Win32+Mat:  CV_64F is 2X slower than CV_32F
    static constexpr const int BRISQUE_CALC_MAT_TYPE = CV_32F;
    // brisque intermediate matrix element type.  float if BRISQUE_CALC_MAT_TYPE == CV_32F, double if BRISQUE_CALC_MAT_TYPE == CV_64F
    using brisque_calc_element_type = float;

    // convert mat to grayscale, range [0-1]
    brisque_mat_type mat_convert( const brisque_mat_type& mat )
    {
        brisque_mat_type result = mat;
        switch (mat.channels())
        {
        case 1:
            break;
        case 3:
            cv::cvtColor(result, result, cv::COLOR_BGR2GRAY, 1);
            break;
        case 4:
            cv::cvtColor(result, result, cv::COLOR_BGRA2GRAY, 1);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "Unknown/unsupported channel count");
        };//switch

        // scale to 0-1 range
        result.convertTo(result, BRISQUE_CALC_MAT_TYPE, 1. / 255.);
        return result;
    }

    // function to compute best fit parameters from AGGDfit
    void AGGDfit(const brisque_mat_type& structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
    {
        long int poscount = 0, negcount = 0;
        double possqsum = 0, negsqsum = 0, abssum = 0;
        for (int i = 0; i < structdis.rows; i++)
        {
            for (int j = 0; j < structdis.cols; j++)
            {
                double pt = structdis.at<brisque_calc_element_type>(i, j);
                if (pt > 0)
                {
                    poscount++;
                    possqsum += pt * pt;
                    abssum += pt;
                }
                else if (pt < 0)
                {
                    negcount++;
                    negsqsum += pt * pt;
                    abssum -= pt;
                }
            }
        }

        lsigma_best = cv::pow(negsqsum / negcount, 0.5);
        rsigma_best = cv::pow(possqsum / poscount, 0.5);

        double gammahat = lsigma_best / rsigma_best;
        long int totalcount = (structdis.cols)*(structdis.rows);
        double rhat = cv::pow(abssum / totalcount, static_cast<double>(2)) / ((negsqsum + possqsum) / totalcount);
        double rhatnorm = rhat * (cv::pow(gammahat, 3) + 1)*(gammahat + 1) / pow(pow(gammahat, 2) + 1, 2);

        double prevgamma = 0;
        double prevdiff = 1e10;
        double sampling = 0.001;
        for (double gam = 0.2; gam < 10; gam += sampling) //possible to coarsen sampling to quicken the code, with some loss of accuracy
        {
            double r_gam = tgamma(2 / gam)*tgamma(2 / gam) / (tgamma(1 / gam)*tgamma(3 / gam));
            double diff = abs(r_gam - rhatnorm);
            if (diff > prevdiff) break;
            prevdiff = diff;
            prevgamma = gam;
        }
        gamma_best = prevgamma;

        // return structdis.clone();
    }

    std::vector<brisque_calc_element_type> ComputeBrisqueFeature( const brisque_mat_type& orig )
    {
        CV_DbgAssert(orig.channels() == 1);

        std::vector<brisque_calc_element_type> featurevector;

        auto orig_bw = orig;

        // orig_bw now contains the grayscale image normalized to the range 0,1
        int scalenum = 2; // number of times to scale the image
        for (int itr_scale = 1; itr_scale <= scalenum; itr_scale++)
        {
            // resize image
            cv::Size dst_size( int( orig_bw.cols / cv::pow((double)2, itr_scale - 1) ), int( orig_bw.rows / pow((double)2, itr_scale - 1)));
            brisque_mat_type imdist_scaled;
            cv::resize(orig_bw, imdist_scaled, dst_size, 0, 0, cv::INTER_CUBIC); // INTER_CUBIC

            // calculating MSCN coefficients
            // compute mu (local mean)
            brisque_mat_type mu;//  (imdist_scaled.size(), CV_64FC1, 1);
            cv::GaussianBlur(imdist_scaled, mu, cv::Size(7, 7), 7. / 6., 0., cv::BORDER_REPLICATE );

            brisque_mat_type mu_sq;
            cv::pow(mu, double(2.0), mu_sq);

            //compute sigma (local sigma)
            brisque_mat_type sigma;// (imdist_scaled.size(), CV_64FC1, 1);
            cv::multiply(imdist_scaled, imdist_scaled, sigma);

            cv::GaussianBlur(sigma, sigma, cv::Size(7, 7), 7./6., 0., cv::BORDER_REPLICATE );

            cv::subtract(sigma, mu_sq, sigma);
            cv::pow(sigma, double(0.5), sigma);
            cv::add(sigma, Scalar(1.0 / 255), sigma); // to avoid DivideByZero Error

            brisque_mat_type structdis;// (imdist_scaled.size(), CV_64FC1, 1);
            cv::subtract(imdist_scaled, mu, structdis);
            cv::divide(structdis, sigma, structdis);  // structdis is MSCN image

            // Compute AGGD fit to MSCN image
            double lsigma_best, rsigma_best, gamma_best;

            //structdis = AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);
            AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);
            featurevector.push_back( (brisque_calc_element_type) gamma_best);
            featurevector.push_back(( (brisque_calc_element_type)(  lsigma_best*lsigma_best + rsigma_best * rsigma_best) / 2 ));

            // Compute paired product images
            // indices for orientations (H, V, D1, D2)
            int shifts[4][2] = { {0,1},{1,0},{1,1},{-1,1} };

            for (int itr_shift = 1; itr_shift <= 4; itr_shift++)
            {
                // select the shifting index from the 2D array
                int* reqshift = shifts[itr_shift - 1];

                // declare, create shifted_structdis as pairwise image
                brisque_mat_type shifted_structdis(imdist_scaled.size(), BRISQUE_CALC_MAT_TYPE); //(imdist_scaled.size(), CV_64FC1, 1);

                // create pair-wise product for the given orientation (reqshift)
                for (int i = 0; i < structdis.rows; i++)
                {
                    for (int j = 0; j < structdis.cols; j++)
                    {
                        if (i + reqshift[0] >= 0 && i + reqshift[0] < structdis.rows && j + reqshift[1] >= 0 && j + reqshift[1] < structdis.cols)
                        {
                            shifted_structdis.at<brisque_calc_element_type>(i,j) = structdis.at<brisque_calc_element_type>(i + reqshift[0], j + reqshift[1]);
                        }
                        else
                        {
                            shifted_structdis.at<brisque_calc_element_type>(i, j) = (brisque_calc_element_type) 0;
                        }
                    }
                }

                // calculate the products of the pairs
                cv::multiply(structdis, shifted_structdis, shifted_structdis);

                // fit the pairwise product to AGGD
                // shifted_structdis = AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);
                AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);

                double constant = sqrt(tgamma(1 / gamma_best)) / sqrt(tgamma(3 / gamma_best));
                double meanparam = (rsigma_best - lsigma_best)*(tgamma(2 / gamma_best) / tgamma(1 / gamma_best))*constant;

                // push the calculated parameters from AGGD fit to pair-wise products
                featurevector.push_back((brisque_calc_element_type)gamma_best);
                featurevector.push_back((brisque_calc_element_type)meanparam);
                featurevector.push_back( (brisque_calc_element_type) cv::pow(lsigma_best, 2));
                featurevector.push_back( (brisque_calc_element_type) cv::pow(rsigma_best, 2));
            }
        }

        return featurevector;
    }

    brisque_calc_element_type computescore(const cv::Ptr<cv::ml::SVM>& model, const cv::Mat& range, const brisque_mat_type& img ) {

        const auto brisqueFeatures = ComputeBrisqueFeature( img ); // compute brisque features

        cv::Mat feat_mat( 1,(int)brisqueFeatures.size(), CV_32FC1, (void*)brisqueFeatures.data() ); // load to mat
        quality_utils::scale(feat_mat, range, -1.f, 1.f);// scale to range [-1,1]

        cv::Mat result;
        model->predict(feat_mat, result);
        return std::min( std::max( result.at<float>(0), 0.f ), 100.f ); // clamp to [0-100]
    }

    // computes score for a single frame
    cv::Scalar compute(const cv::Ptr<cv::ml::SVM>& model, const cv::Mat& range, const brisque_mat_type& img)
    {
        auto result = cv::Scalar{ 0. };
        result[0] = computescore(model, range, img);
        return result;
    }
}

// static
cv::Ptr<QualityBRISQUE> QualityBRISQUE::create(const cv::String& model_file_path, const cv::String& range_file_path)
{
    return cv::Ptr<QualityBRISQUE>(new QualityBRISQUE(model_file_path, range_file_path));
}

// static
cv::Ptr<QualityBRISQUE> QualityBRISQUE::create(const cv::Ptr<cv::ml::SVM>& model, const cv::Mat& range)
{
    return cv::Ptr<QualityBRISQUE>(new QualityBRISQUE(model, range));
}

// static
cv::Scalar QualityBRISQUE::compute( InputArray img, const cv::String& model_file_path, const cv::String& range_file_path)
{
    return QualityBRISQUE(model_file_path, range_file_path).compute(img);
}

// QualityBRISQUE() constructor
QualityBRISQUE::QualityBRISQUE(const cv::String& model_file_path, const cv::String& range_file_path)
    : QualityBRISQUE(
        cv::ml::SVM::load(model_file_path)
        , cv::FileStorage(range_file_path, cv::FileStorage::READ)["range"].mat()
    )
{}

cv::Scalar QualityBRISQUE::compute( InputArray img )
{
    auto mat = quality_utils::extract_mat<brisque_mat_type>(img); // extract input mats

    mat = mat_convert(mat);// convert to gs, scale to [0,1]

    return ::compute(this->_model, this->_range, mat );
}

//static
void QualityBRISQUE::computeFeatures(InputArray img, OutputArray features)
{
    CV_Assert(features.needed());
    CV_Assert(img.isMat());
    CV_Assert(!img.getMat().empty());

    auto mat = mat_convert(img.getMat());

    const auto vals = ComputeBrisqueFeature(mat);
    cv::Mat valmat( cv::Size( (int)vals.size(), 1 ), CV_32FC1, (void*)vals.data()); // create row vector, type depends on brisque_calc_element_type

    if (features.isUMat())
        valmat.copyTo(features.getUMatRef());
    else if (features.isMat())
        // how to move data instead?
        // if calling this:
        //      features.getMatRef() = valmat;
        //  then shared data is erased when valmat is released, corrupting the data in the outputarray for the caller
        valmat.copyTo(features.getMatRef());
    else
        CV_Error(cv::Error::StsNotImplemented, "Unsupported output type");
}