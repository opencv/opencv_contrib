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
#include <fstream>
#include "opencv2/imgproc.hpp"
#include "opencv2/quality/qualitybrisque.hpp"
#include "opencv2/quality/quality_utils.hpp"
#include "opencv2/quality/libsvm/svm.hpp"   // libsvm
#include <iostream>

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
    using brique_calc_element_type = float;

    template<class T> class Image {
    private:
        brisque_mat_type imgP;
    public:
        Image(brisque_mat_type img) {
            imgP = img.clone();
        }
        ~Image() {
            imgP = 0;
        }
        brisque_mat_type equate(brisque_mat_type img) {
            img = imgP.clone();
            return img;
        }

        inline T* operator[](const int rowIndx) {
            // return (T*)(imgP.getMat(ACCESS_READ).data + rowIndx * imgP.step);   // UMat version
            return (T*)(imgP.data + rowIndx * imgP.step);    // Mat version
        }
    };

    typedef Image<brique_calc_element_type> BwImage;

    // function to compute best fit parameters from AGGDfit
    brisque_mat_type AGGDfit(brisque_mat_type structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
    {
        // create a copy of an image using BwImage constructor (brisque.h - more info)
        BwImage ImArr(structdis);

        long int poscount = 0, negcount = 0;
        double possqsum = 0, negsqsum = 0, abssum = 0;
        for (int i = 0; i < structdis.rows; i++)
        {
            for (int j = 0; j < structdis.cols; j++)
            {
                double pt = ImArr[i][j]; // BwImage provides [][] access
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

        return structdis.clone();
    }

    void ComputeBrisqueFeature(brisque_mat_type& orig, std::vector<double>& featurevector)
    {
        CV_Assert(orig.channels() == 1);

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
            cv::GaussianBlur(imdist_scaled, mu, cv::Size(7, 7), 1.16666, 0., cv::BORDER_REPLICATE );

            brisque_mat_type mu_sq;
            cv::pow(mu, double(2.0), mu_sq);

            //compute sigma (local sigma)
            brisque_mat_type sigma;// (imdist_scaled.size(), CV_64FC1, 1);
            cv::multiply(imdist_scaled, imdist_scaled, sigma);

            cv::GaussianBlur(sigma, sigma, cv::Size(7, 7), 1.16666, 0., cv::BORDER_REPLICATE );

            cv::subtract(sigma, mu_sq, sigma);
            cv::pow(sigma, double(0.5), sigma);
            cv::add(sigma, Scalar(1.0 / 255), sigma); // to avoid DivideByZero Error

            brisque_mat_type structdis;// (imdist_scaled.size(), CV_64FC1, 1);
            cv::subtract(imdist_scaled, mu, structdis);
            cv::divide(structdis, sigma, structdis);  // structdis is MSCN image

            // Compute AGGD fit to MSCN image
            double lsigma_best, rsigma_best, gamma_best;

            structdis = AGGDfit(structdis, lsigma_best, rsigma_best, gamma_best);
            featurevector.push_back(gamma_best);
            featurevector.push_back((lsigma_best*lsigma_best + rsigma_best * rsigma_best) / 2);

            // Compute paired product images
            // indices for orientations (H, V, D1, D2)
            int shifts[4][2] = { {0,1},{1,0},{1,1},{-1,1} };

            for (int itr_shift = 1; itr_shift <= 4; itr_shift++)
            {
                // select the shifting index from the 2D array
                int* reqshift = shifts[itr_shift - 1];

                // declare, create shifted_structdis as pairwise image
                brisque_mat_type shifted_structdis(imdist_scaled.size(), BRISQUE_CALC_MAT_TYPE); //(imdist_scaled.size(), CV_64FC1, 1);

                // create copies of the images using BwImage constructor
                // utility constructor for better subscript access (for pixels)
                BwImage OrigArr(structdis);
                BwImage ShiftArr(shifted_structdis);

                // create pair-wise product for the given orientation (reqshift)
                for (int i = 0; i < structdis.rows; i++)
                {
                    for (int j = 0; j < structdis.cols; j++)
                    {
                        if (i + reqshift[0] >= 0 && i + reqshift[0] < structdis.rows && j + reqshift[1] >= 0 && j + reqshift[1] < structdis.cols)
                        {
                            ShiftArr[i][j] = OrigArr[i + reqshift[0]][j + reqshift[1]];
                        }
                        else
                        {
                            ShiftArr[i][j] = 0;
                        }
                    }
                }

                // Mat structdis_pairwise;
                shifted_structdis = ShiftArr.equate(shifted_structdis);

                // calculate the products of the pairs
                cv::multiply(structdis, shifted_structdis, shifted_structdis);

                // fit the pairwise product to AGGD
                shifted_structdis = AGGDfit(shifted_structdis, lsigma_best, rsigma_best, gamma_best);

                double constant = sqrt(tgamma(1 / gamma_best)) / sqrt(tgamma(3 / gamma_best));
                double meanparam = (rsigma_best - lsigma_best)*(tgamma(2 / gamma_best) / tgamma(1 / gamma_best))*constant;

                // push the calculated parameters from AGGD fit to pair-wise products
                featurevector.push_back(gamma_best);
                featurevector.push_back(meanparam);
                featurevector.push_back(cv::pow(lsigma_best, 2));
                featurevector.push_back(cv::pow(rsigma_best, 2));
            }
        }
    }

    double computescore(const svm_model* model, const float* range_min, const float* range_max, brisque_mat_type& orig) {
        double qualityscore;
        int i;

        std::vector<double> brisqueFeatures; // feature vector initialization
        ComputeBrisqueFeature(orig, brisqueFeatures); // compute brisque features

        struct svm_node x[37];

        // rescale the brisqueFeatures vector from -1 to 1
        // also convert vector to svm node array object
        for (i = 0; i < 36; ++i) {
            const float
                min = range_min[i]
                , max = range_max[i]
                ;

            x[i].value = -1 + (2.0 / (max - min) * (brisqueFeatures[i] - min));
            x[i].index = i + 1;
        }
        x[36].index = -1;


        int nr_class = svm_get_nr_class(model);

        std::vector<double> prob_estimates = std::vector<double>(nr_class);

        // predict quality score using libsvm class
        qualityscore = svm_predict_probability(model, x, prob_estimates.data());

        return qualityscore;
    }

    // computes score for a single frame
    cv::Scalar compute(const svm_model* model, const float* range_min, const float* range_max, brisque_mat_type& img)
    {
        auto result = cv::Scalar{ 0. };
        result[0] = computescore(model, range_min, range_max, img);
        return result;
    }

    cv::Scalar compute(const svm_model* model, const float* range_min, const float* range_max, std::vector<brisque_mat_type>& imgs)
    {
        CV_Assert(imgs.size() > 0);

        cv::Scalar result = {};

        const auto sz = imgs.size();

        for (unsigned i = 0; i < sz; ++i)
        {
            auto cmp = compute(model, range_min, range_max, imgs[i]);
            cv::add(result, cmp, result);
        }

        if (sz > 1)
            result /= (cv::Scalar::value_type)sz;   // average result

        return result;
    }
}

// static
Ptr<QualityBRISQUE> QualityBRISQUE::create(const cv::String& model_file_path, const cv::String& range_file_path)
{
    return Ptr<QualityBRISQUE>(new QualityBRISQUE(model_file_path, range_file_path));
}

cv::Scalar QualityBRISQUE::compute(InputArrayOfArrays imgs, const cv::String& model_file_path, const cv::String& range_file_path)
{
    auto obj = create(model_file_path, range_file_path);
    return obj->compute(imgs);
}

// QualityBRISQUE() constructor
QualityBRISQUE::QualityBRISQUE(const cv::String& model_file_path, const cv::String& range_file_path)
{
    // construct data file path from OPENCV_DIR env var and quality subdir
    const auto get_data_path = [](const cv::String& fname)
    {
        cv::String path{ std::getenv("OPENCV_DIR") };
        return path.empty()
            ? path  // empty
            : path + "/testdata/contrib/quality/" + fname
            ;
    };

    const auto
        modelpath = model_file_path.empty() ? get_data_path("brisque_allmodel.dat") : model_file_path
        , rangepath = range_file_path.empty() ? get_data_path("brisque_allrange.dat") : range_file_path
        ;

    if (modelpath.empty())
        CV_Error(cv::Error::StsObjectNotFound, "BRISQUE model data not found");

    if (rangepath.empty())
        CV_Error(cv::Error::StsObjectNotFound, "BRISQUE range data not found");

    // load svm data
    this->_svm_model = svm_load_model(modelpath.c_str());
    if (!this->_svm_model)
        CV_Error(cv::Error::StsParseError, "Error loading BRISQUE model file");

    // load range data
    // based on original brisque impl
    //check if file exists
    char buff[100];
    FILE* range_file = fopen(rangepath.c_str(), "r");
    if (range_file == NULL)
        CV_Error(cv::Error::StsParseError, "Error loading BRISQUE range file");

    //assume standard file format for this program
    CV_Assert(fgets(buff, 100, range_file) != NULL);
    CV_Assert(fgets(buff, 100, range_file) != NULL);

    //now we can fill the array
    for (std::size_t i = 0; i < _SVM_RANGE_SIZE; ++i) {
        float a, b, c;
        CV_Assert(fscanf(range_file, "%f %f %f", &a, &b, &c) == 3);
        this->_svm_range_min[i] = (_svm_range_type)b;
        this->_svm_range_max[i] = (_svm_range_type)c;
    }
    fclose(range_file);
}

cv::Scalar QualityBRISQUE::compute(InputArrayOfArrays imgs)
{
    auto vec = quality_utils::expand_mats<brisque_mat_type>(imgs);// convert inputarrayofarrays to vector of brisque_mat_type

    // convert all mats to single channel/bgr2gray as needed, scale to 0-1
    for (auto& mat : vec)
    {
        switch (mat.channels())
        {
        case 1:
            break;
        case 3:
            cv::cvtColor(mat, mat, cv::COLOR_RGB2GRAY, 1);
            break;
        case 4:
            cv::cvtColor(mat, mat, cv::COLOR_RGBA2GRAY, 1);
            break;
        default:
            CV_Error(cv::Error::StsNotImplemented, "Unknown/unsupported channel count");
        };//switch

        // scale to 0-1 range
        mat.convertTo(mat, BRISQUE_CALC_MAT_TYPE, 1. / 255.);
    }

    // const brisque_svm_data* data_ptr = static_cast<const brisque_svm_data*>(this->_svm_data.get());
    return ::compute( (const svm_model*)this->_svm_model, this->_svm_range_min.data(), this->_svm_range_max.data(), vec);
}

QualityBRISQUE::~QualityBRISQUE()
{
    if (this->_svm_model != nullptr)
    {
        svm_model* ptr = (svm_model*)this->_svm_model;
        svm_free_and_destroy_model(&ptr);
        this->_svm_model = nullptr;
    }
}