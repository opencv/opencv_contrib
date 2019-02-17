// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/quality/qualitybrisque.hpp"
#include "opencv2/quality/quality_utils.hpp"
#include "opencv2/quality/libsvm/svm.hpp"   // libsvm

namespace
{
    using namespace cv;
    using namespace cv::quality;

    // type of mat we're working with internally; use cv::Mat for debugging
    using brisque_mat_type = Mat;

    // the type of quality map we'll generate (if brisque generates one)
    using _quality_map_type = brisque_mat_type;

    template<class T> class Image {
    private:
        Mat imgP;
    public:
        Image(Mat img = 0) {
            imgP = img.clone();
        }
        ~Image() {
            imgP = 0;
        }
        Mat equate(Mat img) {
            img = imgP.clone();
            return img;
        }
        inline T* operator[](const int rowIndx) {
            return (T*)(imgP.data + rowIndx * imgP.step);
        }
    };

    typedef Image<double> BwImage;

    // function to compute best fit parameters from AGGDfit 
    cv::Mat AGGDfit(cv::Mat structdis, double& lsigma_best, double& rsigma_best, double& gamma_best)
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
        float sampling = 0.001;
        for (float gam = 0.2; gam < 10; gam += sampling) //possible to coarsen sampling to quicken the code, with some loss of accuracy
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

    void ComputeBrisqueFeature(cv::Mat& orig, std::vector<double>& featurevector)
    {

        Mat orig_bw_int(orig.size(), CV_64F, 1);
        // convert to grayscale 
        cv::cvtColor(orig, orig_bw_int, cv::COLOR_BGR2GRAY);
        // create a copy of original image
        Mat orig_bw(orig_bw_int.size(), CV_64FC1, 1);
        orig_bw_int.convertTo(orig_bw, 1.0 / 255);
        orig_bw_int.release();

        // orig_bw now contains the grayscale image normalized to the range 0,1
        int scalenum = 2; // number of times to scale the image
        for (int itr_scale = 1; itr_scale <= scalenum; itr_scale++)
        {
            // resize image
            cv::Size dst_size(orig_bw.cols / cv::pow((double)2, itr_scale - 1), orig_bw.rows / pow((double)2, itr_scale - 1));
            cv::Mat imdist_scaled;
            cv::resize(orig_bw, imdist_scaled, dst_size, 0, 0, cv::INTER_CUBIC); // INTER_CUBIC
            imdist_scaled.convertTo(imdist_scaled, CV_64FC1, 1.0 / 255.0);
            // calculating MSCN coefficients
            // compute mu (local mean)
            cv::Mat mu(imdist_scaled.size(), CV_64FC1, 1);
            cv::GaussianBlur(imdist_scaled, mu, cv::Size(7, 7), 1.166);

            cv::Mat mu_sq;
            cv::pow(mu, double(2.0), mu_sq);

            //compute sigma (local sigma)
            cv::Mat sigma(imdist_scaled.size(), CV_64FC1, 1);
            cv::multiply(imdist_scaled, imdist_scaled, sigma);
            cv::GaussianBlur(sigma, sigma, cv::Size(7, 7), 1.166);

            cv::subtract(sigma, mu_sq, sigma);
            cv::pow(sigma, double(0.5), sigma);
            cv::add(sigma, Scalar(1.0 / 255), sigma); // to avoid DivideByZero Error

            cv::Mat structdis(imdist_scaled.size(), CV_64FC1, 1);
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

                // declare shifted_structdis as pairwise image
                cv::Mat shifted_structdis(imdist_scaled.size(), CV_64F, 1);

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

    // float computescore(String imagename) {
    double computescore( const cv::String& model_, const cv::String& range_, cv::Mat& orig ) {
        // pre-loaded vectors from allrange file 
        float min_[36] = { 0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351 };
        float max_[36] = { 9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484 };

        double qualityscore;
        int i;
        struct svm_model* model; // create svm model object
        // cv::Mat orig = cv::imread(imagename, 1); // read image (color mode)

        std::vector<double> brisqueFeatures; // feature vector initialization
        ComputeBrisqueFeature(orig, brisqueFeatures); // compute brisque features

        // use the pre-trained allmodel file

        // TODO:  Use model and range parameters provided to this function instead

        String modelfile = "allmodel";
        if ((model = svm_load_model(modelfile.c_str())) == 0) {
            fprintf(stderr, "can't open model file allmodel\n");
            exit(1);
        }

        // float min_[37];
        // float max_[37];
        //
        struct svm_node x[37];
        // rescale the brisqueFeatures vector from -1 to 1 
        // also convert vector to svm node array object
        for (i = 0; i < 36; ++i) {
            float min = min_[i];
            float max = max_[i];

            x[i].value = -1 + (2.0 / (max - min) * (brisqueFeatures[i] - min));
            x[i].index = i + 1;
        }
        x[36].index = -1;


        int nr_class = svm_get_nr_class(model);
        double *prob_estimates = (double *)malloc(nr_class * sizeof(double));
        // predict quality score using libsvm class
        qualityscore = svm_predict_probability(model, x, prob_estimates);

        free(prob_estimates);
        svm_free_and_destroy_model(&model);
        return qualityscore;
    }

    // computes score and quality map for single frame
    std::pair<cv::Scalar, _quality_map_type> compute( const cv::String& model, const cv::String& range, brisque_mat_type& img )
    {
        std::pair<cv::Scalar, _quality_map_type> result;

        // TODO:  calculate score and quality map for the input image, which is a single frame
        //  place score in result.first, and quality map in result.second

        result.first = cv::Scalar{ 0. };
        result.first[0] = computescore( model, range, img);  

        // is there a quality map that can be generated?

        return result;
    }

    // computes score and quality maps for multiple frames
    cv::Scalar compute( const cv::String& model, const cv::String& range, std::vector<brisque_mat_type>& imgs, OutputArrayOfArrays qualityMaps )
    {
        CV_Assert(imgs.size() > 0);

        cv::Scalar result = {};
        std::vector<_quality_map_type> quality_maps = {};
        const auto sz = imgs.size();

        // future optimization:  convert model and range to libsvm types here instead of every time we compute for a frame
        //  ideally, these would be created in the constructor so they are only created once per instantiation

        for (unsigned i = 0; i < sz; ++i)
        {
            auto cmp = compute( model, range, imgs[i] );
            cv::add(result, cmp.first, result);

            if ( qualityMaps.needed() )
                quality_maps.emplace_back(std::move(cmp.second));
        }

        if (qualityMaps.needed())
        {
            auto qMaps = InputArray(quality_maps);
            qualityMaps.create(qMaps.size(), qMaps.type());
            qualityMaps.assign(quality_maps);
        }

        if (sz > 1)
            result /= (cv::Scalar::value_type)sz;   // average result

        return result;
    }
}

// static
Ptr<QualityBRISQUE> QualityBRISQUE::create(cv::String model, cv::String range)
{
    return Ptr<QualityBRISQUE>(new QualityBRISQUE( std::move(model), std::move(range)));
}

// static
cv::Scalar QualityBRISQUE::compute( const cv::String& model, const cv::String& range, InputArrayOfArrays imgs, OutputArrayOfArrays qualityMaps)
{
    auto vec = quality_utils::expand_mats<brisque_mat_type>(imgs);// convert inputarrayofarrays to vector of brisque_mat_type
    return ::compute(model, range, vec, qualityMaps);
}

// QualityBRISQUE() constructor
QualityBRISQUE::QualityBRISQUE( cv::String model, cv::String range )
    : _model(std::move(model))
    , _range(std::move(range))
{
    // would be nice to convert the model/range strings to libsvm models
    //  and store in a unique_ptr<void> with a custom deleter in the qualitybrisque object so we don't expose libsvm headers
}

cv::Scalar QualityBRISQUE::compute( InputArrayOfArrays imgs )
{
    auto vec = quality_utils::expand_mats<brisque_mat_type>(imgs);// convert inputarrayofarrays to vector of brisque_mat_type
    return ::compute( this->getModel(), this->getRange(), vec, this->_qualityMaps);
}