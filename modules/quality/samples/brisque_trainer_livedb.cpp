#include  <sstream>
#include  <iostream>

#include "opencv2/quality.hpp"
#include "opencv2/quality/quality_utils.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/ml.hpp"

/*
BRISQUE Trainer using LIVE DB R2
http://live.ece.utexas.edu/research/Quality/subjective.htm
H.R. Sheikh, Z.Wang, L. Cormack and A.C. Bovik, "LIVE Image Quality Assessment Database Release 2", http://live.ece.utexas.edu/research/quality .
H.R. Sheikh, M.F. Sabir and A.C. Bovik, "A statistical evaluation of recent full reference image quality assessment algorithms", IEEE Transactions on Image Processing, vol. 15, no. 11, pp. 3440-3451, Nov. 2006.
Z. Wang, A.C. Bovik, H.R. Sheikh and E.P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," IEEE Transactions on Image Processing , vol.13, no.4, pp. 600- 612, April 2004.
*/

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

namespace {

    #define CATEGORIES 5
    #define IMAGENUM   982
    #define JP2KNUM    227
    #define JPEGNUM    233
    #define WNNUM      174
    #define GBLURNUM   174
    #define FFNUM      174

    // collects training data from LIVE R2 database
    // returns {features, responses}, 1 row per image
    std::pair<cv::Mat, cv::Mat> collect_data_live_r2(const std::string& foldername)
    {
        FILE* fid = nullptr;

        //----------------------------------------------------
        // class is the distortion category, there are 982 images in LIVE database
        std::vector<std::string> distortionlabels;
        distortionlabels.push_back("jp2k");
        distortionlabels.push_back("jpeg");
        distortionlabels.push_back("wn");
        distortionlabels.push_back("gblur");
        distortionlabels.push_back("fastfading");

        int imnumber[5] = { 0,227,460,634,808 };

        std::vector<int>categorylabels;
        categorylabels.insert(categorylabels.end(), JP2KNUM, 0);
        categorylabels.insert(categorylabels.end(), JPEGNUM, 1);
        categorylabels.insert(categorylabels.end(), WNNUM, 2);
        categorylabels.insert(categorylabels.end(), GBLURNUM, 3);
        categorylabels.insert(categorylabels.end(), FFNUM, 4);

        int  iforg[IMAGENUM];
        fid = fopen((foldername + "orgs.txt").c_str(), "r");
        for (int itr = 0; itr < IMAGENUM; itr++)
            CV_Assert( fscanf(fid, "%d", iforg + itr) > 0);
        fclose(fid);

        float dmosscores[IMAGENUM];
        fid = fopen((foldername + "dmos.txt").c_str(), "r");
        for (int itr = 0; itr < IMAGENUM; itr++)
            CV_Assert( fscanf(fid, "%f", dmosscores + itr) > 0 );
        fclose(fid);

        // features vector, 1 row per image
        cv::Mat features(0, 0, CV_32FC1);

        // response vector, 1 row per image
        cv::Mat responses(0, 1, CV_32FC1);

        for (int itr = 0; itr < IMAGENUM; itr++)
        {
            //Dont compute features for original images
            if (iforg[itr])
                continue;

            // append dmos score
            float score = dmosscores[itr];
            responses.push_back(cv::Mat(1, 1, CV_32FC1, (void*)&score));

            // load image, calc features
            std::string imname = "";
            imname.append(foldername);
            imname.append("/");
            imname.append(distortionlabels[categorylabels[itr]].c_str());
            imname.append("/img");
            imname += std::to_string((itr - imnumber[categorylabels[itr]] + 1));
            imname.append(".bmp");

            cv::Mat im_features;
            cv::quality::QualityBRISQUE::computeFeatures(cv::imread(imname), im_features);    // outputs a row vector

            features.push_back(im_features.row(0)); // append row vector
        }

        return std::make_pair(std::move(features), std::move(responses));
    }    //    collect_data_live_r2
}

inline void printHelp()
{
    using namespace std;
    cout << "    Demo of training BRISQUE quality assessment model using LIVE R2 database." << endl;
    cout << "    A. Mittal, A. K. Moorthy and A. C. Bovik, 'No Reference Image Quality Assessment in the Spatial Domain'" << std::endl << std::endl;

    cout << "    Usage: program <live_r2_db_path> <output_model_path> <output_range_path>" << endl << endl;
}

int main(int argc, const char * argv[])
{
    using namespace cv::ml;

    if (argc != 4)
    {
        printHelp();
        exit(1);
    }

    std::cout << "Training BRISQUE on database at " << argv[1] << "..." << std::endl;

    // collect data from the data set
    auto data = collect_data_live_r2( std::string( argv[1] ) + "/" );

    // extract column ranges for features
    const auto range = cv::quality::quality_utils::get_column_range(data.first);

    // scale all features from -1 to 1
    cv::quality::quality_utils::scale<float>(data.first, range, -1.f, 1.f);

    // do training, output train file
    // libsvm call from original BRISQUE impl:   svm-train  -s 3 -g 0.05 -c 1024 -b 1 -q train_scale allmodel
    auto svm = SVM::create();
    svm->setType(SVM::Types::EPS_SVR);
    svm->setKernel(SVM::KernelTypes::RBF);
    svm->setGamma(0.05);
    svm->setC(1024.);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::Type::EPS, 1000, 0.001));
    svm->setP(.1);// default p (epsilon) from libsvm

    svm->train(data.first, cv::ml::ROW_SAMPLE, data.second);
    svm->save( argv[2] );   // save to location specified in argv[2]

    // output scale file to argv[3]
    cv::Mat range_mat(range);
    cv::FileStorage fs(argv[3], cv::FileStorage::WRITE );
    fs << "range" << range_mat;

    return 0;
}
