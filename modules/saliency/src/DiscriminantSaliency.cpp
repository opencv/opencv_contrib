// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "precomp.hpp"
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <numeric>
#include <fstream>



using namespace std;
using namespace cv;
using namespace cv::saliency;

namespace cv
{
namespace saliency
{
/*DiscriminantSaliency::DiscriminantSaliency()
{
    imgProcessingSize = Size(127, 127);
    hiddenSpaceDimension = 10;
    centerSize = 8;
    windowSize = 96;
    patchSize = 400;
    temporalSize = 11;
    stride = 1;
    CV_Assert( hiddenSpaceDimension <= temporalSize && temporalSize <= (unsigned)imgProcessingSize.width * imgProcessingSize.height );
}*/

DiscriminantSaliency::DiscriminantSaliency(unsigned _stride, Size _imgProcessingSize, unsigned _hidden, unsigned _center, unsigned _window, unsigned _patch, unsigned _temporal)
{
    imgProcessingSize = _imgProcessingSize;
    hiddenSpaceDimension = _hidden;
    centerSize = _center;
    windowSize = _window;
    patchSize = _patch;
    temporalSize = _temporal;
    stride = _stride;
    CV_Assert( hiddenSpaceDimension <= temporalSize && temporalSize <= (unsigned)imgProcessingSize.width * imgProcessingSize.height && stride <= (centerSize - 1) / 2 );
}

DiscriminantSaliency::~DiscriminantSaliency(){}

bool DiscriminantSaliency::computeSaliencyImpl( InputArray image, OutputArray saliencyMap )
{
    vector<Mat> img_sq;
    image.getMatVector(img_sq);
    vector<Mat>& saliency_sq = *( std::vector<Mat>* ) saliencyMap.getObj();
    CV_Assert( !(img_sq.empty()) || !(img_sq[0].empty()) );
    saliencyMapGenerator(img_sq, saliency_sq);
    return true;
}

vector<Mat> DiscriminantSaliency::saliencyMapGenerator( vector<Mat> img_sq, vector<Mat>& saliency_sq )
{
    CV_Assert( img_sq.size() >= temporalSize );
    for ( unsigned i = 0; i < img_sq.size(); i++ )
    {
        resize(img_sq[i], img_sq[i], imgProcessingSize);
    }
    //vector<Mat> saliency_sq;
    for ( unsigned i = temporalSize - 1; i < img_sq.size(); i++ )
    {
        saliency_sq.push_back(Mat(imgProcessingSize, CV_64F, Scalar::all(0.0)));
        for ( unsigned r = (centerSize - 1) / 2; r < imgProcessingSize.height - (centerSize - (centerSize - 1) / 2 - 1); r+= (2 * stride + 1) )
        {
            for ( unsigned c = (centerSize - 1) / 2; c < imgProcessingSize.width - (centerSize - (centerSize - 1) / 2 - 1); c+= (2 * stride + 1) )
            {
                Mat center, surround, all;
                DT para_c0, para_c1, para_w;
                patchGenerator(img_sq, i, r, c, center, surround, all);
                //dynamicTextureEstimator( surround, para_c0 );
                dynamicTextureEstimator( center, para_c1 );
                dynamicTextureEstimator( all, para_w );
                double kl1 = KLdivDT( para_c1, para_w );//, kl0 = KLdivDT( para_c0, para_w );

                for ( unsigned rn = r - stride; rn <= r + stride; rn++)
                {
                    for ( unsigned cn = c - stride; cn <= c + stride; cn++)
                    {
                        saliency_sq.back().at<double>(rn, cn) = max(saliency_sq.back().at<double>(rn, cn), kl1);
                    }
                }
                //saliency_sq.back().at<double>(r, c) = kl1;
                //cout << r << " " << c << endl;
             }
         }
    }
    return saliency_sq;
}

void DiscriminantSaliency::patchGenerator( const vector<Mat>& img_sq, unsigned index, unsigned r, unsigned c, Mat& center, Mat& surround, Mat& all )
{
    unsigned r1 = max((int)r - ((int)windowSize - 1) / 2, 0), c1 = max((int)c - ((int)windowSize - 1) / 2, 0);
    unsigned r2 = min(r1 + windowSize, (unsigned)imgProcessingSize.height), c2 = min(c1 + windowSize, (unsigned)imgProcessingSize.width);
    all = Mat(patchSize, temporalSize, CV_64F, Scalar::all(0.0));
    surround = Mat(patchSize, temporalSize, CV_64F, Scalar::all(0.0));
    center = Mat(patchSize, temporalSize, CV_64F, Scalar::all(0.0));
    srand(0);
    for ( int i = 0; i < all.size[0]; i++ )
    {
        unsigned rt = rand() % (r2 - r1) + r1, ct = rand() % (c2 - c1) + c1;
        for ( int j = 0; j < all.size[1]; j++ )
        {
            all.at<double>(i, j) = (double)img_sq[j + 1 + index - temporalSize].at<uchar>(rt, ct);
        }
    }
    srand(0);
    for ( int i = 0; i < center.size[0]; i++ )
    {
        unsigned rt = rand() % centerSize + r - (centerSize - 1) / 2, ct = rand() % centerSize + c - (centerSize - 1) / 2;
        for ( int j = 0; j < center.size[1]; j++ )
        {
            center.at<double>(i, j) = (double)img_sq[j + 1 + index - temporalSize].at<uchar>(rt, ct);
        }
    }
    srand(0);
    for ( int i = 0; i < surround.size[0]; i++ )
    {
        unsigned rt = rand() % (r2 - r1) + r1, ct = rand() % (c2 - c1) + c1;
        while ((abs(rt - r) < (centerSize / 2)) && (abs(ct - c) < (centerSize / 2)))
        {
            rt = rand() % (r2 - r1) + r1;
            ct = rand() % (c2 - c1) + c1;
        }
        for ( int j = 0; j < surround.size[1]; j++ )
        {
            surround.at<double>(i, j) = (double)img_sq[j + 1 + index - temporalSize].at<uchar>(rt, ct);
        }
    }
}

double DiscriminantSaliency::KLdivDT( const DT& para_c, const DT& para_w )
{
    Mat temp1, temp2, Eig;
    Mat MU_c = para_c.MU.clone();//1
    Mat MU_w = para_w.MU.clone();//1
    Mat S_c = para_c.S.clone();//1
    Mat S_w = para_w.S.clone();//1
    Mat Beta = ((para_w.S.inv()) + Mat::eye(para_w.S.size(), CV_64F) / (para_w.VAR)).inv();//1
    Mat Beta_c = ((para_c.S.inv()) + Mat::eye(para_c.S.size(), CV_64F) / (para_c.VAR)).inv();//1
    Mat Omega = -1 * para_w.Q.inv() * para_w.A;//const
    Mat Omega_c = -1 * para_c.Q.inv() * para_c.A;
    Mat Theta = (para_w.S.inv()) + (para_w.A.t()) * (para_w.Q.inv()) * (para_w.A);//const
    Mat Theta_c = (para_c.S.inv()) + (para_c.A.t()) * (para_c.Q.inv()) * (para_c.A);
    Mat Tc = para_w.C.t() * para_c.C;//const
    Mat Vc = para_w.C.t() * para_c.C * MU_c - MU_w;//1
    double Omegac = trace(para_c.S).val[0] / para_w.VAR + imgProcessingSize.width * imgProcessingSize.height * para_c.VAR / para_w.VAR - para_c.VAR / pow(para_w.VAR, 2) * trace(Beta).val[0];
    Mat Psic = 1 / pow(para_w.VAR, 2) * Tc * para_c.S * Tc.t();

    Mat U_c = para_c.A * para_c.S;//2
    Mat U_w = para_w.A * para_w.S;//2
    Mat G = -1 * Beta * Omega;//2
    Mat G_c = -1 * Beta_c * Omega_c;
    Mat H = Theta + Mat::eye(Theta.size(), CV_64F) / (para_w.VAR) - Omega.t() * Beta * Omega;//2
    Mat H_c = Theta_c + Mat::eye(Theta_c.size(), CV_64F) / (para_c.VAR) - Omega_c.t() * Beta_c * Omega_c;
    S_w = para_w.A * S_w * para_w.A.t() + para_w.Q;//2
    MU_c = para_c.A * MU_c;//2
    MU_w = para_w.A * MU_w;//2
    Mat Zc = 1 / para_w.VAR * para_w.C * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * Vc - para_c.C * MU_c + para_w.C * MU_w;//2

    PCA pt_pca((S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()), cv::Mat(),  1, 0);
    Eig =  pt_pca.eigenvalues;
    //eigen((S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()), Eig);
    Eig /= para_w.VAR;
    Eig += Mat::ones(Eig.size(), CV_64F);
    log(Eig, Eig);
    double det_w = imgProcessingSize.width * imgProcessingSize.height * log(para_w.VAR) + sum(Eig).val[0];

    pt_pca = PCA((S_c - 1 / para_c.VAR * U_c * (Mat::eye(Beta_c.size(), CV_64F) - Beta_c / para_c.VAR) * U_c.t()), cv::Mat(),  1, 0);
    Eig =  pt_pca.eigenvalues;
    //eigen((S_c - 1 / para_c.VAR * U_c * (Mat::eye(Beta.size(), CV_64F) - Beta_c / para_c.VAR) * U_c.t()), Eig);
    Eig /= para_c.VAR;
    Eig += Mat::ones(Eig.size(), CV_64F);
    log(Eig, Eig);
    double det_c = imgProcessingSize.width * imgProcessingSize.height * log(para_c.VAR) + sum(Eig).val[0];

    Mat Gama = (S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()).inv() + Mat::eye(S_w.size(), CV_64F) / para_w.VAR;//2
    temp1 = (1 / para_w.VAR / para_w.VAR * Zc.t() * para_w.C * Gama.inv() * para_w.C.t() * Zc);//2

    double update_term = pow(norm(Zc), 2) / para_w.VAR - temp1.at<double>(0, 0);
    Mat Xic = 1 / pow(para_w.VAR, 2) * para_c.A * para_c.S * Tc.t();

    S_c = para_c.A * S_c * para_c.A.t() + para_c.Q;//2

    Omegac += 1 / para_w.VAR * trace(S_c).val[0] + imgProcessingSize.width * imgProcessingSize.height * para_c.VAR / para_w.VAR - para_c.VAR / pow(para_w.VAR, 2) * trace(H.inv()).val[0] - para_c.VAR / pow(para_w.VAR, 2) * trace(H.inv() * G.t() * G).val[0];

    hconcat(Psic, Xic.t() * Tc.t(), temp1);
    hconcat(Tc * Xic, 1 / pow(para_w.VAR, 2) * Tc * S_c * Tc.t(), temp2);
    vconcat(temp1, temp2, Psic);//Psic 2

    hconcat(H.inv(), H.inv() * G.t(), temp1);
    hconcat(G * H.inv(), Beta + G * H.inv() * G.t(), temp2);
    vconcat(temp1, temp2, Beta);//Beta 2

    hconcat(H_c.inv(), H_c.inv() * G_c.t(), temp1);
    hconcat(G_c * H_c.inv(), Beta_c + G_c * H_c.inv() * G_c.t(), temp2);
    vconcat(temp1, temp2, Beta_c);

    vconcat(Vc, para_w.C.t() * para_c.C * MU_c - MU_w, Vc);//2

    for ( unsigned i = 2; i < temporalSize; i++)
    {
        hconcat(para_c.A * U_c, para_c.A * S_c, U_c);
        hconcat(para_w.A * U_w, para_w.A * S_w, U_w);
        vconcat(-1 * H.inv() * Omega, -1 * G * H.inv() * Omega, G);
        vconcat(-1 * H_c.inv() * Omega_c, -1 * G_c * H_c.inv() * Omega_c, G_c);
        H = Theta + Mat::eye(Theta.size(), CV_64F) / para_w.VAR - Omega.t() * H.inv() * Omega;
        H_c = Theta_c + Mat::eye(Theta_c.size(), CV_64F) / para_c.VAR - Omega_c.t() * H_c.inv() * Omega_c;

        S_w = para_w.A * S_w * para_w.A.t() + para_w.Q;
        MU_c = para_c.A * MU_c;
        MU_w = para_w.A * MU_w;
        Zc = 1 / para_w.VAR * para_w.C * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * Vc - para_c.C * MU_c + para_w.C * MU_w;

        pt_pca = PCA((S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()), cv::Mat(),  1, 0);
        //eigen((S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()), Eig);
        Eig = pt_pca.eigenvalues;
        Eig /= para_w.VAR;
        Eig += Mat::ones(Eig.size(), CV_64F);
        log(Eig, Eig);
        det_w += imgProcessingSize.width * imgProcessingSize.height * log(para_w.VAR) + sum(Eig).val[0];

        pt_pca = PCA((S_c - 1 / para_c.VAR * U_c * (Mat::eye(Beta_c.size(), CV_64F) - Beta_c / para_c.VAR) * U_c.t()), cv::Mat(),  1, 0);
        //eigen((S_c - 1 / para_c.VAR * U_c * (Mat::eye(Beta.size(), CV_64F) - Beta_c / para_c.VAR) * U_c.t()), Eig);
        Eig = pt_pca.eigenvalues;
        Eig /= para_c.VAR;
        Eig += Mat::ones(Eig.size(), CV_64F);
        log(Eig, Eig);
        det_c += imgProcessingSize.width * imgProcessingSize.height * log(para_c.VAR) + sum(Eig).val[0];

        Gama = (S_w - 1 / para_w.VAR * U_w * (Mat::eye(Beta.size(), CV_64F) - Beta / para_w.VAR) * U_w.t()).inv() + Mat::eye(S_w.size(), CV_64F) / para_w.VAR;
        temp1 = pow(norm(Zc), 2) / para_w.VAR - 1 / para_w.VAR / para_w.VAR * Zc.t() * para_w.C * Gama.inv() * para_w.C.t() * Zc;
        update_term += temp1.at<double>(0, 0);
        hconcat(1/pow(para_w.VAR, 2) * para_c.A * Xic, 1/pow(para_w.VAR, 2) * para_c.A * S_c * Tc.t(), Xic);
        S_c = para_c.A * S_c * para_c.A.t() + para_c.Q;
        Omegac += 1 / para_w.VAR * trace(S_c).val[0] + imgProcessingSize.width * imgProcessingSize.height * para_c.VAR / para_w.VAR;
        Omegac -= para_c.VAR / pow(para_w.VAR, 2) * trace(H.inv()).val[0] - para_c.VAR / pow(para_w.VAR, 2) * trace((H.inv()) * (G.t()) * G).val[0];

        hconcat(Psic, Xic.t() * Tc.t(), temp1);
        hconcat(Tc * Xic, 1 / pow(para_w.VAR, 2) * Tc * S_c * Tc.t(), temp2);
        vconcat(temp1, temp2, Psic);

        hconcat(H.inv(), H.inv() * G.t(), temp1);
        hconcat(G * H.inv(), Beta + G * H.inv() * G.t(), temp2);
        vconcat(temp1, temp2, Beta);

        hconcat(H_c.inv(), H_c.inv() * G_c.t(), temp1);
        hconcat(G_c * H_c.inv(), Beta_c + G_c * H_c.inv() * G_c.t(), temp2);
        vconcat(temp1, temp2, Beta_c);

        vconcat(Vc, para_w.C.t() * para_c.C * MU_c - MU_w, Vc);
    }
    return det_w - det_c + update_term + Omegac - trace(Beta * Psic).val[0] - imgProcessingSize.width * imgProcessingSize.height * temporalSize;
}

void DiscriminantSaliency::dynamicTextureEstimator( const Mat img_sq, DT& para)
{

    unsigned tau = img_sq.size[1]; //row represents pixel location and column represents temporal location
    Mat me( img_sq.size[0], 1, CV_64F, Scalar::all(0.0) );
    Mat temp( img_sq.size[0], img_sq.size[1], CV_64F, Scalar::all(0.0) );
    Mat X, V, B, W, Y;

    for ( unsigned i = 0; i < tau; i++ )
    {
        me += img_sq.col(i) / tau;
    }
    temp += me * Mat::ones(1, temp.size[1], CV_64F) * (-1);
    temp += img_sq;
    Y = temp.clone();
    temp = temp.colRange(0, tau);

    SVD s = SVD(temp, SVD::MODIFY_A);
    para.C = s.u.colRange(0, hiddenSpaceDimension);
    para.C = para.C.clone();
    X = Mat::diag(s.w.rowRange(0, hiddenSpaceDimension)) * s.vt.rowRange(0, hiddenSpaceDimension);
    para.A = (X.colRange(1, tau) * (X.colRange(0, tau - 1)).inv(DECOMP_SVD));
    para.A = para.A.clone();

    V = (X.colRange(1, tau) - para.A * X.colRange(0, tau - 1));
    V = V.clone();
    SVD sv = SVD(V, SVD::MODIFY_A);
    B = sv.u * Mat::diag(sv.w) / sqrt(tau - 1);
    para.Q = (B * B.t());
    para.Q = para.Q.clone();


    /*Mat rnd = Mat(B.size[1], 1, CV_64F, Scalar::all(0.0));
    cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigma= cv::Mat::ones(1,1,CV_64FC1);
    cv::randn(rnd,  mean, sigma);
    Mat Xt = X.col(tau - 1);
    W = (Y.col(tau) - para.C * (para.A * Xt + B * rnd));

    Mat mea, std;
    meanStdDev(W, me, std);
    W -= me;
    //para.VAR = pow(std.at<double>(0, 0), 2);
    SVD sw = SVD(W, SVD::MODIFY_A);
    B = sw.u * Mat::diag(sw.w) / sqrt(tau - 1);
    para.R = (B * B.t());
    para.R = para.R.clone();
    para.VAR = (trace(para.R).val[0] / para.R.size[0]);*/
    para.VAR = 1;

    para.MU = Mat( hiddenSpaceDimension, 1, CV_64F, Scalar::all(0.0) );
    for (unsigned i = 0; i < tau; i++ )
    {
        para.MU += X.col(i) / tau;
    }
    para.S = (X - para.MU * Mat::ones(1, tau, CV_64F)) * ((X - para.MU * Mat::ones(1, tau, CV_64F)).t()) / (tau - 1);
    para.S = para.S.clone();
}

void DiscriminantSaliency::saliencyMapVisualize( InputArray _saliencyMap )
{
    Mat saliency = _saliencyMap.getMat().clone();
    double mi = 0, ma = 0;
    minMaxLoc( saliency, &mi, &ma );
    saliency -= mi;
    saliency /= ( ma - mi );
    imshow( "saliencyVisual", saliency );
    waitKey( 0 );
}

}
}
