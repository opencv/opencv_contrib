// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/alphamat.hpp"
#include <fstream>
#include <Eigen/Sparse>

using namespace Eigen;

namespace cv{
namespace alphamat{

void solve(SparseMatrix<double> Wcm,SparseMatrix<double> Wuu,SparseMatrix<double> Wl,SparseMatrix<double> Dcm,
               SparseMatrix<double> Duu,SparseMatrix<double> Dl, SparseMatrix<double> T, Mat& wf, Mat& alpha);

void solve(SparseMatrix<double> Wcm,SparseMatrix<double> Wuu,SparseMatrix<double> Wl,SparseMatrix<double> Dcm,
    SparseMatrix<double> Duu, SparseMatrix<double> Dl, SparseMatrix<double> T,
    Mat &wf, Mat &alpha){

    float suu = 0.01, sl = 0.1, lamd = 100;

    SparseMatrix<double> Lifm = ((Dcm-Wcm).transpose())*(Dcm-Wcm) + sl*(Dl-Wl) + suu*(Duu-Wuu);

    SparseMatrix<double> A;
    int n = wf.rows;
    VectorXd b(n), x(n);

    Eigen::VectorXd wf_;
    cv2eigen(wf, wf_);

    A = Lifm + lamd*T;
    b = (lamd*T)*(wf_);

    ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;

    cg.setMaxIterations(500);
    cg.compute(A);
    x = cg.solve(b);
    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;

    int nRows = alpha.rows;
    int nCols = alpha.cols;
    float pix_alpha;
    for (int j = 0; j < nCols; ++j)
        for (int i = 0; i < nRows; ++i) {
            pix_alpha = x(i+j*nRows);
            if (pix_alpha<0) pix_alpha=0;
            if (pix_alpha>1) pix_alpha=1;
            alpha.at<uchar>(i, j, 0) = uchar(pix_alpha*255);
        }
}

void infoFlow(InputArray image_ia, InputArray tmap_ia, OutputArray result){

    Mat image = image_ia.getMat();
    Mat tmap = tmap_ia.getMat();

    clock_t begin = clock();
    int nRows = image.rows;
    int nCols = image.cols;
    int N = nRows*nCols;

    SparseMatrix<double> T(N, N);
    typedef Triplet<double> Tr;
    std::vector<Tr> triplets;

    //Pre-process trimap
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j){
            float pix = tmap.at<uchar>(i, j);
            if (pix <= 0.2*255) tmap.at<uchar>(i, j) = 0;
            else if (pix >= 0.8*255) tmap.at<uchar>(i, j) = 255;
            else tmap.at<uchar>(i, j) = 128;
        }

    Mat wf = Mat::zeros(nRows*nCols, 1, CV_8U);

    // Column Major Interpretation for working with SparseMatrix
     for (int i = 0; i < nRows; ++i)
         for (int j = 0; j < nCols; ++j){
            float pix = tmap.at<uchar>(i, j);
             if (pix != 128)                // collection of known pixels samples
                 triplets.push_back(Tr(i+j*nRows, i+j*nRows, 1));
             else
                 triplets.push_back(Tr(i+j*nRows, i+j*nRows, 0));
             if (pix > 200)                                //  foreground pixel
                 wf.at<uchar>(i+j*nRows, 0) = 1;
             else
                 wf.at<uchar>(i+j*nRows, 0) = 0;
     }

    SparseMatrix<double> Wl(N, N), Dl(N, N);
    local_info(image, tmap, Wl, Dl);

    SparseMatrix<double> Wcm(N, N), Dcm(N, N);
    cm(image, tmap, Wcm, Dcm);

    Mat new_tmap = tmap.clone();

    SparseMatrix<double> Wuu(N, N), Duu(N, N);
    Mat image_t = image.t();
    Mat tmap_t = tmap.t();
    UU(image, tmap, Wuu, Duu);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    T.setFromTriplets(triplets.begin(), triplets.end());

    Mat alpha = Mat::zeros(nRows, nCols, CV_8UC1);
    solve(Wcm, Wuu, Wl, Dcm, Duu, Dl, T, wf, alpha);

    alpha.copyTo(result);

    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout<<"total time: "<<elapsed_secs<<std::endl;

}
}}
