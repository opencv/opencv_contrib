// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/infoflow.hpp"

namespace cv{
namespace alphamat{

// const int dim = 5;  // dimension of feature vectors

void solve(SparseMatrix<double> Wcm,SparseMatrix<double> Wuu,SparseMatrix<double> Wl,SparseMatrix<double> Dcm,
    SparseMatrix<double> Duu,SparseMatrix<double> Dl,SparseMatrix<double> H,SparseMatrix<double> T,
    Mat &ak, Mat &wf, bool useKU, Mat &alpha){

    float sku = 0.05, suu = 0.01, sl = 1, lamd = 100;
    // float sku = 0, suu = 0, sl = 0, lamd = 100;

    SparseMatrix<double> Lifm = ((Dcm-Wcm).transpose())*(Dcm-Wcm) + suu*(Duu-Wuu) + sl*(Dl-Wl);
    // # Lifm = suu*(Duu-Wuu) + sl*(Dl-Wl)
    SparseMatrix<double> A;
    int n = ak.cols;
    VectorXd b(n), wf_(n), x(n);


    for (int i = 0; i < n; i++)
        wf_(i) = wf.at<uchar>(i, 0);


    if (useKU)
    {
        A = Lifm + lamd*T + sku*H;
        b = (lamd*T + sku*H)*(wf_);
    }else{
        A = Lifm + lamd*T;
        b = (lamd*T)*(wf_);
    }

    ConjugateGradient<SparseMatrix<double>, Lower|Upper> cg;
    cg.setMaxIterations(500);
    cg.compute(A);
    x = cg.solve(b);

    std::cout << "#iterations:     " << cg.iterations() << std::endl;
    std::cout << "estimated error: " << cg.error()      << std::endl;

    int nRows = alpha.rows;
    int nCols = alpha.cols;
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j){
            // cout<<x(i*nCols+j)<<endl;
            alpha.at<uchar>(i, j) = x(i*nCols+j)*255;
        }
    // show(alpha);
    std::cout << "Done" << std::endl;
}

void infoFlow(Mat& image, Mat& tmap, Mat& result, bool useKU, bool trim){
    clock_t begin = clock();
    int nRows = image.rows;
    int nCols = image.cols;
    int N = nRows*nCols;

    Mat ak, wf;
    SparseMatrix<double> T(N, N);
    typedef Triplet<double> Tr;
    std::vector<Tr> triplets;
    // triplets.reserve(N*N);

    ak.create(1, nRows*nCols, CV_8U);
    wf.create(nRows*nCols, 1, CV_8U);
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j){
            float pix = tmap.at<uchar>(i, j);
            if (pix != 128)                // collection of known pixels samples
              triplets.push_back(Tr(i*nCols+j, i*nCols+j, 1));
            else
              triplets.push_back(Tr(i*nCols+j, i*nCols+j, 0));
            if (pix > 200)                                //  foreground pixel
              ak.at<uchar>(0, i*nCols+j) = 1;
            else
              ak.at<uchar>(0, i*nCols+j) = 0;
            wf.at<uchar>(i*nCols+j, 0) = ak.at<uchar>(0, i*nCols+j);
        }


    SparseMatrix<double> Wl(N, N), Dl(N, N);
    local_info(image, tmap, Wl, Dl);

    SparseMatrix<double> Wcm(N, N), Dcm(N, N);
    cm(image, tmap, Wcm, Dcm);

    Mat new_tmap = tmap.clone();  // after pre-processing
    // trimming(image, tmap, new_tmap, tmap, true);
    trimming(image, tmap, new_tmap, trim);
    SparseMatrix<double> H = KtoU(image, new_tmap, wf);

    SparseMatrix<double> Wuu(N, N), Duu(N, N);
    UU(image, tmap, Wuu, Duu);


    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "time for flow calc: " << elapsed_secs << std::endl;

    T.setFromTriplets(triplets.begin(), triplets.end());
    // Mat calc_alpha = solve(Wcm,Wuu,Wl,Dcm,Duu,Dl,H,T,ak,wf,true);

    Mat alpha;
    alpha.create(nRows, nCols, CV_8UC1);
    solve(Wcm, Wuu, Wl, Dcm, Duu, Dl, H, T, ak, wf, useKU, alpha);

    Mat trim_alpha = alpha.clone();
    std::cout << "Solved" << std::endl;

    int i, j;
    for (i = 0; i < image.rows; i++){
        for (j = 0; j < image.cols; j++){
            float pix = new_tmap.at<uchar>(i, j);
            if (pix != 128){
                // cout<<"in "<<float(trim_alpha.at<uchar>(i, j))<<endl;
                trim_alpha.at<uchar>(i, j) = pix;
            }
            if (float(trim_alpha.at<uchar>(i, j)) > 250)
                trim_alpha.at<uchar>(i, j) = 255;
            if (float(trim_alpha.at<uchar>(i, j)) < 5)
                trim_alpha.at<uchar>(i, j) = 0;
        }
    }

    // trim_alpha[trim_alpha > 230] = 255;
    // trim_alpha[trim_alpha < 20] = 0;


    // cout<<"Trimmed"<<endl;
    // char* res_path = argv[3];


    result = trim_alpha;
    // imwrite(res_path, trim_alpha);

    /*
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<<"total time: "<<elapsed_secs<<endl;
    */




}
}} // namespace ccalib, cv


/*
#include "opencv2/highgui.hpp"
int main(int argc, char** argv){
    Mat image, tmap;
    // const char* img_path = "../../data/input_lowres/elephant.png";
    char* img_path = argv[1];
    // cout<<img_path<<endl;
    image = cv::imread(img_path, cv::CV_LOAD_IMAGE_COLOR);   // Read the file

    // check_image(image);
    // show(image);
    // cout<<image.size<<endl;
    // cout<<image.channels()<<endl;

    // const char* tmap_path = "../../data/trimap_lowres/Trimap1/elephant.png";
    char* tmap_path = argv[2];
    tmap = cv::imread(tmap_path, cv::CV_LOAD_IMAGE_GRAYSCALE);
    // check_image(tmap);
    // show(tmap);
    Mat result;
    cv::alphamat::infoFlow(image, tmap, result, false, true);
    imwrite("result1.png", result);
    return 0;
}
*/
