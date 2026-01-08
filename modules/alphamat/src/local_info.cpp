// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// #ifndef local_info
// #define local_info

#include "precomp.hpp"
#include "local_info.hpp"

namespace cv { namespace alphamat {

void local_info(Mat& img, Mat& tmap, SparseMatrix<double>& Wl, SparseMatrix<double>& Dl)
{
    float eps = 0.000001;
    int win_size = 1;

    int nRows = img.rows;
    int nCols = img.cols;
    int N = img.rows * img.cols;
    Mat unk_img = Mat::zeros(cv::Size(nCols, nRows), CV_32FC1);

    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            uchar pix = tmap.at<uchar>(i, j);
            if (pix == 128)  // collection of unknown pixels samples
            {
                unk_img.at<float>(i, j) = 255;
            }
        }
    }

    Mat element = getStructuringElement(MORPH_RECT, Size(2 * win_size + 1, 2 * win_size + 1));
    /// Apply the dilation operation
    Mat dilation_dst = unk_img.clone();
    //dilate(unk_img, dilation_dst, element);

    int num_win = (win_size * 2 + 1) * (win_size * 2 + 1);  // number of pixels in window
    typedef Triplet<double> T;
    std::vector<T> triplets, td, tl;
    int neighInd[9];
    int i, j;
    for (j = win_size; j < nCols - win_size; j++)
    {
        for (i = win_size; i < nRows - win_size; i++)
        {
            uchar pix = tmap.at<uchar>(i, j);
            //std::cout << i+j*nRows << " --> " << pix << std::endl;
            if (pix != 128)
                continue;
            // extract the window out of image
            Mat win = img.rowRange(i - win_size, i + win_size + 1);
            win = win.colRange(j - win_size, j + win_size + 1);
            Mat win_ravel = Mat::zeros(9, 3, CV_64F);  // doubt ??
            double sum1 = 0;
            double sum2 = 0;
            double sum3 = 0;

            int c = 0;
            for (int q = -1; q <= 1; q++)
            {
                for (int p = -1; p <= 1; p++)
                {
                    neighInd[c] = (j + q) * nRows + (i + p);  // column major
                    c++;
                }
            }

            c = 0;
            //parsing column major way in the window
            for (int q = 0; q < win_size * 2 + 1; q++)
            {
                for (int p = 0; p < win_size * 2 + 1; p++)
                {
                    win_ravel.at<double>(c, 0) = win.at<cv::Vec3b>(p, q)[0] / 255.0;
                    win_ravel.at<double>(c, 1) = win.at<cv::Vec3b>(p, q)[1] / 255.0;
                    win_ravel.at<double>(c, 2) = win.at<cv::Vec3b>(p, q)[2] / 255.0;
                    sum1 += win.at<cv::Vec3b>(p, q)[0] / 255.0;
                    sum2 += win.at<cv::Vec3b>(p, q)[1] / 255.0;
                    sum3 += win.at<cv::Vec3b>(p, q)[2] / 255.0;
                    c++;
                }
            }
            win = win_ravel;
            Mat win_mean = Mat::zeros(1, 3, CV_64F);
            win_mean.at<double>(0, 0) = sum1 / num_win;
            win_mean.at<double>(0, 1) = sum2 / num_win;
            win_mean.at<double>(0, 2) = sum3 / num_win;

            // calculate the covariance matrix
            Mat covariance = (win.t() * win / num_win) - (win_mean.t() * win_mean);

            Mat I = Mat::eye(img.channels(), img.channels(), CV_64F);
            Mat I1 = (covariance + (eps / num_win) * I);
            Mat I1_inv = I1.inv();

            Mat X = win - repeat(win_mean, num_win, 1);
            Mat vals = (1 + X * I1_inv * X.t()) / num_win;

            for (int q = 0; q < num_win; q++)
            {
                for (int p = 0; p < num_win; p++)
                {
                    triplets.push_back(T(neighInd[p], neighInd[q], vals.at<double>(p, q)));
                }
            }
        }
    }

    std::vector<T> tsp;
    SparseMatrix<double> W(N, N), Wsp(N, N);
    W.setFromTriplets(triplets.begin(), triplets.end());

    SparseMatrix<double> Wt = W.transpose();
    SparseMatrix<double> Ws = Wt + W;
    W = Ws;

    for (int k = 0; k < W.outerSize(); ++k)
    {
        double sumCol = 0;
        for (SparseMatrix<double>::InnerIterator it(W, k); it; ++it)
        {
            sumCol += it.value();
        }
        if (sumCol < 0.05)
            sumCol = 1;
        tsp.push_back(T(k, k, 1 / sumCol));
    }
    Wsp.setFromTriplets(tsp.begin(), tsp.end());

    Wl = Wsp * W;  // For normalization
    //Wl = W; // No normalization

    SparseMatrix<double> Wlt = Wl.transpose();

    for (int k = 0; k < Wlt.outerSize(); ++k)
    {
        double sumarr = 0;
        for (SparseMatrix<double>::InnerIterator it(Wlt, k); it; ++it)
            sumarr += it.value();
        td.push_back(T(k, k, sumarr));
    }

    Dl.setFromTriplets(td.begin(), td.end());

    CV_LOG_INFO(NULL, "ALPHAMAT: local_info DONE");
}

}}  // namespace cv::alphamat

// #endif
