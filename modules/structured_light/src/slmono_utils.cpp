#include "opencv2/structured_light/slmono_utils.hpp"

namespace cv{
namespace structured_light{

//quadrand swapping for FFT
void circshift(OutputArray out, InputArray in, int xdim, int ydim, bool isFftshift = true) {

    Mat in_ = in.getMat();
    Mat& out_ = *(Mat*) out.getObj();

    if (isFftshift) {
        int xshift = (xdim / 2);
        int yshift = (ydim / 2);

        for (int i = 0; i < xdim; i++) {
            int ii = (i + xshift) % xdim;
            for (int j = 0; j < ydim; j++) {
                int jj = (j + yshift) % ydim;
                out_.at<float>(ii * ydim + jj) = in_.at<float>(i * ydim + j);
            }
        }
    }
    else {
        int xshift = ((xdim + 1) / 2);
        int yshift = ((ydim + 1) / 2);

        for (int i = 0; i < xdim; i++) {
            int ii = (i + xshift) % xdim;
            for (int j = 0; j < ydim; j++) {
                int jj = (j + yshift) % ydim;
                out_.at<float>(ii * ydim + jj) = in_.at<float>(i * ydim + j);
            }
        }
    }
}


void createGrid(OutputArray output, Size size) {
    auto gridX = Mat(size, CV_32FC1);
    auto gridY = Mat(size, CV_32FC1);

    for (auto i = 0; i < size.height; i++) {
        for (auto j = 0; j < size.width; j++) {
            gridX.at<float>(i, j) = float(j + 1);
            gridY.at<float>(i, j) = float(i + 1);
        }
    }
    multiply(gridX, gridX, gridX);
    multiply(gridY, gridY, gridY);

    add(gridX, gridY, output);
}

void wrapSin(InputArray img, OutputArray out) {

    Mat img_ = img.getMat();
    Mat& out_ = *(Mat*) out.getObj();
    out_ = Mat(img_.rows, img_.cols, CV_32FC1);

    for (auto i = 0; i < img_.rows; i++) {
        for (auto j = 0; j < img_.cols; j++) {
            float x = img_.at<float>(i, j);
            while (abs(x) >= M_PI_2) {
                x += ((x > 0) - (x < 0)) * (float(M_PI) - 2 * abs(x));
            }
            out_.at<float>(i, j) = x;
        }
    }
}

void wrapCos(InputArray img, OutputArray out) {

    Mat img_ = img.getMat();
    Mat& out_ = *(Mat*) out.getObj();
    out_ = Mat(img_.rows, img_.cols, CV_32FC1);

    for (auto i = 0; i < img_.rows; i++) {
        for (auto j = 0; j < img_.cols; j++) {
            float x = img_.at<float>(i, j) - (float)M_PI_2;
            while (abs(x) > M_PI_2) {
                x += ((x > 0) - (x < 0)) * ((float)M_PI - 2 * abs(x));
            }
            out_.at<float>(i, j) = -x;
        }
    }
}

void computeAtanDiff(InputOutputArrayOfArrays src, OutputArray dst) {

    std::vector<Mat>& src_ = *( std::vector<Mat>* ) src.getObj();

    Mat& dst_ = *(Mat*) dst.getObj();

    for (int i = 0; i < src_[0].rows; i++) {
        for (int j = 0; j < src_[0].cols; j++) {
            float x = src_[3].at<float>(i, j) - src_[1].at<float>(i, j);
            float y = src_[0].at<float>(i, j) - src_[2].at<float>(i, j);
            dst_.at<float>(i, j) = std::atan2(x, y);
        }
    }
}

void Laplacian(InputArray img, InputArray grid, OutputArray out, int flag = 0) {

    Mat& img_ = *(Mat*) img.getObj();
    Mat& out_ = *(Mat*) out.getObj();

    if (flag == 0){
        dct(img, out_);
        multiply(out_, grid, out_);
        dct(out_, out_, DCT_INVERSE);
        out_ = out_ * (-4 * M_PI * M_PI / (img_.rows * img_.cols));
    }
    else if (flag == 1)
    {
        dct(img, out_);
        divide(out_, grid, out_);
        dct(out_, out_, DCT_INVERSE);
        out_ = out_ * (-img_.rows * img_.cols) / (4 * M_PI * M_PI);
    }

}

void computeDelta(InputArray img, InputArray grid, OutputArray out) {

    Mat x1, x2;
    Mat img_sin, img_cos;

    wrapSin(img, img_sin);
    wrapCos(img, img_cos);

    Mat laplacian1, laplacian2;

    Laplacian(img_sin, grid, laplacian1);
    Laplacian(img_cos, grid, laplacian2);

    multiply(img_cos, laplacian1, x1);
    multiply(img_sin, laplacian2, x2);
    subtract(x1, x2, out);
}


void unwrapPCG(InputArray img, OutputArray out, Size imgSize) {

    Mat g_laplacian;
    Mat phase1;
    Mat error, k1, k2, phase2;
    Mat phiError;

    createGrid(g_laplacian, imgSize);
    computeDelta(img, g_laplacian, phase1);
    Laplacian(phase1, g_laplacian, phase1, 1);

    subtract(phase1, img, k1);
    k1 *= 0.5 / M_PI;
    abs(k1);
    k1 *= 2 * M_PI;
    add(img, k1, out);

    for (auto i = 0; i < 0; i++) {
        subtract(phase2, phase1, error);
        computeDelta(error, g_laplacian, phiError);
        Laplacian(phiError, g_laplacian, phiError, 1);

        add(phase1, phiError, phase1);
        subtract(phase1, img, k2);
        k2 *= 0.5 / M_PI;
        abs(k2);
        k2 *= 2 * M_PI;
        add(img, k2, out);
        k2.copyTo(k1);
    }
}

void unwrapTPU(InputArray phase1, InputArray phase2, OutputArray out, int scale) {

    Mat& phase1_ = *(Mat*) phase1.getObj();
    Mat& phase2_ = *(Mat*) phase2.getObj();

    phase1_.convertTo(phase1_, phase1_.type(), scale);
    subtract(phase1_, phase2_, phase1_);
    phase1_.convertTo(phase1_, phase1_.type(), 0.5f / CV_PI);
    abs(phase1_);
    phase1_.convertTo(phase1_, phase1_.type(), 2 * CV_PI);
    add(phase1_, phase2_, out);
}

void fft2(InputArray in, OutputArray complexI) {

    Mat in_ = in.getMat();
    Mat& complexI_ = *(Mat*) complexI.getObj();

    Mat padded;
    int m = getOptimalDFTSize(in_.rows);
    int n = getOptimalDFTSize(in_.cols);
    copyMakeBorder(in, padded, 0, m - in.rows, 0, n - in.cols,
            BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    merge(planes, 2, complexI_);
    dft(complexI_, complexI_);
}

void lowPassFilter(InputArray image, OutputArray out, int filterSize) {
    Mat& image_ = *(Mat*) image.getObj();
    int rows = image_.rows;
    int cols = image_.cols;

    Mat greyMat;
    cvtColor(image, greyMat, COLOR_BGR2GRAY);
    Mat result;
    fft2(greyMat, result);
    Mat matrix_(Size(rows, cols), CV_64FC1);
    circshift(matrix_, result, result.rows, result.cols, true);


    Mat lowPass(Size(rows, cols), CV_64FC1, Scalar(0));

    lowPass(Rect_<int>((int)(0.5*rows-filterSize), (int)(0.5 * cols - filterSize),
                      (int)(0.5*rows+filterSize), (int)(0.5 * cols + filterSize))) = 1;

    Mat pass = matrix_.mul(lowPass);

    Mat J1(Size(rows, cols), CV_64FC1);
    circshift(J1, pass, rows, cols, false);

    idft(J1, out);

}

void highPassFilter(InputArray image, OutputArray out, int filterSize) {

    Mat& image_ = *(Mat*) image.getObj();
    int rows = image_.rows;
    int cols = image_.cols;

    Mat greyMat;
    cvtColor(image, greyMat, COLOR_BGR2GRAY);
    Mat result;
    fft2(greyMat, result);
    Mat matrix_(Size(rows, cols), CV_64FC1);
    circshift(matrix_, result, result.rows, result.cols, true);

    Mat highPass(Size(rows, cols), CV_64FC1, Scalar(1));

    highPass(Rect_<int>((int)(0.5*rows-filterSize), (int)(0.5 * cols - filterSize),
                        (int)(0.5*rows+filterSize), (int)(0.5 * cols + filterSize))) = 0;

    Mat pass = matrix_.mul(highPass);

    Mat filter(Size(rows, cols), CV_64FC1);
    circshift(filter, pass, rows, cols, false);

    idft(filter, out);
}


}
}