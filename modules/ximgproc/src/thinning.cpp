#include "precomp.hpp"

using namespace std;

namespace cv {
namespace ximgproc {

// look up table - there is one entry for each of the 2^8=256 possible
// combinations of 8 binary neighbors.
static uint8_t lut_zhang_iter0[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1};

static uint8_t lut_zhang_iter1[] = {
    1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
    0, 1, 1, 1};

static uint8_t lut_guo_iter0[] = {
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1};

static uint8_t lut_guo_iter1[] = {
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1,
    1, 1, 1, 1};

// Applies a thinning iteration to a binary image
static void thinningIteration(Mat img, int iter, int thinningType){
    Mat marker = Mat::zeros(img.size(), CV_8UC1);
    int rows = img.rows;
    int cols = img.cols;
    marker.col(0).setTo(1);
    marker.col(cols - 1).setTo(1);
    marker.row(0).setTo(1);
    marker.row(rows - 1).setTo(1);

    if(thinningType == THINNING_ZHANGSUEN){
        marker.forEach<uchar>([=](uchar& value, const int postion[]) {
            int i = postion[0];
            int j = postion[1];
            if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1)
                return;

            auto ptr = img.ptr(i, j); // p1

            // p9 p2 p3
            // p8 p1 p4
            // p7 p6 p5
            uchar p2 = ptr[-cols];
            uchar p3 = ptr[-cols + 1];
            uchar p4 = ptr[1];
            uchar p5 = ptr[cols + 1];
            uchar p6 = ptr[cols];
            uchar p7 = ptr[cols - 1];
            uchar p8 = ptr[-1];
            uchar p9 = ptr[-cols - 1];

            int neighbors = p9 | (p2 << 1) | (p3 << 2) | (p4 << 3) | (p5 << 4) | (p6 << 5) | (p7 << 6) | (p8 << 7);

            if (iter == 0)
                value = lut_zhang_iter0[neighbors];
            else
                value = lut_zhang_iter1[neighbors];

            //int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
            //         (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
            //         (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
            //         (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            //int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            //int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            //int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
            //if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) value = 0;
            // else value = 1;
        });
    }
    if(thinningType == THINNING_GUOHALL){
        marker.forEach<uchar>([=](uchar& value, const int postion[]) {
            int i = postion[0];
            int j = postion[1];
            if (i == 0 || j == 0 || i == rows - 1 || j == cols - 1)
                return;

            auto ptr = img.ptr(i, j); // p1

            // p9 p2 p3
            // p8 p1 p4
            // p7 p6 p5
            uchar p2 = ptr[-cols];
            uchar p3 = ptr[-cols + 1];
            uchar p4 = ptr[1];
            uchar p5 = ptr[cols + 1];
            uchar p6 = ptr[cols];
            uchar p7 = ptr[cols - 1];
            uchar p8 = ptr[-1];
            uchar p9 = ptr[-cols - 1];

            int neighbors = p9 | (p2 << 1) | (p3 << 2) | (p4 << 3) | (p5 << 4) | (p6 << 5) | (p7 << 6) | (p8 << 7);

            if (iter == 0)
                value = lut_guo_iter0[neighbors];
            else
                value = lut_guo_iter1[neighbors];

            //int C  = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
            //         ((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
            //int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            //int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            //int N  = N1 < N2 ? N1 : N2;
            //int m  = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);
            //if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0))) value = 0;
            // else value = 1;
        });
    }

    img &= marker;
}

// Apply the thinning procedure to a given image
void thinning(InputArray input, OutputArray output, int thinningType){
    Mat processed = input.getMat().clone();
    CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
    // Enforce the range of the input image to be in between 0 - 255
    processed /= 255;

    Mat prev = processed.clone();
    Mat diff;

    do {
        thinningIteration(processed, 0, thinningType);
        thinningIteration(processed, 1, thinningType);
        absdiff(processed, prev, diff);
        if (!hasNonZero(diff)) break;
        processed.copyTo(prev);
    }
    while (true);

    processed *= 255;

    output.assign(processed);
}

} //namespace ximgproc
} //namespace cv
