// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "tracking_utils.hpp"

using namespace cv;

double tracking_internal::computeNCC(const Mat& patch1, const Mat& patch2)
{
    CV_Assert( patch1.rows == patch2.rows,
               patch1.cols == patch2.cols);

    int N = patch1.rows * patch1.cols;

    if(N <= 1000 && patch1.type() == CV_8U && patch2.type() == CV_8U)
    {
        unsigned s1 = 0, s2 = 0;
        unsigned n1 = 0, n2 = 0;
        unsigned prod = 0;

        if(patch1.isContinuous() && patch2.isContinuous())
        {
            const uchar* p1Ptr = patch1.ptr<uchar>(0);
            const uchar* p2Ptr = patch2.ptr<uchar>(0);

            for(int j = 0; j < N; j++)
            {
              s1 += p1Ptr[j];
              s2 += p2Ptr[j];
              n1 += p1Ptr[j]*p1Ptr[j];
              n2 += p2Ptr[j]*p2Ptr[j];
              prod += p1Ptr[j]*p2Ptr[j];
            }
        }
        else
        {
            for(int i = 0; i < patch1.rows; i++)
            {
              const uchar* p1Ptr = patch1.ptr<uchar>(i);
              const uchar* p2Ptr = patch2.ptr<uchar>(i);

              for(int j = 0; j < patch1.cols; j++)
              {
                s1 += p1Ptr[j];
                s2 += p2Ptr[j];
                n1 += p1Ptr[j]*p1Ptr[j];
                n2 += p2Ptr[j]*p2Ptr[j];
                prod += p1Ptr[j]*p2Ptr[j];
              }
            }
        }

        double sq1 = sqrt(std::max(0.0, n1 - 1.0 * s1 * s1 / N));
        double sq2 = sqrt(std::max(0.0, n2 - 1.0 * s2 * s2 / N));
        return (sq2 == 0) ? sq1 / abs(sq1) : (prod - 1.0 * s1 * s2 / N) / sq1 / sq2;
    }
    else
    {
        double s1 = sum(patch1)(0);
        double s2 = sum(patch2)(0);
        double n1 = norm(patch1, NORM_L2SQR);
        double n2 = norm(patch2, NORM_L2SQR);
        double prod=patch1.dot(patch2);
        double sq1 = sqrt(std::max(0.0, n1 - 1.0 * s1 * s1 / N));
        double sq2 = sqrt(std::max(0.0, n2 - 1.0 * s2 * s2 / N));
        return (sq2 == 0) ? sq1 / abs(sq1) : (prod - s1 * s2 / N) / sq1 / sq2;
    }
}
