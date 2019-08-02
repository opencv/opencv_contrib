// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/xfeatures2d.hpp>

#ifndef VERIFY_CORNERS
#define VERIFY_CORNERS 0
#endif

namespace {
    using namespace cv;
    #if VERIFY_CORNERS
    void testCorner(const uchar* ptr, const int pixel[], int K, int N, int threshold) {
        // check that with the computed "threshold" the pixel is still a corner
        // and that with the increased-by-1 "threshold" the pixel is not a corner anymore
        for( int delta = 0; delta <= 1; delta++ )
        {
            int v0 = std::min(ptr[0] + threshold + delta, 255);
            int v1 = std::max(ptr[0] - threshold - delta, 0);
            int c0 = 0, c1 = 0;

            for( int k = 0; k < N; k++ )
            {
                int x = ptr[pixel[k]];
                if(x > v0)
                {
                    if( ++c0 > K )
                        break;
                    c1 = 0;
                }
                else if( x < v1 )
                {
                    if( ++c1 > K )
                        break;
                    c0 = 0;
                }
                else
                {
                    c0 = c1 = 0;
                }
            }
            CV_Assert( (delta == 0 && std::max(c0, c1) > K) ||
                        (delta == 1 && std::max(c0, c1) <= K) );
        }
    }
    #endif

    template<int patternSize>
    int cornerScore(const uchar* ptr, const int pixel[], int threshold);

    template<>
    int cornerScore<16>(const uchar* ptr, const int pixel[], int threshold)
    {
        const int K = 8, N = K*3 + 1;
        int k, v = ptr[0];
        short d[N];
        for( k = 0; k < N; k++ )
            d[k] = (short)(v - ptr[pixel[k]]);

    #if CV_SSE2
        __m128i q0 = _mm_set1_epi16(-1000), q1 = _mm_set1_epi16(1000);
        for( k = 0; k < 16; k += 8 )
        {
            __m128i v0 = _mm_loadu_si128((__m128i*)(d+k+1));
            __m128i v1 = _mm_loadu_si128((__m128i*)(d+k+2));
            __m128i a = _mm_min_epi16(v0, v1);
            __m128i b = _mm_max_epi16(v0, v1);
            v0 = _mm_loadu_si128((__m128i*)(d+k+3));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+4));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+5));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+6));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+7));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+8));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k));
            q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
            q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
            v0 = _mm_loadu_si128((__m128i*)(d+k+9));
            q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
            q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
        }
        q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
        q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
        threshold = (short)_mm_cvtsi128_si32(q0) - 1;
    #else
        int a0 = threshold;
        for( k = 0; k < 16; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            a = std::min(a, (int)d[k+3]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+4]);
            a = std::min(a, (int)d[k+5]);
            a = std::min(a, (int)d[k+6]);
            a = std::min(a, (int)d[k+7]);
            a = std::min(a, (int)d[k+8]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+9]));
        }

        int b0 = -a0;
        for( k = 0; k < 16; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            b = std::max(b, (int)d[k+4]);
            b = std::max(b, (int)d[k+5]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+6]);
            b = std::max(b, (int)d[k+7]);
            b = std::max(b, (int)d[k+8]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+9]));
        }

        threshold = -b0-1;
    #endif

    #if VERIFY_CORNERS
        testCorner(ptr, pixel, K, N, threshold);
    #endif
        return threshold;
    }

    template<>
    int cornerScore<12>(const uchar* ptr, const int pixel[], int threshold)
    {
        const int K = 6, N = K*3 + 1;
        int k, v = ptr[0];
        short d[N + 4];
        for( k = 0; k < N; k++ )
            d[k] = (short)(v - ptr[pixel[k]]);
    #if CV_SSE2
        for( k = 0; k < 4; k++ )
            d[N+k] = d[k];
    #endif

    #if CV_SSE2
        __m128i q0 = _mm_set1_epi16(-1000), q1 = _mm_set1_epi16(1000);
        for( k = 0; k < 16; k += 8 )
        {
            __m128i v0 = _mm_loadu_si128((__m128i*)(d+k+1));
            __m128i v1 = _mm_loadu_si128((__m128i*)(d+k+2));
            __m128i a = _mm_min_epi16(v0, v1);
            __m128i b = _mm_max_epi16(v0, v1);
            v0 = _mm_loadu_si128((__m128i*)(d+k+3));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+4));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+5));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k+6));
            a = _mm_min_epi16(a, v0);
            b = _mm_max_epi16(b, v0);
            v0 = _mm_loadu_si128((__m128i*)(d+k));
            q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
            q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
            v0 = _mm_loadu_si128((__m128i*)(d+k+7));
            q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
            q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
        }
        q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
        q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
        threshold = (short)_mm_cvtsi128_si32(q0) - 1;
    #else
        int a0 = threshold;
        for( k = 0; k < 12; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+3]);
            a = std::min(a, (int)d[k+4]);
            a = std::min(a, (int)d[k+5]);
            a = std::min(a, (int)d[k+6]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+7]));
        }

        int b0 = -a0;
        for( k = 0; k < 12; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            b = std::max(b, (int)d[k+4]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+5]);
            b = std::max(b, (int)d[k+6]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+7]));
        }

        threshold = -b0-1;
    #endif

    #if VERIFY_CORNERS
        testCorner(ptr, pixel, K, N, threshold);
    #endif
        return threshold;
    }

    template<>
    int cornerScore<8>(const uchar* ptr, const int pixel[], int threshold)
    {
        const int K = 4, N = K*3 + 1;
        int k, v = ptr[0];
        short d[N];
        for( k = 0; k < N; k++ )
            d[k] = (short)(v - ptr[pixel[k]]);

    #if CV_SSE2
        __m128i v0 = _mm_loadu_si128((__m128i*)(d+1));
        __m128i v1 = _mm_loadu_si128((__m128i*)(d+2));
        __m128i a = _mm_min_epi16(v0, v1);
        __m128i b = _mm_max_epi16(v0, v1);
        v0 = _mm_loadu_si128((__m128i*)(d+3));
        a = _mm_min_epi16(a, v0);
        b = _mm_max_epi16(b, v0);
        v0 = _mm_loadu_si128((__m128i*)(d+4));
        a = _mm_min_epi16(a, v0);
        b = _mm_max_epi16(b, v0);
        v0 = _mm_loadu_si128((__m128i*)(d));
        __m128i q0 = _mm_min_epi16(a, v0);
        __m128i q1 = _mm_max_epi16(b, v0);
        v0 = _mm_loadu_si128((__m128i*)(d+5));
        q0 = _mm_max_epi16(q0, _mm_min_epi16(a, v0));
        q1 = _mm_min_epi16(q1, _mm_max_epi16(b, v0));
        q0 = _mm_max_epi16(q0, _mm_sub_epi16(_mm_setzero_si128(), q1));
        q0 = _mm_max_epi16(q0, _mm_unpackhi_epi64(q0, q0));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 4));
        q0 = _mm_max_epi16(q0, _mm_srli_si128(q0, 2));
        threshold = (short)_mm_cvtsi128_si32(q0) - 1;
    #else
        int a0 = threshold;
        for( k = 0; k < 8; k += 2 )
        {
            int a = std::min((int)d[k+1], (int)d[k+2]);
            if( a <= a0 )
                continue;
            a = std::min(a, (int)d[k+3]);
            a = std::min(a, (int)d[k+4]);
            a0 = std::max(a0, std::min(a, (int)d[k]));
            a0 = std::max(a0, std::min(a, (int)d[k+5]));
        }

        int b0 = -a0;
        for( k = 0; k < 8; k += 2 )
        {
            int b = std::max((int)d[k+1], (int)d[k+2]);
            b = std::max(b, (int)d[k+3]);
            if( b >= b0 )
                continue;
            b = std::max(b, (int)d[k+4]);

            b0 = std::min(b0, std::max(b, (int)d[k]));
            b0 = std::min(b0, std::max(b, (int)d[k+5]));
        }

        threshold = -b0-1;
    #endif

    #if VERIFY_CORNERS
        testCorner(ptr, pixel, K, N, threshold);
    #endif
        return threshold;
    }

    void makeOffsets(int pixel[25], int rowStride, int patternSize)
    {
        static const int offsets16[][2] =
        {
            {0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3},
            {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
        };

        static const int offsets12[][2] =
        {
            {0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
            {0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
        };

        static const int offsets8[][2] =
        {
            {0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
            {0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
        };

        const int (*offsets)[2] = patternSize == 16 ? offsets16 :
                                    patternSize == 12 ? offsets12 :
                                    patternSize == 8  ? offsets8  : 0;

        CV_Assert(pixel && offsets);

        int k = 0;
        for( ; k < patternSize; k++ )
            pixel[k] = offsets[k][0] + offsets[k][1] * rowStride;
        for( ; k < 25; k++ )
            pixel[k] = pixel[k - patternSize];
    }

    template<int patternSize>
    void FASTForPointSet_t( InputArray image, std::vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression ) {

        Mat img = image.getMat();
        const int K = patternSize/2, N = patternSize + K + 1;

        int i, k, pixel[25];
        makeOffsets(pixel, (int)img.step, patternSize);

        threshold = std::min(std::max(threshold, 0), 255);

        uchar threshold_tab[512];
        for( i = -255; i <= 255; i++ )
            threshold_tab[i+255] = (uchar)(i < -threshold ? 1 : i > threshold ? 2 : 0);

        AutoBuffer<uchar> _buf((img.cols+16)*3*(sizeof(int) + sizeof(uchar)) + 128);
        uchar* buf[3];
        buf[0] = _buf.data(); buf[1] = buf[0] + img.cols; buf[2] = buf[1] + img.cols;
        int* cpbuf[3];
        cpbuf[0] = (int*)alignPtr(buf[2] + img.cols, sizeof(int)) + 1;
        cpbuf[1] = cpbuf[0] + img.cols + 1;
        cpbuf[2] = cpbuf[1] + img.cols + 1;
        memset(buf[0], 0, img.cols*3);

        // Calculate threshold for the keypoints
        for (size_t keyPointIdx=0; keyPointIdx < keypoints.size(); keyPointIdx++) {
            // Set response to -1:
            // All keypoints with response <= 0 will be removed afterwards
            keypoints[keyPointIdx].response = -1;

            // Poiter to keyPoint in image
            Point keyPoint = keypoints[keyPointIdx].pt;
            const uchar* ptr = img.ptr<uchar>(keyPoint.y, keyPoint.x);

            // value of the pixel at certain position
            int v = ptr[0];

            // Initialize Lookup table
            // If k=v --> tab[k] is at the center of the thrshold table
            // The threshold table is made as follows:
            // -255         -threshold         0        +threshold        255
            // 111111111111111111|0000000000000|0000000000000|222222222222222
            const uchar* tab = &threshold_tab[0] - v + 255;

            // Calculate the fast value
            int d = tab[ptr[pixel[0]]] | tab[ptr[pixel[8]]];

            if( d == 0 )
                continue;

            d &= tab[ptr[pixel[2]]] | tab[ptr[pixel[10]]];
            d &= tab[ptr[pixel[4]]] | tab[ptr[pixel[12]]];
            d &= tab[ptr[pixel[6]]] | tab[ptr[pixel[14]]];

            if( d == 0 )
                continue;

            d &= tab[ptr[pixel[1]]] | tab[ptr[pixel[9]]];
            d &= tab[ptr[pixel[3]]] | tab[ptr[pixel[11]]];
            d &= tab[ptr[pixel[5]]] | tab[ptr[pixel[13]]];
            d &= tab[ptr[pixel[7]]] | tab[ptr[pixel[15]]];

            // For at least half pixels darker than v count the number
            if( d & 1 )
            {
                int vt = v - threshold, count = 0;

                for(k = 0; k < N; k++ )
                {
                    int x = ptr[pixel[k]];
                    if(x < vt)
                    {
                        if( ++count > K )
                        {
                            // Calculate score
                            keypoints[keyPointIdx].response = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                            // Non Maxima Supression I
                            if (nonmaxSuppression && keyPointIdx>0 && keypoints[keyPointIdx-1].response < keypoints[keyPointIdx].response) {
                                keypoints[keyPointIdx-1].response = -1;
                            }
                            break;
                        }
                    }
                    else
                        count = 0;
                }
            }

            // For at least half pixels brighter than v count the number
            if(d & 2 )
            {
                int vt = v + threshold, count = 0;

                for(k = 0; k < N; k++ )
                {
                    int x = ptr[pixel[k]];
                    if(x > vt)
                    {
                        if( ++count > K )
                        {
                            // Calculate score
                            keypoints[keyPointIdx].response = (uchar)cornerScore<patternSize>(ptr, pixel, threshold);
                            // Non Maxima Suppression I
                            if (nonmaxSuppression && keyPointIdx>0 &&keypoints[keyPointIdx-1].response < keypoints[keyPointIdx].response) {
                                keypoints[keyPointIdx-1].response = -1;
                            }
                            break;
                        }
                    }
                    else
                        count = 0;
                }
            }

        }

        // Remove unused Keypoints
        size_t maxKeypointSize = keypoints.size();
        for (size_t keyPointIdx=maxKeypointSize; keyPointIdx > 0;) {
            keyPointIdx--;
            if (keypoints[keyPointIdx].response <= 0) {
                keypoints.erase(keypoints.begin() + keyPointIdx);
            } else if (nonmaxSuppression && keyPointIdx>0 && keypoints[keyPointIdx-1].response > keypoints[keyPointIdx].response) {
                // Non Maxima Suppression II
                keypoints.erase(keypoints.begin() + keyPointIdx);
            }
        }
    }

}

namespace cv {
    namespace xfeatures2d {

        void FASTForPointSet(InputArray _img, std::vector<KeyPoint>& keypoints, int threshold, bool nonmax_suppression, FastFeatureDetector::DetectorType type)
        {
            if (keypoints.empty()) {
                FAST(_img, keypoints, threshold, nonmax_suppression, type);
                return;
            }

            switch(type) {
            case FastFeatureDetector::TYPE_5_8:
                FASTForPointSet_t<8>(_img, keypoints, threshold, nonmax_suppression);
                break;
            case FastFeatureDetector::TYPE_7_12:
                FASTForPointSet_t<12>(_img, keypoints, threshold, nonmax_suppression);
                break;
            case FastFeatureDetector::TYPE_9_16:
                FASTForPointSet_t<16>(_img, keypoints, threshold, nonmax_suppression);
                break;
            }
        }

    }
}
