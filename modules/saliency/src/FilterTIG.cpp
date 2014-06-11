#include "kyheader.h"
#include "FilterTIG.h"
#include "CmShow.h"


void FilterTIG::update(CMat &w1f){
    CV_Assert(w1f.cols * w1f.rows == D && w1f.type() == CV_32F && w1f.isContinuous());
    float b[D], residuals[D];
    memcpy(residuals, w1f.data, sizeof(float)*D);
    for (int i = 0; i < NUM_COMP; i++){
        float avg = 0;
        for (int j = 0; j < D; j++){
            b[j] = residuals[j] >= 0.0f ? 1.0f : -1.0f;
            avg += residuals[j] * b[j];
        }
        avg /= D;
        _coeffs1[i] = avg, _coeffs2[i] = avg*2, _coeffs4[i] = avg*4, _coeffs8[i] = avg*8;
        for (int j = 0; j < D; j++)
            residuals[j] -= avg*b[j];
        UINT64 tig = 0;
        for (int j = 0; j < D; j++)
            tig = (tig << 1) | (b[j] > 0 ? 1 : 0);
        _bTIGs[i] = tig;
    }
}

void FilterTIG::reconstruct(Mat &w1f){
    w1f = Mat::zeros(8, 8, CV_32F);
    float *weight = (float*)w1f.data;
    for (int i = 0; i < NUM_COMP; i++){
        UINT64 tig = _bTIGs[i];
        for (int j = 0; j < D; j++)
            weight[j] += _coeffs1[i] * (((tig >> (63-j)) & 1) ? 1 : -1);
    }
}

// For a W by H gradient magnitude map, find a W-7 by H-7 CV_32F matching score map
// Please refer to my paper for definition of the variables used in this function
Mat FilterTIG::matchTemplate(const Mat &mag1u){
    const int H = mag1u.rows, W = mag1u.cols;
    const Size sz(W+1, H+1); // Expand original size to avoid dealing with boundary conditions
    Mat_<INT64> Tig1 = Mat_<INT64>::zeros(sz), Tig2 = Mat_<INT64>::zeros(sz);
    Mat_<INT64> Tig4 = Mat_<INT64>::zeros(sz), Tig8 = Mat_<INT64>::zeros(sz);
    Mat_<byte> Row1 = Mat_<byte>::zeros(sz), Row2 = Mat_<byte>::zeros(sz);
    Mat_<byte> Row4 = Mat_<byte>::zeros(sz), Row8 = Mat_<byte>::zeros(sz);
    Mat_<float> scores(sz);
    for(int y = 1; y <= H; y++){
        const byte* G = mag1u.ptr<byte>(y-1);
        INT64* T1 = Tig1.ptr<INT64>(y); // Binary TIG of current row
        INT64* T2 = Tig2.ptr<INT64>(y);
        INT64* T4 = Tig4.ptr<INT64>(y);
        INT64* T8 = Tig8.ptr<INT64>(y);
        INT64* Tu1 = Tig1.ptr<INT64>(y-1); // Binary TIG of upper row
        INT64* Tu2 = Tig2.ptr<INT64>(y-1);
        INT64* Tu4 = Tig4.ptr<INT64>(y-1);
        INT64* Tu8 = Tig8.ptr<INT64>(y-1);
        byte* R1 = Row1.ptr<byte>(y);
        byte* R2 = Row2.ptr<byte>(y);
        byte* R4 = Row4.ptr<byte>(y);
        byte* R8 = Row8.ptr<byte>(y);
        float *s = scores.ptr<float>(y);
        for (int x = 1; x <= W; x++) {
            byte g = G[x-1];
            R1[x] = (R1[x-1] << 1) | ((g >> 4) & 1);
            R2[x] = (R2[x-1] << 1) | ((g >> 5) & 1);
            R4[x] = (R4[x-1] << 1) | ((g >> 6) & 1);
            R8[x] = (R8[x-1] << 1) | ((g >> 7) & 1);
            T1[x] = (Tu1[x] << 8) | R1[x];
            T2[x] = (Tu2[x] << 8) | R2[x];
            T4[x] = (Tu4[x] << 8) | R4[x];
            T8[x] = (Tu8[x] << 8) | R8[x];
            s[x] = dot(T1[x], T2[x], T4[x], T8[x]);
        }
    }
    Mat matchCost1f;
    scores(Rect(8, 8, W-7, H-7)).copyTo(matchCost1f);
    return matchCost1f;
}
