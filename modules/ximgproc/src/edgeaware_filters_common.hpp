#ifndef __EDGEAWAREFILTERS_COMMON_HPP__
#define __EDGEAWAREFILTERS_COMMON_HPP__
#ifdef __cplusplus

namespace cv
{

Ptr<DTFilter> createDTFilterRF(InputArray adistHor, InputArray adistVert, double sigmaSpatial, double sigmaColor, int numIters);

int getTotalNumberOfChannels(InputArrayOfArrays src);

void checkSameSizeAndDepth(InputArrayOfArrays src, Size &sz, int &depth);

namespace eaf
{  
    void add_(register float *dst, register float *src1, int w);

    void mul(register float *dst, register float *src1, register float *src2, int w);

    void mul(register float *dst, register float *src1, float src2, int w);

    //dst = alpha*src + beta
    void mad(register float *dst, register float *src1, float alpha, float beta, int w);

    void add_mul(register float *dst, register float *src1, register float *src2, int w);

    void sub_mul(register float *dst, register float *src1, register float *src2, int w);

    void sub_mad(register float *dst, register float *src1, register float *src2, float c0, int w);

    void det_2x2(register float *dst, register float *a00, register float *a01, register float *a10, register float *a11, int w);

    void div_det_2x2(register float *a00, register float *a01, register float *a11, int w);

    void div_1x(register float *a1, register float *b1, int w);

    void inv_self(register float *src, int w);

    
    void sqr_(register float *dst, register float *src1, int w);

    void sqrt_(register float *dst, register float *src, int w);

    void sqr_dif(register float *dst, register float *src1, register float *src2, int w);

    void add_sqr_dif(register float *dst, register float *src1, register float *src2, int w);

    void add_sqr(register float *dst, register float *src1, int w);

    void min_(register float *dst, register float *src1, register float *src2, int w);

    void rf_vert_row_pass(register float *curRow, register float *prevRow, float alphaVal, int w);
}
}

#endif
#endif