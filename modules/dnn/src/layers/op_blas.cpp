#include "op_blas.hpp"

#if HAVE_CBLAS
#include "opencv_cblas.hpp"
#endif

#include <iostream>

namespace cv
{
namespace dnn
{

void gemm(InputArray A, InputArray B, double alpha, InputOutputArray C, double beta, int flags)
{
    if (C.isMat())
        gemmCPU(A.getMat(), B.getMat(), alpha, C.getMatRef(), beta, flags);
    else
    {
        cv::gemm(A, B, alpha, (beta == 0) ? noArray() : C, beta, C, flags);
    }
}

inline void SwapRowCols(const Mat &A, int &rows, int &cols, bool isTrans)
{
    CV_DbgAssert(A.dims == 2);
    rows = (isTrans) ? A.cols : A.rows;
    cols = (isTrans) ? A.rows : A.cols;
}


class GEMMInvoker : public ParallelLoopBody
{
public:
    GEMMInvoker(const Mat* _a, const Mat* _b, double _alpha, Mat* _c, double _beta)
    {
        a = _a;
        b = _b;
        c = _c;
        alpha = _alpha;
        beta = _beta;
    }

    void operator()(const Range& range) const
    {
        int mmax = a->rows;
        int nmax = range.end - range.start;
        int kmax = a->cols;
        int m, n, k;

        for( m = 0; m < mmax; m++ )
        {
            float* dst = c->ptr<float>(m) + range.start;
            if( beta != 1 )
            {
                if( beta == 0 )
                    for( n = 0; n < nmax; n++ )
                        dst[n] = 0.f;
                else
                    for( n = 0; n < nmax; n++ )
                        dst[n] *= (float)beta;
            }
            const float* aptr = a->ptr<float>(m);

            for( k = 0; k < kmax; k++ )
            {
                float alpha1 = (float)(alpha*aptr[k]);
                const float* bptr = b->ptr<float>(k) + range.start;

                for( n = 0; n < nmax; n++ )
                    dst[n] += alpha1*bptr[n];
            }
        }
    }

    const Mat *a, *b;
    Mat* c;
    double alpha, beta;
};


void gemmCPU(const Mat &A, const Mat &B, double alpha, Mat &C, double beta, int flags)
{
    if( C.type() == CV_32F && flags == 0 )
    {
        GEMMInvoker invoker(&A, &B, alpha, &C, beta);
        double granularity = 10000000./((double)A.rows*A.cols);
        parallel_for_(Range(0, B.cols), invoker, granularity);
    }
    else
    {
    #if HAVE_CBLAS
    bool transA = static_cast<bool>(flags & GEMM_1_T);
    bool transB = static_cast<bool>(flags & GEMM_2_T);
    bool transC = static_cast<bool>(flags & GEMM_3_T);

    int Arows, Acols, Brows, Bcols, Crows, Ccols;
    SwapRowCols(A, Arows, Acols, transA);
    SwapRowCols(B, Brows, Bcols, transB);
    SwapRowCols(C, Crows, Ccols, transC);

    CV_Assert(!(flags & GEMM_3_T));
    CV_Assert(Acols == Brows && Arows == Crows && Bcols == Ccols);
    CV_Assert(A.isContinuous() && B.isContinuous() && C.isContinuous());
    CV_Assert(A.type() == B.type() && B.type() == C.type());
    CV_Assert(A.data != C.data && B.data != C.data);

    if (C.type() == CV_32F)
    {
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                    Arows, Bcols, Acols,
                    (float)alpha, A.ptr<float>(), A.cols,
                    B.ptr<float>(), B.cols,
                    (float)beta, C.ptr<float>(), C.cols);
    }
    else if (C.type() == CV_64F)
    {
        //TODO: Should be tested
        cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                    Arows, Bcols, Acols,
                    alpha, A.ptr<double>(), A.cols,
                    B.ptr<double>(), B.cols,
                    beta, C.ptr<double>(), C.cols);
    }
    else
    {
        CV_Error(Error::BadDepth, "Only floating point types are supported");
    }
    #else
    cv::gemm(A, B, alpha, C, beta, C, flags);
    #endif
    }
}

int getBlasThreads()
{
    #ifdef OPENBLAS_VERSION
    return openblas_get_num_threads();
    #else
    return 1;
    #endif
}

void setBlasThreads(int numThreads)
{
    #ifdef OPENBLAS_VERSION
    openblas_set_num_threads(numThreads);
    goto_set_num_threads(numThreads);
    #else
    (void)numThreads;   //suppress compilers' warning
    #endif
}

}
}
