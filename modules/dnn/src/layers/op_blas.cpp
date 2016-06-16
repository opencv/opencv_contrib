#include "op_blas.hpp"

#if HAVE_CBLAS
#include "cblas.h"
#endif

namespace cv
{
namespace dnn
{

void gemm(InputArray A, InputArray B, double alpha, InputOutputArray C, double beta, int flags /*= 0*/)
{
    cv::gemm(A, B, alpha, C, beta, C, flags);
}

inline void SwapRowCols(const Mat &A, int &rows, int &cols, bool transA)
{
    rows = (transA) ? A.cols : A.rows;
    cols = (transA) ? A.rows : A.cols;
}

void gemmCPU(const Mat &A, const Mat &B, double alpha, Mat &C, double beta, int flags /*= 0*/)
{
    #if HAVE_CBLAS
    int transA = flags & GEMM_1_T;
    int transB = flags & GEMM_2_T;
    int transC = flags & GEMM_3_T;

    int Arows, Acols, Brows, Bcols, Crows, Ccols;
    SwapRowCols(A, Arows, Acols, transA);
    SwapRowCols(B, Brows, Bcols, transB);
    SwapRowCols(C, Crows, Ccols, transC);

    CV_DbgAssert(!(flags & GEMM_3_T));
    CV_Assert(Acols == Brows && Arows == Crows && Bcols == Ccols);
    CV_DbgAssert(A.isContinuous() && B.isContinuous() && C.isContinuous());
    CV_DbgAssert(A.type() == CV_32F || A.type() == CV_64F);
    CV_DbgAssert(A.type() == B.type() && B.type() == C.type());

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
    #else
    cv::gemm(A, B, alpha, C, beta, C, flags);
    #endif
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
    numThreads = 0;
    #endif
}

}
}
