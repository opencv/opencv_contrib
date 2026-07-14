#pragma once

#include "cuda_runtime_api.h"

namespace cv{
    namespace kfusion
    {
        namespace cuda
        {
            void error(const char *error_string, const char *file, const int line, const char *func);
        }
    }

#if defined(__GNUC__)
#define cudaSafeCall(expr)  kfusion::cuda::___cudaSafeCall(expr, __FILE__, __LINE__, __func__)
#else /* defined(__CUDACC__) || defined(__MSVC__) */
#define cudaSafeCall(expr)  kfusion::cuda::___cudaSafeCall(expr, __FILE__, __LINE__)
#endif

    namespace kfusion
    {
        namespace cuda
        {
            static inline void ___cudaSafeCall(cudaError_t err, const char *file, const int line, const char *func = "")
            {
                if (cudaSuccess != err)
                    error(cudaGetErrorString(err), file, line, func);
            }

            static inline int divUp(int total, int grain) { return (total + grain - 1) / grain; }
        }

        namespace device
        {
            using kfusion::cuda::divUp;
        }
    }
}
