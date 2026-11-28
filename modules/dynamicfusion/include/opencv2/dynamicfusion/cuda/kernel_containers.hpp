#pragma once

#if defined(__CUDACC__)
    #define __kf_hdevice__ __host__ __device__ __forceinline__
    #define __kf_device__ __device__ __forceinline__
#else
    #define __kf_hdevice__
    #define __kf_device__
#endif  

#include <cstddef>

namespace cv
{
    namespace kfusion
    {
        namespace cuda
        {
            template<typename T> struct DevPtr
            {
                typedef T elem_type;
                const static size_t elem_size = sizeof(elem_type);

                T* data;

                __kf_hdevice__ DevPtr() : data(0) {}
                __kf_hdevice__ DevPtr(T* data_arg) : data(data_arg) {}

                __kf_hdevice__ size_t elemSize() const { return elem_size; }
                __kf_hdevice__ operator       T*()       { return data; }
                __kf_hdevice__ operator const T*() const { return data; }
            };

            template<typename T> struct PtrSz : public DevPtr<T>
            {
                __kf_hdevice__ PtrSz() : size(0) {}
                __kf_hdevice__ PtrSz(T* data_arg, size_t size_arg) : DevPtr<T>(data_arg), size(size_arg) {}

                size_t size;
            };

            template<typename T>  struct PtrStep : public DevPtr<T>
            {
                __kf_hdevice__ PtrStep() : step(0) {}
                __kf_hdevice__ PtrStep(T* data_arg, size_t step_arg) : DevPtr<T>(data_arg), step(step_arg) {}

                /** \brief stride between two consecutive rows in bytes. Step is stored always and everywhere in bytes!!! */
                size_t step;

                __kf_hdevice__       T* ptr(int y = 0)       { return (      T*)( (      char*)DevPtr<T>::data + y * step); }
                __kf_hdevice__ const T* ptr(int y = 0) const { return (const T*)( (const char*)DevPtr<T>::data + y * step); }

                __kf_hdevice__       T& operator()(int y, int x)       { return ptr(y)[x]; }
                __kf_hdevice__ const T& operator()(int y, int x) const { return ptr(y)[x]; }
            };

            template <typename T> struct PtrStepSz : public PtrStep<T>
            {
                __kf_hdevice__ PtrStepSz() : cols(0), rows(0) {}
                __kf_hdevice__ PtrStepSz(int rows_arg, int cols_arg, T* data_arg, size_t step_arg)
                        : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg) {}

                int cols;
                int rows;
            };
        }

        namespace device
        {
            using kfusion::cuda::PtrSz;
            using kfusion::cuda::PtrStep;
            using kfusion::cuda::PtrStepSz;
        }
    }
}

namespace kf = cv::kfusion;

