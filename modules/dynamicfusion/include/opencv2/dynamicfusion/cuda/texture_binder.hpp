#pragma once

#include "device_array.hpp"
#include "safe_call.hpp"

namespace kfusion
{
    namespace cuda
    {
        class TextureBinder
        {
        public:
            template<class T, enum cudaTextureReadMode readMode>
            TextureBinder(const DeviceArray2D<T>& arr, const struct texture<T, 2, readMode>& tex) : texref(&tex)
            {
                cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
                cudaSafeCall( cudaBindTexture2D(0, tex, arr.ptr(), desc, arr.cols(), arr.rows(), arr.step()) );
            }

            template<class T, enum cudaTextureReadMode readMode>
            TextureBinder(const DeviceArray<T>& arr, const struct texture<T, 1, readMode> &tex) : texref(&tex)
            {
                cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
                cudaSafeCall( cudaBindTexture(0, tex, arr.ptr(), desc, arr.sizeBytes()) );
            }

            template<class T, enum cudaTextureReadMode readMode>
            TextureBinder(const PtrStepSz<T>& arr, const struct texture<T, 2, readMode>& tex) : texref(&tex)
            {
                cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
                cudaSafeCall( cudaBindTexture2D(0, tex, arr.data, desc, arr.cols, arr.rows, arr.step) );
            }

            template<class A, class T, enum cudaTextureReadMode readMode>
            TextureBinder(const A& arr, const struct texture<T, 2, readMode>& tex, const cudaChannelFormatDesc& desc) : texref(&tex)
            {
                cudaSafeCall( cudaBindTexture2D(0, tex, arr.data, desc, arr.cols, arr.rows, arr.step) );
            }

            template<class T, enum cudaTextureReadMode readMode>
            TextureBinder(const PtrSz<T>& arr, const struct texture<T, 1, readMode> &tex) : texref(&tex)
            {
                cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
                cudaSafeCall( cudaBindTexture(0, tex, arr.data, desc, arr.size * arr.elemSize()) );
            }

            ~TextureBinder()
            {
                cudaSafeCall( cudaUnbindTexture(texref) );
            }
        private:
            const struct textureReference *texref;
        };
    }

    namespace device
    {
        using kfusion::cuda::TextureBinder;
    }
}
