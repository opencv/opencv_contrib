// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CUDEV_PTR2D_TEXTURE_OBJECT_HPP
#define OPENCV_CUDEV_PTR2D_TEXTURE_OBJECT_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/core/cuda_types.hpp>
#include <opencv2/cudev/common.hpp>
#include <opencv2/cudev/ptr2d/traits.hpp>

/** \file texture.hpp
*/

namespace cv {  namespace cudev {

//! @addtogroup cudev
//! @{

    /** @brief Simple lightweight structures that encapsulate information about an image texture on the device.
    * They are intended to be passed to nvcc-compiled code.
    */
    template<class T, class R = T>
    struct TexturePtr {
        typedef R     elem_type, value_type;
        typedef float index_type;
        __host__ TexturePtr() {};
        __host__ TexturePtr(const cudaTextureObject_t tex_) : tex(tex_) {};
        __device__ __forceinline__ R operator ()(index_type y, index_type x) const {
            return tex2D<R>(tex, x, y);
        }
        __device__ __forceinline__ R operator ()(index_type x) const {
            return tex1Dfetch<R>(tex, x);
        }
    private:
        cudaTextureObject_t tex;
    };

    // textures are a maximum of 32 bits wide, 64 bits is read as two 32 bit wide values
    template <class R>
    struct TexturePtr<uint64, R> {
        typedef float index_type;
        __host__ TexturePtr() {};
        __host__ TexturePtr(const cudaTextureObject_t tex_) : tex(tex_) {};
        __device__ __forceinline__ R operator ()(index_type y, index_type x) const {
            const uint2 retVal = tex2D<uint2>(tex, x, y);
            return *(reinterpret_cast<const R*>(&retVal));
        }
        __device__ __forceinline__ R operator ()(index_type x) const {
            const uint2 retVal = tex1Dfetch<uint2>(tex, x);
            return *(reinterpret_cast<const R*>(&retVal));
        }
    private:
        cudaTextureObject_t tex;
    };

    template<class T, class R = T>
    struct TextureOffPtr {
        typedef R     elem_type;
        typedef float index_type;
        __host__ TextureOffPtr(const cudaTextureObject_t tex_, const int yoff_, const int xoff_) : tex(tex_), yoff(yoff_), xoff(xoff_) {};
        __device__ __forceinline__ R operator ()(index_type y, index_type x) const {
            return tex2D<R>(tex, x + xoff, y + yoff);
        }
    private:
        cudaTextureObject_t tex;
        int xoff = 0;
        int yoff = 0;
    };

    /** @brief non-copyable smart CUDA texture object
    *
    * UniqueTexture is a smart non-sharable wrapper for a cudaTextureObject_t handle which ensures that the handle is destroyed after use.
    */
    template<class T, class R = T>
    class UniqueTexture {
    public:
        __host__ UniqueTexture() noexcept { }
        __host__ UniqueTexture(UniqueTexture&) = delete;
        __host__ UniqueTexture(UniqueTexture&& other) noexcept {
            tex = other.tex;
            other.tex = 0;
        }

        __host__ UniqueTexture(const int rows, const int cols, T* data, const size_t step, const bool normalizedCoords = false,
            const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
            const cudaTextureReadMode readMode = cudaReadModeElementType)
        {
            create(rows, cols, data, step, normalizedCoords, filterMode, addressMode, readMode);
        }

        __host__ UniqueTexture(const size_t sizeInBytes, T* data, const bool normalizedCoords = false, const cudaTextureFilterMode filterMode = cudaFilterModePoint,
            const cudaTextureAddressMode addressMode = cudaAddressModeClamp, const cudaTextureReadMode readMode = cudaReadModeElementType)
        {
            create(1, static_cast<int>(sizeInBytes/sizeof(T)), data, sizeInBytes, normalizedCoords, filterMode, addressMode, readMode);
        }

        __host__ ~UniqueTexture() {
            if (tex != cudaTextureObject_t()) {
                try {
                    CV_CUDEV_SAFE_CALL(cudaDestroyTextureObject(tex));
                }
                catch (const cv::Exception& ex) {
                    std::ostringstream os;
                    os << "Exception caught during CUDA texture object destruction.\n";
                    os << ex.what();
                    os << "Exception will be ignored.\n";
                    CV_LOG_WARNING(0, os.str().c_str());
                }
            }

        }

        __host__ UniqueTexture& operator=(const UniqueTexture&) = delete;
        __host__ UniqueTexture& operator=(UniqueTexture&& other) noexcept {
            CV_Assert(other);
            if (&other != this) {
                UniqueTexture(std::move(*this)); /* destroy current texture object */
                tex = other.tex;
                other.tex = cudaTextureObject_t();
            }
            return *this;
        }

        __host__ cudaTextureObject_t get() const noexcept {
            CV_Assert(tex);
            return tex;
        }

        __host__ explicit operator bool() const noexcept { return tex != cudaTextureObject_t(); }

    private:

        template <class T1>
        __host__ void create(const int rows, const int cols, T1* data, const size_t step, const bool normalizedCoords, const cudaTextureFilterMode filterMode,
            const cudaTextureAddressMode addressMode, const cudaTextureReadMode readMode)
        {
            cudaResourceDesc texRes;
            std::memset(&texRes, 0, sizeof(texRes));
            if (rows == 1) {
                CV_Assert(rows == 1 && cols*sizeof(T) == step);
                texRes.resType = cudaResourceTypeLinear;
                texRes.res.linear.devPtr = data;
                texRes.res.linear.sizeInBytes = step;
                texRes.res.linear.desc = cudaCreateChannelDesc<T1>();
            }
            else {
                texRes.resType = cudaResourceTypePitch2D;
                texRes.res.pitch2D.devPtr = data;
                texRes.res.pitch2D.height = rows;
                texRes.res.pitch2D.width = cols;
                texRes.res.pitch2D.pitchInBytes = step;
                texRes.res.pitch2D.desc = cudaCreateChannelDesc<T1>();
            }

            cudaTextureDesc texDescr;
            std::memset(&texDescr, 0, sizeof(texDescr));
            texDescr.normalizedCoords = normalizedCoords;
            texDescr.filterMode = filterMode;
            texDescr.addressMode[0] = addressMode;
            texDescr.addressMode[1] = addressMode;
            texDescr.addressMode[2] = addressMode;
            texDescr.readMode = readMode;

            CV_CUDEV_SAFE_CALL(cudaCreateTextureObject(&tex, &texRes, &texDescr, 0));
        }

        __host__ void create(const int rows, const int cols, uint64* data, const size_t step, const bool normalizedCoords, const cudaTextureFilterMode filterMode,
            const cudaTextureAddressMode addressMode, const cudaTextureReadMode readMode)
        {
            create<uint2>(rows, cols, (uint2*)data, step, normalizedCoords, filterMode, addressMode, readMode);
        }

    private:
        cudaTextureObject_t tex;
    };

    /** @brief sharable smart CUDA texture object
    *
    * Texture is a smart sharable wrapper for a cudaTextureObject_t handle which ensures that the handle is destroyed after use.
    */
    template<class T, class R = T>
    class Texture {
    public:
        Texture() = default;
        Texture(const Texture&) = default;
        Texture(Texture&&) = default;

        __host__ Texture(const int rows_, const int cols_, T* data, const size_t step, const bool normalizedCoords = false, const cudaTextureFilterMode filterMode = cudaFilterModePoint,
            const cudaTextureAddressMode addressMode = cudaAddressModeClamp, const cudaTextureReadMode readMode = cudaReadModeElementType) :
            rows(rows_), cols(cols_), texture(std::make_shared<UniqueTexture<T,R>>(rows, cols, data, step, normalizedCoords, filterMode, addressMode, readMode))
        {
        }

        __host__ Texture(const size_t sizeInBytes, T* data, const bool normalizedCoords = false, const cudaTextureFilterMode filterMode = cudaFilterModePoint,
            const cudaTextureAddressMode addressMode = cudaAddressModeClamp, const cudaTextureReadMode readMode = cudaReadModeElementType) :
            rows(1), cols(static_cast<int>(sizeInBytes/sizeof(T))), texture(std::make_shared<UniqueTexture<T, R>>(sizeInBytes, data, normalizedCoords, filterMode, addressMode, readMode))
        {
        }

        __host__ Texture(PtrStepSz<T> src, const bool normalizedCoords = false, const cudaTextureFilterMode filterMode = cudaFilterModePoint,
            const cudaTextureAddressMode addressMode = cudaAddressModeClamp, const cudaTextureReadMode readMode = cudaReadModeElementType) :
            Texture(src.rows, src.cols, src.data, src.step, normalizedCoords, filterMode, addressMode, readMode)
        {
        }

        Texture& operator=(const Texture&) = default;
        Texture& operator=(Texture&&) = default;

        __host__ explicit operator bool() const noexcept {
            if (!texture)
                return false;
            return texture->operator bool();
        }

        __host__ operator TexturePtr<T, R>() const {
            if (texture)
                return TexturePtr<T, R>(texture->get());
            else
                return TexturePtr<T, R>(cudaTextureObject_t());
        }

        int rows = 0;
        int cols = 0;

    protected:
        std::shared_ptr<UniqueTexture<T, R>> texture = 0;
    };

    template <typename T, typename R> struct PtrTraits<Texture<T, R>> : PtrTraitsBase<Texture<T, R>, TexturePtr<T, R>>
    {
    };


    /** @brief sharable smart CUDA texture object with offset
    * TextureOff is a smart sharable wrapper for a cudaTextureObject_t handle which ensures that the handle is destroyed after use.
    */
    template<class T, class R = T>
    class TextureOff {
    public:
        TextureOff(const TextureOff&) = default;
        TextureOff(TextureOff&&) = default;

        __host__ TextureOff(const int rows, const int cols, T* data, const size_t step, const int yoff_ = 0, const int xoff_ = 0, const bool normalizedCoords = false,
            const cudaTextureFilterMode filterMode = cudaFilterModePoint, const cudaTextureAddressMode addressMode = cudaAddressModeClamp,
            const cudaTextureReadMode readMode = cudaReadModeElementType) :
            texture(std::make_shared<UniqueTexture<T, R>>(rows, cols, data, step, normalizedCoords, filterMode, addressMode, readMode)), xoff(xoff_), yoff(yoff_)
        {
        }

        __host__ TextureOff(PtrStepSz<T> src, const int yoff = 0, const int xoff = 0, const bool normalizedCoords = false, const cudaTextureFilterMode filterMode = cudaFilterModePoint,
            const cudaTextureAddressMode addressMode = cudaAddressModeClamp, const cudaTextureReadMode readMode = cudaReadModeElementType) :
            TextureOff(src.rows, src.cols, src.data, src.step, yoff, xoff, normalizedCoords, filterMode, addressMode, readMode)
        {
        }

        TextureOff& operator=(const TextureOff&) = default;
        TextureOff& operator=(TextureOff&&) = default;

        __host__ operator TextureOffPtr<T, R>() const {
            return TextureOffPtr<T, R>(texture->get(), yoff, xoff);
        }

    private:
        int xoff = 0;
        int yoff = 0;
        std::shared_ptr<UniqueTexture<T, R>> texture = 0;
    };
}}

#endif
