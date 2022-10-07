// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_NVENCODERCUDA_HPP
#define OPENCV_NVENCODERCUDA_HPP
#include <vector>
#include <stdint.h>
#include <mutex>
#include <cuda.h>
#include "NvEncoder.h"

namespace cv { namespace cudacodec {

/**
* @brief Encoder for CUDA device memory.
*/
class NvEncoderCuda : public NvEncoder
{
public:
    NvEncoderCuda(CUcontext cuContext, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
        uint32_t nExtraOutputDelay = 3);
    virtual ~NvEncoderCuda();

    /**
    * @brief This is a static function to copy input data from host memory to device memory.
    * This function assumes YUV plane is a single contiguous memory segment.
    */
    static void CopyToDeviceFrame(CUcontext device,
        void* pSrcFrame,
        uint32_t nSrcPitch,
        CUdeviceptr pDstFrame,
        uint32_t dstPitch,
        int width,
        int height,
        CUmemorytype srcMemoryType,
        NV_ENC_BUFFER_FORMAT pixelFormat,
        const uint32_t dstChromaOffsets[],
        uint32_t numChromaPlanes,
        bool bUnAlignedDeviceCopy = false,
        CUstream stream = NULL);

    /**
    * @brief This function sets input and output CUDA streams
    */
    void SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream);

protected:
    /**
    * @brief This function is used to release the input buffers allocated for encoding.
    * This function is an override of virtual function NvEncoder::ReleaseInputBuffers().
    */
    virtual void ReleaseInputBuffers() override;

private:
    /**
    * @brief This function is used to allocate input buffers for encoding.
    * This function is an override of virtual function NvEncoder::AllocateInputBuffers().
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) override;

private:
    /**
    * @brief This is a private function to release CUDA device memory used for encoding.
    */
    void ReleaseCudaResources();

protected:
    CUcontext m_cuContext;

private:
    size_t m_cudaPitch = 0;
};
}}
#endif