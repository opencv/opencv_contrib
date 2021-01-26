//
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
//M*/
#if !defined CUDA_DISABLER

#include <cuda_runtime.h>
#include <stdio.h>

typedef unsigned char   uint8_t;
typedef unsigned short  uint16_t;
typedef unsigned int    uint32_t;
typedef   signed short  int16_t;
typedef   signed int    int32_t;

#define BLOCKDIM_X 32
#define BLOCKDIM_Y 16

// data required to do 2x upsampling.  Same can be used for 4x upsampling also
#define SMEM_COLS  ((BLOCKDIM_X)/2)
#define SMEM_ROWS  ((BLOCKDIM_Y)/2)

#if defined(__CUDACC_VER_MAJOR__) && (10 <= __CUDACC_VER_MAJOR__)
namespace cv { namespace cuda { namespace device { namespace optflow_nvidia
{
static const char *_cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }

template <typename T>
void check(T result, char const *const func, const char *const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<uint32_t>(result), _cudaGetErrorEnum(result), func);
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <class T>
static __device__ void ReadDevPtrData(void* devptr, uint32_t x0, uint32_t y0, uint32_t src_w, uint32_t src_h, uint32_t src_pitch,
                                      T src[][SMEM_COLS], uint32_t i, uint32_t j)
{
    uint32_t shift = (sizeof(T) == sizeof(int32_t)) ? 2 : 1;
    src[j][i] = *(T*)((uint8_t*)devptr + y0 * src_pitch + (x0 << shift));
}


extern "C"
__global__ void NearestNeighborFlowKernel(cudaSurfaceObject_t srcSurf, void* srcDevPtr, uint32_t src_w, uint32_t src_pitch, uint32_t src_h,
                                          cudaSurfaceObject_t dstSurf, void* dstDevPtr, uint32_t dst_w, uint32_t dst_pitch, uint32_t dst_h,
                                          uint32_t nScaleFactor)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int x0 = x / nScaleFactor;
    int y0 = y / nScaleFactor;

    __shared__ short2 src[SMEM_ROWS][SMEM_COLS];

    int i = threadIdx.x / nScaleFactor;
    int j = threadIdx.y / nScaleFactor;

    if ((x % nScaleFactor == 0) && (y % nScaleFactor == 0))
    {
        ReadDevPtrData<short2>(srcDevPtr, x0, y0, src_w, src_h, src_pitch, src, i, j);
    }
    __syncthreads();

    if (x < dst_w && y < dst_h)
    {
        if (dstDevPtr == NULL)
        {
            surf2Dwrite<short2>(src[j][i], dstSurf, x * sizeof(short2), y, cudaBoundaryModeClamp);
        }
        else
        {
            *(short2*)((uint8_t*)dstDevPtr + y * dst_pitch + (x << 2)) = src[j][i];
        }
    }
}

void FlowUpsample(void* srcDevPtr, uint32_t nSrcWidth, uint32_t nSrcPitch, uint32_t nSrcHeight,
                  void* dstDevPtr, uint32_t nDstWidth, uint32_t nDstPitch, uint32_t nDstHeight,
                  uint32_t nScaleFactor)
{

        dim3 blockDim(BLOCKDIM_X, BLOCKDIM_Y);
        dim3 gridDim((nDstWidth + blockDim.x - 1) / blockDim.x, (nDstHeight + blockDim.y - 1) / blockDim.y);
        NearestNeighborFlowKernel << <gridDim, blockDim >> > (0, srcDevPtr, nSrcWidth, nSrcPitch, nSrcHeight,
            0, dstDevPtr, nDstWidth, nDstPitch, nDstHeight,
            nScaleFactor);

        checkCudaErrors(cudaGetLastError());
}}}}}
#endif //defined(__CUDACC_VER_MAJOR__) && (10 <= __CUDACC_VER_MAJOR__)

#endif