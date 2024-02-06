// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if !defined CUDA_DISABLER

#include <opencv2/core/cuda/common.hpp>
#include <opencv2/cudev/util/atomic.hpp>
#include "moments.cuh"

namespace cv { namespace cuda { namespace device { namespace imgproc {

constexpr int blockSizeX = 32;
constexpr int blockSizeY = 16;

template <typename T>
__device__ T butterflyWarpReduction(T value) {
    for (int i = 16; i >= 1; i /= 2)
#if (CUDART_VERSION >= 9000)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
#else
        value += __shfl_xor(value, i, 32);
#endif
    return value;
}

template <typename T>
__device__ T butterflyHalfWarpReduction(T value) {
    for (int i = 8; i >= 1; i /= 2)
#if (CUDART_VERSION >= 9000)
        value += __shfl_xor_sync(0xffff, value, i, 16);
#else
        value += __shfl_xor(value, i, 16);
#endif
    return value;
}

template<typename T, int nMoments>
__device__ void updateSums(const T val, const unsigned int x, T r[4]) {
    const T x2 = x * x;
    const T x3 = static_cast<T>(x) * x2;
    r[0] += val;
    r[1] += val * x;
    if (nMoments >= n12) r[2] += val * x2;
    if (nMoments >= n123) r[3] += val * x3;
}

template<typename TSrc, typename TMoments, int nMoments>
__device__ void rowReductions(const PtrStepSz<TSrc> img, const bool binary, const unsigned int y, TMoments r[4], TMoments smem[][nMoments + 1]) {
    for (int x = threadIdx.x; x < img.cols; x += blockDim.x) {
        const TMoments val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
        updateSums<TMoments,nMoments>(val, x, r);
    }
}

template<typename TSrc, typename TMoments, bool fourByteAligned, int nMoments>
__device__ void rowReductionsCoalesced(const PtrStepSz<TSrc> img, const bool binary, const unsigned int y, TMoments r[4], const int offsetX, TMoments smem[][nMoments + 1]) {
    const int alignedOffset = fourByteAligned ? 0 : 4 - offsetX;
    // load uncoalesced head
    if (!fourByteAligned && threadIdx.x == 0) {
        for (int x = 0; x < ::min(alignedOffset, static_cast<int>(img.cols)); x++) {
            const TMoments val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            updateSums<TMoments, nMoments>(val, x, r);
        }
    }

    // coalesced loads
    const unsigned int* rowPtrIntAligned = (const unsigned int*)(fourByteAligned ? img.ptr(y) : img.ptr(y) + alignedOffset);
    const int cols4 = fourByteAligned ? img.cols / 4 : (img.cols - alignedOffset) / 4;
    for (int x = threadIdx.x; x < cols4; x += blockDim.x) {
        const unsigned int data = rowPtrIntAligned[x];
#pragma unroll 4
        for (int i = 0; i < 4; i++) {
            const int iX = alignedOffset + 4 * x + i;
            const uchar ucharVal = ((data >> i * 8) & 0xFFU);
            const TMoments val = (!binary || ucharVal == 0) ? ucharVal : 1;
            updateSums<TMoments, nMoments>(val, iX, r);
        }
    }

    // load uncoalesced tail
    if (threadIdx.x == 0) {
        const int iTailStart = fourByteAligned ? cols4 * 4 : cols4 * 4 + alignedOffset;
        for (int x = iTailStart; x < img.cols; x++) {
            const TMoments val = (!binary || img(y, x) == 0) ? img(y, x) : 1;
            updateSums<TMoments, nMoments>(val, x, r);
        }
    }
}

template <typename TSrc, typename TMoments, bool coalesced = false, bool fourByteAligned = false, int nMoments>
__global__ void spatialMoments(const PtrStepSz<TSrc> img, const bool binary, TMoments* moments, const int offsetX = 0) {
    const unsigned int y = blockIdx.x * blockDim.y + threadIdx.y;
    __shared__ TMoments smem[blockSizeY][nMoments + 1];
    if (threadIdx.y < nMoments && threadIdx.x < blockSizeY)
        smem[threadIdx.x][threadIdx.y] = 0;
    __syncthreads();

    TMoments r[4] = { 0 };
    if (y < img.rows) {
        if (coalesced)
            rowReductionsCoalesced<TSrc, TMoments, fourByteAligned, nMoments>(img, binary, y, r, offsetX, smem);
        else
            rowReductions<TSrc, TMoments, nMoments>(img, binary, y, r, smem);
    }

    const unsigned long y2 = y * y;
    const TMoments y3 = static_cast<TMoments>(y2) * y;
    const TMoments res = butterflyWarpReduction<float>(r[0]);
    if (res) {
        smem[threadIdx.y][0] = res; //0th
        smem[threadIdx.y][1] = butterflyWarpReduction(r[1]); //1st
        smem[threadIdx.y][2] = y * res; //1st
        if (nMoments >= n12) {
            smem[threadIdx.y][3] = butterflyWarpReduction(r[2]); //2nd
            smem[threadIdx.y][4] = smem[threadIdx.y][1] * y; //2nd
            smem[threadIdx.y][5] = y2 * res; //2nd
        }
        if (nMoments >= n123) {
            smem[threadIdx.y][6] = butterflyWarpReduction(r[3]); //3rd
            smem[threadIdx.y][7] = smem[threadIdx.y][3] * y; //3rd
            smem[threadIdx.y][8] = smem[threadIdx.y][1] * y2; //3rd
            smem[threadIdx.y][9] = y3 * res; //3rd
        }
    }
    __syncthreads();

    if (threadIdx.x < blockSizeY && threadIdx.y < nMoments)
        smem[threadIdx.y][nMoments] = butterflyHalfWarpReduction(smem[threadIdx.x][threadIdx.y]);
    __syncthreads();

    if (threadIdx.y == 0 && threadIdx.x < nMoments) {
        if (smem[threadIdx.x][nMoments])
            cudev::atomicAdd(&moments[threadIdx.x], smem[threadIdx.x][nMoments]);
    }
}

template <typename TSrc, typename TMoments, int nMoments> struct momentsDispatcherNonChar {
    static void call(const PtrStepSz<TSrc> src, PtrStepSz<TMoments> moments, const bool binary, const int offsetX, const cudaStream_t stream) {
        dim3 blockSize(blockSizeX, blockSizeY);
        dim3 gridSize = dim3(divUp(src.rows, blockSizeY));
        spatialMoments<TSrc, TMoments, false, false, nMoments> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());
        if (stream == 0)
            cudaSafeCall(cudaStreamSynchronize(stream));
    };
};

template <typename TSrc, int nMoments> struct momentsDispatcherChar {
    static void call(const PtrStepSz<TSrc> src, PtrStepSz<float> moments, const bool binary, const int offsetX, const cudaStream_t stream) {
        dim3 blockSize(blockSizeX, blockSizeY);
        dim3 gridSize = dim3(divUp(src.rows, blockSizeY));
        if (offsetX)
            spatialMoments<TSrc, float, true, false, nMoments> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr(), offsetX);
        else
            spatialMoments<TSrc, float, true, true, nMoments> << <gridSize, blockSize, 0, stream >> > (src, binary, moments.ptr());

        if (stream == 0)
            cudaSafeCall(cudaStreamSynchronize(stream));
    };
};

template <typename TSrc, typename TMoments, int nMoments> struct momentsDispatcher : momentsDispatcherNonChar<TSrc, TMoments, nMoments> {};
template <int nMoments> struct momentsDispatcher<uchar, float, nMoments> : momentsDispatcherChar<uchar, nMoments> {};
template <int nMoments> struct momentsDispatcher<schar, float, nMoments> : momentsDispatcherChar<schar, nMoments> {};

template <typename TSrc, typename TMoments>
void moments(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream) {
    if (order == 1)
        momentsDispatcher<TSrc, TMoments, n1>::call(static_cast<PtrStepSz<TSrc>>(src), static_cast<PtrStepSz<TMoments>>(moments), binary, offsetX, stream);
    else if (order == 2)
        momentsDispatcher<TSrc, TMoments, n12>::call(static_cast<PtrStepSz<TSrc>>(src), static_cast<PtrStepSz<TMoments>>(moments), binary, offsetX, stream);
    else if (order == 3)
        momentsDispatcher<TSrc, TMoments, n123>::call(static_cast<PtrStepSz<TSrc>>(src), static_cast<PtrStepSz<TMoments>>(moments), binary, offsetX, stream);
};

template void moments<uchar, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<schar, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<ushort, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<short, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<int, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<float, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<double, float>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);

template void moments<uchar, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<schar, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<ushort, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<short, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<int, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<float, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);
template void moments<double, double>(const PtrStepSzb src, PtrStepSzb moments, const bool binary, const int order, const int offsetX, const cudaStream_t stream);

}}}}

#endif /* CUDA_DISABLER */
