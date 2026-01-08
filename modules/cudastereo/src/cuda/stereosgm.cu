// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Author: The "adaskit Team" at Fixstars Corporation

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include <cuda.h>
#include "stereosgm.hpp"
#include "opencv2/cudev/common.hpp"
#include "opencv2/cudev/warp/warp.hpp"
#include "opencv2/cudastereo.hpp"

namespace cv { namespace cuda { namespace device {
namespace stereosgm
{

static constexpr uint16_t INVALID_DISP = static_cast<uint16_t>(-1);

namespace detail
{

template <typename T>
__device__ __forceinline__ static T ldg(const T* const p)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

template <unsigned int WARPS_PER_BLOCK, typename T>
__device__ __forceinline__ static T shfl(T var, int srcLane, int width = cudev::WARP_SIZE, uint32_t mask = 0xFFFFFFFFU)
{
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
    return __shfl_sync(mask, var, srcLane, width);
#else
    return __shfl(var, srcLane, width);
#endif // CUDA_VERSION
#else
    static __shared__ T smem[WARPS_PER_BLOCK][cudev::WARP_SIZE];
    srcLane %= width;
    smem[cudev::Warp::warpId()][cudev::Warp::laneId()] = var;
    T ret = smem[cudev::Warp::warpId()][srcLane + (cudev::Warp::laneId() / width) * width];
    return ret;
#endif // __CUDA_ARCH__
}

template <unsigned int WARPS_PER_BLOCK, typename T>
__device__ __forceinline__ static T shfl_up(T var, unsigned int delta, int width = cudev::WARP_SIZE, uint32_t mask = 0xFFFFFFFFU)
{
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
    return __shfl_up_sync(mask, var, delta, width);
#else
    return __shfl_up(var, delta, width);
#endif // CUDA_VERSION
#else
    static __shared__ T smem[WARPS_PER_BLOCK][cudev::WARP_SIZE];
    smem[cudev::Warp::warpId()][cudev::Warp::laneId()] = var;
    T ret = var;
    if (cudev::Warp::laneId() % width >= delta)
    {
        ret = smem[cudev::Warp::warpId()][cudev::Warp::laneId() - delta];
    }
    return ret;
#endif // __CUDA_ARCH__
}

template <unsigned int WARPS_PER_BLOCK, typename T>
__device__ __forceinline__ static T shfl_down(T var, unsigned int delta, int width = cudev::WARP_SIZE, uint32_t mask = 0xFFFFFFFFU)
{
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, var, delta, width);
#else
    return __shfl_down(var, delta, width);
#endif // CUDA_VERSION
#else
    static __shared__ T smem[WARPS_PER_BLOCK][cudev::WARP_SIZE];
    smem[cudev::Warp::warpId()][cudev::Warp::laneId()] = var;
    T ret = var;
    if (cudev::Warp::laneId() % width + delta < width)
    {
        ret = smem[cudev::Warp::warpId()][cudev::Warp::laneId() + delta];
    }
    return ret;
#endif // __CUDA_ARCH__
}

template <unsigned int WARPS_PER_BLOCK, typename T>
__device__ __forceinline__ static T shfl_xor(T var, int laneMask, int width = cudev::WARP_SIZE, uint32_t mask = 0xFFFFFFFFU)
{
#if __CUDA_ARCH__ >= 300
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, var, laneMask, width);
#else
    return __shfl_xor(var, laneMask, width);
#endif // CUDA_VERSION
#else
    static __shared__ T smem[WARPS_PER_BLOCK][cudev::WARP_SIZE];
    smem[cudev::Warp::warpId()][cudev::Warp::laneId()] = var;
    T ret = var;
    if (((cudev::Warp::laneId() % width) ^ laneMask) < width)
    {
        ret = smem[cudev::Warp::warpId()][cudev::Warp::laneId() ^ laneMask];
    }
    return ret;
#endif // __CUDA_ARCH__
}


template <typename T, unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE, unsigned int STEP>
struct subgroup_min_impl
{
    static __device__ T call(T x, uint32_t mask)
    {
        x = ::min(x, shfl_xor<WARPS_PER_BLOCK, T>(x, STEP / 2, GROUP_SIZE, mask));
        return subgroup_min_impl<T, WARPS_PER_BLOCK, GROUP_SIZE, STEP / 2>::call(x, mask);
    }
};
template <typename T, unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE>
struct subgroup_min_impl<T, WARPS_PER_BLOCK, GROUP_SIZE, 1u>
{
    static __device__ T call(T x, uint32_t)
    {
        return x;
    }
};

template <unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE, unsigned int STEP>
struct subgroup_and_impl
{
    static __device__ bool call(bool x, uint32_t mask)
    {
        x &= shfl_xor<WARPS_PER_BLOCK>(x, STEP / 2, GROUP_SIZE, mask);
        return subgroup_and_impl<WARPS_PER_BLOCK, GROUP_SIZE, STEP / 2>::call(x, mask);
    }
};
template <unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE>
struct subgroup_and_impl<WARPS_PER_BLOCK, GROUP_SIZE, 1u>
{
    static __device__ bool call(bool x, uint32_t)
    {
        return x;
    }
};
} // namespace detail


template <unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE, typename T>
__device__ inline T subgroup_min(T x, uint32_t mask)
{
    return detail::subgroup_min_impl<T, WARPS_PER_BLOCK, GROUP_SIZE, GROUP_SIZE>::call(x, mask);
}

template <unsigned int WARPS_PER_BLOCK, unsigned int GROUP_SIZE>
__device__ inline bool subgroup_and(bool x, uint32_t mask)
{
    return detail::subgroup_and_impl<WARPS_PER_BLOCK, GROUP_SIZE, GROUP_SIZE>::call(x, mask);
}


template <typename T, typename S>
__device__ inline T load_as(const S *p)
{
    return *reinterpret_cast<const T *>(p);
}

template <typename T, typename S>
__device__ inline void store_as(S *p, const T& x)
{
    *reinterpret_cast<T *>(p) = x;
}


template <typename T>
__device__ inline uint32_t pack_uint8x4(T x, T y, T z, T w)
{
    uchar4 uint8x4;
    uint8x4.x = static_cast<uint8_t>(x);
    uint8x4.y = static_cast<uint8_t>(y);
    uint8x4.z = static_cast<uint8_t>(z);
    uint8x4.w = static_cast<uint8_t>(w);
    return load_as<uint32_t>(&uint8x4);
}


template <unsigned int N>
__device__ inline void load_uint8_vector(uint32_t *dest, const uint8_t *ptr);

template <>
__device__ inline void load_uint8_vector<1u>(uint32_t *dest, const uint8_t *ptr)
{
    dest[0] = static_cast<uint32_t>(ptr[0]);
}

template <>
__device__ inline void load_uint8_vector<2u>(uint32_t *dest, const uint8_t *ptr)
{
    const auto uint8x2 = load_as<uchar2>(ptr);
    dest[0] = uint8x2.x; dest[1] = uint8x2.y;
}

template <>
__device__ inline void load_uint8_vector<4u>(uint32_t *dest, const uint8_t *ptr)
{
    const auto uint8x4 = load_as<uchar4>(ptr);
    dest[0] = uint8x4.x; dest[1] = uint8x4.y; dest[2] = uint8x4.z; dest[3] = uint8x4.w;
}

template <>
__device__ inline void load_uint8_vector<8u>(uint32_t *dest, const uint8_t *ptr)
{
    const auto uint32x2 = load_as<uint2>(ptr);
    load_uint8_vector<4u>(dest + 0, reinterpret_cast<const uint8_t *>(&uint32x2.x));
    load_uint8_vector<4u>(dest + 4, reinterpret_cast<const uint8_t *>(&uint32x2.y));
}

template <>
__device__ inline void load_uint8_vector<16u>(uint32_t *dest, const uint8_t *ptr)
{
    const auto uint32x4 = load_as<uint4>(ptr);
    load_uint8_vector<4u>(dest + 0, reinterpret_cast<const uint8_t *>(&uint32x4.x));
    load_uint8_vector<4u>(dest + 4, reinterpret_cast<const uint8_t *>(&uint32x4.y));
    load_uint8_vector<4u>(dest + 8, reinterpret_cast<const uint8_t *>(&uint32x4.z));
    load_uint8_vector<4u>(dest + 12, reinterpret_cast<const uint8_t *>(&uint32x4.w));
}


template <unsigned int N>
__device__ inline void store_uint8_vector(uint8_t *dest, const uint32_t *ptr);

template <>
__device__ inline void store_uint8_vector<1u>(uint8_t *dest, const uint32_t *ptr)
{
    dest[0] = static_cast<uint8_t>(ptr[0]);
}

template <>
__device__ inline void store_uint8_vector<2u>(uint8_t *dest, const uint32_t *ptr)
{
    uchar2 uint8x2;
    uint8x2.x = static_cast<uint8_t>(ptr[0]);
    uint8x2.y = static_cast<uint8_t>(ptr[0]);
    store_as<uchar2>(dest, uint8x2);
}

template <>
__device__ inline void store_uint8_vector<4u>(uint8_t *dest, const uint32_t *ptr)
{
    store_as<uint32_t>(dest, pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]));
}

template <>
__device__ inline void store_uint8_vector<8u>(uint8_t *dest, const uint32_t *ptr)
{
    uint2 uint32x2;
    uint32x2.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
    uint32x2.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
    store_as<uint2>(dest, uint32x2);
}

template <>
__device__ inline void store_uint8_vector<16u>(uint8_t *dest, const uint32_t *ptr)
{
    uint4 uint32x4;
    uint32x4.x = pack_uint8x4(ptr[0], ptr[1], ptr[2], ptr[3]);
    uint32x4.y = pack_uint8x4(ptr[4], ptr[5], ptr[6], ptr[7]);
    uint32x4.z = pack_uint8x4(ptr[8], ptr[9], ptr[10], ptr[11]);
    uint32x4.w = pack_uint8x4(ptr[12], ptr[13], ptr[14], ptr[15]);
    store_as<uint4>(dest, uint32x4);
}


template <unsigned int N>
__device__ inline void load_uint16_vector(uint32_t *dest, const uint16_t *ptr);

template <>
__device__ inline void load_uint16_vector<1u>(uint32_t *dest, const uint16_t *ptr)
{
    dest[0] = static_cast<uint32_t>(ptr[0]);
}

template <>
__device__ inline void load_uint16_vector<2u>(uint32_t *dest, const uint16_t *ptr)
{
    const auto uint16x2 = load_as<ushort2>(ptr);
    dest[0] = uint16x2.x; dest[1] = uint16x2.y;
}

template <>
__device__ inline void load_uint16_vector<4u>(uint32_t *dest, const uint16_t *ptr)
{
    const auto uint16x4 = load_as<ushort4>(ptr);
    dest[0] = uint16x4.x; dest[1] = uint16x4.y; dest[2] = uint16x4.z; dest[3] = uint16x4.w;
}

template <>
__device__ inline void load_uint16_vector<8u>(uint32_t *dest, const uint16_t *ptr)
{
    const auto uint32x4 = load_as<uint4>(ptr);
    load_uint16_vector<2u>(dest + 0, reinterpret_cast<const uint16_t *>(&uint32x4.x));
    load_uint16_vector<2u>(dest + 2, reinterpret_cast<const uint16_t *>(&uint32x4.y));
    load_uint16_vector<2u>(dest + 4, reinterpret_cast<const uint16_t *>(&uint32x4.z));
    load_uint16_vector<2u>(dest + 6, reinterpret_cast<const uint16_t *>(&uint32x4.w));
}


template <unsigned int N>
__device__ inline void store_uint16_vector(uint16_t *dest, const uint32_t *ptr);

template <>
__device__ inline void store_uint16_vector<1u>(uint16_t *dest, const uint32_t *ptr)
{
    dest[0] = static_cast<uint16_t>(ptr[0]);
}

template <>
__device__ inline void store_uint16_vector<2u>(uint16_t *dest, const uint32_t *ptr)
{
    ushort2 uint16x2;
    uint16x2.x = static_cast<uint16_t>(ptr[0]);
    uint16x2.y = static_cast<uint16_t>(ptr[1]);
    store_as<ushort2>(dest, uint16x2);
}

template <>
__device__ inline void store_uint16_vector<4u>(uint16_t *dest, const uint32_t *ptr)
{
    ushort4 uint16x4;
    uint16x4.x = static_cast<uint16_t>(ptr[0]);
    uint16x4.y = static_cast<uint16_t>(ptr[1]);
    uint16x4.z = static_cast<uint16_t>(ptr[2]);
    uint16x4.w = static_cast<uint16_t>(ptr[3]);
    store_as<ushort4>(dest, uint16x4);
}

template <>
__device__ inline void store_uint16_vector<8u>(uint16_t *dest, const uint32_t *ptr)
{
    uint4 uint32x4;
    store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.x), &ptr[0]);
    store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.y), &ptr[2]);
    store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.z), &ptr[4]);
    store_uint16_vector<2u>(reinterpret_cast<uint16_t *>(&uint32x4.w), &ptr[6]);
    store_as<uint4>(dest, uint32x4);
}

template <>
__device__ inline void store_uint16_vector<16u>(uint16_t *dest, const uint32_t *ptr)
{
    store_uint16_vector<8u>(dest + 0, ptr + 0);
    store_uint16_vector<8u>(dest + 8, ptr + 8);
}

namespace census_transform
{
namespace
{
static constexpr int WINDOW_WIDTH = 9;
static constexpr int WINDOW_HEIGHT = 7;

static constexpr int BLOCK_SIZE = 128;
static constexpr int LINES_PER_BLOCK = 16;

template <typename T>
__global__ void census_transform_kernel(
    PtrStepSz<T> src,
    PtrStep<int32_t> dest)
{
    using pixel_type = T;
    static const int SMEM_BUFFER_SIZE = WINDOW_HEIGHT + 1;

    const int half_kw = WINDOW_WIDTH / 2;
    const int half_kh = WINDOW_HEIGHT / 2;

    __shared__ pixel_type smem_lines[SMEM_BUFFER_SIZE][BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int x0 = blockIdx.x * (BLOCK_SIZE - WINDOW_WIDTH + 1) - half_kw;
    const int y0 = blockIdx.y * LINES_PER_BLOCK;

    for (int i = 0; i < WINDOW_HEIGHT; ++i)
    {
        const int x = x0 + tid, y = y0 - half_kh + i;
        pixel_type value = 0;
        if (0 <= x && x < src.cols && 0 <= y && y < src.rows)
        {
            value = src(y, x);
        }
        smem_lines[i][tid] = value;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < LINES_PER_BLOCK; ++i)
    {
        if (i + 1 < LINES_PER_BLOCK)
        {
            // Load to smem
            const int x = x0 + tid, y = y0 + half_kh + i + 1;
            pixel_type value = 0;
            if (0 <= x && x < src.cols && 0 <= y && y < src.rows)
            {
                value = src(y, x);
            }
            const int smem_x = tid;
            const int smem_y = (WINDOW_HEIGHT + i) % SMEM_BUFFER_SIZE;
            smem_lines[smem_y][smem_x] = value;
        }

        if (half_kw <= tid && tid < BLOCK_SIZE - half_kw)
        {
            // Compute and store
            const int x = x0 + tid, y = y0 + i;
            if (half_kw <= x && x < src.cols - half_kw && half_kh <= y && y < src.rows - half_kh)
            {
                const int smem_x = tid;
                const int smem_y = (half_kh + i) % SMEM_BUFFER_SIZE;
                int32_t f = 0;
                for (int dy = -half_kh; dy < 0; ++dy)
                {
                    const int smem_y1 = (smem_y + dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                    const int smem_y2 = (smem_y - dy + SMEM_BUFFER_SIZE) % SMEM_BUFFER_SIZE;
                    for (int dx = -half_kw; dx <= half_kw; ++dx)
                    {
                        const int smem_x1 = smem_x + dx;
                        const int smem_x2 = smem_x - dx;
                        const auto a = smem_lines[smem_y1][smem_x1];
                        const auto b = smem_lines[smem_y2][smem_x2];
                        f = (f << 1) | (a > b);
                    }
                }
                for (int dx = -half_kw; dx < 0; ++dx)
                {
                    const int smem_x1 = smem_x + dx;
                    const int smem_x2 = smem_x - dx;
                    const auto a = smem_lines[smem_y][smem_x1];
                    const auto b = smem_lines[smem_y][smem_x2];
                    f = (f << 1) | (a > b);
                }
                dest(y, x) = f;
            }
        }
        __syncthreads();
    }
}
} // anonymous namespace

void censusTransform(const GpuMat& src, GpuMat& dest, Stream& _stream)
{
    CV_Assert(src.size() == dest.size());
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_16UC1);
    const int width_per_block = BLOCK_SIZE - WINDOW_WIDTH + 1;
    const int height_per_block = LINES_PER_BLOCK;
    const dim3 gdim(
        cudev::divUp(src.cols, width_per_block),
        cudev::divUp(src.rows, height_per_block));
    const dim3 bdim(BLOCK_SIZE);
    cudaStream_t stream = StreamAccessor::getStream(_stream);
    switch (src.type())
    {
    case CV_8UC1:
        census_transform_kernel<uint8_t><<<gdim, bdim, 0, stream>>>(src, dest);
        break;
    case CV_16UC1:
        census_transform_kernel<uint16_t><<<gdim, bdim, 0, stream>>>(src, dest);
        break;
    }
}
} // namespace census_transform

namespace path_aggregation
{

template <
    unsigned int DP_BLOCK_SIZE,
    unsigned int SUBGROUP_SIZE,
    unsigned int WARPS_PER_BLOCK>
    struct DynamicProgramming
{
    static_assert(
        DP_BLOCK_SIZE >= 2,
        "DP_BLOCK_SIZE must be greater than or equal to 2");
    static_assert(
        (SUBGROUP_SIZE & (SUBGROUP_SIZE - 1)) == 0,
        "SUBGROUP_SIZE must be a power of 2");

    uint32_t last_min;
    uint32_t dp[DP_BLOCK_SIZE];

    __device__ DynamicProgramming()
        : last_min(0)
    {
        for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
        {
            dp[i] = 0;
        }
    }

    __device__ void update(
        uint32_t *local_costs, uint32_t p1, uint32_t p2, uint32_t mask)
    {
        const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;

        const auto dp0 = dp[0];
        uint32_t lazy_out = 0, local_min = 0;
        {
            const unsigned int k = 0;
            const uint32_t prev = detail::shfl_up<WARPS_PER_BLOCK>(dp[DP_BLOCK_SIZE - 1], 1, cudev::WARP_SIZE, mask);
            uint32_t out = ::min(dp[k] - last_min, p2);
            if (lane_id != 0)
            {
                out = ::min(out, prev - last_min + p1);
            }
            out = ::min(out, dp[k + 1] - last_min + p1);
            lazy_out = local_min = out + local_costs[k];
        }
        for (unsigned int k = 1; k + 1 < DP_BLOCK_SIZE; ++k)
        {
            uint32_t out = ::min(dp[k] - last_min, p2);
            out = ::min(out, dp[k - 1] - last_min + p1);
            out = ::min(out, dp[k + 1] - last_min + p1);
            dp[k - 1] = lazy_out;
            lazy_out = out + local_costs[k];
            local_min = ::min(local_min, lazy_out);
        }
        {
            const unsigned int k = DP_BLOCK_SIZE - 1;
            const uint32_t next = detail::shfl_down<WARPS_PER_BLOCK>(dp0, 1, cudev::WARP_SIZE, mask);
            uint32_t out = ::min(dp[k] - last_min, p2);
            out = ::min(out, dp[k - 1] - last_min + p1);
            if (lane_id + 1 != SUBGROUP_SIZE)
            {
                out = ::min(out, next - last_min + p1);
            }
            dp[k - 1] = lazy_out;
            dp[k] = out + local_costs[k];
            local_min = ::min(local_min, dp[k]);
        }
        last_min = subgroup_min<WARPS_PER_BLOCK, SUBGROUP_SIZE>(local_min, mask);
    }
};

template <unsigned int SIZE>
__device__ unsigned int generate_mask()
{
    static_assert(SIZE <= 32, "SIZE must be less than or equal to 32");
    return static_cast<unsigned int>((1ull << SIZE) - 1u);
}

namespace horizontal
{
namespace
{
static constexpr unsigned int DP_BLOCK_SIZE = 8u;
static constexpr unsigned int DP_BLOCKS_PER_THREAD = 1u;

static constexpr unsigned int WARPS_PER_BLOCK = 4u;
static constexpr unsigned int BLOCK_SIZE = cudev::WARP_SIZE * WARPS_PER_BLOCK;


template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_horizontal_path_kernel(
    PtrStep<int32_t> left,
    PtrStep<int32_t> right,
    PtrStep<uint8_t> dest,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int SUBGROUPS_PER_WARP = cudev::WARP_SIZE / SUBGROUP_SIZE;
    static const unsigned int PATHS_PER_WARP =
        cudev::WARP_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;
    static const unsigned int PATHS_PER_BLOCK =
        BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

    static_assert(DIRECTION == 1 || DIRECTION == -1, "");
    if (width == 0 || height == 0)
    {
        return;
    }

    int32_t right_buffer[DP_BLOCKS_PER_THREAD][DP_BLOCK_SIZE];
    DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE, WARPS_PER_BLOCK> dp[DP_BLOCKS_PER_THREAD];

    const unsigned int warp_id = cudev::Warp::warpId();
    const unsigned int group_id = cudev::Warp::laneId() / SUBGROUP_SIZE;
    const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
    const unsigned int shfl_mask =
        generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

    const unsigned int y0 =
        PATHS_PER_BLOCK * blockIdx.x +
        PATHS_PER_WARP  * warp_id +
        group_id;
    const unsigned int feature_step = SUBGROUPS_PER_WARP;
    const unsigned int dest_step = SUBGROUPS_PER_WARP * MAX_DISPARITY * width;
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;
    left = PtrStep<int32_t>(left.ptr(y0), left.step);
    right = PtrStep<int32_t>(right.ptr(y0), right.step);
    dest = PtrStep<uint8_t>(&dest(0, y0 * width * MAX_DISPARITY), dest.step);

    if (y0 >= height)
    {
        return;
    }

    if (DIRECTION > 0)
    {
        for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i)
        {
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                right_buffer[i][j] = 0;
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < DP_BLOCKS_PER_THREAD; ++i)
        {
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                const int x = static_cast<int>(width - (min_disp + j + dp_offset));
                if (0 <= x && x < static_cast<int>(width))
                {
                    right_buffer[i][j] = detail::ldg(&right(i * feature_step, x));
                }
                else
                {
                    right_buffer[i][j] = 0;
                }
            }
        }
    }

    int x0 = (DIRECTION > 0) ? 0 : static_cast<int>((width - 1) & ~(DP_BLOCK_SIZE - 1));
    for (unsigned int iter = 0; iter < width; iter += DP_BLOCK_SIZE)
    {
        for (unsigned int i = 0; i < DP_BLOCK_SIZE; ++i)
        {
            const unsigned int x = x0 + (DIRECTION > 0 ? i : (DP_BLOCK_SIZE - 1 - i));
            if (x >= width)
            {
                continue;
            }
            for (unsigned int j = 0; j < DP_BLOCKS_PER_THREAD; ++j)
            {
                const unsigned int y = y0 + j * SUBGROUPS_PER_WARP;
                if (y >= height)
                {
                    continue;
                }
                const int32_t left_value = detail::ldg(&left(j * feature_step, x));
                if (DIRECTION > 0)
                {
                    const int32_t t = right_buffer[j][DP_BLOCK_SIZE - 1];
                    for (unsigned int k = DP_BLOCK_SIZE - 1; k > 0; --k)
                    {
                        right_buffer[j][k] = right_buffer[j][k - 1];
                    }
                    right_buffer[j][0] = detail::shfl_up<WARPS_PER_BLOCK>(t, 1, SUBGROUP_SIZE, shfl_mask);
                    if (lane_id == 0 && x >= min_disp)
                    {
                        right_buffer[j][0] =
                            detail::ldg(&right(j * feature_step, x - min_disp));
                    }
                }
                else
                {
                    const int32_t t = right_buffer[j][0];
                    for (unsigned int k = 1; k < DP_BLOCK_SIZE; ++k)
                    {
                        right_buffer[j][k - 1] = right_buffer[j][k];
                    }
                    right_buffer[j][DP_BLOCK_SIZE - 1] = detail::shfl_down<WARPS_PER_BLOCK>(t, 1, SUBGROUP_SIZE, shfl_mask);
                    if (lane_id + 1 == SUBGROUP_SIZE)
                    {
                        if (x >= min_disp + dp_offset + DP_BLOCK_SIZE - 1)
                        {
                            right_buffer[j][DP_BLOCK_SIZE - 1] =
                                detail::ldg(&right(j * feature_step, x - (min_disp + dp_offset + DP_BLOCK_SIZE - 1)));
                        }
                        else
                        {
                            right_buffer[j][DP_BLOCK_SIZE - 1] = 0;
                        }
                    }
                }
                uint32_t local_costs[DP_BLOCK_SIZE];
                for (unsigned int k = 0; k < DP_BLOCK_SIZE; ++k)
                {
                    local_costs[k] = __popc(left_value ^ right_buffer[j][k]);
                }
                dp[j].update(local_costs, p1, p2, shfl_mask);
                store_uint8_vector<DP_BLOCK_SIZE>(
                    &dest(0, j * dest_step + x * MAX_DISPARITY + dp_offset),
                    dp[j].dp);
            }
        }
        x0 += static_cast<int>(DP_BLOCK_SIZE) * DIRECTION;
    }
}
} // anonymous namespace

template <unsigned int MAX_DISPARITY>
void aggregateLeft2RightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    CV_Assert(left.size() == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK =
        BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

    const int gdim = cudev::divUp(left.rows, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    aggregate_horizontal_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, left.cols, left.rows, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void aggregateRight2LeftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    CV_Assert(left.size() == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK =
        BLOCK_SIZE * DP_BLOCKS_PER_THREAD / SUBGROUP_SIZE;

    const int gdim = cudev::divUp(left.rows, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    aggregate_horizontal_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, left.cols, left.rows, p1, p2, min_disp);
}


template CV_EXPORTS void aggregateLeft2RightPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);

template CV_EXPORTS void aggregateLeft2RightPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);

template CV_EXPORTS void aggregateLeft2RightPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);

template CV_EXPORTS void aggregateRight2LeftPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);

template CV_EXPORTS void aggregateRight2LeftPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);

template CV_EXPORTS void aggregateRight2LeftPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream);
} // namespace horizontal

namespace vertical
{
namespace
{
static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int WARPS_PER_BLOCK = 8u;
static constexpr unsigned int BLOCK_SIZE = cudev::WARP_SIZE * WARPS_PER_BLOCK;

template <int DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_vertical_path_kernel(
    PtrStep<int32_t> left,
    PtrStep<int32_t> right,
    PtrStep<uint8_t> dest,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_WARP = cudev::WARP_SIZE / SUBGROUP_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
    static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

    static_assert(DIRECTION == 1 || DIRECTION == -1, "");
    if (width == 0 || height == 0)
    {
        return;
    }

    __shared__ int32_t right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS + 1];
    DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE, WARPS_PER_BLOCK> dp;

    const unsigned int warp_id = cudev::Warp::warpId();
    const unsigned int group_id = cudev::Warp::laneId() / SUBGROUP_SIZE;
    const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
    const unsigned int shfl_mask =
        generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

    const unsigned int x =
        blockIdx.x * PATHS_PER_BLOCK +
        warp_id    * PATHS_PER_WARP +
        group_id;
    const unsigned int right_x0 = blockIdx.x * PATHS_PER_BLOCK;
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

    const unsigned int right0_addr =
        (right_x0 + PATHS_PER_BLOCK - 1) - x + dp_offset;
    const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
    const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

    for (unsigned int iter = 0; iter < height; ++iter)
    {
        const unsigned int y = (DIRECTION > 0 ? iter : height - 1 - iter);
        // Load left to register
        int32_t left_value;
        if (x < width)
        {
            left_value = left(y, x);
        }
        // Load right to smem
        for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
        {
            const unsigned int i = i0 + threadIdx.x;
            if (i < RIGHT_BUFFER_SIZE)
            {
                const int x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
                int32_t right_value = 0;
                if (0 <= x && x < static_cast<int>(width))
                {
                    right_value = right(y, x);
                }
                const unsigned int lo = i % DP_BLOCK_SIZE;
                const unsigned int hi = i / DP_BLOCK_SIZE;
                right_buffer[lo][hi] = right_value;
                if (hi > 0)
                {
                    right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                }
            }
        }
        __syncthreads();
        // Compute
        if (x < width)
        {
            int32_t right_values[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
            }
            uint32_t local_costs[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                local_costs[j] = __popc(left_value ^ right_values[j]);
            }
            dp.update(local_costs, p1, p2, shfl_mask);
            store_uint8_vector<DP_BLOCK_SIZE>(
                &dest(0, dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width),
                dp.dp);
        }
        __syncthreads();
    }
}
} // anonymous namespace

template <unsigned int MAX_DISPARITY>
void aggregateUp2DownPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    aggregate_vertical_path_kernel<1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void aggregateDown2UpPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    aggregate_vertical_path_kernel<-1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}


template CV_EXPORTS void aggregateUp2DownPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUp2DownPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUp2DownPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDown2UpPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDown2UpPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDown2UpPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

} // namespace vertical

namespace oblique
{
namespace
{
static constexpr unsigned int DP_BLOCK_SIZE = 16u;
static constexpr unsigned int WARPS_PER_BLOCK = 8u;
static constexpr unsigned int BLOCK_SIZE = cudev::WARP_SIZE * WARPS_PER_BLOCK;

template <int X_DIRECTION, int Y_DIRECTION, unsigned int MAX_DISPARITY>
__global__ void aggregate_oblique_path_kernel(
    PtrStep<int32_t> left,
    PtrStep<int32_t> right,
    PtrStep<uint8_t> dest,
    int width,
    int height,
    unsigned int p1,
    unsigned int p2,
    int min_disp)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_WARP = cudev::WARP_SIZE / SUBGROUP_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    static const unsigned int RIGHT_BUFFER_SIZE = MAX_DISPARITY + PATHS_PER_BLOCK;
    static const unsigned int RIGHT_BUFFER_ROWS = RIGHT_BUFFER_SIZE / DP_BLOCK_SIZE;

    static_assert(X_DIRECTION == 1 || X_DIRECTION == -1, "");
    static_assert(Y_DIRECTION == 1 || Y_DIRECTION == -1, "");
    if (width == 0 || height == 0)
    {
        return;
    }

    __shared__ int32_t right_buffer[2 * DP_BLOCK_SIZE][RIGHT_BUFFER_ROWS];
    DynamicProgramming<DP_BLOCK_SIZE, SUBGROUP_SIZE, WARPS_PER_BLOCK> dp;

    const unsigned int warp_id = cudev::Warp::warpId();
    const unsigned int group_id = cudev::Warp::laneId() / SUBGROUP_SIZE;
    const unsigned int lane_id = threadIdx.x % SUBGROUP_SIZE;
    const unsigned int shfl_mask =
        generate_mask<SUBGROUP_SIZE>() << (group_id * SUBGROUP_SIZE);

    const int x0 =
        blockIdx.x * PATHS_PER_BLOCK +
        warp_id * PATHS_PER_WARP +
        group_id +
        (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
    const int right_x00 =
        blockIdx.x * PATHS_PER_BLOCK +
        (X_DIRECTION > 0 ? -static_cast<int>(height - 1) : 0);
    const unsigned int dp_offset = lane_id * DP_BLOCK_SIZE;

    const unsigned int right0_addr =
        static_cast<unsigned int>(right_x00 + PATHS_PER_BLOCK - 1 - x0) + dp_offset;
    const unsigned int right0_addr_lo = right0_addr % DP_BLOCK_SIZE;
    const unsigned int right0_addr_hi = right0_addr / DP_BLOCK_SIZE;

    for (unsigned int iter = 0; iter < height; ++iter)
    {
        const int y = static_cast<int>(Y_DIRECTION > 0 ? iter : height - 1 - iter);
        const int x = x0 + static_cast<int>(iter) * X_DIRECTION;
        const int right_x0 = right_x00 + static_cast<int>(iter) * X_DIRECTION;
        // Load right to smem
        for (unsigned int i0 = 0; i0 < RIGHT_BUFFER_SIZE; i0 += BLOCK_SIZE)
        {
            const unsigned int i = i0 + threadIdx.x;
            if (i < RIGHT_BUFFER_SIZE)
            {
                const int x = static_cast<int>(right_x0 + PATHS_PER_BLOCK - 1 - i - min_disp);
                int32_t right_value = 0;
                if (0 <= x && x < static_cast<int>(width))
                {
                    right_value = right(y, x);
                }
                const unsigned int lo = i % DP_BLOCK_SIZE;
                const unsigned int hi = i / DP_BLOCK_SIZE;
                right_buffer[lo][hi] = right_value;
                if (hi > 0)
                {
                    right_buffer[lo + DP_BLOCK_SIZE][hi - 1] = right_value;
                }
            }
        }
        __syncthreads();
        // Compute
        if (0 <= x && x < static_cast<int>(width))
        {
            const int32_t left_value = detail::ldg(&left(y, x));
            int32_t right_values[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                right_values[j] = right_buffer[right0_addr_lo + j][right0_addr_hi];
            }
            uint32_t local_costs[DP_BLOCK_SIZE];
            for (unsigned int j = 0; j < DP_BLOCK_SIZE; ++j)
            {
                local_costs[j] = __popc(left_value ^ right_values[j]);
            }
            dp.update(local_costs, p1, p2, shfl_mask);
            store_uint8_vector<DP_BLOCK_SIZE>(
                &dest(0, dp_offset + x * MAX_DISPARITY + y * MAX_DISPARITY * width),
                dp.dp);
        }
        __syncthreads();
    }
}
} // anonymous namespace

template <unsigned int MAX_DISPARITY>
void aggregateUpleft2DownrightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width + size.height - 1, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = StreamAccessor::getStream(_stream);
    aggregate_oblique_path_kernel<1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void aggregateUpright2DownleftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width + size.height - 1, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = StreamAccessor::getStream(_stream);
    aggregate_oblique_path_kernel<-1, 1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void aggregateDownright2UpleftPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width + size.height - 1, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = StreamAccessor::getStream(_stream);
    aggregate_oblique_path_kernel<-1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}

template <unsigned int MAX_DISPARITY>
void aggregateDownleft2UprightPath(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& _stream)
{
    static const unsigned int SUBGROUP_SIZE = MAX_DISPARITY / DP_BLOCK_SIZE;
    static const unsigned int PATHS_PER_BLOCK = BLOCK_SIZE / SUBGROUP_SIZE;

    const Size size = left.size();
    const int gdim = cudev::divUp(size.width + size.height - 1, PATHS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = StreamAccessor::getStream(_stream);
    aggregate_oblique_path_kernel<1, -1, MAX_DISPARITY><<<gdim, bdim, 0, stream>>>(
        left, right, dest, size.width, size.height, p1, p2, min_disp);
}

template CV_EXPORTS void aggregateUpleft2DownrightPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUpleft2DownrightPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUpleft2DownrightPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUpright2DownleftPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUpright2DownleftPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateUpright2DownleftPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownright2UpleftPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownright2UpleftPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownright2UpleftPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownleft2UprightPath<64u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownleft2UprightPath<128u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

template CV_EXPORTS void aggregateDownleft2UprightPath<256u>(
    const GpuMat& left,
    const GpuMat& right,
    GpuMat& dest,
    unsigned int p1,
    unsigned int p2,
    int min_disp,
    Stream& stream);

} // namespace oblique

template <size_t MAX_DISPARITY>
void PathAggregation::operator() (const GpuMat& left, const GpuMat& right, GpuMat& dest, int mode, int p1, int p2, int min_disp, Stream& stream)
{
    CV_Assert(left.size() == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(left.type() == CV_32SC1);

    const int num_paths = mode == StereoSGBM::MODE_HH4 ? 4 : 8;

    stream.waitForCompletion();

    const Size size = left.size();
    const int buffer_step = size.width * size.height * static_cast<int>(MAX_DISPARITY);
    CV_Assert(dest.rows == 1 && buffer_step * num_paths == dest.cols);

    for (int i = 0; i < num_paths; ++i)
    {
        subs[i] = dest.colRange(i * buffer_step, (i + 1) * buffer_step);
    }

    vertical::aggregateUp2DownPath<MAX_DISPARITY>(left, right, subs[0], p1, p2, min_disp, streams[0]);
    vertical::aggregateDown2UpPath<MAX_DISPARITY>(left, right, subs[1], p1, p2, min_disp, streams[1]);
    horizontal::aggregateLeft2RightPath<MAX_DISPARITY>(left, right, subs[2], p1, p2, min_disp, streams[2]);
    horizontal::aggregateRight2LeftPath<MAX_DISPARITY>(left, right, subs[3], p1, p2, min_disp, streams[3]);

    if (mode == StereoSGBM::MODE_HH)
    {
        oblique::aggregateUpleft2DownrightPath<MAX_DISPARITY>(left, right, subs[4], p1, p2, min_disp, streams[4]);
        oblique::aggregateUpright2DownleftPath<MAX_DISPARITY>(left, right, subs[5], p1, p2, min_disp, streams[5]);
        oblique::aggregateDownright2UpleftPath<MAX_DISPARITY>(left, right, subs[6], p1, p2, min_disp, streams[6]);
        oblique::aggregateDownleft2UprightPath<MAX_DISPARITY>(left, right, subs[7], p1, p2, min_disp, streams[7]);
    }

    // synchronization
    for (int i = 0; i < num_paths; ++i)
    {
        events[i].record(streams[i]);
        stream.waitEvent(events[i]);
        streams[i].waitForCompletion();
    }
}

template void PathAggregation::operator()< 64>(const GpuMat& left, const GpuMat& right, GpuMat& dest, int mode, int p1, int p2, int min_disp, Stream& stream);
template void PathAggregation::operator()<128>(const GpuMat& left, const GpuMat& right, GpuMat& dest, int mode, int p1, int p2, int min_disp, Stream& stream);
template void PathAggregation::operator()<256>(const GpuMat& left, const GpuMat& right, GpuMat& dest, int mode, int p1, int p2, int min_disp, Stream& stream);

} // namespace path_aggregation

namespace winner_takes_all
{
namespace
{
static constexpr unsigned int WARPS_PER_BLOCK = 8u;
static constexpr unsigned int BLOCK_SIZE = WARPS_PER_BLOCK * cudev::WARP_SIZE;

__device__ inline uint32_t pack_cost_index(uint32_t cost, uint32_t index)
{
    union
    {
        uint32_t uint32;
        ushort2 uint16x2;
    } u;
    u.uint16x2.x = static_cast<uint16_t>(index);
    u.uint16x2.y = static_cast<uint16_t>(cost);
    return u.uint32;
}

__device__ uint32_t unpack_cost(uint32_t packed)
{
    return packed >> 16;
}

__device__ int unpack_index(uint32_t packed)
{
    return packed & 0xffffu;
}

using ComputeDisparity = uint32_t(*)(uint32_t, uint32_t, uint16_t*);

__device__ inline uint32_t compute_disparity_normal(uint32_t disp, uint32_t cost = 0, uint16_t* smem = nullptr)
{
    return disp;
}

template <size_t MAX_DISPARITY>
__device__ inline uint32_t compute_disparity_subpixel(uint32_t disp, uint32_t cost, uint16_t* smem)
{
    uint32_t subp = disp;
    subp <<= StereoSGBM::DISP_SHIFT;
    if (disp > 0 && disp < MAX_DISPARITY - 1)
    {
        const int left = smem[disp - 1];
        const int right = smem[disp + 1];
        const int numer = left - right;
        const int denom = left - 2 * cost + right;
        subp += ((numer << StereoSGBM::DISP_SHIFT) + denom) / (2 * denom);
    }
    return subp;
}


template <unsigned int MAX_DISPARITY, unsigned int NUM_PATHS, ComputeDisparity compute_disparity = compute_disparity_normal>
__global__ void winner_takes_all_kernel(
    const PtrStep<uint8_t> _src,
    PtrStep<int16_t> _left_dest,
    PtrStep<int16_t> _right_dest,
    int width,
    int height,
    float uniqueness)
{
    static const unsigned int ACCUMULATION_PER_THREAD = 16u;
    static const unsigned int REDUCTION_PER_THREAD = MAX_DISPARITY / cudev::WARP_SIZE;
    static const unsigned int ACCUMULATION_INTERVAL = ACCUMULATION_PER_THREAD / REDUCTION_PER_THREAD;
    static const unsigned int UNROLL_DEPTH =
        (REDUCTION_PER_THREAD > ACCUMULATION_INTERVAL)
        ? REDUCTION_PER_THREAD
        : ACCUMULATION_INTERVAL;

    const unsigned int cost_step = MAX_DISPARITY * width * height;
    const unsigned int warp_id = cudev::Warp::warpId();
    const unsigned int lane_id = cudev::Warp::laneId();

    const unsigned int y = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const PtrStep<uint8_t> src{ (uint8_t*)&_src(0, y * MAX_DISPARITY * width), height * width * MAX_DISPARITY * NUM_PATHS };
    PtrStep<int16_t> left_dest{ _left_dest.ptr(y), _left_dest.step };
    PtrStep<int16_t> right_dest{ _right_dest.ptr(y), _right_dest.step };

    if (y >= height)
    {
        return;
    }

    __shared__ uint16_t smem_cost_sum[WARPS_PER_BLOCK][ACCUMULATION_INTERVAL][MAX_DISPARITY];

    uint32_t right_best[REDUCTION_PER_THREAD];
    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
    {
        right_best[i] = 0xffffffffu;
    }

    for (unsigned int x0 = 0; x0 < width; x0 += UNROLL_DEPTH)
    {
#pragma unroll
        for (unsigned int x1 = 0; x1 < UNROLL_DEPTH; ++x1)
        {
            if (x1 % ACCUMULATION_INTERVAL == 0)
            {
                const unsigned int k = lane_id * ACCUMULATION_PER_THREAD;
                const unsigned int k_hi = k / MAX_DISPARITY;
                const unsigned int k_lo = k % MAX_DISPARITY;
                const unsigned int x = x0 + x1 + k_hi;
                if (x < width)
                {
                    const unsigned int offset = x * MAX_DISPARITY + k_lo;
                    uint32_t sum[ACCUMULATION_PER_THREAD];
                    for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                    {
                        sum[i] = 0;
                    }
                    for (unsigned int p = 0; p < NUM_PATHS; ++p)
                    {
                        uint32_t load_buffer[ACCUMULATION_PER_THREAD];
                        load_uint8_vector<ACCUMULATION_PER_THREAD>(
                            load_buffer, &src(0, p * cost_step + offset));
                        for (unsigned int i = 0; i < ACCUMULATION_PER_THREAD; ++i)
                        {
                            sum[i] += load_buffer[i];
                        }
                    }
                    store_uint16_vector<ACCUMULATION_PER_THREAD>(
                        &smem_cost_sum[warp_id][k_hi][k_lo], sum);
                }
#if CUDA_VERSION >= 9000
                __syncwarp();
#else
                __threadfence_block();
#endif
            }
            const unsigned int x = x0 + x1;
            if (x < width)
            {
                // Load sum of costs
                const unsigned int smem_x = x1 % ACCUMULATION_INTERVAL;
                const unsigned int k0 = lane_id * REDUCTION_PER_THREAD;
                uint32_t local_cost_sum[REDUCTION_PER_THREAD];
                load_uint16_vector<REDUCTION_PER_THREAD>(
                    local_cost_sum, &smem_cost_sum[warp_id][smem_x][k0]);
                // Pack sum of costs and dispairty
                uint32_t local_packed_cost[REDUCTION_PER_THREAD];
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    local_packed_cost[i] = pack_cost_index(local_cost_sum[i], k0 + i);
                }
                // Update left
                uint32_t best = 0xffffffffu;
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    best = ::min(best, local_packed_cost[i]);
                }
                best = subgroup_min<WARPS_PER_BLOCK, cudev::WARP_SIZE>(best, 0xffffffffu);
                // Update right
#pragma unroll
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
                    const int p = static_cast<int>(((x - k) & ~(MAX_DISPARITY - 1)) + k);
                    const unsigned int d = static_cast<unsigned int>(x - p);
                    uint32_t recv = detail::shfl<WARPS_PER_BLOCK>(local_packed_cost[(REDUCTION_PER_THREAD - i + x1) % REDUCTION_PER_THREAD], d / REDUCTION_PER_THREAD);
                    right_best[i] = ::min(right_best[i], recv);
                    if (d == MAX_DISPARITY - 1)
                    {
                        if (0 <= p)
                        {
                            right_dest(0, p) = compute_disparity_normal(unpack_index(right_best[i]));
                        }
                        right_best[i] = 0xffffffffu;
                    }
                }
                // Resume updating left to avoid execution dependency
                const uint32_t bestCost = unpack_cost(best);
                const int bestDisp = unpack_index(best);
                bool uniq = true;
                for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
                {
                    const uint32_t x = local_packed_cost[i];
                    const bool uniq1 = unpack_cost(x) * uniqueness >= bestCost;
                    const bool uniq2 = ::abs(unpack_index(x) - bestDisp) <= 1;
                    uniq &= uniq1 || uniq2;
                }
                uniq = subgroup_and<WARPS_PER_BLOCK, cudev::WARP_SIZE>(uniq, 0xffffffffu);
                if (lane_id == 0)
                {
                    left_dest(0, x) = uniq ? compute_disparity(bestDisp, bestCost, smem_cost_sum[warp_id][smem_x]) : INVALID_DISP;
                }
            }
        }
    }
    for (unsigned int i = 0; i < REDUCTION_PER_THREAD; ++i)
    {
        const unsigned int k = lane_id * REDUCTION_PER_THREAD + i;
        const int p = static_cast<int>(((width - k) & ~(MAX_DISPARITY - 1)) + k);
        if (0 <= p && p < width)
        {
            right_dest(0, p) = compute_disparity_normal(unpack_index(right_best[i]));
        }
    }
}
} // anonymous namespace

template <size_t MAX_DISPARITY>
void winnerTakesAll(const GpuMat& src, GpuMat& left, GpuMat& right, float uniqueness, bool subpixel, int mode, cv::cuda::Stream& _stream)
{
    cv::Size size = left.size();
    int num_paths = mode == StereoSGBM::MODE_HH4 ? 4 : 8;
    CV_Assert(src.rows == 1 && src.cols == size.width * size.height * static_cast<int>(MAX_DISPARITY) * num_paths);
    CV_Assert(size == right.size());
    CV_Assert(left.type() == right.type());
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(mode == StereoSGBM::MODE_HH || mode == StereoSGBM::MODE_HH4);
    const int gdim = cudev::divUp(size.height, WARPS_PER_BLOCK);
    const int bdim = BLOCK_SIZE;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    if (subpixel && mode == StereoSGBM::MODE_HH)
    {
        winner_takes_all_kernel<MAX_DISPARITY, 8, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    }
    else if (subpixel && mode == StereoSGBM::MODE_HH4)
    {
        winner_takes_all_kernel<MAX_DISPARITY, 4, compute_disparity_subpixel<MAX_DISPARITY>><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    }
    else if (!subpixel && mode == StereoSGBM::MODE_HH)
    {
        winner_takes_all_kernel<MAX_DISPARITY, 8, compute_disparity_normal><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    }
    else /* if (!subpixel && mode == StereoSGBM::MODE_HH4) */
    {
        winner_takes_all_kernel<MAX_DISPARITY, 4, compute_disparity_normal><<<gdim, bdim, 0, stream>>>(
            src, left, right, size.width, size.height, uniqueness);
    }
}

template CV_EXPORTS void winnerTakesAll< 64>(const GpuMat&, GpuMat&, GpuMat&, float, bool, int, cv::cuda::Stream&);
template CV_EXPORTS void winnerTakesAll<128>(const GpuMat&, GpuMat&, GpuMat&, float, bool, int, cv::cuda::Stream&);
template CV_EXPORTS void winnerTakesAll<256>(const GpuMat&, GpuMat&, GpuMat&, float, bool, int, cv::cuda::Stream&);
} // namespace winner_takes_all

namespace median_filter
{
namespace
{
const int BLOCK_X = 16;
const int BLOCK_Y = 16;
const int KSIZE = 3;
const int RADIUS = KSIZE / 2;
const int KSIZE_SQ = KSIZE * KSIZE;

template <typename T>
__device__ inline void swap(T& x, T& y)
{
    T tmp(x);
    x = y;
    y = tmp;
}

// sort, min, max of 1 element
template <typename T, int V = 1> __device__ inline void dev_sort(T& x, T& y)
{
    if (x > y) swap(x, y);
}
template <typename T, int V = 1> __device__ inline void dev_min(T& x, T& y)
{
    x = ::min(x, y);
}
template <typename T, int V = 1> __device__ inline void dev_max(T& x, T& y)
{
    y = ::max(x, y);
}

// sort, min, max of 2 elements
__device__ inline void dev_sort_2(uint32_t& x, uint32_t& y)
{
    const uint32_t mask = __vcmpgtu2(x, y);
    const uint32_t tmp = (x ^ y) & mask;
    x ^= tmp;
    y ^= tmp;
}
__device__ inline void dev_min_2(uint32_t& x, uint32_t& y)
{
    x = __vminu2(x, y);
}
__device__ inline void dev_max_2(uint32_t& x, uint32_t& y)
{
    y = __vmaxu2(x, y);
}

template <> __device__ inline void dev_sort<uint32_t, 2>(uint32_t& x, uint32_t& y)
{
    dev_sort_2(x, y);
}
template <> __device__ inline void dev_min<uint32_t, 2>(uint32_t& x, uint32_t& y)
{
    dev_min_2(x, y);
}
template <> __device__ inline void dev_max<uint32_t, 2>(uint32_t& x, uint32_t& y)
{
    dev_max_2(x, y);
}

// sort, min, max of 4 elements
__device__ inline void dev_sort_4(uint32_t& x, uint32_t& y)
{
    const uint32_t mask = __vcmpgtu4(x, y);
    const uint32_t tmp = (x ^ y) & mask;
    x ^= tmp;
    y ^= tmp;
}
__device__ inline void dev_min_4(uint32_t& x, uint32_t& y)
{
    x = __vminu4(x, y);
}
__device__ inline void dev_max_4(uint32_t& x, uint32_t& y)
{
    y = __vmaxu4(x, y);
}

template <> __device__ inline void dev_sort<uint32_t, 4>(uint32_t& x, uint32_t& y)
{
    dev_sort_4(x, y);
}
template <> __device__ inline void dev_min<uint32_t, 4>(uint32_t& x, uint32_t& y)
{
    dev_min_4(x, y);
}
template <> __device__ inline void dev_max<uint32_t, 4>(uint32_t& x, uint32_t& y)
{
    dev_max_4(x, y);
}

template <typename T, int V = 1>
__device__ inline void median_selection_network_9(T* buf)
{
#define SWAP_OP(i, j) dev_sort<T, V>(buf[i], buf[j])
#define MIN_OP(i, j) dev_min<T, V>(buf[i], buf[j])
#define MAX_OP(i, j) dev_max<T, V>(buf[i], buf[j])

    SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
    SWAP_OP(1, 2); SWAP_OP(4, 5); SWAP_OP(7, 8);
    SWAP_OP(0, 1); SWAP_OP(3, 4); SWAP_OP(6, 7);
    MAX_OP(0, 3); MAX_OP(3, 6);
    SWAP_OP(1, 4); MIN_OP(4, 7); MAX_OP(1, 4);
    MIN_OP(5, 8); MIN_OP(2, 5);
    SWAP_OP(2, 4); MIN_OP(4, 6); MAX_OP(2, 4);

#undef SWAP_OP
#undef MIN_OP
#undef MAX_OP
}

__global__ void median_kernel_3x3_8u(const PtrStepSz<uint8_t> src, PtrStep<uint8_t> dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < RADIUS || x >= src.cols - RADIUS || y < RADIUS || y >= src.rows - RADIUS)
        return;

    uint8_t buf[KSIZE_SQ];
    for (int i = 0; i < KSIZE_SQ; i++)
        buf[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

    median_selection_network_9(buf);

    dst(y, x) = buf[KSIZE_SQ / 2];
}

__global__ void median_kernel_3x3_16u(const PtrStepSz<uint16_t> src, PtrStep<uint16_t> dst)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < RADIUS || x >= src.cols - RADIUS || y < RADIUS || y >= src.rows - RADIUS)
        return;

    uint16_t buf[KSIZE_SQ];
    for (int i = 0; i < KSIZE_SQ; i++)
        buf[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

    median_selection_network_9(buf);

    dst(y, x) = buf[KSIZE_SQ / 2];
}

__global__ void median_kernel_3x3_8u_v4(const PtrStepSz<uint8_t> src, PtrStep<uint8_t> dst)
{
    const int x_4 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < RADIUS || y >= src.rows - RADIUS)
        return;

    uint32_t buf[KSIZE_SQ];
    if (x_4 >= 4 && x_4 + 7 < src.cols)
    {
        buf[0] = *((const uint32_t*)&src(y - 1, x_4 - 4));
        buf[1] = *((const uint32_t*)&src(y - 1, x_4 - 0));
        buf[2] = *((const uint32_t*)&src(y - 1, x_4 + 4));

        buf[3] = *((const uint32_t*)&src(y - 0, x_4 - 4));
        buf[4] = *((const uint32_t*)&src(y - 0, x_4 - 0));
        buf[5] = *((const uint32_t*)&src(y - 0, x_4 + 4));

        buf[6] = *((const uint32_t*)&src(y + 1, x_4 - 4));
        buf[7] = *((const uint32_t*)&src(y + 1, x_4 - 0));
        buf[8] = *((const uint32_t*)&src(y + 1, x_4 + 4));

        buf[0] = (buf[1] << 8) | (buf[0] >> 24);
        buf[2] = (buf[1] >> 8) | (buf[2] << 24);

        buf[3] = (buf[4] << 8) | (buf[3] >> 24);
        buf[5] = (buf[4] >> 8) | (buf[5] << 24);

        buf[6] = (buf[7] << 8) | (buf[6] >> 24);
        buf[8] = (buf[7] >> 8) | (buf[8] << 24);

        median_selection_network_9<uint32_t, 4>(buf);

        *((uint32_t*)&dst(y, x_4)) = buf[KSIZE_SQ / 2];
    }
    else if (x_4 == 0)
    {
        for (int x = RADIUS; x < 4; x++)
        {
            uint8_t* buf_u8 = (uint8_t*)buf;
            for (int i = 0; i < KSIZE_SQ; i++)
                buf_u8[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

            median_selection_network_9(buf_u8);

            dst(y, x) = buf_u8[KSIZE_SQ / 2];
        }
    }
    else if (x_4 < src.cols)
    {
        for (int x = x_4; x < ::min(x_4 + 4, src.cols - RADIUS); x++)
        {
            uint8_t* buf_u8 = (uint8_t*)buf;
            for (int i = 0; i < KSIZE_SQ; i++)
                buf_u8[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

            median_selection_network_9(buf_u8);

            dst(y, x) = buf_u8[KSIZE_SQ / 2];
        }
    }
}

__global__ void median_kernel_3x3_16u_v2(const PtrStepSz<uint16_t> src, PtrStep<uint16_t> dst)
{
    const int x_2 = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < RADIUS || y >= src.rows - RADIUS)
        return;

    uint32_t buf[KSIZE_SQ];
    if (x_2 >= 2 && x_2 + 3 < src.cols)
    {
        buf[0] = *((const uint32_t*)&src(y - 1, x_2 - 2));
        buf[1] = *((const uint32_t*)&src(y - 1, x_2 - 0));
        buf[2] = *((const uint32_t*)&src(y - 1, x_2 + 2));

        buf[3] = *((const uint32_t*)&src(y - 0, x_2 - 2));
        buf[4] = *((const uint32_t*)&src(y - 0, x_2 - 0));
        buf[5] = *((const uint32_t*)&src(y - 0, x_2 + 2));

        buf[6] = *((const uint32_t*)&src(y + 1, x_2 - 2));
        buf[7] = *((const uint32_t*)&src(y + 1, x_2 - 0));
        buf[8] = *((const uint32_t*)&src(y + 1, x_2 + 2));

        buf[0] = (buf[1] << 16) | (buf[0] >> 16);
        buf[2] = (buf[1] >> 16) | (buf[2] << 16);

        buf[3] = (buf[4] << 16) | (buf[3] >> 16);
        buf[5] = (buf[4] >> 16) | (buf[5] << 16);

        buf[6] = (buf[7] << 16) | (buf[6] >> 16);
        buf[8] = (buf[7] >> 16) | (buf[8] << 16);

        median_selection_network_9<uint32_t, 2>(buf);

        *((uint32_t*)&dst(y, x_2)) = buf[KSIZE_SQ / 2];
    }
    else if (x_2 == 0)
    {
        for (int x = RADIUS; x < 2; x++)
        {
            uint8_t* buf_u8 = (uint8_t*)buf;
            for (int i = 0; i < KSIZE_SQ; i++)
                buf_u8[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

            median_selection_network_9(buf_u8);

            dst(y, x) = buf_u8[KSIZE_SQ / 2];
        }
    }
    else if (x_2 < src.cols)
    {
        for (int x = x_2; x < ::min(x_2 + 2, src.cols - RADIUS); x++)
        {
            uint8_t* buf_u8 = (uint8_t*)buf;
            for (int i = 0; i < KSIZE_SQ; i++)
                buf_u8[i] = src(y - RADIUS + i / KSIZE, x - RADIUS + i % KSIZE);

            median_selection_network_9(buf_u8);

            dst(y, x) = buf_u8[KSIZE_SQ / 2];
        }
    }
}

template <typename T>
void median_filter(const PtrStepSz<T> d_src, PtrStep<T> d_dst, Stream& _stream);

template <>
void median_filter<uint8_t>(const PtrStepSz<uint8_t> d_src, PtrStep<uint8_t> d_dst, Stream& _stream)
{
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

    if ((d_src.step / sizeof(uint8_t)) % 4 == 0)
    {
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(cudev::divUp(d_src.cols / 4, block.x), cudev::divUp(d_src.rows, block.y));
        median_kernel_3x3_8u_v4<<<grid, block, 0, stream>>>(d_src, d_dst);
    }
    else
    {
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(cudev::divUp(d_src.cols, block.x), cudev::divUp(d_src.rows, block.y));
        median_kernel_3x3_8u<<<grid, block, 0, stream>>>(d_src, d_dst);
    }

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
}

template <>
void median_filter(const PtrStepSz<uint16_t> d_src, PtrStep<uint16_t> d_dst, Stream& _stream)
{
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

    if ((d_src.step / sizeof(uint16_t)) % 2 == 0)
    {
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(cudev::divUp(d_src.cols / 2, block.x), cudev::divUp(d_src.rows, block.y));
        median_kernel_3x3_16u_v2<<<grid, block, 0, stream>>>(d_src, d_dst);
    }
    else
    {
        const dim3 block(BLOCK_X, BLOCK_Y);
        const dim3 grid(cudev::divUp(d_src.cols, block.x), cudev::divUp(d_src.rows, block.y));
        median_kernel_3x3_16u<<<grid, block, 0, stream>>>(d_src, d_dst);
    }

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
}
} // anonymous namespace

void medianFilter(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    CV_Assert(src.size() == dst.size());
    CV_Assert(src.type() == CV_16SC1);
    CV_Assert(src.type() == dst.type());

    switch (src.type())
    {
    case CV_8UC1:
        median_filter<uint8_t>(src, dst, stream);
        break;
    case CV_16SC1:
    case CV_16UC1:
        median_filter<uint16_t>(src, dst, stream);
        break;
    default:
        CV_Error(cv::Error::BadDepth, "Unsupported depth");
    }
}
} // namespace median_filter

namespace check_consistency
{
namespace
{
template<typename SRC_T, typename DST_T>
__global__ void check_consistency_kernel(PtrStep<DST_T> d_leftDisp, const PtrStep<DST_T> d_rightDisp, const PtrStep<SRC_T> d_left, int width, int height, bool subpixel)
{

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    // left-right consistency check, only on leftDisp, but could be done for rightDisp too

    SRC_T mask = d_left(i, j);
    DST_T org = d_leftDisp(i, j);
    int d = org;
    if (subpixel)
    {
        d >>= StereoMatcher::DISP_SHIFT;
    }
    int k = j - d;
    if (mask == 0 || org == INVALID_DISP || (k >= 0 && k < width && abs(d_rightDisp(i, k) - d) > 1))
    {
        // masked or left-right inconsistent pixel -> invalid
        d_leftDisp(i, j) = static_cast<DST_T>(INVALID_DISP);
    }
}

template <typename disp_type, typename image_type>
void check_consistency(PtrStep<disp_type> d_left_disp, const PtrStep<disp_type> d_right_disp, const PtrStep<image_type> d_src_left, int width, int height, bool subpixel, Stream& _stream)
{
    const dim3 blocks(width / 16, height / 16);
    const dim3 threads(16, 16);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

    check_consistency_kernel<image_type, disp_type><<<blocks, threads, 0, stream>>>(d_left_disp, d_right_disp, d_src_left, width, height, subpixel);

    CV_CUDEV_SAFE_CALL(cudaGetLastError());
}
} // anonymous namespace

void checkConsistency(GpuMat& left_disp, const GpuMat& right_disp, const GpuMat& src_left, bool subpixel, Stream& stream)
{
    Size size = left_disp.size();
    CV_Assert(size == right_disp.size());
    CV_Assert(size == src_left.size());
    CV_Assert(left_disp.type() == CV_16SC1);
    CV_Assert(left_disp.type() == right_disp.type());
    CV_Assert(src_left.type() == CV_8UC1 || src_left.type() == CV_16UC1);

    switch (src_left.type())
    {
    case CV_8UC1:
        check_consistency<uint16_t, uint8_t>(left_disp, right_disp, src_left, size.width, size.height, subpixel, stream);
        break;
    case CV_16SC1:
    case CV_16UC1:
        check_consistency<uint16_t, uint16_t>(left_disp, right_disp, src_left, size.width, size.height, subpixel, stream);
        break;
    default:
        CV_Error(cv::Error::BadDepth, "Unsupported depth");
    }
}
} // namespace check_consistency

namespace correct_disparity_range
{
namespace
{
__global__ void correct_disparity_range_kernel(
    PtrStepSz<uint16_t> disp,
    int min_disp_scaled,
    int invalid_disp_scaled)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= disp.cols || y >= disp.rows)
    {
        return;
    }

    uint16_t d = disp(y, x);
    if (d == INVALID_DISP)
    {
        d = invalid_disp_scaled;
    }
    else
    {
        d += min_disp_scaled;
    }
    disp(y, x) = d;
}
} // anonymous namespace

void correctDisparityRange(
    GpuMat& disp,
    bool subpixel,
    int min_disp,
    Stream& _stream)
{
    CV_Assert(disp.type() == CV_16SC1);

    static constexpr int SIZE = 16;
    cv::Size size = disp.size();

    const dim3 blocks(cudev::divUp(size.width, SIZE), cudev::divUp(size.height, SIZE));
    const dim3 threads(SIZE, SIZE);
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

    const int scale = subpixel ? StereoSGBM::DISP_SCALE : 1;
    const int     min_disp_scaled =  min_disp      * scale;
    const int invalid_disp_scaled = (min_disp - 1) * scale;

    correct_disparity_range_kernel<<<blocks, threads, 0, stream>>>(disp, min_disp_scaled, invalid_disp_scaled);
}
} // namespace correct_disparity_range

} // namespace stereosgm
}}} // namespace cv { namespace cuda { namespace device {

#endif // HAVE_OPENCV_CUDEV
