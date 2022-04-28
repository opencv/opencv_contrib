// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if !defined CUDA_DISABLER

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda.hpp"

using namespace cv::cuda;
using namespace cv::cuda::device;


namespace cv { namespace cuda { namespace device { namespace imgproc {

constexpr int kblock_rows = 16;
constexpr int kblock_cols = 16;

namespace {
enum class Info : unsigned char { a = 0, b = 1, c = 2, d = 3, P = 4, Q = 5, R = 6, S = 7 };

// Only use it with unsigned numeric types
template <typename T>
__device__ __forceinline__ unsigned char HasBit(T bitmap, Info pos) {
    return (bitmap >> static_cast<unsigned char>(pos)) & 1;
}

template <typename T>
__device__ __forceinline__ unsigned char HasBit(T bitmap, unsigned char pos) {
    return (bitmap >> pos) & 1;
}

// Only use it with unsigned numeric types
__device__ __forceinline__ void SetBit(unsigned char& bitmap, Info pos) {
    bitmap |= (1 << static_cast<unsigned char>(pos));
}

// Returns the root index of the UFTree
__device__ unsigned Find(const int* s_buf, unsigned n) {
    while (s_buf[n] != n) {
        n = s_buf[n];
    }
    return n;
}

__device__ unsigned FindAndCompress(int* s_buf, unsigned n) {
    unsigned id = n;
    while (s_buf[n] != n) {
        n = s_buf[n];
        s_buf[id] = n;
    }
    return n;
}

// Merges the UFTrees of a and b, linking one root to the other
__device__ void Union(int* s_buf, unsigned a, unsigned b) {

    bool done;

    do {

        a = Find(s_buf, a);
        b = Find(s_buf, b);

        if (a < b) {
            int old = atomicMin(s_buf + b, a);
            done = (old == b);
            b = old;
        }
        else if (b < a) {
            int old = atomicMin(s_buf + a, b);
            done = (old == a);
            a = old;
        }
        else {
            done = true;
        }

    } while (!done);

}


__global__ void InitLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels, unsigned char* last_pixel) {
    unsigned row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    unsigned img_index = row * img.step + col;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        unsigned P = 0;

        // Bitmask representing two kinds of information
        // Bits 0, 1, 2, 3 are set if pixel a, b, c, d are foreground, respectively
        // Bits 4, 5, 6, 7 are set if block P, Q, R, S need to be merged to X in Merge phase
        unsigned char info = 0;

        char buffer alignas(int)[4];
        *(reinterpret_cast<int*>(buffer)) = 0;

        // Read pairs of consecutive values in memory at once
        if (col + 1 < img.cols) {
            // This does not depend on endianness
            *(reinterpret_cast<int16_t*>(buffer)) = *(reinterpret_cast<int16_t*>(img.data + img_index));

            if (row + 1 < img.rows) {
                *(reinterpret_cast<int16_t*>(buffer + 2)) = *(reinterpret_cast<int16_t*>(img.data + img_index + img.step));
            }
        }
        else {
            buffer[0] = img.data[img_index];

            if (row + 1 < img.rows) {
                buffer[2] = img.data[img_index + img.step];
            }
        }

        if (buffer[0]) {
            P |= 0x777;
            SetBit(info, Info::a);
        }
        if (buffer[1]) {
            P |= (0x777 << 1);
            SetBit(info, Info::b);
        }
        if (buffer[2]) {
            P |= (0x777 << 4);
            SetBit(info, Info::c);
        }
        if (buffer[3]) {
            SetBit(info, Info::d);
        }

        if (col == 0) {
            P &= 0xEEEE;
        }
        if (col + 1 >= img.cols) {
            P &= 0x3333;
        }
        else if (col + 2 >= img.cols) {
            P &= 0x7777;
        }

        if (row == 0) {
            P &= 0xFFF0;
        }
        if (row + 1 >= img.rows) {
            P &= 0x00FF;
        }
        else if (row + 2 >= img.rows) {
            P &= 0x0FFF;
        }

        // P is now ready to be used to find neighbor blocks
        // P value avoids range errors

        int father_offset = 0;

        // P square
        if (HasBit(P, 0) && img.data[img_index - img.step - 1]) {
            father_offset = -(2 * (labels.step / labels.elem_size) + 2);
        }

        // Q square
        if ((HasBit(P, 1) && img.data[img_index - img.step]) || (HasBit(P, 2) && img.data[img_index + 1 - img.step])) {
            if (!father_offset) {
                father_offset = -(2 * (labels.step / labels.elem_size));
            }
            else {
                SetBit(info, Info::Q);
            }
        }

        // R square
        if (HasBit(P, 3) && img.data[img_index + 2 - img.step]) {
            if (!father_offset) {
                father_offset = -(2 * (labels.step / labels.elem_size) - 2);
            }
            else {
                SetBit(info, Info::R);
            }
        }

        // S square
        if ((HasBit(P, 4) && img.data[img_index - 1]) || (HasBit(P, 8) && img.data[img_index + img.step - 1])) {
            if (!father_offset) {
                father_offset = -2;
            }
            else {
                SetBit(info, Info::S);
            }
        }

        labels.data[labels_index] = labels_index + father_offset;
        if (col + 1 < labels.cols) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + 1);
        }
        else if (row + 1 < labels.rows) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + labels.step / labels.elem_size);
        }
        *last_pixel = info;
    }
}

__global__ void Merge(cuda::PtrStepSzi labels, unsigned char* last_pixel) {

    unsigned row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        if (col + 1 < labels.cols) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + 1);
        }
        else if (row + 1 < labels.rows) {
            last_pixel = reinterpret_cast<unsigned char*>(labels.data + labels_index + labels.step / labels.elem_size);
        }
        unsigned char info = *last_pixel;

        if (HasBit(info, Info::Q)) {
            Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size));
        }
        if (HasBit(info, Info::R)) {
            Union(labels.data, labels_index, labels_index - 2 * (labels.step / labels.elem_size) + 2);
        }
        if (HasBit(info, Info::S)) {
            Union(labels.data, labels_index, labels_index - 2);
        }
    }
}

__global__ void Compression(cuda::PtrStepSzi labels) {
    unsigned row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {
        FindAndCompress(labels.data, labels_index);
    }
}

__global__ void FinalLabeling(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {

    unsigned row = (blockIdx.y * kblock_rows + threadIdx.y) * 2;
    unsigned col = (blockIdx.x * kblock_cols + threadIdx.x) * 2;
    unsigned labels_index = row * (labels.step / labels.elem_size) + col;

    if (row < labels.rows && col < labels.cols) {

        int label;
        unsigned char info;
        unsigned long long buffer;

        if (col + 1 < labels.cols) {
            buffer = *reinterpret_cast<unsigned long long*>(labels.data + labels_index);
            label = (buffer & (0xFFFFFFFF)) + 1;
            info = (buffer >> 32) & 0xFFFFFFFF;
        }
        else {
            label = labels[labels_index] + 1;
            if (row + 1 < labels.rows) {
                info = labels[labels_index + labels.step / labels.elem_size];
            }
            else {
                // Read from the input image
                // "a" is already in position 0
                info = img[row * img.step + col];
            }
        }

        if (col + 1 < labels.cols) {
            *reinterpret_cast<unsigned long long*>(labels.data + labels_index) =
                (static_cast<unsigned long long>(HasBit(info, Info::b) * label) << 32) | (HasBit(info, Info::a) * label);

            if (row + 1 < labels.rows) {
                *reinterpret_cast<unsigned long long*>(labels.data + labels_index + labels.step / labels.elem_size) =
                    (static_cast<unsigned long long>(HasBit(info, Info::d) * label) << 32) | (HasBit(info, Info::c) * label);
            }
        }
        else {
            labels[labels_index] = HasBit(info, Info::a) * label;

            if (row + 1 < labels.rows) {
                labels[labels_index + (labels.step / labels.elem_size)] = HasBit(info, Info::c) * label;
            }
        }

    }

}

}


void BlockBasedKomuraEquivalence(const cv::cuda::GpuMat& img, cv::cuda::GpuMat& labels) {

    dim3 grid_size;
    dim3 block_size;
    unsigned char* last_pixel;
    bool last_pixel_allocated;

    last_pixel_allocated = false;
    if ((img.rows == 1 || img.cols == 1) && !((img.rows + img.cols) % 2)) {
        cudaSafeCall(cudaMalloc(&last_pixel, sizeof(unsigned char)));
        last_pixel_allocated = true;
    }
    else {
        last_pixel = labels.data + ((labels.rows - 2) * labels.step) + (labels.cols - 2) * labels.elemSize();
    }

    grid_size = dim3((((img.cols + 1) / 2) - 1) / kblock_cols + 1, (((img.rows + 1) / 2) - 1) / kblock_rows + 1, 1);
    block_size = dim3(kblock_cols, kblock_rows, 1);

    InitLabeling << <grid_size, block_size >> > (img, labels, last_pixel);
    cudaSafeCall(cudaGetLastError());

    Compression << <grid_size, block_size >> > (labels);
    cudaSafeCall(cudaGetLastError());

    Merge << <grid_size, block_size >> > (labels, last_pixel);
    cudaSafeCall(cudaGetLastError());

    Compression << <grid_size, block_size >> > (labels);
    cudaSafeCall(cudaGetLastError());

    FinalLabeling << <grid_size, block_size >> > (img, labels);
    cudaSafeCall(cudaGetLastError());

    if (last_pixel_allocated) {
        cudaSafeCall(cudaFree(last_pixel));
    }
    cudaSafeCall(cudaDeviceSynchronize());

}

}}}}


#endif /* CUDA_DISABLER */
