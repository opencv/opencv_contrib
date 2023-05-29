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

constexpr int blockSizeX = 16;
constexpr int blockSizeY = 16;
constexpr int momentsSize = sizeof(cv::Moments) / sizeof(double);

constexpr int m00 = offsetof(cv::Moments, m00) / sizeof(double);
constexpr int m10 = offsetof(cv::Moments, m10) / sizeof(double);
constexpr int m01 = offsetof(cv::Moments, m01) / sizeof(double);
constexpr int m20 = offsetof(cv::Moments, m20) / sizeof(double);
constexpr int m11 = offsetof(cv::Moments, m11) / sizeof(double);
constexpr int m02 = offsetof(cv::Moments, m02) / sizeof(double);
constexpr int m30 = offsetof(cv::Moments, m30) / sizeof(double);
constexpr int m21 = offsetof(cv::Moments, m21) / sizeof(double);
constexpr int m12 = offsetof(cv::Moments, m12) / sizeof(double);
constexpr int m03 = offsetof(cv::Moments, m03) / sizeof(double);

constexpr int mu20 = offsetof(cv::Moments, mu20) / sizeof(double);
constexpr int mu11 = offsetof(cv::Moments, mu11) / sizeof(double);
constexpr int mu02 = offsetof(cv::Moments, mu02) / sizeof(double);
constexpr int mu30 = offsetof(cv::Moments, mu30) / sizeof(double);
constexpr int mu21 = offsetof(cv::Moments, mu21) / sizeof(double);
constexpr int mu12 = offsetof(cv::Moments, mu12) / sizeof(double);
constexpr int mu03 = offsetof(cv::Moments, mu03) / sizeof(double);

__global__ void ComputeSpatialMoments(const cuda::PtrStepSzb img, bool binary, double* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
      const unsigned int img_index = y * img.step + x;
      const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
      if (val > 0) {
        const unsigned long x2 = x * x, x3 = x2 * x;
        const unsigned long y2 = y * y, y3 = y2 * y;

        atomicAdd(&moments[m00],           val);
        atomicAdd(&moments[m10], x       * val);
        atomicAdd(&moments[m01],      y  * val);
        atomicAdd(&moments[m20], x2      * val);
        atomicAdd(&moments[m11], x  * y  * val);
        atomicAdd(&moments[m02],      y2 * val);
        atomicAdd(&moments[m30], x3      * val);
        atomicAdd(&moments[m21], x2 * y  * val);
        atomicAdd(&moments[m12], x  * y2 * val);
        atomicAdd(&moments[m03],      y3 * val);
      }
    }
}

__global__ void ComputeCenteroid(const double* moments, double2* centroid) {
    centroid->x = moments[m10] / moments[m00];
    centroid->y = moments[m01] / moments[m00];
}

__global__ void ComputeCenteralMoments(const cuda::PtrStepSzb img, bool binary,
                                       const double2* centroid, double* moments) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < img.rows && x < img.cols) {
      const unsigned int img_index = y * img.step + x;
      const unsigned char val = (!binary || img.data[img_index] == 0) ? img.data[img_index] : 1;
      if (val > 0) {
        const double x1 = x - centroid->x, x2 = x1 * x1, x3 = x2 * x1;
        const double y1 = y - centroid->y, y2 = y1 * y1, y3 = y2 * y1;

        atomicAdd(&moments[mu20], x2      * val);
        atomicAdd(&moments[mu11], x1 * y1 * val);
        atomicAdd(&moments[mu02],      y2 * val);
        atomicAdd(&moments[mu30], x3      * val);
        atomicAdd(&moments[mu21], x2 * y1 * val);
        atomicAdd(&moments[mu12], x1 * y2 * val);
        atomicAdd(&moments[mu03],      y3 * val);
      }
    }
}

void ComputeCenteralNormalizedMoments(cv::Moments& moments_cpu) {
    const double m00_pow2 = pow(moments_cpu.m00, 2), m00_pow2p5 = pow(moments_cpu.m00, 2.5);

    moments_cpu.nu20 = moments_cpu.mu20 / m00_pow2;
    moments_cpu.nu11 = moments_cpu.mu11 / m00_pow2;
    moments_cpu.nu02 = moments_cpu.mu02 / m00_pow2;
    moments_cpu.nu30 = moments_cpu.mu30 / m00_pow2p5;
    moments_cpu.nu21 = moments_cpu.mu21 / m00_pow2p5;
    moments_cpu.nu12 = moments_cpu.mu12 / m00_pow2p5;
    moments_cpu.nu03 = moments_cpu.mu03 / m00_pow2p5;
}

cv::Moments Moments(const cv::cuda::GpuMat& img, bool binary) {
    const dim3 blockSize(blockSizeX, blockSizeY, 1);
    const dim3 gridSize((img.cols + blockSize.x - 1) / blockSize.x,
                        (img.rows + blockSize.y - 1) / blockSize.y, 1);

    double2* centroid;
    cudaSafeCall(cudaMalloc(&centroid, sizeof(double2)));
    cv::cuda::GpuMat moments_gpu(1, momentsSize, CV_64F, cv::Scalar(0));
    ComputeSpatialMoments <<<gridSize, blockSize>>>(img, binary, moments_gpu.ptr<double>(0));
    cudaSafeCall(cudaGetLastError());

    ComputeCenteroid <<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(moments_gpu.ptr<double>(0), centroid);
    cudaSafeCall(cudaGetLastError());

    ComputeCenteralMoments <<<gridSize, blockSize>>>(img, binary, centroid, moments_gpu.ptr<double>(0));
    cudaSafeCall(cudaFree(centroid));
    cudaSafeCall(cudaGetLastError());

    cv::Moments moments_cpu;
    cv::Mat moments_map(1, momentsSize, CV_64F, reinterpret_cast<double*>(&moments_cpu));
    moments_gpu.download(moments_map);
    cudaSafeCall(cudaDeviceSynchronize());

    ComputeCenteralNormalizedMoments(moments_cpu);

    return moments_cpu;
}

}}}}


#endif /* CUDA_DISABLER */
