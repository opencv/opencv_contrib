/*
 * Copyright 2014 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "opencv2/graphmodels/cudamat/cudamat_conv.cuh"

#include "cudamat_conv_util.cuh"

__device__ inline float square(const float a) {
    return a * a;
}

__global__ void kTile(const float* src, float* tgt, const uint srcWidth, const uint srcHeight, const uint tgtWidth, const uint tgtHeight);
__global__ void kDotProduct_r(float* a, float* b, float* target, const uint numElements);

template<typename T> 
__device__ T shfl_down(T a, int b, int c=WARP_SIZE) {
#if __CUDA_ARCH__ >= 300
    return __shfl_down(a, b, c);
#else
    return 0;
#endif
}



/*
 * Horizontal reflection.
 * imgs:    (numColors, imgSize, imgSize, numCases)
 * targets: (numColors, imgSize, imgSize, numCases)
 * 
 * targets should be a different array from imgs.
 * 
 * Block size: (4, 32)
 * blockIdx.y * 4 + threadIdx.y determines pixel
 * blockIdx.x * 32 * imgsPerThread + threadIdx.x determines case batch
 * 
 */
template<int numColors, int imgsPerThread, bool checkCaseBounds>
__global__ void kReflectH(float * imgs, float * targets,
                          const int imgSize, const int numCases) {
    const int pxIdx = blockIdx.y * 4 + threadIdx.y;
    const int imgPixels = imgSize * imgSize;
    
    if (pxIdx < imgPixels) {
        const int caseIdx = blockIdx.x * 32 * imgsPerThread + threadIdx.x;
        const int pxIdxY = pxIdx / imgSize;
        const int pxIdxX = pxIdx % imgSize;

        const int pxIdxXR = imgSize - 1 - pxIdxX; // reflected coordinate
        const int pxIdxR = pxIdxY * imgSize + pxIdxXR;

        imgs += pxIdx * numCases + caseIdx;
        targets += pxIdxR * numCases + caseIdx;

#pragma unroll
        for (int i = 0; i < imgsPerThread; ++i) {
            if (!checkCaseBounds || caseIdx + i * 32 < numCases) {
#pragma unroll
                for (int c = 0; c < numColors; ++c) {
                    targets[c * imgPixels * numCases + i * 32] = imgs[c * imgPixels * numCases + i * 32];
                }
            }
        }
    }
}
/*
 * Horizontal reflection.
 * imgs:    (numColors, imgSize, imgSize, numCases)
 * targets: (numColors, imgSize, imgSize, numCases)
 */
void convReflectHorizontal(cudamat* images, cudamat* targets, int imgSize) {
    int numCases = images->size[0]; 
    int imgPixels = imgSize * imgSize;
    int numColors = images->size[1] / imgPixels;
    assert(numColors * imgPixels == images->size[1]);
    assert(numColors > 0 && numColors <= 3);
    
    //targets.resize(images);
    int imgsPerThread = numCases % 128 == 0 ? 4 : numCases % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numCases % (32 * imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numCases, imgsPerThread * 32), DIVUP(imgPixels, 4));
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (checkCaseBounds) {
        if (numColors == 1) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<1, 1, true>, cudaFuncCachePreferL1);
                kReflectH<1, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<1, 2, true>, cudaFuncCachePreferL1);
                kReflectH<1, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<1, 4, true>, cudaFuncCachePreferL1);
                kReflectH<1, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        } else if (numColors == 2) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<2, 1, true>, cudaFuncCachePreferL1);
                kReflectH<2, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<2, 2, true>, cudaFuncCachePreferL1);
                kReflectH<2, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<2, 4, true>, cudaFuncCachePreferL1);
                kReflectH<2, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        } else if (numColors == 3) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<3, 1, true>, cudaFuncCachePreferL1);
                kReflectH<3, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<3, 2, true>, cudaFuncCachePreferL1);
                kReflectH<3, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<3, 4, true>, cudaFuncCachePreferL1);
                kReflectH<3, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        }
    } else {
        if (numColors == 1) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<1, 1, false>, cudaFuncCachePreferL1);
                kReflectH<1, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<1, 2, false>, cudaFuncCachePreferL1);
                kReflectH<1, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<1, 4, false>, cudaFuncCachePreferL1);
                kReflectH<1, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        } else if (numColors == 2) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<2, 1, false>, cudaFuncCachePreferL1);
                kReflectH<2, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<2, 2, false>, cudaFuncCachePreferL1);
                kReflectH<2, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<2, 4, false>, cudaFuncCachePreferL1);
                kReflectH<2, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        } else if (numColors == 3) {
            if (imgsPerThread == 1) {
                cudaFuncSetCacheConfig(kReflectH<3, 1, false>, cudaFuncCachePreferL1);
                kReflectH<3, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 2) {
                cudaFuncSetCacheConfig(kReflectH<3, 2, false>, cudaFuncCachePreferL1);
                kReflectH<3, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            } else if (imgsPerThread == 4) {
                cudaFuncSetCacheConfig(kReflectH<3, 4, false>, cudaFuncCachePreferL1);
                kReflectH<3, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, targets->data_device, imgSize, numCases);
            }
        }
    }
    getLastCudaError("kReflectH: kernel execution failed");
}

/*
 * blockIdx.y determines module in batches of B_Y
 * blockIdx.x determines filter in batches of B_X * filtersPerThread
 * 
 * weights: (numModules, numColors, filterPixels, numFilters)
 * Not fully coalesced if B_X < 32, so use cache.
 */
template <int B_Y, int B_X, int filtersPerThread>
__global__ void kNormalizeLCWeights(float* weights, const uint numFilters, const int numModules, const uint weightsPerFilter, const float norm) {
    const uint moduleIdx = B_Y * blockIdx.y + threadIdx.y;
    const uint filterIdx = B_X * blockIdx.x + threadIdx.x;
    
    float prod[filtersPerThread];
    #pragma unroll
    for (uint i = 0; i < filtersPerThread; ++i) {
        prod[i] = 0;
    }
    if (moduleIdx < numModules) {
        weights += moduleIdx * weightsPerFilter * numFilters + filterIdx;
        for (uint p = 0; p < weightsPerFilter; ++p) {
            #pragma unroll
            for (uint i = 0; i < filtersPerThread; ++i) {
                prod[i] += square(weights[p * numFilters + i * B_X]);
            }
        }
        
        #pragma unroll
        for (uint i = 0; i < filtersPerThread; ++i) {
            prod[i] = sqrtf(prod[i]);
            prod[i] = prod[i] > norm ? __fdividef(norm, prod[i]) : 1.0f;
        }
        
        for (uint p = 0; p < weightsPerFilter; ++p) {
            #pragma unroll
            for (uint i = 0; i < filtersPerThread; ++i) {
                weights[p * numFilters + i * B_X] *= prod[i];
            }
        }
    }
}

/*
 * weights: (numModules, numColors, filterPixels, numFilters)
 */
void normalizeLocalWeights(cudamat* weights, int numModules, float norm) {
    int numFilters = weights->size[0];
    int weightsPerFilter = weights->size[1] / numModules;
    assert(numModules * weightsPerFilter == weights->size[1]);
    
    assert(!weights->is_trans);
   // assert(weights.isContiguous());
    assert(numFilters % 16 == 0);
    
    int bx = numFilters % 32 == 0 ? 32 : 16;
    int by = bx == 32 ? 4 : 8;
    
    int filtersPerThread = numFilters % 128 == 0 ? 4 : numFilters % 64 == 0 ? 2 : 1;
    dim3 blocks(numFilters / (bx * filtersPerThread), DIVUP(numModules, by));
    dim3 threads(bx, by);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (filtersPerThread == 4) {
        cudaFuncSetCacheConfig(kNormalizeLCWeights<4, 32, 4>, cudaFuncCachePreferL1);
        kNormalizeLCWeights<4, 32, 4><<<blocks, threads, 0, stream>>>(weights->data_device, numFilters, numModules, weightsPerFilter, norm);
    } else if (filtersPerThread == 2) {
        cudaFuncSetCacheConfig(kNormalizeLCWeights<4, 32, 2>, cudaFuncCachePreferL1);
        kNormalizeLCWeights<4, 32, 2><<<blocks, threads, 0, stream>>>(weights->data_device, numFilters, numModules, weightsPerFilter, norm);
    } else {
        if (numFilters % 32 == 0) {
            cudaFuncSetCacheConfig(kNormalizeLCWeights<4, 32, 1>, cudaFuncCachePreferL1);
            kNormalizeLCWeights<4, 32, 1><<<blocks, threads, 0, stream>>>(weights->data_device, numFilters, numModules, weightsPerFilter, norm);
        } else {
            cudaFuncSetCacheConfig(kNormalizeLCWeights<8, 16, 1>, cudaFuncCachePreferL1);
            kNormalizeLCWeights<8, 16, 1><<<blocks, threads, 0, stream>>>(weights->data_device, numFilters, numModules, weightsPerFilter, norm);
        }
    }
}

/*
 * Block size 4x32
 * blockIdx.x determines img idx in batches of 32*imgsPerThread
 * blockIdx.y determines channel idx, pixel idx in batches of 4
 * 
 * threadIdx.x determins case idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:    (numChannels, imgPixels, numImages) with given imgStride
 * target:  (numChannels, tgtPixels, numImages)
 */
template <int imgsPerThread, bool checkCaseBounds>
__global__ void kCrop(float* imgs, float* target, const uint numImages, const int imgStride,
                      const uint imgSize, const uint tgtSize, const uint startY, const uint startX) {
    const uint imgPixels = imgSize * imgSize;
    const uint tgtPixels = tgtSize * tgtSize;
    const uint caseIdx = blockIdx.x * 32 * imgsPerThread + threadIdx.x;
    const uint blockChanIdx = blockIdx.y / DIVUP(tgtPixels, 4);
    const uint tgtPixelIdx = 4*(blockIdx.y % DIVUP(tgtPixels, 4)) + threadIdx.y;
    const uint tgtPxY = tgtPixelIdx / tgtSize;
    const uint tgtPxX = tgtPixelIdx % tgtSize;
    const uint srcPixelIdx = (startY + tgtPxY) * imgSize + startX + tgtPxX;
    
    if (tgtPixelIdx < tgtPixels) {
        imgs += (blockChanIdx * imgPixels + srcPixelIdx) * imgStride + caseIdx;
        target += (blockChanIdx * tgtPixels + tgtPixelIdx) * numImages + caseIdx;

        #pragma unroll
        for (uint i = 0; i < imgsPerThread; ++i) {
            if (!checkCaseBounds || (caseIdx + 32 * i < numImages)) {
                target[i * 32] = imgs[i * 32];
            }
        }
    }
}

/*
 * Block size 4x32
 * blockIdx.y determines pixel idx in batches of 4
 * blockIdx.x determines case idx in batches of 32*imgsPerThread
 * threadIdx.y determines pixel idx
 * threadIdx.x determines case idx
 * 
 * imgs:        (3, imgPixels, numImages) with given imgStride
 * target:      (3, imgPixels, numImages)
 * 
 * Each thread produces (y,u,v) values for a particular (r,g,b) pixel
 * 
 * The RGB --> YUV transform is (http://en.wikipedia.org/wiki/YUV):
 * 
 * [Y]      [ 0.2126     0.7152      0.0722 ][R]
 * [U]  =   [-0.09991   -0.33609     0.436  ][G]
 * [V]      [ 0.615     -0.55861    -0.05639][B]
 */
template <int imgsPerThread, bool checkCaseBounds>
__global__ void kRGBToYUV(float* imgs, float* target, const int imgPixels, const int numImages, const int imgStride) {
    const int caseIdx = blockIdx.x * 32 * imgsPerThread + threadIdx.x;
    const int pxIdx = blockIdx.y * 4 + threadIdx.y;

    if (pxIdx < imgPixels) {
        const int imgChannelStride = imgPixels * imgStride;
        const int tgtChannelStride = imgPixels * numImages;
        imgs += pxIdx * imgStride + caseIdx;
        target += pxIdx * numImages + caseIdx;
        
        #pragma unroll
        for (int i = 0; i < imgsPerThread; ++i) {
            if (!checkCaseBounds || caseIdx + i * 32 < numImages) {
                const float R = imgs[0 * imgChannelStride + i * 32];
                const float G = imgs[1 * imgChannelStride + i * 32];
                const float B = imgs[2 * imgChannelStride + i * 32];
                target[0 * tgtChannelStride + i * 32] = 0.2126f * R + 0.7152f * G + 0.0722f * B;      // Y
                target[1 * tgtChannelStride + i * 32] = -0.09991f * R + -0.33609f * G + 0.436f * B;   // U
                target[2 * tgtChannelStride + i * 32] = 0.615f * R + -0.55861f * G + -0.05639f * B;   // V
            }
        }
    }
}

__device__ inline float labf(const float x) {
    if (x > 0.0088564517f) {
        return __powf(x, 0.3333f);
    }
    return 7.787037f * x + 0.13793103f;
}

/*
 * Block size 4x32
 * blockIdx.y determines pixel idx in batches of 4
 * blockIdx.x determines case idx in batches of 32*imgsPerThread
 * threadIdx.y determines pixel idx
 * threadIdx.x determines case idx
 * 
 * imgs:        (3, imgPixels, numImages) with given imgStride
 * target:      (3, imgPixels, numImages)
 * 
 * This proceeds in two steps.
 * 
 * - First, RGB values are linearly transformed to XYZ as per
 *   http://en.wikipedia.org/wiki/CIE_XYZ_color_space
 * - Second, XYZ values are nonlinearly transformed to L*a*b* as per
 *   http://en.wikipedia.org/wiki/Lab_color_space#The_forward_transformation
 * 
 * Each thread produces (L*,a*,b*) values for a particular (r,g,b) pixel
 * 
 * The RGB --> XYZ transform is:
 * 
 * [X]                  [0.49       0.31        0.2     ][R]
 * [Y]  =   5.6506753 * [0.17697    0.8124      0.01063 ][G]
 * [Z]                  [0          0.01        0.99    ][B]
 * 
 * NOTE: The input should be in the range 0-1. Don't do mean-subtraction beforehand.
 * 
 * Then X_max, Y_max, Z_max = 5.6506753.
 * 
 * The range of the L* values is [0, 100].
 * If the center flag is given, the range will be [-50, 50].
 * 
 */
template <int imgsPerThread, bool checkCaseBounds, bool center>
__global__ void kRGBToLAB(float* imgs, float* target, const int imgPixels, const int numImages, const int imgStride) {
    const int caseIdx = blockIdx.x * 32 * imgsPerThread + threadIdx.x;
    const int pxIdx = blockIdx.y * 4 + threadIdx.y;

    if (pxIdx < imgPixels) {
        const int imgChannelStride = imgPixels * imgStride;
        const int tgtChannelStride = imgPixels * numImages;
        imgs += pxIdx * imgStride + caseIdx;
        target += pxIdx * numImages + caseIdx;
        
        #pragma unroll
        for (int i = 0; i < imgsPerThread; ++i) {
            if (!checkCaseBounds || caseIdx + i * 32 < numImages) {
                const float R = imgs[0 * imgChannelStride + i * 32];
                const float G = imgs[1 * imgChannelStride + i * 32];
                const float B = imgs[2 * imgChannelStride + i * 32];
                
                const float X = (0.49f * R + 0.31f * G + 0.2f * B);
                const float Y = (0.17697f * R + 0.8124f * G + 0.01063f * B);
                const float Z = (0.01f * G + 0.99f * B);
                
                const float labX = labf(X);
                const float labY = labf(Y);
                const float labZ = labf(Z);
                
                target[0 * tgtChannelStride + i * 32] = 116.0f * labY - 16.0f - (center ? 50.0f : 0);  // L*
                target[1 * tgtChannelStride + i * 32] = 500.0f * (labX - labY); // a*
                target[2 * tgtChannelStride + i * 32] = 200.0f * (labY - labZ); // b*
            }
        }
    }
}

/*
 * Block size 16x32.
 * Each block produces a 4x4 chunk of the output image.
 * threadIdx.y determines pixel idx in 4x4 chunk.
 * threadIdx.x determines case idx.
 * blockIdx.x determines case idx in batches of 32*imgsPerThread.
 * blockIdx.y determines 4x4 chunk idx, channel idx.
 * 
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * target:      (numChannels, tgtPixels, numImages)
 * 
 * imgSize = scale * tgtSize (roughly)
 * 
 * This is a rather naive kernel that relies on cache for speed. But all it's doing
 * is basic texture manipulation, which is very local in nature, so it should be ok.
 * Also, it will in practice be a tiny fraction of the runtime of a large convnet.
 * 
 * So that is my justification for being lazy here.
 */
template <int imgsPerThread, bool checkCaseBounds>
__global__ void kResizeBilinear(float* imgs, float* target, const int imgSize, const int tgtSize,
                                const int numImages, const int imgStride, const float scale,
                                const float centerScale) {
    const int numChunksX = DIVUP(tgtSize, 4);
    const int numChunks = numChunksX * numChunksX;
    const int channelIdx = blockIdx.y / numChunks;
    const int chunkIdx = blockIdx.y % numChunks;
    const int chunkIdxX = chunkIdx % numChunksX;
    const int chunkIdxY = chunkIdx / numChunksX;
    const int caseIdx = blockIdx.x * 32 * imgsPerThread + threadIdx.x;
    const int imgPixels = imgSize * imgSize;
    const int tgtPixels = tgtSize * tgtSize;
    
    const int pxX = 4 * chunkIdxX + threadIdx.y % 4;
    const int pxY = 4 * chunkIdxY + threadIdx.y / 4;

    if (pxY < tgtSize && pxX < tgtSize) {
        const int pxIdx = pxY * tgtSize + pxX;

        imgs += channelIdx * imgPixels * imgStride + caseIdx;
        target += channelIdx * tgtPixels * numImages + pxIdx * numImages + caseIdx;

        // This will cause slight distortions at the edges when upsampling in some cases.
        // But I think that's not a big deal.
        const float srcPxX = fmaxf(0.0f, fminf(__int2float_rn(imgSize) - 1.01f, __int2float_rn(pxX) * scale + centerScale));
        const float srcPxY = fmaxf(0.0f, fminf(__int2float_rn(imgSize) - 1.01f, __int2float_rn(pxY) * scale + centerScale));

        const float u = floorf(srcPxX + 1) - srcPxX;
        const float w = srcPxY - floorf(srcPxY);

        // Consider doing max(0, min(imgSize, x)) here
        const int srcPx0 = (__float2int_rd(srcPxY) * imgSize + __float2int_rd(srcPxX)); // top-left
        const int srcPx1 = srcPx0 + 1; // top-right
        const int srcPx2 = srcPx0 + imgSize; // bottom-left
        const int srcPx3 = srcPx2 + 1; // bottom-right

        #pragma unroll
        for (int c = 0; c < imgsPerThread; ++c) {
            if (!checkCaseBounds || caseIdx + c * 32 < numImages) {
                const float val0 = imgs[srcPx0 * imgStride + c * 32];
                const float val1 = imgs[srcPx1 * imgStride + c * 32];
                const float val2 = imgs[srcPx2 * imgStride + c * 32];
                const float val3 = imgs[srcPx3 * imgStride + c * 32];

                const float c0 = u * (val0 - val1) + val1;
                const float c1 = u * (val2 - val3) + val3;

                target[32 * c] = w * (c1 - c0) + c0;
            }
        }
    }
}

/*
 * Block size B_YxB_X.
 * B_X*imgsPerThread*blockIdx.x + threadIdx.x determines img idx 
 * B_Y*blockIdx.y + threadIdx.y determines img row (col if !horiz), channel idx
 * 
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * filter:      (1, 2*radius + 1)
 * target:      (numChannels, imgPixels, numImages)
 * 
 * target can be the same matrix as imgs.
 * radius must be one of 3, 5, 7, 9.
 * 
 * Tried imgsPerThread, slower.
 */
template<int B_Y, int B_X, int radius>
__global__ void kGaussianBlur(float* imgs, float* filter, float* target, const int imgSize,
                              const int numImages, const int imgStride,
                              const bool horiz,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shFilter[radius];
    
    const int imgPixels = imgSize * imgSize;
    const int ty = B_Y * blockIdx.y + threadIdx.y;
    const int channelIdx = ty / imgSize;
    const int rowIdx = ty % imgSize;
    const int imgIdx = B_X*blockIdx.x + threadIdx.x;
    const int filterWidth = 2*radius+1;
//    const int tidx = B_Y * threadIdx.y + threadIdx.x;
    if (horiz) {
        imgs += channelIdx * imgPixels * imgStride + rowIdx * imgSize * imgStride + imgIdx;
        target += channelIdx * imgPixels * numImages + rowIdx * imgSize * numImages + imgIdx;
    } else {
        imgs += channelIdx * imgPixels * imgStride + rowIdx * imgStride + imgIdx;
        target += channelIdx * imgPixels * numImages + rowIdx * numImages + imgIdx;
    }
    float outputs[filterWidth-1];
    #pragma unroll
    for (int r = 0; r < filterWidth-1; r++) {
        outputs[r] = 0;
    }
    if (threadIdx.x < filterWidth-1) {
        shFilter[threadIdx.x] = filter[threadIdx.x];
    }
    __syncthreads();

    if (imgIdx < numImages) {
        // This writes radius*2 = filterWidth - 1 values to outputs 
        #pragma unroll
        for (int col = 0; col < radius; col++) {
            float px = imgs[0];
            #pragma unroll
            for (int r = 0; r < radius + 1 + col; r++) {
                outputs[r] += px * shFilter[radius + col - r];
            }
            imgs += horiz ? imgStride : imgStride * imgSize;
        }

        // Unfortunately this has to be at this level of granularity
        if (scaleTargets != 0) {
            for (int col = radius; col < imgSize ; col++) { // loop over img columns
                float px = imgs[0];
                target[0] = scaleTargets * target[0] + scaleOutputs * (outputs[0] + px * shFilter[0]);

                #pragma unroll
                for (int r = 1; r < radius*2; r++) {
                    outputs[r-1] = outputs[r] + px * shFilter[r];
                }
                outputs[filterWidth - 2] = px * shFilter[0];

                imgs += horiz ? imgStride : imgStride * imgSize;
                target += horiz ? numImages : numImages * imgSize;
            }

            #pragma unroll
            for (int r = 0; r < radius; r++) {
                float* t = &target[0];
                t[0] = scaleTargets * t[0] + scaleOutputs * outputs[r];
                target += horiz ? numImages : numImages * imgSize;
            }
        } else {
            for (int col = radius; col < imgSize ; col++) { // loop over img columns
                float px = imgs[0];
                target[0] = scaleOutputs * (outputs[0] + px * shFilter[0]);
                #pragma unroll
                for (int r = 1; r < radius*2; r++) {
                    outputs[r-1] = outputs[r] + px * shFilter[r];
                }
                outputs[filterWidth - 2] = px * shFilter[0];

                imgs += horiz ? imgStride : imgStride * imgSize;
                target += horiz ? numImages : numImages * imgSize;
            }

            #pragma unroll
            for (int r = 0; r < radius; r++) {
                target[0] = scaleOutputs * outputs[r];
                target += horiz ? numImages : numImages * imgSize;
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numChannels, imgPixels, numImages)
 * target:      (numChannels, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int chansPerThread, bool checkCaseBounds>
__global__ void kBedOfNails(float* imgs, float* target, const int imgSize, const int numChannels,
                           const int numImages, const int startX, const int strideX, const int outputsX,
                           const bool reverse, const float scaleTargets, const float scaleOutput) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numChanBlocks = DIVUP(numChannels, B_Y*chansPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numChanBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockChanIdx = (blockIdx.y % numChanBlocks) * B_Y * chansPerThread;
    const int myChanIdx = (blockChanIdx + threadIdx.y*chansPerThread);
    if (myChanIdx >= numChannels) {
        return;
    }
//    if (blockIdx.x != 0 || blockIdx.y != 0) {
//        return;
//    }
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    const int imgPx = startImgPxY * imgSize + startImgPxX;
    
    imgs += myChanIdx * imgPixels * numImages + imgPx * numImages + imgIdx;
    target += (myChanIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    if (scaleTargets != 0) {
        if (!reverse) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        target[c * numOutputs * numImages + i * B_X] = scaleTargets * target[c * numOutputs * numImages + i * B_X] + scaleOutput * imgs[c * imgPixels * numImages + i * B_X]; 
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        imgs[c * imgPixels * numImages + i * B_X] = scaleTargets * imgs[c * imgPixels * numImages + i * B_X] + scaleOutput * target[c * numOutputs * numImages + i * B_X]; 
                    }
                }
            }
        }
    } else {
        if (!reverse) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        target[c * numOutputs * numImages + i * B_X] = scaleOutput * imgs[c * imgPixels * numImages + i * B_X]; 
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int c = 0; c < chansPerThread; c++) {
                        imgs[c * imgPixels * numImages + i * B_X] = scaleOutput * target[c * numOutputs * numImages + i * B_X]; 
                    }
                }
            }
        }
    }

}

/*
 * imgs:        (numChannels, imgPixels, numImages)
 * target:      (numChannels, outputs, numImages)
 */
void _convBedOfNails(cudamat* images, cudamat* target, int numChannels, int imgSize, int startX, int strideX,
                     bool reverse, float scaleTargets, float scaleOutput) {
    int numImages = reverse ? target->size[0] : images->size[0];
    int imgPixels = imgSize * imgSize;

    assert(!images->is_trans);
    assert(!target->is_trans);
    //assert(images.isContiguous());
    //assert(target.isContiguous());
    assert(strideX > 1);
    
    int outputsX = DIVUP(imgSize, strideX);
    int outputs = outputsX * outputsX;
    if (reverse) {
        assert(target->size[1] == numChannels * outputs);
    } else  {
        assert(images->size[1] == numChannels * imgPixels);
    }
    
    //if (scaleTargets == 0) {
    //    if (reverse) {
    //        images.resize(numChannels * imgPixels, numImages);
     //       images.apply(NVMatrixOps::Zero());
    //    } else {
    //        target.resize(numChannels*outputs, numImages);
    //    }
    //} else {
        if (reverse) {
            assert(images->size[1] == numChannels * outputs);
            assert(images->size[0] == numImages);
        } else {
            assert(target->size[1] == numChannels * outputs);
            assert(target->size[0] == numImages);
        }
    //}
    
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    int chansPerThread = numChannels % 8 == 0 ? 2 : 1;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numChannels, 4 * chansPerThread) * outputsX);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (imgsPerThread == 4) {
        if (chansPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 4, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 4, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 4, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 4, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if (chansPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 2, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 2, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 2, 2, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 2, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 2, 2, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 2, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (chansPerThread == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 1, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 1, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 1, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                    imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                    reverse, scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kBedOfNails<4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                kBedOfNails<4, 32, 1, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                     imgSize, numChannels, numImages, startX, strideX, outputsX,
                                                                     reverse, scaleTargets, scaleOutput);
            }
        }
    }
}

/*
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * filter:      (1, 2*radius + 1)
 * target:      (numChannels, imgPixels, numImages)
 */
void convGaussianBlur(cudamat* images, cudamat* filter, cudamat* target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs) {
    int numImages = images->size[0];
    int radius = filter->size[0] / 2;
    int imgPixels = images->size[1] / numChannels;
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgPixels == imgSize * imgSize);
    assert(radius >= 1 && radius <= 4);
    assert(imgSize >= 2 * radius + 1);
    assert(filter->size[1] == 1);
    assert(images->size[1] == numChannels * imgPixels);
    assert(!images->is_trans);
    assert(!filter->is_trans);
    assert(!target->is_trans);
    //assert(target.isContiguous());
    //if (scaleTargets == 0) {
    //    target.resize(images);
    //} else {
        assert(target->size[0] == images->size[0] && target->size[1] == images->size[1]);
    //}

    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages, threads.x), DIVUP(numChannels*imgSize, threads.y));
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (radius == 1) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 1>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 1><<<blocks, threads, 0, stream>>>(images->data_device, filter->data_device, target->data_device,
                                                           imgSize, numImages, images->size[0], horiz, scaleTargets, scaleOutputs);

    } else if (radius == 2) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 2>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 2><<<blocks, threads, 0, stream>>>(images->data_device, filter->data_device, target->data_device,
                                                           imgSize, numImages, images->size[0], horiz, scaleTargets, scaleOutputs);

    } else if (radius == 3) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 3>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 3><<<blocks, threads, 0, stream>>>(images->data_device, filter->data_device, target->data_device,
                                                           imgSize, numImages, images->size[0], horiz, scaleTargets, scaleOutputs);
    } else if (radius == 4) {
        cudaFuncSetCacheConfig(kGaussianBlur<4, 32, 4>, cudaFuncCachePreferL1);
        kGaussianBlur<4, 32, 4><<<blocks, threads, 0, stream>>>(images->data_device, filter->data_device, target->data_device,
                                                           imgSize, numImages, images->size[0], horiz, scaleTargets, scaleOutputs);
    }
}

/*
 * Block size 1x128
 * blockIdx.x determines pixel.x, image idx in batches of 128*imgsPerThread
 * blockIdx.y determines pixel.y
 * 
 * So each block does one output for some number of images and all the fliters.
 * 
 * threadIdx.x determines img idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int imgsPerThread, int numFilters, bool checkCaseBounds>
__global__ void kCNorm_fewfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                                  const int numImages, const int sizeX, const float addScale, const float powScale, const float minDiv) {

    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, 128*imgsPerThread);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * 128 * imgsPerThread;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    
    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += pxIdx * numImages + imgIdx;
    denoms += pxIdx * numImages + imgIdx;
    meanDiffs  += imgIdx;
    target += pxIdx * numImages + imgIdx;
    
    float prod[numFilters][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
            #pragma unroll
            for (int f = 0; f < numFilters; f++) {
                prod[f][i] = 0; 
            }
        }
    }
    const int loopStartY = MAX(0, startPxY);
    const int loopStartX = MAX(0, startPxX);
    const int loopEndY = MIN(imgSize, startPxY + sizeX);
    const int loopEndX = MIN(imgSize, startPxX + sizeX);
        
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
                    #pragma unroll
                    for (int f = 0; f < numFilters; f++) {
                        prod[f][i] += square(meanDiffs[(f * imgPixels + imgPx) * numImages + i * 128]);
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * 128 < numImages) {
            #pragma unroll
            for (int f = 0; f < numFilters; f++) {
                prod[f][i] = minDiv + addScale * prod[f][i];
                denoms[f * imgPixels * numImages + i * 128] = prod[f][i];
                target[f * imgPixels * numImages + i * 128] = imgs[f * imgPixels * numImages + i * 128] * __powf(prod[f][i], -powScale);
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines filter idx in batches of B_Y*filtersPerThread
 * blockIdx.z determines pixel
 * 
 * So each block does one pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm_manyfilter(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                                  const int numFilters, const int numImages, const int sizeX, 
                                  const float addScale, const float powScale, const float minDiv) {
    const int imgPixels = imgSize * imgSize;

    const int pxIdxX = blockIdx.z % imgSize;
    const int pxIdxY = blockIdx.z / imgSize;
    const int blockImgIdx = blockIdx.x * B_X * imgsPerThread;
    const int blockFilterIdx = blockIdx.y * B_Y * filtersPerThread;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    
    const int startPxX = -sizeX/2 + pxIdxX;
    const int startPxY = -sizeX/2 + pxIdxY;
    const int imgIdx = blockImgIdx + threadIdx.x;
    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    meanDiffs += (blockFilterIdx + threadIdx.y) * imgPixels * numImages + imgIdx;
    denoms += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = 0;
            }
        }
    }

    const int loopStartY = max(0, startPxY);
    const int loopStartX = max(0, startPxX);
    const int loopEndY = min(imgSize, startPxY + sizeX);
    const int loopEndX = min(imgSize, startPxX + sizeX);
    
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        prod[f][i] += square(meanDiffs[(f * B_Y * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[f][i] = minDiv + addScale * prod[f][i];
                denoms[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                target[f * B_Y * imgPixels * numImages + i * B_X] = imgs[f * B_Y * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
            }
        }
    }
}


/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region of pixels for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * means:       (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kCNorm2(float* imgs, float* meanDiffs, float* denoms, float* target, const int imgSize,
                         const int numFilters, const int numImages, const int sizeX, const float addScale, const float powScale, const float minDiv) {
    __shared__ float shDiffs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int startPxX = MAX(0, -sizeX/2 + blockPxX);
    const int startPxY = MAX(0, -sizeX/2 + blockPxY);
    const int endPxX = MIN(imgSize, blockPxX + DIVUP(sizeX, 2) + 3);
    const int endPxY = MIN(imgSize, blockPxY + DIVUP(sizeX, 2) + 3);
    
    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -sizeX/2 + myPxY;
    const int myStartPxX = -sizeX/2 + myPxX;
    const int myEndPxY = myPxY + DIVUP(sizeX, 2);
    const int myEndPxX = myPxX + DIVUP(sizeX, 2);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
        
    imgs        += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    meanDiffs   += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                prod[f][i] = 0;
            }
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        const bool isInY = y >= myStartPxY && y < myEndPxY;
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shDiffs[ly + loadY][lx + loadX] = meanDiffs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();
            
            // Each row of threads decides if it's interested in this pixel
            if (isInY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += square(shDiffs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
//    imgs -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
//    imgs += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    prod[f][i] = minDiv + addScale * prod[f][i];
                    denoms[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    target[f * imgPixels * numImages + i * B_X] = imgs[f * imgPixels * numImages + i * B_X] * __powf(prod[f][i], -powScale);
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 *
 * So each block does one pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by B_Y
 */
template<int B_Y, int B_X, int imgsPerThread, bool checkCaseBounds, bool blocked>
__global__ void kFCNorm(cudaTextureObject_t imgs, cudaTextureObject_t meanDiffs, float* target, const int imgSize,
                          const int numFilters, const int numImages, const int sizeF,
                          const float addScale, const float powScale, const float minDiv) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;
    
    const int pxIdx = pxIdxY * imgSize + pxIdxX;

    
    const int imgIdx = blockImgIdx + threadIdx.x;
    const int imgOffset = ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    const int meanDiffsOffset = pxIdx * numImages + imgIdx;
//    imgs += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
//    meanDiffs += pxIdx * numImages + imgIdx;
    target += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    
    float prod[imgsPerThread];
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = 0;
        }
    }

    const int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF/2 + filterIdx;
    const int loopStartF = blocked ? startF : MAX(0, startF);
    const int loopEndF = MIN(numFilters, startF + sizeF);
 
    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] += square(tex1Dfetch<float>(meanDiffs, meanDiffsOffset + f * imgPixels * numImages + i * B_X));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            prod[i] = minDiv + addScale * prod[i];
            target[i * B_X] = tex1Dfetch<float>(imgs, imgOffset + i * B_X) * __powf(prod[i], -powScale);
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:                (numFilters, imgPixels, numImages)
 * maxGrads:            (numOutputs, imgPixels, numImages)
 * maxActs:             (numOutputs, imgPixels, numImages)
 * target:              (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y
 * 
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, bool add, bool checkCaseBounds>
__global__ void kCrossMapMaxPoolUndo(float* imgs, float* maxGrads, float* maxActs, float* target, const int imgSize, const int numFilters,
                                     const int numImages, const int startF, const int poolSize,
                                     const int numOutputs, const int stride, const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
//    const int numOutputs = DIVUP(numFilters, stride);
    const int numFilterBlocks = numFilters/B_Y;

    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;

    const int imgPixels = imgSize * imgSize;
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;

    imgs            += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;
    maxGrads        += (/*(filterIdx) * imgPixels +*/ pxIdx) * numImages + imgIdx;
    maxActs         += (/*(filterIdx) * imgPixels +*/ pxIdx) * numImages + imgIdx;
    target          += ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;

    float prod[imgsPerThread];
//    if (imgIdx != 0 || pxIdx != 0 || filterIdx != 0) {
//        return;
//    }
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        prod[i] = 0;
    }

    if (filterIdx < numFilters) {
//        const int startOut = max(0, (filterIdx-startF-poolSize)/ stride + 1);
        const int loopStartOut = max(0, (filterIdx-startF-poolSize)/ stride + 1);
        const int loopEndOut = min(numOutputs, (filterIdx - startF)/ stride + 1);

        for (int o = loopStartOut; o < loopEndOut; ++o) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    const float ma = maxActs[o * imgPixels * numImages + i * B_X];
                    const float mg = maxGrads[o * imgPixels * numImages + i * B_X];
                    const float img = imgs[i*B_X];
                    prod[i] += (img == ma) * mg;
                }
            }
        }
    //    printf("gpu f start: %d, end: %d\n", loopStartF, loopEndF);

        if (!add) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    target[i * B_X] = prod[i];
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    target[i * B_X] = scaleTargets * target[i * B_X] + scaleOutputs * prod[i];
                }
            }
        }
    }
}

/*
 * images:              (numFilters, imgPixels, numImages)
 * maxGrads:            (numOutputs, imgPixels, numImages)
 * maxActs:             (numOutputs, imgPixels, numImages)
 * target:              (numFilters, imgPixels, numImages)
 */
void convCrossMapMaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* target,
                             const int imgSize, const int startF, const int poolSize,
                             const int stride, const float scaleTargets, const float scaleOutputs) {
    int numImages = images->size[0];
    int imgPixels = imgSize * imgSize;
    int numFilters = images->size[1] / imgPixels;
    int numOutputs = maxActs->size[1] / imgPixels;
    assert(images->size[1] == numFilters * imgPixels);
    assert(maxGrads->size[1] == numOutputs * imgPixels);
    assert(maxGrads->size[0] == numImages);
    assert(maxGrads->size[0] == maxActs->size[0] && maxGrads->size[1] == maxActs->size[1]);

    assert(images->size[1] == numFilters * imgPixels);

    assert(!images->is_trans);
    assert(!target->is_trans);
    assert(!maxGrads->is_trans);
    assert(!maxActs->is_trans);
    //assert(images.isContiguous());
    //assert(maxGrads.isContiguous());
    //assert(maxActs.isContiguous());
//    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);

    assert(stride <= poolSize);
    assert(startF <= 0);
    assert(startF + (numOutputs-1) * stride + poolSize >= numFilters); // All filters must be covered

    dim3 threads(32, 4);

    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    dim3 blocks(imgSize * DIVUP(numImages, threads.x * imgsPerThread), imgSize * DIVUP(numFilters, threads.y));
    bool checkCaseBounds = numImages % (threads.x*imgsPerThread) != 0;

    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    assert(target->size[0] == images->size[0] && target->size[0] == images->size[1]);
    if (scaleTargets == 0) {
        //target.resize(images);
        if (!checkCaseBounds) {
            if (imgsPerThread == 4) {
                kCrossMapMaxPoolUndo<4, 32, 4, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            } else if (imgsPerThread == 2) {
                kCrossMapMaxPoolUndo<4, 32, 2, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            } else {
                kCrossMapMaxPoolUndo<4, 32, 1, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            }
        } else {
            kCrossMapMaxPoolUndo<4, 32, 1, false, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                         imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                         scaleTargets, scaleOutputs);
        }
    } else {
        if (!checkCaseBounds) {
            if (imgsPerThread == 4) {
                kCrossMapMaxPoolUndo<4, 32, 4, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            } else if (imgsPerThread == 2) {
                kCrossMapMaxPoolUndo<4, 32, 2, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            } else {
                kCrossMapMaxPoolUndo<4, 32, 1, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                             imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                             scaleTargets, scaleOutputs);
            }
        } else {
            kCrossMapMaxPoolUndo<4, 32, 1, true, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                                         imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride,
                                                                                         scaleTargets, scaleOutputs);
        }
    }
    getLastCudaError("convCrossMapMaxPoolUndo: kernel execution failed");
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 *
 * So each block does one output pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y
 *
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, bool add, bool checkCaseBounds, bool blocked>
__global__ void kFRNormUndo(cudaTextureObject_t outGrads, cudaTextureObject_t denoms, cudaTextureObject_t inputs, cudaTextureObject_t acts,
                            float* target, const int imgSize, const int numFilters, const int numImages, const int sizeF, const float powScale,
                            const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;
    
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;
    
    const int imgPixels = imgSize * imgSize;
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    const int actsOffset = pxIdx * numImages + imgIdx;
    const int inputOffset = ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;

    target      += inputOffset;
    
    float prod[imgsPerThread];

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        prod[i] = 0;
    }
    
    const int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF + sizeF/2 + 1 + filterIdx;
    const int loopStartF = blocked ? startF : MAX(0, startF);
    const int loopEndF = MIN(numFilters, startF + sizeF);
    
    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] += tex1Dfetch<float>(acts, actsOffset + f * imgPixels * numImages + i * B_X);
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, inputOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, inputOffset + i * B_X);
                const float den = tex1Dfetch<float>(denoms, inputOffset + i * B_X);
                prod[i] = inp * prod[i] + out * __powf(den, -powScale);
                target[i * B_X] = prod[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, inputOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, inputOffset + i * B_X);
                const float den = tex1Dfetch<float>(denoms, inputOffset + i * B_X);
                prod[i] = inp * prod[i] + out * __powf(den, -powScale);
                target[i * B_X] = scaleTargets * target[i * B_X] + scaleOutputs * prod[i];
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y
 *
 * So each block does one output pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 *
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y
 *
 * TODO: this is pretty wasteful of computation. a lot of threads basically compute the same products.
 */
template<int B_Y, int B_X, int imgsPerThread, bool add, bool checkCaseBounds, bool blocked>
//__launch_bounds__(128,16)
__global__ void kFRNormUndo2(cudaTextureObject_t outGrads, cudaTextureObject_t inputs, cudaTextureObject_t acts, float* target, const int imgSize, const int numFilters,
                            const int numImages, const int sizeF, const float addScale, const float powScale, const float minDiv,
                            const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/B_Y;

    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int filterIdx = (blockIdx.y % numFilterBlocks) * B_Y + threadIdx.y;

    const int imgPixels = imgSize * imgSize;
    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;

    const int inpOffset = pxIdx * numImages + imgIdx;
    const int outOffset = ((filterIdx) * imgPixels + pxIdx) * numImages + imgIdx;

    target      += outOffset;

    float prod[imgsPerThread];
    float denoms[imgsPerThread];

    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        prod[i] = 0;
        denoms[i] = 0;
    }

    int startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF + sizeF/2 + 1 + filterIdx;
    int loopStartF = blocked ? startF : MAX(0, startF);
    int loopEndF = MIN(numFilters, startF + sizeF);

    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                // If an input is zero, then we shuldn't divide by it.
                const float grad = tex1Dfetch<float>(outGrads, inpOffset + f * imgPixels * numImages + i * B_X);
                const float act = tex1Dfetch<float>(acts, inpOffset + f * imgPixels * numImages + i * B_X);
                const float inp = tex1Dfetch<float>(inputs, inpOffset + f * imgPixels * numImages + i * B_X) + (act == 0);
                prod[i] += grad * act * __powf(__fdividef(act, inp), 1.0f/powScale);
            }
        }
    }

    startF = blocked ? (filterIdx / sizeF) * sizeF : -sizeF/2 + filterIdx;
    loopStartF = blocked ? startF : MAX(0, startF);
    loopEndF = MIN(numFilters, startF + sizeF);

    for (int f = loopStartF; f < loopEndF; ++f) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                denoms[i] += square(tex1Dfetch<float>(inputs, inpOffset + f * imgPixels * numImages + i * B_X));
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, outOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, outOffset + i * B_X);
                denoms[i] = addScale * denoms[i] + minDiv;
                prod[i] = (-2 * powScale * addScale * inp * prod[i] + out * __powf(denoms[i], -powScale));
                target[i * B_X] = prod[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                const float inp = tex1Dfetch<float>(inputs, outOffset + i * B_X);
                const float out = tex1Dfetch<float>(outGrads, outOffset + i * B_X);
                denoms[i] = addScale * denoms[i] + minDiv;
                prod[i] = (-2 * powScale * addScale * inp * prod[i] + out * __powf(denoms[i], -powScale));
                target[i * B_X] = scaleTargets * target[i * B_X] + scaleOutputs * prod[i];
            }
        }
    }
}


/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */

template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalAvgUndo(float* avgGrads, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    avgGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages + imgIdx;
    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX
            && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {

        for (int my = startOutputY; my < endOutputY; my++) {
            const float regionStartY = fmaxf(0, startX + my * strideX);
            const float regionEndY = fminf(imgSize, startX + my * strideX + subsX);
            const float regionSizeY = regionEndY - regionStartY;
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                const float regionStartX = fmaxf(0, startX + mx * strideX);
                const float regionEndX = fminf(imgSize, startX + mx * strideX + subsX);
                const float regionSizeX = regionEndX - regionStartX;
                // It's important to do the division here, because pushing division into the below
                // loops makes the code 4x slower.
                const float regionSizeInv = 1.0f / (regionSizeX * regionSizeY);
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += avgGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X] * regionSizeInv;
                        }
                    }
                }
            }
        }
    }

    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * maxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kLocalMaxUndo(float* imgs, float* maxGrads, float* maxActs, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int subsX, const int startX, const int strideX, const int outputsX,
                              const float scaleTargets, const float scaleOutputs) {
    __shared__ float shImgs[B_Y*filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / (numFilters/(B_Y*filtersPerThread));

    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % (numFilters/(B_Y*filtersPerThread))) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;

    const int startOutputY = blockPxY - startX < subsX ? 0 : 1 + (blockPxY - startX - subsX) / strideX;
    const int endOutputY = MIN(outputsX, 1 + (blockPxY - startX) / strideX);
    const int startOutputX = blockPxX - startX < subsX ? 0 : 1 + (blockPxX - startX - subsX) / strideX;
    const int endOutputX = MIN(outputsX, 1 + (blockPxX - startX) / strideX);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    maxGrads += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages
            + imgIdx;
    maxActs += ((blockFilterIdx + threadIdx.y) * numOutputs) * numImages
            + imgIdx;

    target += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    if  (blockPxX >= startX && blockPxX < startX + strideX * (outputsX-1) + subsX
         && blockPxY >= startX && blockPxY < startX + strideX * (outputsX-1) + subsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i] = imgs[f * B_Y * imgPixels * numImages + i * B_X];
                }
            }
        }
        for (int my = startOutputY; my < endOutputY; my++) {
            for (int mx = startOutputX; mx < endOutputX; mx++) {
                const int outputIdx = my * outputsX + mx;
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            const float ma = maxActs[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const float mg = maxGrads[(f * B_Y * numOutputs + outputIdx) * numImages + i * B_X];
                            const float img = shImgs[threadIdx.y + B_Y * f][threadIdx.x + B_X * i];

                            prod[f][i] += (img == ma) * mg;
                        }
                    }
                }
            }
        }
    }
    if (!add) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * B_Y * imgPixels * numImages + i * B_X] = scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * acts := -2 x scale x acts x outGrads / denoms
 */
template<int B_X, int eltsPerThread>
__global__ void kRNormUndoPrelims(float* acts, cudaTextureObject_t denoms, cudaTextureObject_t outGrads,
                                  const uint numElements, const float scale) {
    const uint e = B_X * blockIdx.x * eltsPerThread + threadIdx.x;
    const uint numThreads = B_X * gridDim.x;
    for (uint i = e; i < numElements; i += numThreads*eltsPerThread) {
        #pragma unroll
        for (uint k = 0; k < eltsPerThread; k++) {
            if (i + k * B_X < numElements) {
                acts[i + k * B_X] = __fdividef(scale * tex1Dfetch<float>(outGrads, i + k * B_X) * acts[i + k * B_X],
                                               tex1Dfetch<float>(denoms, i + k * B_X));
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output pixel for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread
 * numFilters must be divisible by B_Y*filtersPerThread
 *
 * TODO: this isn't really ideal
 */
template<int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kRNormUndo(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
                              const int numImages, const int sizeX, const float powScale, const float scaleTargets, const float scaleOutputs) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(B_Y*filtersPerThread);

    const int blockPxX = blockIdx.x / numImgBlocks;
    const int blockPxY = blockIdx.y / numFilterBlocks;
    
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    
    const int blockPx = blockPxY * imgSize + blockPxX;
    const int imgPixels = imgSize * imgSize;

    const int startY = MAX(0, blockPxY + sizeX/2 - sizeX + 1);
    const int startX = MAX(0, blockPxX + sizeX/2 - sizeX + 1);
    const int endY = MIN(imgSize, blockPxY + sizeX/2 + 1);
    const int endX = MIN(imgSize, blockPxX + sizeX/2 + 1);

    const int imgIdx = blockImgIdx + threadIdx.x;
    
    acts        += ((blockFilterIdx + threadIdx.y) * imgPixels) * numImages + imgIdx;
    inputs      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    denoms      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    outGrads    += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    target      += ((blockFilterIdx + threadIdx.y) * imgPixels + blockPx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }
    
    for (int sy = startY; sy < endY; sy++) {
        for (int sx = startX; sx < endX; sx++) {
            const int outPx = sy * imgSize + sx;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] += acts[(f * B_Y * imgPixels + outPx) * numImages + i * B_X];
                    }
                }
            }
        }
    }
//    outGrads += blockPx * numImages;
    if (scaleTargets == 0) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                    target[f * B_Y * imgPixels * numImages + i * B_X] = prod[f][i];
                }
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    const float inp = inputs[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float out = outGrads[(f * B_Y * imgPixels) * numImages + i * B_X];
                    const float den = denoms[(f * B_Y * imgPixels) * numImages + i * B_X];
                    prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                    target[f * B_Y * imgPixels * numImages + i * B_X] =
                                                scaleTargets * target[f * B_Y * imgPixels * numImages + i * B_X]
                                                + scaleOutputs * prod[f][i];
                }
            }
        }
    }
}

/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 *
 * outGrads:        (numFilters, imgPixels, numImages)
 * denoms:          (numFilters, imgPixels, numImages)
 * inputs:          (numFilters, imgPixels, numImages)
 * acts:            (numFilters, imgPixels, numImages)
 * target:          (numFilters, imgPixels, numImages)
 *
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 *
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * numFilters must be divisible by filtersPerThread
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 */
template<int B_X, int imgsPerThread, int filtersPerThread, bool add, bool checkCaseBounds>
__global__ void kRNormUndo2(float* outGrads, float* denoms, float* inputs, float* acts, float* target, const int imgSize, const int numFilters,
                            const int numImages, const int sizeX, const float powScale, const float scaleTargets, const float scaleOutputs) {
    __shared__ float shActs[filtersPerThread][B_X*imgsPerThread];
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockPxX = 4*(blockIdx.x / numImgBlocks);
    const int blockPxY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int startPxX = MAX(0, -DIVUP(sizeX,2) + blockPxX + 1);
    const int startPxY = MAX(0, -DIVUP(sizeX,2) + blockPxY + 1);
    const int endPxX = MIN(imgSize, blockPxX + sizeX/2 + 4);
    const int endPxY = MIN(imgSize, blockPxY + sizeX/2 + 4);

    const int myPxX = blockPxX + threadIdx.y % 4;
    const int myPxY = blockPxY + threadIdx.y / 4;
    const int myPxIdx = myPxY * imgSize + myPxX;
//    const bool doWork = myPxX < imgSize && myPxY < imgSize;
    const int myStartPxY = -DIVUP(sizeX,2) + myPxY + 1;
    const int myStartPxX = -DIVUP(sizeX,2) + myPxX + 1;
    const int myEndPxY = myPxY + sizeX/2 + 1;
    const int myEndPxX = myPxX + sizeX/2 + 1;
    
    const int imgIdx = blockImgIdx + threadIdx.x;

    acts        += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    denoms      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    inputs      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    outGrads    += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    target      += (blockFilterIdx * imgPixels + myPxIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = 0;
        }
    }

    for (int y = startPxY; y < endPxY; y++) {
        const bool isInY = y >= myStartPxY && y < myEndPxY;
        for (int x = startPxX; x < endPxX; x++) {
            const int px = y * imgSize + x;
            // All the threads load a pixel from memory
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shActs[ly + loadY][lx + loadX] = acts[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Each row of threads decides if it's interested in this pixel
            if (isInY && x >= myStartPxX && x < myEndPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] += shActs[f][threadIdx.x + i * B_X];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    acts -= (loadY * imgPixels - myPxIdx) * numImages + loadX;
    acts += threadIdx.x;
    if (myPxX < imgSize && myPxY < imgSize) {
        if (!add) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const float out = outGrads[f * imgPixels * numImages + i * B_X];
                        const float den = denoms[f * imgPixels * numImages + i * B_X];
                        const float inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                        target[f * imgPixels * numImages + i * B_X] = prod[f][i];
                    }
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        const float out = outGrads[f * imgPixels * numImages + i * B_X];
                        const float den = denoms[f * imgPixels * numImages + i * B_X];
                        const float inp = inputs[f * imgPixels * numImages + i * B_X];
                        prod[f][i] = inp * prod[f][i] + out * __powf(den, -powScale);
                        target[f * imgPixels * numImages + i * B_X] = scaleTargets * target[f * imgPixels * numImages + i * B_X] + scaleOutputs * prod[f][i];
                    }
                }
            }
        }

    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * maxGrads:    (numFilters, numOutputs, numImages)
 * rMaxActs:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalMaxUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput) {
    int outputs = outputsX * outputsX;
    int numImages = images->size[0];
    int numFilters = maxGrads->size[1] / outputs;
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    
    assert(imgSize * imgSize == imgPixels);
    assert(maxGrads->size[1] == numFilters * outputs);
    assert(maxGrads->size[0] == numImages);
    assert(!images->is_trans);
    assert(!target->is_trans);
    assert(!maxGrads->is_trans);
    assert(!maxActs->is_trans);
    //assert(images.isContiguous());
    //assert(maxGrads.isContiguous());
    //assert(maxActs.isContiguous());
    assert(maxGrads->size[0] == maxActs->size[0] && maxGrads->size[1] == maxActs->size[1]);
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    //target.resize(images);
    //assert(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 2)) * imgSize);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (imgsPerThread == 4) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 4, 2, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 4, 2, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 2, 2, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 2, 2, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if  (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, true><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalMaxUndo<4, 32, 1, 2, false, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalMaxUndo<4, 32, 1, 2, true, false><<<blocks, threads, 0, stream>>>(images->data_device, maxGrads->data_device, maxActs->data_device, target->data_device,
                                                                imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalMaxUndo: kernel execution failed");
}

/*
 * avgGrads:    (numFilters, numOutputs, numImages)
 * target:      (numFilters, imgPixels, numImages)
 */
void convLocalAvgUndo(cudamat* avgGrads, cudamat* target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput) {
    int numImages = avgGrads->size[0];

    int outputs = outputsX * outputsX;
    //int imgPixels = imgSize * imgSize;
    int numFilters = avgGrads->size[1] / outputs;
    assert(avgGrads->size[1] == numFilters * outputs);

    assert(!target->is_trans);
    assert(!avgGrads->is_trans);
    //assert(avgGrads.isContiguous());
    assert(numFilters % 16 == 0);
//    assert(numImages % 128 == 0);
    
    assert(strideX <= subsX);
    
    //target.resize(numFilters * imgPixels, numImages);
    //assert(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    int checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 4)) * imgSize);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 4, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 4, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 2, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 2, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    } else {
        if (checkCaseBounds) {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, true><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        } else {
            if (scaleTargets == 0 && scaleOutput == 1) {
                kLocalAvgUndo<4, 32, 1, 4, false, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                       imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                       outputsX, scaleTargets, scaleOutput);
            } else {
                kLocalAvgUndo<4, 32, 1, 4, true, false><<<blocks, threads, 0, stream>>>(avgGrads->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX,
                                                                      outputsX, scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convLocalAvgUndo: kernel execution failed");
}



/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)
 */
void convContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* target, int numFilters, int sizeX, float addScale, float powScale, float minDiv) {
    int numImages = images->size[0];
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(meanDiffs->size[0] == images->size[0] && meanDiffs->size[1] == images->size[1]);
    
    assert(!meanDiffs->is_trans);
    assert(!images->is_trans);
//    assert(images.isContiguous());
//    assert(meanDiffs.isContiguous());
    assert(numFilters % 16 == 0 || numFilters <= 8);

    //target.resize(images);
    //denoms.resize(images);
//    assert(target.isContiguous());
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        int imgsPerThread = 8;
        int filtersPerThread = 4;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        assert(numFilters % filtersPerThread == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);

        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, true>, cudaFuncCachePreferL1); // L1 faster here
            kCNorm2<8, 8, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                           imgSize, numFilters, numImages, sizeX, addScale, powScale, minDiv);
        } else {
            cudaFuncSetCacheConfig(kCNorm2<8, 8, 4, false>, cudaFuncCachePreferL1); // L1 faster here
            kCNorm2<8, 8, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                           imgSize, numFilters, numImages, sizeX, addScale, powScale, minDiv);
        }
    } else {
        bool checkCaseBounds = numImages % 128 != 0;
        if (numFilters <= 8) {
            dim3 threads(128);
            dim3 blocks(DIVUP(numImages,128) * imgSize, imgSize);
            if (numFilters == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 1, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 2, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 3, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 3, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 4, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 5) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 5, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 5, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 6) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 6, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 6, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 7) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 7, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 7, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            } else  if (numFilters == 8) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, true>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                } else {
                    cudaFuncSetCacheConfig(kCNorm_fewfilter<1, 8, false>, cudaFuncCachePreferL1);
                    kCNorm_fewfilter<1, 8, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                      imgSize, numImages, sizeX, addScale, powScale, minDiv);
                }
            }
        } else {
            dim3 threads(32, 4);
            dim3 blocks(DIVUP(numImages,threads.x*4), (numFilters / (threads.y * 2)), imgPixels);
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale, minDiv);
            } else {
                cudaFuncSetCacheConfig(kCNorm_manyfilter<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kCNorm_manyfilter<4, 32, 4, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device,
                                                                  imgSize, numFilters, numImages, sizeX, addScale, powScale, minDiv);
            }
        }
    }
    getLastCudaError("convResponseNorm: kernel execution failed");
}


/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void convResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput) {
    int numImages = outGrads->size[0];
    int imgPixels = outGrads->size[1] / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);

    assert(outGrads->size[1] == numFilters * imgPixels);

    assert(denoms->size[0] == outGrads->size[0] && denoms->size[1] == outGrads->size[1]);
    assert(denoms->size[0] == acts->size[0] && denoms->size[1] == acts->size[1]);
    assert(!denoms->is_trans);
    assert(!outGrads->is_trans);
    assert(!acts->is_trans);
    assert(!target->is_trans);
//    assert(outGrads.isContiguous());

    assert(numFilters % 16 == 0);
    
    //target.resize(outGrads);
//    assert(target.isContiguous());
    // First do acts := -2 x scale x acts x outGrads / denoms
    // so that the main routine only has to do an addition in its inner loop.
    int prelimEltsPerThread = 8;
    dim3 threads(128);
    dim3 blocks(DIVUP(outGrads->size[0] * outGrads->size[1],(threads.x * prelimEltsPerThread)));
    //bool checkPrelimBounds = (outGrads->size[0] * outGrads->size[1]) % (threads.x * prelimEltsPerThread) != 0;
    //printf("num elts: %d, blocks: %d\n", outGrads.getNumElements(), blocks.x);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    kRNormUndoPrelims<128, 8><<<blocks, threads, 0, stream>>>(acts->data_device, getTextureObject(denoms), getTextureObject(outGrads), outGrads->size[0] * outGrads->size[1], -2*addScale*powScale);

    // Now the main routine
    if (sizeX >= 6 && numFilters % 4 == 0) {
        // This one is faster for large regions (my tests show regions >= 6...)
        // NOTE: this stuff is not optimized for Kepler. Only kRNormUndo is.
        int imgsPerThread = numImages % 128 == 0 ? 8 : numImages % 64 == 0 ? 4 : 2;
        int filtersPerThread = 4;
        int bx = 16;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);

        threads = dim3(bx, 16);
        blocks = dim3(DIVUP(imgSize, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(imgSize, 4) * numFilters / filtersPerThread);
        if (imgsPerThread == 8) {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, true, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, false, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, true, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, true, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 8, 4, false, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 8, 4, false, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        } else if (imgsPerThread == 4) {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, true, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, true, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, false, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, false, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, true, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, true, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 4, 4, false, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 4, 4, false, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        } else {
            if (checkCaseBounds) {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, true, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, true, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, false, true>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, false, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            } else {
                if (scaleTargets == 0 && scaleOutput == 1) {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, true, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, true, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                } else {
                    cudaFuncSetCacheConfig(kRNormUndo2<16, 2, 4, false, false>, cudaFuncCachePreferL1);
                    kRNormUndo2<16, 2, 4, false, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                                  target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                                  scaleTargets, scaleOutput);
                }
            }
        }
    } else {
        int imgsPerThread = numImages % 128 == 0 ? 4 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        threads = dim3(32, 4);
        blocks = dim3(DIVUP(numImages,32*imgsPerThread) * imgSize, (numFilters / (4 * 2)) * imgSize);

        if (imgsPerThread == 4) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 4, 2, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 4, 2, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 4, 2, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 4, 2, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, true>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 1, 2, true><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kRNormUndo<4, 32, 1, 2, false>, cudaFuncCachePreferL1);
                kRNormUndo<4, 32, 1, 2, false><<<blocks, threads, 0, stream>>>(outGrads->data_device, denoms->data_device, inputs->data_device, acts->data_device,
                                                                          target->data_device, imgSize, numFilters, numImages, sizeX, powScale,
                                                                          scaleTargets, scaleOutput);
            }
        }
    }
    getLastCudaError("kRNormUndo: kernel execution failed");
}

/*
 * imgs:        (numChannels, imgPixels, numImages) with given imgStride
 * target:      (numChannels, tgtPixels, numImages)
 *
 * imgSize = scale * tgtSize
 */
void convResizeBilinear(cudamat* images, cudamat* target, int imgSize, int tgtSize, float scale) {
    assert(!images->is_trans);
    assert(!target->is_trans);
    int imgPixels = imgSize * imgSize;
    int numChannels = images->size[1] / imgPixels;
    int numImages = images->size[0];
    assert(images->size[1] == numChannels * imgPixels);
    
    //int tgtPixels = tgtSize * tgtSize;
    //target.resize(numChannels * tgtPixels, numImages);
//    assert(target.isContiguous());
    int numChunksX = DIVUP(tgtSize, 4);
    int numChunks = numChunksX * numChunksX;
    double imgCenter = imgSize * 0.5;
    double tgtCenter = tgtSize * 0.5;
    double centerScale = imgCenter - tgtCenter * scale;

    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    dim3 threads(32, 16);
    dim3 blocks(DIVUP(numImages, imgsPerThread * 32), numChannels * numChunks);
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kResizeBilinear<4, true>, cudaFuncCachePreferL1);
            kResizeBilinear<4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        } else {
            cudaFuncSetCacheConfig(kResizeBilinear<4, false>, cudaFuncCachePreferL1);
            kResizeBilinear<4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kResizeBilinear<2, true>, cudaFuncCachePreferL1);
            kResizeBilinear<2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        } else {
            cudaFuncSetCacheConfig(kResizeBilinear<2, false>, cudaFuncCachePreferL1);
            kResizeBilinear<2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kResizeBilinear<1, true>, cudaFuncCachePreferL1);
            kResizeBilinear<1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        } else {
            cudaFuncSetCacheConfig(kResizeBilinear<1, false>, cudaFuncCachePreferL1);
            kResizeBilinear<1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgSize, tgtSize, numImages, images->size[0], scale, centerScale);
        }
    }
    getLastCudaError("convResizeBilinear: kernel execution failed");
}

/*
 * imgs:        (3, imgPixels, numImages) with given imgStride
 * target:      (3, imgPixels, numImages)
 */
void convRGBToYUV(cudamat* images, cudamat* target) {
    assert(!images->is_trans);
    assert(!target->is_trans);
    int imgPixels = images->size[1] / 3;
    int numImages = images->size[0];
    assert(images->size[1] == 3 * imgPixels);
    
    //target.resize(3 * imgPixels, numImages);
//    assert(target.isContiguous());
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages, imgsPerThread * 32), DIVUP(imgPixels, 4));
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<4, true>, cudaFuncCachePreferL1);
            kRGBToYUV<4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<4, false>, cudaFuncCachePreferL1);
            kRGBToYUV<4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<2, true>, cudaFuncCachePreferL1);
            kRGBToYUV<2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<2, false>, cudaFuncCachePreferL1);
            kRGBToYUV<2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kRGBToYUV<1, true>, cudaFuncCachePreferL1);
            kRGBToYUV<1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        } else {
            cudaFuncSetCacheConfig(kRGBToYUV<1, false>, cudaFuncCachePreferL1);
            kRGBToYUV<1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
        }
    }
    getLastCudaError("convRGBToYUV: kernel execution failed");
}

/*
 * imgs:        (3, imgPixels, numImages) with given imgStride
 * target:      (3, imgPixels, numImages)
 */
void convRGBToLAB(cudamat* images, cudamat* target, bool center) {
    assert(!images->is_trans);
    assert(!target->is_trans);
    int imgPixels = images->size[1] / 3;
    int numImages = images->size[0];
    assert(images->size[1] == 3 * imgPixels);
    
    //target.resize(3 * imgPixels, numImages);
//    assert(target.isContiguous());
    
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages, imgsPerThread * 32), DIVUP(imgPixels, 4));
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (imgsPerThread == 4) {
        if (center) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<4, true, true>, cudaFuncCachePreferL1);
                kRGBToLAB<4, true, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<4, false, true>, cudaFuncCachePreferL1);
                kRGBToLAB<4, false, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<4, true, false>, cudaFuncCachePreferL1);
                kRGBToLAB<4, true, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<4, false, false>, cudaFuncCachePreferL1);
                kRGBToLAB<4, false, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        }
    } else if (imgsPerThread == 2) {
        if (center) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<2, true, true>, cudaFuncCachePreferL1);
                kRGBToLAB<2, true, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<2, false, true>, cudaFuncCachePreferL1);
                kRGBToLAB<2, false, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<2, true, false>, cudaFuncCachePreferL1);
                kRGBToLAB<2, true, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<2, false, false>, cudaFuncCachePreferL1);
                kRGBToLAB<2, false, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        }
    } else {
        if (center) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<1, true, true>, cudaFuncCachePreferL1);
                kRGBToLAB<1, true, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<1, false, true>, cudaFuncCachePreferL1);
                kRGBToLAB<1, false, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kRGBToLAB<1, true, false>, cudaFuncCachePreferL1);
                kRGBToLAB<1, true, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            } else {
                cudaFuncSetCacheConfig(kRGBToLAB<1, false, false>, cudaFuncCachePreferL1);
                kRGBToLAB<1, false, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device, imgPixels, numImages, images->size[0]);
            }
        }
    }
    getLastCudaError("convRGBToLAB: kernel execution failed");
}

/*
 * imgs:    (numChannels, imgPixels, numImages) with given imgStride
 * target:  (numChannels, tgtPixels, numImages)
 */
void convCrop(cudamat* imgs, cudamat* target, int imgSize, int tgtSize, int startY, int startX) {
    int numImages = imgs->size[0];
    int imgPixels = imgSize * imgSize;
    int tgtPixels = tgtSize * tgtSize;

    int numChannels = imgs->size[1] / imgPixels;
    assert(imgs->size[1] == imgPixels * numChannels);
    assert(imgPixels == imgSize * imgSize);
    assert(imgSize - startY >= tgtSize);
    assert(imgSize - startX >= tgtSize);
    assert(startY >= 0);
    assert(startX >= 0);
    //target.resize(numChannels * tgtPixels, numImages);
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
    dim3 blocks(DIVUP(numImages, 32 * imgsPerThread), numChannels * DIVUP(tgtPixels, 4));
    dim3 threads(32, 4);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    if (imgsPerThread == 4) {
        if (checkCaseBounds) {
            kCrop<4, true><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        } else {
            kCrop<4, false><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        }
    } else if (imgsPerThread == 2) {
        if (checkCaseBounds) {
            kCrop<2, true><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        } else {
            kCrop<2, false><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        }
    } else {
        if (checkCaseBounds) {
            kCrop<1, true><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        } else {
            kCrop<1, false><<<blocks, threads, 0, stream>>>(imgs->data_device, target->data_device, numImages, imgs->size[0], imgSize, tgtSize, startY, startX);
        }
    }
    getLastCudaError("convCrop: kernel execution failed");
}

/*
 * images:      (numFilters, imgPixels, numImages)
 * meanDiffs:   (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages) (out)
 * target:      (numFilters, imgPixels, numImages) (out)

 * Note: at present, I have no code to compute the meanDiffs. So it should be set
 * to be equal to images. In other words, this isn't really doing contrast normalization,
 * just response normalization.
 */
void convContrastNormCrossMap(cudamat* images, cudamat* meanDiffs, cudamat* target,
                             int numFilters, int sizeF, float addScale, float powScale, float minDiv, bool blocked) {
    int numImages = images->size[0];
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    //assert(meanDiffs.isSameDims(images));
    assert(sizeF > 0 && sizeF <= numFilters);
    
    assert(!meanDiffs->is_trans);
    assert(!images->is_trans);
//    assert(images.isContiguous());
//    assert(meanDiffs.isContiguous());
    assert(numFilters % 16 == 0);

    //target.resize(images);
//    denoms.resize(images);
//    assert(target.isContiguous());

    bool checkCaseBounds = numImages % 128 != 0;

    dim3 threads(32, 4);
    dim3 blocks(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
//    printf("convContrastNormCrossMap imgs: %p, meanDiffs: %p, denoms: %p, target: %p, imgSize: %d, numFilters: %d, numImages: %d, sizeF: %d, addScale: %f, powScale: %f, minDiv: %f, blocked: %d\n",
//            images->data_device, meanDiffs->data_device, denoms->data_device, target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv, blocked);
    if (blocked) {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, true><<<blocks, threads, 0, stream>>>(getTextureObject(images), getTextureObject(meanDiffs), target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, true>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, true><<<blocks, threads, 0, stream>>>(getTextureObject(images), getTextureObject(meanDiffs), target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        }
    } else {
        if (checkCaseBounds) {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, true, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, true, false><<<blocks, threads, 0, stream>>>(getTextureObject(images), getTextureObject(meanDiffs), target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        } else {
            cudaFuncSetCacheConfig(kFCNorm<4, 32, 4, false, false>, cudaFuncCachePreferL1);
            kFCNorm<4, 32, 4, false, false><<<blocks, threads, 0, stream>>>(getTextureObject(images), getTextureObject(meanDiffs), target->data_device,
                                                                imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv);
        }
    }

    getLastCudaError("convContrastNormCrossMap: kernel execution failed");
}

/*
 * outGrads:    (numFilters, imgPixels, numImages)
 * denoms:      (numFilters, imgPixels, numImages)
 * inputs:      (numFilters, imgPixels, numImages)
 * acts:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, imgPixels, numImages)
 *
 * THIS WILL OVERWRITE THE ACTS MATRIX.
 */
void convResponseNormCrossMapUndo(cudamat* outGrads, cudamat* inputs, cudamat* acts, cudamat* target, int numFilters,
                         int sizeF, float addScale, float powScale, float minDiv, bool blocked, float scaleTargets, float scaleOutput) {
    int numImages = outGrads->size[0];
    int imgPixels = outGrads->size[1] / numFilters;

    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    assert(sizeF > 0 && sizeF <= numFilters);
    assert(outGrads->size[1] == numFilters * imgPixels);

    assert(!outGrads->is_trans);
    assert(!acts->is_trans);
    assert(!target->is_trans);
//    assert(outGrads.isContiguous());

    assert(numFilters % 16 == 0);

    //target.resize(outGrads);
//    assert(target.isContiguous());

    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();

    dim3 threads2 = dim3(32, 4);
    dim3 blocks2 = dim3(DIVUP(numImages,32*4) * imgSize, (numFilters / 4) * imgSize);

    bool checkCaseBounds = (numImages % 128) != 0;
    if (blocked) {
        if (scaleTargets == 0 && scaleOutput == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, true, true><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, false, true><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, true, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, true, true><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, false, true>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, false, true><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            }
        }
    } else {
        if (scaleTargets == 0 && scaleOutput == 1) {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, true, false><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, false, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, false, false, false><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            }
        } else {
            if (checkCaseBounds) {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, true, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, true, false><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            } else {
                cudaFuncSetCacheConfig(kFRNormUndo2<4, 32, 4, true, false, false>, cudaFuncCachePreferL1);
                kFRNormUndo2<4, 32, 4, true, false, false><<<blocks2, threads2, 0, stream>>>(getTextureObject(outGrads), getTextureObject(inputs), getTextureObject(acts),
                                                                        target->data_device, imgSize, numFilters, numImages, sizeF, addScale, powScale, minDiv,
                                                                        scaleTargets, scaleOutput);
            }
        }
    }

    getLastCudaError("convResponseNormCrossMapUndo: kernel execution failed");
}


class AvgPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a / regionSize;
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fmaxf(a, b);
    }
    __device__ inline float getBaseValue() const {
        return -2e38; 
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

class MaxAbsPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return 0.0f;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

/*
 * Block size B_YxB_X
 * blockIdx.x determines output.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines output.y, filter idx in batches of B_Y*filtersPerThread
 * 
 * So each block does one output for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines filter idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */

template<class Agg, int B_Y, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX, const int strideX,
                           const int outputsX, Agg agg) {
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = DIVUP(numFilters, B_Y*filtersPerThread);
    const int outputIdxX = blockIdx.x / numImgBlocks;
    const int outputIdxY = blockIdx.y / numFilterBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * B_Y * filtersPerThread;
    const int myFilterIdx = (blockFilterIdx + threadIdx.y*filtersPerThread);
    if (myFilterIdx >= numFilters) {
        return;
    }
    
    const int outputIdx = outputIdxY * outputsX + outputIdxX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int startImgPxX = startX + outputIdxX * strideX;
    const int startImgPxY = startX + outputIdxY * strideX;
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += myFilterIdx * imgPixels * numImages + imgIdx;
    target += (myFilterIdx * numOutputs + outputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    
    const int loopStartY = MAX(0, startImgPxY);
    const int loopStartX = MAX(0, startImgPxX);
    const int loopEndY = MIN(imgSize, startImgPxY + subsX);
    const int loopEndX = MIN(imgSize, startImgPxX + subsX);
    const int regionSize = (loopEndY - loopStartY) * (loopEndX - loopStartX);
    for (int y = loopStartY; y < loopEndY; y++) {
        for (int x = loopStartX; x < loopEndX; x++) {
            const int imgPx = y * imgSize + x;
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    #pragma unroll
                    for (int f = 0; f < filtersPerThread; f++) {
                        prod[f][i] = agg(prod[f][i], imgs[(f * imgPixels + imgPx) * numImages + i * B_X]);
                    }
                }
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < imgsPerThread; i++) {
        if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
            #pragma unroll
            for (int f = 0; f < filtersPerThread; f++) {
                target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
            }
        }
    }
}

/*
 * Block size B_YxB_X
 * blockIdx.x determines pixel.x, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines pixel.y, output idx in batches of B_Y
 *
 * So each block does one pixel for some number of images/filters.
 *
 * threadIdx.x determines img idx
 * threadIdx.y determines output idx
 *
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numOutputs, imgPixels, numImages) (out)
 *
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 */
template<class Agg, int B_Y, int B_X, int imgsPerThread, bool checkCaseBounds>
__global__ void kPoolCrossMap(float* imgs, float* target, const int imgSize,
                              const int numFilters, const int numImages, const int startF, const int poolSize,
                              const int numOutputs, const int stride, Agg agg) {
    const int imgPixels = imgSize * imgSize;
    const int numImgBlocks = DIVUP(numImages, B_X*imgsPerThread);
//    const int numOutputs = DIVUP(numFilters, stride);
    const int numOutputBlocks = DIVUP(numOutputs,B_Y);
    const int pxIdxX = blockIdx.x / numImgBlocks;
    const int pxIdxY = blockIdx.y / numOutputBlocks;
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int outputIdx = (blockIdx.y % numOutputBlocks) * B_Y + threadIdx.y;
//    const int filterIdx = outputIdx * stride;

    const int pxIdx = pxIdxY * imgSize + pxIdxX;
    const int imgIdx = blockImgIdx + threadIdx.x;

    if (outputIdx < numOutputs) {
        imgs += (pxIdx) * numImages + imgIdx;
        target += (outputIdx * imgPixels + pxIdx) * numImages + imgIdx;

        float prod[imgsPerThread];
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                prod[i] = agg.getBaseValue();
            }
        }

        const int myStartF = startF + outputIdx * stride;
        const int loopStartF = max(0, myStartF);
        const int loopEndF = min(numFilters, myStartF + poolSize);

        for (int f = loopStartF; f < loopEndF; ++f) {
            #pragma unroll
            for (int i = 0; i < imgsPerThread; i++) {
                if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                    prod[i] = agg(prod[i], imgs[f * imgPixels * numImages + i * B_X]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                target[i * B_X] = agg.output(prod[i], poolSize);
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numOutputs, imgPixels, numImages)
 */
template<class Pooler>
void convPoolCrossMap(cudamat* images, cudamat* target, const int startF, const int poolSize,
                      const int numOutputs, const int stride, const int imgSize, Pooler pooler) {
    int numImages = images->size[0];
    int imgPixels = imgSize * imgSize;
    int numFilters = images->size[1] / imgPixels;
    assert(images->size[1] == numFilters * imgPixels);

    assert(!images->is_trans);
    assert(!target->is_trans);
//    assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    assert(stride <= poolSize);
    assert(startF <= 0);
    assert(startF + (numOutputs-1) * stride + poolSize >= numFilters); // All filters must be covered

    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    //target.resize(imgPixels*numOutputs, numImages);
    int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;

    dim3 threads(32, 4);
    dim3 blocks(imgSize * DIVUP(numImages, threads.x * imgsPerThread), imgSize * DIVUP(numOutputs, threads.y));
    bool checkCaseBounds = numImages % (threads.x*imgsPerThread) != 0;
    if (!checkCaseBounds) {
        if (imgsPerThread == 4) {
            cudaFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 4, false>, cudaFuncCachePreferShared);
            kPoolCrossMap<Pooler, 4, 32, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);

        } else if (imgsPerThread == 2) {
            cudaFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 2, false>, cudaFuncCachePreferShared);
            kPoolCrossMap<Pooler, 4, 32, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);

        } else if (imgsPerThread == 1) {
            cudaFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 1, false>, cudaFuncCachePreferShared);
            kPoolCrossMap<Pooler, 4, 32, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);
        }
    } else {
        if (imgsPerThread == 1) {
            cudaFuncSetCacheConfig(kPoolCrossMap<Pooler, 4, 32, 1, true>, cudaFuncCachePreferShared);
            kPoolCrossMap<Pooler, 4, 32, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                              imgSize, numFilters, numImages, startF, poolSize, numOutputs, stride, pooler);
        } else {
            assert(false);
        }
    }
    getLastCudaError("convPoolCrossMap: kernel execution failed");
}

/*
 * Block size 16xB_X
 * blockIdx.x determines 4x4 pixel.x region, image idx in batches of B_X*imgsPerThread
 * blockIdx.y determines 4x4 pixel.y region, filter idx in batches of filtersPerThread
 * 
 * So each block does a 4x4 region for some number of images/filters.
 * 
 * threadIdx.x determines img idx
 * threadIdx.y determines pixel idx
 * 
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, numOutputs, numImages)
 * 
 * B_X one of 8, 16, 32
 * imgsPerThread one of 1, 2, 4, 8, 16
 * 
 * B_XximgsPerThread MUST be divisible by 32.
 * Number of filters MUST be divisible by filtersPerThread.
 * 
 * numImages must be divisible by B_X*imgsPerThread if checkCaseBounds is false
 * 
 * Final write-out will not be fully coalesced unless B_X is 32. But there's a lot more
 * reading than writing here, and the reading is all coalesced, so it should be OK.
 * 
 * To be used when the stride is 1 and the pooling region is fairly large.
 */
template<class Agg, int B_X, int imgsPerThread, int filtersPerThread, bool checkCaseBounds>
__global__ void kLocalPool2(float* imgs, float* target, const int imgSize, const int numFilters,
                           const int numImages, const int subsX, const int startX,
                           const int outputsX, Agg agg) {
    __shared__ float shImgs[filtersPerThread][B_X*imgsPerThread];
    const int numImgBlocks = DIVUP(numImages,B_X*imgsPerThread);
    const int numFilterBlocks = numFilters/(filtersPerThread);
    const int blockOutputX = 4*(blockIdx.x / numImgBlocks);
    const int blockOutputY = 4*(blockIdx.y / numFilterBlocks);
    const int blockImgIdx = (blockIdx.x % numImgBlocks) * B_X * imgsPerThread;
    const int blockFilterIdx = (blockIdx.y % numFilterBlocks) * filtersPerThread;
    
//    const int blockOutputIdx = blockOutputY * outputsX + blockOutputX;
    const int numOutputs = outputsX * outputsX;
    const int imgPixels = imgSize * imgSize;
    
    const int tidx = threadIdx.y * B_X + threadIdx.x;
    const int loadY = tidx / 32, loadX = tidx % 32;
    
    const int myX = threadIdx.y % 4;
    const int myY = threadIdx.y / 4;
    
    const int myOutputIdxY = blockOutputY + myY;
    const int myOutputIdxX = blockOutputX + myX;
    const int myOutputIdx = myOutputIdxY * outputsX + myOutputIdxX;
    
    const int startImgPxX = startX + blockOutputX;
    const int startImgPxY = startX + blockOutputY;
    const int endImgPxX = startImgPxX + subsX;
    const int endImgPxY = startImgPxY + subsX;
    
    const int myStartImgPxY = startImgPxY + myY;
    const int myStartImgPxX = startImgPxX + myX;
    const int myEndImgPxY = endImgPxY + myY;
    const int myEndImgPxX = endImgPxX + myX;

    const int loopStartY = MAX(startImgPxY, 0);
    const int loopStartX = MAX(startImgPxX, 0);
    const int loopEndY = MIN(imgSize, endImgPxY + 3);
    const int loopEndX = MIN(imgSize, endImgPxX + 3);
    
    const int imgIdx = blockImgIdx + threadIdx.x;
    
    imgs += (blockFilterIdx + loadY) * imgPixels * numImages + blockImgIdx + loadX;
    target += (blockFilterIdx * numOutputs + myOutputIdx) * numImages + imgIdx;
    
    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for (int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            prod[f][i] = agg.getBaseValue(); 
        }
    }
    int regionSize = 0;
    for (int y = loopStartY; y < loopEndY; y++) {
        const bool isInY = y >= myStartImgPxY && y < myEndImgPxY ;
        for (int x = loopStartX; x < loopEndX; x++) {
            // Load a pixel
            const int px = y * imgSize + x;
            #pragma unroll
            for (int ly = 0; ly < filtersPerThread; ly += B_X/2) {
                if (filtersPerThread % (B_X/2) == 0 || ly + loadY < filtersPerThread) {
                    #pragma unroll
                    for (int lx = 0; lx < B_X*imgsPerThread; lx += 32) {
                        if (!checkCaseBounds || lx + loadX + blockImgIdx < numImages) {
                            shImgs[ly + loadY][lx + loadX] = imgs[(ly * imgPixels + px) * numImages + lx];
                        }
                    }
                }
            }
            __syncthreads();

            // Is this pixel in my region?
            if (isInY && x >= myStartImgPxX && x < myEndImgPxX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int f = 0; f < filtersPerThread; f++) {
                            prod[f][i] = agg(prod[f][i], shImgs[f][threadIdx.x + i * B_X]);
                        }
                    }
                }
                ++regionSize;
            }
            __syncthreads();

        }
    }
    if (myOutputIdxY < outputsX && myOutputIdxX < outputsX) {
        #pragma unroll
        for (int i = 0; i < imgsPerThread; i++) {
            if (!checkCaseBounds || imgIdx + i * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    target[f * numOutputs * numImages + i * B_X] = agg.output(prod[f][i], regionSize); 
                }
            }
        }
    }
}

/*
 * imgs:        (numFilters, imgPixels, numImages)
 * target:      (numFilters, outputs, numImages)
 */
template<class Pooler>
void convLocalPool(cudamat* images, cudamat* target, int numFilters,
                   int subsX, int startX, int strideX, int outputsX, Pooler pooler) {
    int numImages = images->size[0];
    int imgPixels = images->size[1] / numFilters;
    assert(images->size[1] == numFilters * imgPixels);
    int imgSize = int(sqrt(imgPixels));
    assert(imgSize * imgSize == imgPixels);
    
    assert(!images->is_trans);
    assert(!target->is_trans);
    //assert(images.isContiguous());
//    assert(numFilters % 4 == 0);
//    assert(numImages % 128 == 0);
    cudaStream_t stream = 0;  // NVMatrix::getDefaultStream();
    //int outputs = outputsX * outputsX;
    //target.resize(numFilters*outputs, numImages);

    if (strideX == 1 && subsX >= 6) {
        // NOTE: this part has not been optimized for Kepler
        int imgsPerThread = numImages % 128 == 0 ? 8 : 4;
        int filtersPerThread = numFilters % 4 == 0 ? 4 : numFilters % 3 == 0 ? 3 : numFilters % 2 == 0 ? 2 : 1;
        int bx = 8;
        bool checkCaseBounds = numImages % (bx*imgsPerThread) != 0;
        assert((imgsPerThread * bx) % 32 == 0);
        assert(numFilters % filtersPerThread == 0);
        dim3 threads(bx, 16);
        dim3 blocks(DIVUP(outputsX, 4) * DIVUP(numImages, bx*imgsPerThread), DIVUP(outputsX, 4) * numFilters / filtersPerThread);
//        printf("threads: %dx%d, blocks: %dx%d, imgSize: %d, numFilters: %d, numImages: %d, subsX: %d, startX: %d, outputsX: %d\n",
//                threads.y, threads.x, blocks.y, blocks.x, imgSize, numFilters, numImages, subsX, startX, outputsX);
        if (imgsPerThread == 8) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 3, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 8, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 8, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                 if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 1, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 2) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 2, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 2, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 3) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 3, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 3, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            } else if (filtersPerThread == 4) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, true>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool2<Pooler, 8, 4, 4, false>, cudaFuncCachePreferShared);
                    kLocalPool2<Pooler, 8, 4, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, outputsX, pooler);
                }
            }
        }
    } else {
        int filtersPerThread = numFilters % 16 == 0 ? 4 : 1;
        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkCaseBounds = numImages % (32*imgsPerThread) != 0;
        dim3 threads(32, 4);
        dim3 blocks(DIVUP(numImages,32*imgsPerThread) * outputsX, DIVUP(numFilters, 4 * filtersPerThread) * outputsX);
        if (imgsPerThread == 4) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 4, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 4, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else if (imgsPerThread == 2) {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 2, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 2, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        } else {
            if (filtersPerThread == 1) {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 1, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 1, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            } else {
                if (checkCaseBounds) {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, true>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 4, true><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                } else {
                    cudaFuncSetCacheConfig(kLocalPool<Pooler, 4, 32, 1, 4, false>, cudaFuncCachePreferL1);
                    kLocalPool<Pooler, 4, 32, 1, 4, false><<<blocks, threads, 0, stream>>>(images->data_device, target->data_device,
                                                                      imgSize, numFilters, numImages, subsX, startX, strideX, outputsX, pooler);
                }
            }
        }
    }
    getLastCudaError("convLocalPool: kernel execution failed");
}


#ifdef __cplusplus
extern "C" {
#endif
// Response Normalization cross map - computes : images / ((1 + addScale * (sum sq images over neighbourhood))^{powScale})
// blocked : true means divide input into blocks and compete within each, false means compete within a running window centered at self.
void ResponseNormCrossMap(cudamat* images, cudamat* targets, int numFilters, int sizeF, float addScale, float powScale, bool blocked){
  convContrastNormCrossMap(images, images, targets, numFilters, sizeF, addScale, powScale, 1, blocked);
}

// overwrites acts.
void ResponseNormCrossMapUndo(cudamat* outGrads, cudamat* inputs, cudamat* acts, cudamat* targets, int numFilters, int sizeF, float addScale, float powScale, bool blocked){
  convResponseNormCrossMapUndo(outGrads, inputs, acts, targets, numFilters, sizeF, addScale, powScale, 1, blocked, 0, 1);
}

void ResponseNorm(cudamat* images, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNorm(images, images, denoms, targets, numFilters, sizeX, addScale, powScale, 1);
}

void ResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNormUndo(outGrads, denoms, inputs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Contrast Normalization.
void ContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convContrastNorm(images, meanDiffs, denoms, targets, numFilters, sizeX, addScale, powScale, 1);
}
void ContrastNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs, cudamat* acts, cudamat* targets, int numFilters, int sizeX, float addScale, float powScale){
  convResponseNormUndo(outGrads, denoms, meanDiffs, acts, targets, numFilters, sizeX, addScale, powScale, 0, 1);
}

// Pooling.
void MaxPool(cudamat* images, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  MaxPooler mpooler;
  convLocalPool<MaxPooler>(images, targets, numFilters, subsX, startX, strideX, outputsX, mpooler);
}

void AvgPool(cudamat* images, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  AvgPooler pooler = AvgPooler();
  convLocalPool<AvgPooler>(images, targets, numFilters, subsX, startX, strideX, outputsX, pooler);
}
/*
void ProbMaxPool(cudamat* images, cudamat* rnd, cudamat* targets, int numFilters, int subsX,	int startX,	int strideX, int outputsX){
  ProbMaxPooler mpooler;
  convLocalProbPool<ProbMaxPooler>(images, rnd, targets, numFilters, subsX, startX, strideX, outputsX, mpooler);
}
*/


void MaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs, cudamat* targets, int subsX, int startX, int strideX, int outputsX){
  convLocalMaxUndo(images, maxGrads, maxActs, targets, subsX, startX, strideX, outputsX, 0, 1);
}

void AvgPoolUndo(cudamat* avgGrads, cudamat* targets, int subsX, int startX, int strideX, int outputsX, int imgSize) {
  convLocalAvgUndo(avgGrads, targets, subsX, startX, strideX, outputsX, imgSize, 0, 1);
}

void UpSample(cudamat* images, cudamat* targets, int factor, int input_image_size, float scaleTargets) { 
  convLocalAvgUndo(images, targets, factor, 0, factor, input_image_size,
                   factor * input_image_size, scaleTargets, factor * factor);
}

void DownSample(cudamat* images, cudamat* targets, int factor, int input_image_size) {
  AvgPooler pooler = AvgPooler();
  int num_filters = images->size[1] / (input_image_size * input_image_size);
  convLocalPool<AvgPooler>(images, targets, num_filters, factor, 0, factor,
                           input_image_size / factor, pooler);
}

void RGBToYUV(cudamat* images, cudamat* targets) {
  convRGBToYUV(images, targets);
}


void convBedOfNails(cudamat* images, cudamat* target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput) {
  _convBedOfNails(images, target, numChannels, imgSize, startX, strideX, false, scaleTargets, scaleOutput);
}

void convBedOfNailsUndo(cudamat* actsGrad, cudamat* target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput) {

  _convBedOfNails(target, actsGrad, numChannels, imgSize, startX, strideX, true, scaleTargets, scaleOutput);
}
#ifdef __cplusplus
}
#endif
