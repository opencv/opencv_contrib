#ifndef CUDAMAT_CONV_CUH
#define CUDAMAT_CONV_CUH

#include "cudamat.cuh"
#ifdef __cplusplus
extern "C" {
#endif

void convUp(cudamat* images, cudamat* filters, cudamat* targets, int imgSizeY, int numModulesY,
            int numModulesX, int paddingStart, int moduleStride,
            int numImgColors, int numGroups, float scaleTargets);

void convDown(cudamat* images, cudamat* filters, cudamat* targets,
              int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
              int numImgColors, int numGroups, float scaleTargets);

void convOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int imgSizeY, int numModulesY,
              int numModulesX, int filterSize, int paddingStart,
              int moduleStride, int numImgColors, int numGroups,
              int partialSum, float scaleTargets, float scaleOutput);

void localUp(cudamat* images, cudamat* filters, cudamat* targets, int imgSizeY, int numModulesY,
             int numModulesX, int paddingStart, int moduleStride,
             int numImgColors, int numGroups, float scaleTargets);

void localDown(cudamat* images, cudamat* filters, cudamat* targets,
               int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride,
               int numImgColors, int numGroups, float scaleTargets);

void localOutp(cudamat* images, cudamat* hidSums, cudamat* targets, int imgSizeY, int numModulesY,
               int numModulesX, int filterSize, int paddingStart,
               int moduleStride, int numImgColors, int numGroups,
               float scaleTargets, float scaleOutput);

void ResponseNormCrossMap(cudamat* images, cudamat* targets,
                          int numFilters, int sizeF, float addScale,
                          float powScale, bool blocked);

void ResponseNormCrossMapUndo(cudamat* outGrads,
                              cudamat* inputs, cudamat* acts, cudamat* targets,
                              int numFilters, int sizeF, float addScale,
                              float powScale, bool blocked);

void ResponseNorm(cudamat* images, cudamat* denoms, cudamat* targets,
                  int numFilters, int sizeX, float addScale, float powScale);

void ResponseNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* inputs,
                      cudamat* acts, cudamat* targets, int numFilters,
                      int sizeX, float addScale, float powScale);

void ContrastNorm(cudamat* images, cudamat* meanDiffs, cudamat* denoms,
                  cudamat* targets, int numFilters, int sizeX, float addScale,
                  float powScale);

void ContrastNormUndo(cudamat* outGrads, cudamat* denoms, cudamat* meanDiffs,
                      cudamat* acts, cudamat* targets, int numFilters,
                      int sizeX, float addScale, float powScale);

void MaxPool(cudamat* images, cudamat* targets, int numFilters, int subsX,
             int startX,	int strideX, int outputsX);

void AvgPool(cudamat* images, cudamat* targets, int numFilters, int subsX,
             int startX,	int strideX, int outputsX);

void ProbMaxPool(cudamat* images, cudamat* rnd, cudamat* targets,
                 int numFilters, int subsX,	int startX,	int strideX,
                 int outputsX);

void MaxPoolUndo(cudamat* images, cudamat* maxGrads, cudamat* maxActs,
                 cudamat* targets, int subsX, int startX, int strideX,
                 int outputsX);

void AvgPoolUndo(cudamat* avgGrads, cudamat* targets, int subsX, int startX,
                 int strideX, int outputsX, int imgSize);

void UpSample(cudamat* images, cudamat* targets, int factor,
              int input_image_size, float scaleTargets);
 
void DownSample(cudamat* images, cudamat* targets, int factor,
                int input_image_size);

void RGBToYUV(cudamat* images, cudamat* targets);
#ifdef __cplusplus
}
#endif
#endif
