#ifndef NVMATRIX_KERNEL_H_
#define NVMATRIX_KERNEL_H_

#define NUM_RND_BLOCKS                      96
#define NUM_RND_THREADS_PER_BLOCK           128
#define NUM_RND_STREAMS                     (NUM_RND_BLOCKS * NUM_RND_THREADS_PER_BLOCK)

/*
 * Defines for getting the values at the lower and upper 32 bits
 * of a 64-bit number.
 */
#define LOW_BITS(x)                         ((x) & 0xffffffff)
#define HIGH_BITS(x)                        ((x) >> 32)

/*
 * Number of iterations to run random number generator upon initialization.
 */
#define NUM_RND_BURNIN                      100

#define COPY_BLOCK_SIZE                     16
#
#define NUM_VECTOR_OP_BLOCKS                4096
#define NUM_VECTOR_OP_THREADS_PER_BLOCK     512
#define NUM_VECTOR_OP_LOOPS_PER_THREAD      1

#define PI 3.1415926535897932f
#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif
#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif
#ifndef MAX
#define MAX(x,y) ((x > y) ? x : y)
#endif


__device__ float device_val;
__device__ void reduceToSumLocal(float* sdata, unsigned int tid);
__device__ void reduceToMax(float* sdata, unsigned int tid);
__global__ void kSeedRandom(unsigned int* randMults, unsigned long long* randWords, unsigned int seed);
__global__ void kRandomUniform(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements);
__global__ void kRandomGaussian(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements);
__global__ void kRandomDropout(unsigned int* randMults, unsigned long long* randWords, float* gData, unsigned int numElements, float dropprob, float val, float scale);
__global__ void kRandomGaussianDropout(unsigned int* rndMults, unsigned long long* rndWords, float* gData, unsigned int numElements, float scale);
__global__ void kSampleBernoulli(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSampleBernoulliTanh(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSamplePoisson(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kSampleGaussian(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements, float mult);
__global__ void kPerturbProb(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);
__global__ void kPerturbEnergy(unsigned int* randMults, unsigned long long* randWords, float* gData, float* target, unsigned int numElements);

__global__ void kGetRowSlice(float* source, float* target, int start, int end, int width, int height);
__global__ void kTranspose(float *odata, float *idata, int width, int height);
__global__ void kTransposeBig(float *odata, float *idata, int height, int width);
__global__ void kSetRowSlice(float* source, float* target, int start, int end, int width, int height);

__global__ void kLessThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kLessThanEq(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kLessThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kLessThanEqScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kGreaterThan(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kGreaterThanEq(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kGreaterThanScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kGreaterThanEqScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kUpperBound(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kLowerBound(float* mat1, float* mat2, float* target, unsigned int len);
__global__ void kUpperBoundScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kLowerBoundScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kUpperBoundModScalar(float* mat, float val, float* target, unsigned int len);
__global__ void kMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kArgMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kSqSumColumnwise(float* mat, float* target, unsigned int width, unsigned int height, float mult, float p);
__global__ void kSqSumRowwise(float* mat, float* target, unsigned int width, unsigned int height, float mult, float p);
__global__ void kNormLimitColumnwise(float* mat, float* target, float norm, unsigned int width, unsigned int height, int constraint);
__global__ void kNormLimitRowwise(float* mat, float* target, float norm, unsigned int width, unsigned int height, int constraint);
__global__ void kSumAll(float* mat, unsigned int len);
__global__ void kSparseDot(int m, int n, int k, float *data, int* indptr, int* indices, float *dense_data, float* target, float beta, float alpha);
__global__ void kSign(float* mat, float* target, unsigned int len);
__global__ void kApplySigmoid(float* mat, float* target, unsigned int len);
__global__ void kApplySin(float* mat, float* target, unsigned int len);
__global__ void kApplyCos(float* mat, float* target, unsigned int len);
__global__ void kApplyTanh(float* mat, float* target, unsigned int len);
__global__ void kApplyAbs(float* mat, float* target, unsigned int len);
__global__ void kApplyLog1PlusExp(float* mat, float* target, unsigned int len);
__global__ void kApplyLog1PlusExpExact(float* mat, float* target, unsigned int len);
__global__ void kSquashRelu(float* mat, float* target, unsigned int len, float lambda);
__global__ void kLog(float* mat, float* target, unsigned int len, float tiny);
__global__ void kExp(float* mat, float* target, unsigned int len);
__global__ void kCeil(float* mat, float* target, unsigned int len);
__global__ void kFloor(float* mat, float* target, unsigned int len);
__global__ void kSqrt(float* mat, float* target, unsigned int len);
__global__ void kPow(float* mat, float pow, float* target, unsigned int len);
__global__ void kPowMatrix(float* mat, float* pow, float* target, unsigned int len);
__global__ void kCrossEntropy(float* mat, float* p, float* target, unsigned int len, float tiny);
__global__ void kCrossEntropyBernoulli(float* mat, float* p, float* target, unsigned int len, float tiny);
__global__ void kCorrectPreds(float* mat, float* p, float* target, unsigned int len, float cutoff);
__global__ void kReciprocal(float* mat, float* target, unsigned int len);
__global__ void kAddDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width);
__global__ void kAddDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width);
__global__ void kMultDiagonal(float* mat, float* vec, float* tgtMat, unsigned int width);
__global__ void kMultDiagonalScalar(float* mat, float val, float* tgtMat, unsigned int width);
__global__ void kAddColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddColMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height);
__global__ void kAddRowMult(float* mat, float* vec, float* tgtMat, float mult, unsigned int width, unsigned int height);
__global__ void kAddToEachPixel(float* mat1, float* mat2, float* tgtMat, float mult, unsigned int width, unsigned int height, unsigned int num_pix);
__global__ void kMultByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kMultByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByColVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kDivByRowVector(float* mat, float* vec, float* tgtMat, unsigned int width, unsigned int height);
__global__ void kAddMultSign(float* a, float* b, unsigned int numEls, float mult);
__global__ void kAdd(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kSubtract(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kMult(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kLogisticDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kLogisticGrad(float* mat, float* targets, float* out_grad, unsigned int numEls);
__global__ void kSinDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kCosDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kTanhDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kRectifiedLinearDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kRectifiedLinearSmoothDeriv(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kDivide(float* a, float* b, float* dest, unsigned int numEls);
__global__ void kMultScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void kAssignScalar(float* dest, float alpha, unsigned int len);
__global__ void kDivideScalar(float* mat, float alpha, float* dest, unsigned int len);
__global__ void kAddScalar(float* a, float alpha, float* dest, unsigned int numEls);
__global__ void kSelectRows(float* source, float* target, float* indices, int nRowIs, int nCols, int nSourceRows);
__global__ void kSetSelectedRows(float* target, float* source, float* indices, int nRowIs, int nCols, int nTargetRows);
__global__ void kSwapColumns(float* target, float* source, float* indices1, float* indices2, int cols, int width, int height);
__global__ void kShuffleColumns(float* source, float* target, float* indices, int width, int height);
__global__ void kGenerateTranslationsBigVarOff(float* source, float* target, float* off_x_arr, float* off_y_arr, int source_w, int target_w, int num_channels);
__global__ void kBlockify(float* source, float* target, int numdims, int blocksize);
__global__ void kCumsum(float *mat, float *target, float *temp, unsigned int height);
__global__ void kChooseMaxColumnwise(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kChooseMaxAndAccumulate(float* mat, float* acc, unsigned int width, unsigned int height);


__global__ void kSoftMax(float* mat, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxOverwrite(float* mat, unsigned int width, unsigned int height);
__global__ void kSoftMaxRowMajor(float* mat, unsigned int width, unsigned int height);
__global__ void kSoftMaxGrad(float* mat, float* labels, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxGradCLS(float* mat, int* labels, float* indices, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxGradRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxCorrect(float* mat, float* labels, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxCorrectRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxCorrectCLS(float* mat, int* labels, float* indices, float* target, unsigned int width, unsigned int height);
__global__ void kSoftMaxCrossEntropy(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float tiny);
__global__ void kSoftMaxCrossEntropyRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float tiny);
__global__ void kHingeQuadraticRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float margin);
__global__ void kHingeLinearRowMajor(float* mat, float* labels, float* target, unsigned int width, unsigned int height, float margin);
__global__ void kExpandAndAdd(float* source, float* mat, float* indices, float* target, int width, int height, float mult, int width2);
__global__ void kExpand(float* source, float* indices, float* target, int height, int width, int target_width);
__global__ void kAccumulateColumns(float* mat, float* indices, float* target, int mat_width, int target_width, int height, float mult, int avg);
__global__ void kExtractPatches(float* images, float* patches, float* indices, float* width_offset, float* height_offset, int num_images, int img_width, int img_height, int patch_width, int patch_height, int num_colors);
__global__ void kRectifyBoundingBox(float* boxes, float* width_offset, float* height_offset, float* flip, int num_images, int patch_width, int patch_height, int num_locs);
__global__ void kExtractPatches2(float* images, float* patches, float* width_offset, float* height_offset, float* flip, int num_images, int img_width, int img_height, int patch_width, int patch_height, int num_colors);
__global__ void kAdagrad(float *w, float *grad, float *sum_grad_sq, int len, float decay, float epsilon);
__global__ void kBoundingBoxLogisticGrad(
    float* mat, int* bbox, int* label, int* seg, float* indices,
    float *width_offset, float* height_offset,
    int size, int width, int height, int depth, float scale_width,
    float scale_height, float* grad);

__global__ void kBoundingBoxSoftMaxGrad(
    float* mat, int* bbox, int* label, int* seg, float* indices,
    float *width_offset, float* height_offset,
    int size, int width, int height, int depth, float scale_width,
    float scale_height, float* grad);
__global__ void kSoftMaxCorrectBoundingBox(
    float* mat, int* bbox, int* label, int* seg, float* indices,
    float *width_offset, float* height_offset,
    int size, int width, int height, int depth, float scale_width,
    float scale_height, float* target);
__global__ void kLogisticCorrectBoundingBox(
    float* mat, int* bbox, int* label, int* seg, float* indices,
    float *width_offset, float* height_offset,
    int size, int width, int height, int depth, float scale_width,
    float scale_height, float* target, float cutoff);
__global__ void kLogisticCorrectNormalized(float* mat, float* targets, float* out, unsigned int height, unsigned int width);
#endif
