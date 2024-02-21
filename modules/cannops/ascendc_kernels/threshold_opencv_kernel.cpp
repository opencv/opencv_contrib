#include "kernel_operator.h"
#include "vector_tiling.h"
#include "kernel_tiling_types.h"

using namespace AscendC;

// Make compiler happy. These two function will never be called.
__aicore__ static inline void Cast(const LocalTensor<half>& dstLocal,
                                   const LocalTensor<half>& srcLocal, const RoundMode& round_mode,
                                   const uint32_t calCount){};
__aicore__ static inline void Cast(const LocalTensor<float>& dstLocal,
                                   const LocalTensor<float>& srcLocal, const RoundMode& round_mode,
                                   const uint32_t calCount){};

/**
 * T: input data type.
 * C: data type for calculate.
 * if T != C, data should cast from T to C.
 */
template <typename T, typename C>
class KernelThreshold
{
public:
    __aicore__ inline KernelThreshold() {}
    __aicore__ inline void Init(ThresholdOpencvTilingData* tiling, GM_ADDR x, GM_ADDR y)
    {
        tilingData = tiling;

        /**
         * Calculate memory use per element.
         * 1. InputQueue: sizeof(T) * BUFFER_NUM
         * 2. OutputQueue: sizeof(T) * BUFFER_NUM
         * 3. maskBuffer: 1 byte at most.
         */
        uint64_t bytesPerElem = sizeof(T) * BUFFER_NUM * 2 + sizeof(uint8_t) * 1;

        /**
         * If need cast, should init two more cast buffers.
         * Memory use per element:
         * 1. InputCastBuffer: sizeof(C)
         * 2. OutputCastBuffer: sizeof(C)
         */
        if (!std::is_same<T, C>::value)
        {
            bytesPerElem += sizeof(C) * 2;
        }

        // Most of AscendC APIs need align to 32 Bytes, but Compare and Select need
        // align to 256 Bytes, 256/sizeof(C) means how many element can be process
        // in one loop.
        vecTiling.calculate(tilingData->totalLength, GetBlockNum(), GetBlockIdx(), bytesPerElem,
                            256 / sizeof(C));

        xGM.SetGlobalBuffer((__gm__ T*)x + vecTiling.blockOffset, vecTiling.blockLength);
        yGM.SetGlobalBuffer((__gm__ T*)y + vecTiling.blockOffset, vecTiling.blockLength);

        // Cast buffer.
        if (!std::is_same<T, C>::value)
        {
            pipe.InitBuffer(InputCastBuffer, vecTiling.loopLength * sizeof(C));
            pipe.InitBuffer(outputCastBuffer, vecTiling.loopLength * sizeof(C));
        }

        pipe.InitBuffer(inputQueue, BUFFER_NUM, vecTiling.loopLength * sizeof(T));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, vecTiling.loopLength * sizeof(T));
        pipe.InitBuffer(maskBuffer, vecTiling.loopLength * sizeof(uint8_t));
    }

    __aicore__ inline void Run()
    {
        for (uint32_t loop = 0; loop < vecTiling.loopCount; loop++)
        {
            uint32_t offset = loop * vecTiling.loopLength;
            Compute(offset, vecTiling.loopLength);
        }

        if (vecTiling.loopTailLength != 0)
        {
            uint32_t offset = vecTiling.loopCount * vecTiling.loopLength;
            Compute(offset, vecTiling.loopTailLength);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t offset, uint32_t len)
    {
        CopyIn(offset, len);

        // Get local Tensor, if case is need, local tensors come from
        // cast buffer. otherwise, local tensors come from input/output queue.
        LocalTensor<C> xLocal = CastInput(inputQueue, InputCastBuffer, len);
        LocalTensor<C> yLocal = GetOutput(outputQueue, outputCastBuffer);

        Threshold(xLocal, yLocal, len);

        // Free local input tensor if tensor is not from cast buffer.
        FreeInput(inputQueue, xLocal);
        // Cast output tensor to output queue if output tensor is from cast buffer.
        CastOutput(outputQueue, yLocal, len);

        CopyOut(offset, len);
    }

    /**
     * If need cast:
     * 1. Get data from input queue, this data can't be calculate directly.
     * 2. Get buffer with type C, which satisfied AscendC APIs.
     * 3. Cast data from T to C.
     *
     * If not need cast:
     * 1. Only need get data from queue.
     */
    __aicore__ inline LocalTensor<C> CastInput(TQue<QuePosition::VECIN, BUFFER_NUM>& queue,
                                               TBuf<TPosition::VECCALC>& buffer, uint32_t len)
    {
        LocalTensor<C> xLocal;
        if (std::is_same<T, C>::value)
        {
            xLocal = queue.DeQue<C>();
        }
        else
        {
            xLocal = buffer.Get<C>();
            LocalTensor<T> xCast = queue.DeQue<T>();
            Cast(xLocal, xCast, RoundMode::CAST_NONE, len);
            queue.FreeTensor(xCast);
        }
        return xLocal;
    }

    /**
     * If need cast:
     * 1. Get local tensor from cast buffer.
     *
     * If not need cast:
     * 1. Alloc local tensor from output queue.
     */
    __aicore__ inline LocalTensor<C> GetOutput(TQue<QuePosition::VECOUT, BUFFER_NUM>& queue,
                                               TBuf<TPosition::VECCALC>& buffer)
    {
        if (std::is_same<T, C>::value)
        {
            return queue.AllocTensor<C>();
        }
        else
        {
            return buffer.Get<C>();
        }
    }

    /**
     * If need cast:
     * 1. Input local tensor are get from cast buffer, which do not need free.
     *
     * If not need cast:
     * 1. Input local tensor are alloced from input queue, which need free.
     */
    __aicore__ inline void FreeInput(TQue<QuePosition::VECIN, BUFFER_NUM>& queue,
                                     LocalTensor<C>& xLocal)
    {
        if (std::is_same<T, C>::value)
        {
            queue.FreeTensor(xLocal);
        }
    }

    /**
     * If need cast:
     * 1. Alloc local tensor from output queue.
     * 2. Cast from C to T.
     * 3. Put casted local tensor in queue.
     *
     * If not need cast:
     * 1. Only put local tensor in queue.
     *
     */
    __aicore__ inline void CastOutput(TQue<QuePosition::VECOUT, BUFFER_NUM>& queue,
                                      LocalTensor<C>& yLocal, uint32_t len)
    {
        if (std::is_same<T, C>::value)
        {
            queue.EnQue(yLocal);
        }
        else
        {
            LocalTensor<T> yCast = queue.AllocTensor<T>();
            RoundMode roundMode = RoundMode::CAST_NONE;
            // Ref to AscendC cast API.
            if (std::is_same<T, int16_t>::value)
            {
                roundMode = RoundMode::CAST_RINT;
            }
            else if (std::is_same<T, int32_t>::value)
            {
                roundMode = RoundMode::CAST_ROUND;
            }
            Cast(yCast, yLocal, roundMode, len);
            queue.EnQue(yCast);
        }
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> xLocal = inputQueue.AllocTensor<T>();
        DataCopy(xLocal, xGM[offset], len);
        inputQueue.EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t len)
    {
        LocalTensor<T> yLocal = outputQueue.DeQue<T>();
        DataCopy(yGM[offset], yLocal, len);
        outputQueue.FreeTensor(yLocal);
    }

    /**
     * AscendC API Compare Warpper.
     * AscendC Compare level2 API need input length align to 256, process
     * tail data by level0 API.
     */
    __aicore__ inline void CompareWrap(const LocalTensor<uint8_t>& dstLocal,
                                       const LocalTensor<C>& src0Local,
                                       const LocalTensor<C>& src1Local, CMPMODE cmpMode,
                                       uint32_t calCount)
    {
        // Elements total count for on loop inside Compare.
        uint32_t batchCount = 256 / sizeof(C);

        // Tail elements count.
        uint32_t tailCount = calCount % batchCount;

        // Level2 API, calCount should align to 256.
        Compare(dstLocal, src0Local, src1Local, cmpMode, calCount - tailCount);

        // Data blocks are already cut align to 256, tail count will be 0 for
        // all process loops except last one.
        if (tailCount != 0)
        {
            BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
            uint32_t tailIdx = calCount - tailCount;
            uint32_t maskIdx = tailIdx / sizeof(uint8_t);
            Compare(dstLocal[maskIdx], src0Local[tailIdx], src1Local[tailIdx], cmpMode, tailCount,
                    1, repeatParams);
        }
    }

    /**
     * AscendC API Select Warpper.
     * AscendC Select level2 API need input length align to 256, process
     * tail data by level0 API.
     */
    __aicore__ inline void SelectWrap(const LocalTensor<C>& dstLocal,
                                      const LocalTensor<uint8_t>& selMask,
                                      const LocalTensor<C>& src0Local, C src1Local, SELMODE selMode,
                                      uint32_t calCount)
    {
        uint32_t batchCount = 256 / sizeof(C);
        uint32_t tailCount = calCount % batchCount;

        Select(dstLocal, selMask, src0Local, src1Local, selMode, calCount - tailCount);
        if (tailCount != 0)
        {
            BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
            uint32_t tailIdx = calCount - tailCount;
            uint32_t maskIdx = tailIdx / sizeof(uint8_t);
            Select(dstLocal[tailIdx], selMask[maskIdx], src0Local[tailIdx], src1Local, selMode,
                   tailCount, 1, repeatParams);
        }
    }

    __aicore__ inline void Threshold(LocalTensor<C>& xLocal, LocalTensor<C>& yLocal, uint32_t len)
    {
        LocalTensor<uint8_t> mask = maskBuffer.Get<uint8_t>();
        Duplicate(yLocal, static_cast<C>(tilingData->thresh), len);
        switch (tilingData->threshType)
        {
            case 0:
                CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
                Duplicate(yLocal, static_cast<C>(0), len);
                SelectWrap(yLocal, mask, yLocal, static_cast<C>(tilingData->maxVal),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
                break;
            case 1:
                CompareWrap(mask, xLocal, yLocal, CMPMODE::GT, len);
                Duplicate(yLocal, static_cast<C>(0), len);
                SelectWrap(yLocal, mask, yLocal, static_cast<C>(tilingData->maxVal),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
                break;
            case 2:
                CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
                SelectWrap(yLocal, mask, xLocal, static_cast<C>(tilingData->thresh),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
                break;
            case 3:
                CompareWrap(mask, xLocal, yLocal, CMPMODE::GT, len);
                SelectWrap(yLocal, mask, xLocal, static_cast<C>(0),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
                break;
            case 4:
                CompareWrap(mask, xLocal, yLocal, CMPMODE::LE, len);
                SelectWrap(yLocal, mask, xLocal, static_cast<C>(0),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE, len);
                break;
            default:
                break;
        }
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<TPosition::VECCALC> InputCastBuffer, outputCastBuffer, maskBuffer;

    GlobalTensor<T> xGM, yGM;
    VectorTiling vecTiling;
    ThresholdOpencvTilingData* tilingData;
};

#define LAUNCH_THRESHOLD_KERNEL(NAME, T, C)                                                      \
    __aicore__ inline void launch_threshold_kernel_##NAME(ThresholdOpencvTilingData* tilingData, \
                                                          GM_ADDR x, GM_ADDR y)                  \
    {                                                                                            \
        KernelThreshold<T, C> op;                                                                \
        op.Init(tilingData, x, y);                                                               \
        op.Run();                                                                                \
    }

LAUNCH_THRESHOLD_KERNEL(CV_8U, uint8_t, half)   // CV_8U
LAUNCH_THRESHOLD_KERNEL(CV_8S, int8_t, half)    // CV_8S
                                                // CV_16U
LAUNCH_THRESHOLD_KERNEL(CV_16S, int16_t, half)  // CV_16S
LAUNCH_THRESHOLD_KERNEL(CV_32S, int32_t, float) // CV_32S
LAUNCH_THRESHOLD_KERNEL(CV_32F, float, float)   // CV_32F
                                                // CV_64F
LAUNCH_THRESHOLD_KERNEL(CV_16F, half, half)     // CV_16F

#undef LAUNCH_THRESHOLD_KERNEL

#define CALL_THRESHOLD_KERNEL(NAME) launch_threshold_kernel_##NAME

extern "C" __global__ __aicore__ void threshold_opencv(GM_ADDR tilingGM, GM_ADDR x, GM_ADDR y)
{
    ThresholdOpencvTilingData tilingData;
    auto tempTilingGM = (__gm__ uint8_t*)tilingGM;
    auto tempTiling = (uint8_t*)&tilingData;
    for (int32_t i = 0; i < sizeof(ThresholdOpencvTilingData) / sizeof(uint8_t);
         ++i, ++tempTilingGM, ++tempTiling)
    {
        *tempTiling = *tempTilingGM;
    }

    // AscendC can only call inline functions, function pointer can't be used here.
    // Use Macro and switch case instead.
    switch (tilingData.dtype)
    {
        case 0:
            CALL_THRESHOLD_KERNEL(CV_8U)(&tilingData, x, y);
            break;
        case 1:
            CALL_THRESHOLD_KERNEL(CV_8S)(&tilingData, x, y);
            break;
        case 3:
            CALL_THRESHOLD_KERNEL(CV_16S)(&tilingData, x, y);
            break;
        case 4:
            CALL_THRESHOLD_KERNEL(CV_32S)(&tilingData, x, y);
            break;
        case 5:
            CALL_THRESHOLD_KERNEL(CV_32F)(&tilingData, x, y);
            break;
        case 7:
            CALL_THRESHOLD_KERNEL(CV_16F)(&tilingData, x, y);
            break;
        case 2: case 6: default: // CV_16U, CV_64F
            break;
    }
    // Clear tiling GM cache manually. (cce compiler bug)
    dcci(tilingGM, 1);
}
