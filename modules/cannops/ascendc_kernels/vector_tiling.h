#ifndef TILING_KERNEL_H
#define TILING_KERNEL_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignNCeil(int32_t n, int32_t align) { return ((n + align) & ~(align-1)); }

inline __aicore__ int32_t AlignNFloor(int32_t n, int32_t align) { return (n & ~(align-1)); }

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t UB_BUF_LEN = 248 * 1024;

struct VectorTiling {
  __aicore__ inline void calculate(uint64_t _totalLength, uint64_t _blockNum,
                                   uint64_t _blockIdx, uint64_t _variableBytesPerElem, uint32_t _align) {
    totalLength = _totalLength;
    blockNum = _blockNum;
    blockIdx = _blockIdx;
    variableBytesPerElem = _variableBytesPerElem;
    blockLength = 0;
    blockOffset = 0;
    align = _align;
    GetBlockLengthAndOffset();
    GetLoopLengthAndCount();
#ifdef __CCE_KT_TEST__
    std::cout << "Block(" << blockIdx << "): BlockLength = " << blockLength
              << ", BlockOffset = " << blockOffset
              << ", LoopLength = " << loopLength
              << ", LoopCount = " << loopCount
              << ", LoopTailLength = " << loopTailLength << std::endl;
#endif
  }

  __aicore__ inline void GetBlockLengthAndOffset() {
    // Data should Align by 32B.
    uint32_t fullBlockLength = AlignNCeil(totalLength / blockNum, 32);
    // Some core may get no data after Align32 Ceil.
    uint32_t fullBlockNum = totalLength / fullBlockLength;
    uint32_t blockTailLength = totalLength % fullBlockLength;

    if (blockIdx < fullBlockNum) {
      blockLength = fullBlockLength;
      blockOffset = blockIdx * blockLength;
      // Last block must less than full block num.
    } else if (blockTailLength != 0 && blockIdx == fullBlockNum) {
      blockLength = blockTailLength;
      blockOffset = blockIdx * fullBlockLength;
    }
  }

  /**
   * @brief Get length for one loop and loop count.
   * Use as much UB buf as possible.
   */
  __aicore__ inline void GetLoopLengthAndCount() {
    loopLength = AlignNFloor(UB_BUF_LEN / variableBytesPerElem, align);
    loopCount = blockLength / loopLength;
    loopTailLength = blockLength - (loopLength * loopCount);
  }

  uint64_t totalLength;
  uint64_t blockNum;
  uint64_t blockIdx;
  uint64_t variableBytesPerElem;
  uint32_t blockLength;
  uint32_t blockOffset;
  uint32_t loopLength;
  uint32_t loopCount;
  uint32_t loopTailLength;
  uint32_t align;
};

#endif  // TILING_KERNEL_H
