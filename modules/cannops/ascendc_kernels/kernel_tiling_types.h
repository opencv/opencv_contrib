#ifndef KERNEL_TILING_H
#define KERNEL_TILING_H

/*
 * threshType:
 *   THRESH_BINARY     = 0,
 *   THRESH_BINARY_INV = 1,
 *   THRESH_TRUNC      = 2,
 *   THRESH_TOZERO     = 3,
 *   THRESH_TOZERO_INV = 4,
*/
#pragma pack(push, 8)
struct ThresholdOpencvTilingData
{
    float maxVal;
    float thresh;
    uint32_t totalLength;
    uint8_t threshType;
    uint8_t dtype;
};
#pragma pack(pop)
#endif // KERNEL_TILING_H
