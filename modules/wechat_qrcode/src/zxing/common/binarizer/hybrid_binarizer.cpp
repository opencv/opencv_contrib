// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../../precomp.hpp"
#include "hybrid_binarizer.hpp"

using zxing::HybridBinarizer;
using zxing::BINARIZER_BLOCK;

// This class uses 5*5 blocks to compute local luminance, where each block is
// 8*8 pixels So this is the smallest dimension in each axis we can accept.
namespace {
const int BLOCK_SIZE_POWER = 3;
const int BLOCK_SIZE = 1 << BLOCK_SIZE_POWER;  // ...0100...00
const int BLOCK_SIZE_MASK = BLOCK_SIZE - 1;    // ...0011...11
const int MINIMUM_DIMENSION = BLOCK_SIZE * 5;
#ifdef USE_SET_INT
const int BITS_PER_BYTE = 8;
const int BITS_PER_WORD = BitMatrix::bitsPerWord;
#endif
}  // namespace

HybridBinarizer::HybridBinarizer(Ref<LuminanceSource> source) : GlobalHistogramBinarizer(source) {
    int subWidth = width >> BLOCK_SIZE_POWER;
    if ((width & BLOCK_SIZE_MASK) != 0) {
        subWidth++;
    }
    int subHeight = height >> BLOCK_SIZE_POWER;
    if ((height & BLOCK_SIZE_MASK) != 0) {
        subHeight++;
    }

    grayByte_ = source->getByteMatrix();

    blocks_ = getBlockArray(subWidth * subHeight);

    subWidth_ = subWidth;
    subHeight_ = subHeight;

    initBlocks();
    initBlockIntegral();
}

HybridBinarizer::~HybridBinarizer() {
}

Ref<Binarizer> HybridBinarizer::createBinarizer(Ref<LuminanceSource> source) {
    return Ref<Binarizer>(new GlobalHistogramBinarizer(source));
}

int HybridBinarizer::initBlockIntegral() {
    blockIntegralWidth = subWidth_ + 1;
    blockIntegralHeight = subHeight_ + 1;
    blockIntegral_ = new Array<int>(blockIntegralWidth * blockIntegralHeight);

    int* integral = blockIntegral_->data();

    // unsigned char* therow = grayByte_->getByteRow(0);

    // first row only
    int rs = 0;

    for (int j = 0; j < blockIntegralWidth; j++) {
        integral[j] = 0;
    }

    for (int i = 0; i < blockIntegralHeight; i++) {
        integral[i * blockIntegralWidth] = 0;
    }

    // remaining cells are sum above and to the left
    int offsetBlock = 0;
    int offsetIntegral = 0;

    for (int i = 0; i < subHeight_; ++i) {
        // therow = grayByte_->getByteRow(i);
        offsetBlock = i * subWidth_;
        offsetIntegral = (i + 1) * blockIntegralWidth;
        rs = 0;

        for (int j = 0; j < subWidth_; ++j) {
            rs += blocks_[offsetBlock + j].threshold;
            integral[offsetIntegral + j + 1] = rs + integral[offsetIntegral - blockIntegralWidth + j + 1];
        }
    }

    return 1;
}

/**
 * Calculates the final BitMatrix once for all requests. This could be called
 * once from the constructor instead, but there are some advantages to doing it
 * lazily, such as making profiling easier, and not doing heavy lifting when
 * callers don't expect it.
 */
Ref<BitMatrix> HybridBinarizer::getBlackMatrix(ErrorHandler& err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeByBlock(err_handler);
        if (err_handler.ErrCode()) return Ref<BitMatrix>();
    }

    // First call binarize image in child class to get matrix0_ and binCache
    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackMatrix(err_handler);
}

#if 1
/**
 * Calculate black row from BitMatrix
 * If BitMatrix has been calculated then just get the row
 * If BitMatrix has not been calculated then call getBlackMatrix first
 */
Ref<BitArray> HybridBinarizer::getBlackRow(int y, Ref<BitArray> row, ErrorHandler& err_handler) {
    // First call binarize image in child class to get matrix0_ and binCache
    if (!matrix0_) {
        binarizeByBlock(err_handler);
        if (err_handler.ErrCode()) return Ref<BitArray>();
    }

    // Call parent getBlackMatrix to get current matrix
    return Binarizer::getBlackRow(y, row, err_handler);
}
#endif

namespace {
inline int cap(int value, int min, int max) {
    return value < min ? min : value > max ? max : value;
}
}  // namespace


// For each block in the image, calculates the average black point using a 5*5
// grid of the blocks around it. Also handles the corner cases (fractional
// blocks are computed based on the last pixels in the row/column which are also
// used in the previous block.)

#define THRES_BLOCKSIZE 2

// No use of level now
ArrayRef<int> HybridBinarizer::getBlackPoints() {
    int blackWidth, blackHeight;

    blackWidth = subWidth_;
    blackHeight = subHeight_;

    ArrayRef<int> blackPoints(blackWidth * blackHeight);

    int* blackArray = blackPoints->data();

    int offset = 0;
    for (int i = 0; i < blackHeight; i++) {
        offset = i * blackWidth;
        for (int j = 0; j < blackWidth; j++) {
            blackArray[offset + j] = blocks_[offset + j].threshold;
        }
    }

    return blackPoints;
}

// Original code 20140606
void HybridBinarizer::calculateThresholdForBlock(Ref<ByteMatrix>& _luminances, int subWidth,
                                                 int subHeight, int SIZE_POWER,
                                                 // ArrayRef<int> &blackPoints,
                                                 Ref<BitMatrix> const& matrix,
                                                 ErrorHandler& err_handler) {
    int block_size = 1 << SIZE_POWER;

    int maxYOffset = height - block_size;
    int maxXOffset = width - block_size;

    int* blockIntegral = blockIntegral_->data();

    int blockArea = ((2 * THRES_BLOCKSIZE + 1) * (2 * THRES_BLOCKSIZE + 1));

    for (int y = 0; y < subHeight; y++) {
        int yoffset = y << SIZE_POWER;
        if (yoffset > maxYOffset) {
            yoffset = maxYOffset;
        }
        for (int x = 0; x < subWidth; x++) {
            int xoffset = x << SIZE_POWER;
            if (xoffset > maxXOffset) {
                xoffset = maxXOffset;
            }
            int left = cap(x, THRES_BLOCKSIZE, subWidth - THRES_BLOCKSIZE - 1);
            int top = cap(y, THRES_BLOCKSIZE, subHeight - THRES_BLOCKSIZE - 1);

            int sum = 0;
            // int sum2 = 0;

            int offset1 = (top - THRES_BLOCKSIZE) * blockIntegralWidth + left - THRES_BLOCKSIZE;
            int offset2 = (top + THRES_BLOCKSIZE + 1) * blockIntegralWidth + left - THRES_BLOCKSIZE;

            int blocksize = THRES_BLOCKSIZE * 2 + 1;

            sum = blockIntegral[offset1] - blockIntegral[offset1 + blocksize] -
                  blockIntegral[offset2] + blockIntegral[offset2 + blocksize];

            int average = sum / blockArea;
            thresholdBlock(_luminances, xoffset, yoffset, average, matrix, err_handler);
            if (err_handler.ErrCode()) return;
        }
    }
}

#ifdef USE_SET_INT
void HybridBinarizer::thresholdFourBlocks(Ref<ByteMatrix>& luminances, int xoffset, int yoffset,
                                          int* thresholds, int stride,
                                          Ref<BitMatrix> const& matrix) {
    int setIntCircle = BITS_PER_WORD / BITS_PER_BYTE;
    for (int y = 0; y < BLOCK_SIZE; y++) {
        unsigned char* pTemp = luminances->getByteRow(yoffset + y);
        pTemp = pTemp + xoffset;
        unsigned int valueInt = 0;
        int bitPosition = 0;
        for (int k = 0; k < setIntCircle; k++) {
            for (int x = 0; x < BLOCK_SIZE; x++) {
                int pixel = *pTemp++;
                if (pixel <= thresholds[k]) {
                    // bitPosition=(3-k)*8+x;
                    valueInt |= (unsigned int)1 << bitPosition;
                }
                bitPosition++;
            }
        }
        matrix->setIntOneTime(xoffset, yoffset + y, valueInt);
    }
    return;
}
#endif

// Applies a single threshold to a block of pixels
void HybridBinarizer::thresholdBlock(Ref<ByteMatrix>& _luminances, int xoffset, int yoffset,
                                     int threshold, Ref<BitMatrix> const& matrix,
                                     ErrorHandler& err_handler) {
    int rowBitsSize = matrix->getRowBitsSize();
    int rowSize = width;

    int rowBitStep = rowBitsSize - BLOCK_SIZE;
    int rowStep = rowSize - BLOCK_SIZE;

    unsigned char* pTemp = _luminances->getByteRow(yoffset, err_handler);
    if (err_handler.ErrCode()) return;
    bool* bpTemp = matrix->getRowBoolPtr(yoffset);

    pTemp += xoffset;
    bpTemp += xoffset;

    for (int y = 0; y < BLOCK_SIZE; y++) {
        for (int x = 0; x < BLOCK_SIZE; x++) {
            // comparison needs to be <= so that black == 0 pixels are black
            // even if the threshold is 0.
            *bpTemp++ = (*pTemp++ <= threshold) ? true : false;
        }

        pTemp += rowBitStep;
        bpTemp += rowStep;
    }
}

void HybridBinarizer::thresholdIrregularBlock(Ref<ByteMatrix>& _luminances, int xoffset,
                                              int yoffset, int blockWidth, int blockHeight,
                                              int threshold, Ref<BitMatrix> const& matrix,
                                              ErrorHandler& err_handler) {
    for (int y = 0; y < blockHeight; y++) {
        unsigned char* pTemp = _luminances->getByteRow(yoffset + y, err_handler);
        if (err_handler.ErrCode()) return;
        pTemp = pTemp + xoffset;
        for (int x = 0; x < blockWidth; x++) {
            // comparison needs to be <= so that black == 0 pixels are black
            // even if the threshold is 0.
            int pixel = *pTemp++;
            if (pixel <= threshold) {
                matrix->set(xoffset + x, yoffset + y);
            }
        }
    }
}

namespace {

inline int getBlackPointFromNeighbors(ArrayRef<BINARIZER_BLOCK> block, int subWidth, int x, int y) {
    return (block[(y - 1) * subWidth + x].threshold + 2 * block[y * subWidth + x - 1].threshold +
            block[(y - 1) * subWidth + x - 1].threshold) >>
           2;
}

}  // namespace


#define MIN_DYNAMIC_RANGE 24

// Calculates a single black point for each block of pixels and saves it away.
int HybridBinarizer::initBlocks() {
    Ref<ByteMatrix>& _luminances = grayByte_;
    int subWidth = subWidth_;
    int subHeight = subHeight_;

    unsigned char* bytes = _luminances->bytes;

    const int minDynamicRange = 24;

    for (int y = 0; y < subHeight; y++) {
        int yoffset = y << BLOCK_SIZE_POWER;
        int maxYOffset = height - BLOCK_SIZE;
        if (yoffset > maxYOffset) yoffset = maxYOffset;
        for (int x = 0; x < subWidth; x++) {
            int xoffset = x << BLOCK_SIZE_POWER;
            int maxXOffset = width - BLOCK_SIZE;
            if (xoffset > maxXOffset) xoffset = maxXOffset;
            int sum = 0;
            int min = 0xFF;
            int max = 0;
            for (int yy = 0, offset = yoffset * width + xoffset; yy < BLOCK_SIZE;
                 yy++, offset += width) {
                for (int xx = 0; xx < BLOCK_SIZE; xx++) {
                    // int pixel = luminances->bytes[offset + xx] & 0xFF;
                    int pixel = bytes[offset + xx];
                    sum += pixel;

                    // still looking for good contrast
                    if (pixel < min) {
                        min = pixel;
                    }
                    if (pixel > max) {
                        max = pixel;
                    }
                }

                // short-circuit min/max tests once dynamic range is met
                if (max - min > minDynamicRange) {
                    // finish the rest of the rows quickly
                    for (yy++, offset += width; yy < BLOCK_SIZE; yy++, offset += width) {
                        for (int xx = 0; xx < BLOCK_SIZE; xx += 2) {
                            sum += bytes[offset + xx];
                            sum += bytes[offset + xx + 1];
                        }
                    }
                }
            }

            blocks_[y * subWidth + x].min = min;
            blocks_[y * subWidth + x].max = max;
            blocks_[y * subWidth + x].sum = sum;
            blocks_[y * subWidth + x].threshold =
                getBlockThreshold(x, y, subWidth, sum, min, max, minDynamicRange, BLOCK_SIZE_POWER);
        }
    }

    return 1;
}

int HybridBinarizer::getBlockThreshold(int x, int y, int subWidth, int sum, int min, int max,
                                       int minDynamicRange, int SIZE_POWER) {
    // See
    // http://groups.google.com/group/zxing/browse_thread/thread/d06efa2c35a7ddc0

    // The default estimate is the average of the values in the block.
    int average = sum >> (SIZE_POWER * 2);
    if (max - min <= minDynamicRange) {
        // If variation within the block is low, assume this is a block withe
        // only light or only dark pixels. In that case we do not want to use
        // the average, as it would divide this low contrast area into black and
        // white pixels, essentially creating data out of noise. The default
        // assumption is that the block is light/background. Since no estimate
        // for the level of dark pixels exists locally, use half the min for the
        // block.
        average = min >> 1;
        if (y > 0 && x > 0) {
            // Correct the "white background" assumption for blocks that have
            // neighbors by comparing the pixels in this block to the previously
            // calculated black points. This is based on the fact that dark
            // barcode symbology is always surrounded by some amout of light
            // background for which reasonable black point estimates were made.
            // The bp estimated at the boundaries is used for the interior.
            int bp = getBlackPointFromNeighbors(blocks_, subWidth, x, y);
            // The (min<bp) is arbitrary but works better than other heuristics
            // that were tried.
            if (min < bp) {
                average = bp;
            }
        }
    }

    // blocks_[y * subWidth + x].average = average;
    // blocks_[y * subWidth + x].threshold = average;

    return average;
}


int HybridBinarizer::binarizeByBlock(ErrorHandler& err_handler) {
    if (width >= MINIMUM_DIMENSION && height >= MINIMUM_DIMENSION) {
        Ref<BitMatrix> newMatrix(new BitMatrix(width, height, err_handler));
        if (err_handler.ErrCode()) return -1;

        calculateThresholdForBlock(grayByte_, subWidth_, subHeight_, BLOCK_SIZE_POWER, newMatrix,
                                   err_handler);
        if (err_handler.ErrCode()) return -1;

        matrix0_ = newMatrix;

    } else {
        // If the image is too small, fall back to the global histogram
        // approach.
        matrix0_ = GlobalHistogramBinarizer::getBlackMatrix(err_handler);
        if (err_handler.ErrCode()) return 1;
    }
    // return matrix0_;
    return 1;
}
