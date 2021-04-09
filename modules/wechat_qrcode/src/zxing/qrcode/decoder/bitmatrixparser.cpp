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
#include "bitmatrixparser.hpp"
#include "datamask.hpp"

using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

int BitMatrixParser::copyBit(size_t x, size_t y, int versionBits) {
    bool bit = ((mirror_ ? bitMatrix_->get(y, x) : bitMatrix_->get(x, y)) != (unsigned char)0);
    return bit ? (versionBits << 1) | 0x1 : versionBits << 1;
}

BitMatrixParser::BitMatrixParser(Ref<BitMatrix> bitMatrix, ErrorHandler &err_handler)
    : bitMatrix_(bitMatrix), parsedVersion_(0), parsedFormatInfo_() {
    mirror_ = false;
    size_t dimension = bitMatrix->getHeight();

    if ((dimension < 21) || (dimension & 0x03) != 1) {
        err_handler = zxing::ReaderErrorHandler("Dimension must be 1 mod 4 and >= 21");
        return;
    }
}

Ref<FormatInformation> BitMatrixParser::readFormatInformation(ErrorHandler &err_handler) {
    if (parsedFormatInfo_ != 0) {
        return parsedFormatInfo_;
    }

    // Read top-left format info bits
    int formatInfoBits1 = 0;
    for (int i = 0; i < 6; i++) {
        formatInfoBits1 = copyBit(i, 8, formatInfoBits1);
    }
    // .. and skip a bit in the timing pattern ...
    formatInfoBits1 = copyBit(7, 8, formatInfoBits1);
    formatInfoBits1 = copyBit(8, 8, formatInfoBits1);
    formatInfoBits1 = copyBit(8, 7, formatInfoBits1);
    // .. and skip a bit in the timing pattern ...
    for (int j = 5; j >= 0; j--) {
        formatInfoBits1 = copyBit(8, j, formatInfoBits1);
    }

    // Read the top-right/bottom-left pattern
    int dimension = bitMatrix_->getHeight();
    int formatInfoBits2 = 0;
    int jMin = dimension - 7;
    for (int j = dimension - 1; j >= jMin; j--) {
        formatInfoBits2 = copyBit(8, j, formatInfoBits2);
    }
    for (int i = dimension - 8; i < dimension; i++) {
        formatInfoBits2 = copyBit(i, 8, formatInfoBits2);
    }

    parsedFormatInfo_ =
        FormatInformation::decodeFormatInformation(formatInfoBits1, formatInfoBits2);
    if (parsedFormatInfo_ != 0) {
        return parsedFormatInfo_;
    }
    err_handler = zxing::ReaderErrorHandler("Could not decode format information");
    return Ref<FormatInformation>();
}

Version *BitMatrixParser::readVersion(ErrorHandler &err_handler) {
    if (parsedVersion_ != 0) {
        return parsedVersion_;
    }

    int dimension = bitMatrix_->getHeight();

    int provisionalVersion = (dimension - 17) >> 2;
    if (provisionalVersion <= 6) {
        Version *version = Version::getVersionForNumber(provisionalVersion, err_handler);
        if (err_handler.ErrCode()) return NULL;
        return version;
    }

    // Read top-right version info: 3 wide by 6 tall
    int versionBits = 0;
    for (int y = 5; y >= 0; y--) {
        int xMin = dimension - 11;
        for (int x = dimension - 9; x >= xMin; x--) {
            versionBits = copyBit(x, y, versionBits);
        }
    }

    parsedVersion_ = Version::decodeVersionInformation(versionBits);
    if (parsedVersion_ != 0 && parsedVersion_->getDimensionForVersion(err_handler) == dimension) {
        return parsedVersion_;
    }

    // Hmm, failed. Try bottom left: 6 wide by 3 tall
    versionBits = 0;
    for (int x = 5; x >= 0; x--) {
        int yMin = dimension - 11;
        for (int y = dimension - 9; y >= yMin; y--) {
            versionBits = copyBit(x, y, versionBits);
        }
    }

    parsedVersion_ = Version::decodeVersionInformation(versionBits);
    if (parsedVersion_ == NULL) {
        err_handler = zxing::ReaderErrorHandler("Could not decode version");
        return NULL;
    }

    if (parsedVersion_ != 0 && parsedVersion_->getDimensionForVersion(err_handler) == dimension) {
        return parsedVersion_;
    }

    err_handler = zxing::ReaderErrorHandler("Could not decode version");
    return NULL;
}

/**
 * <p>Reads the bits in the {@link BitMatrix} representing the finder pattern in
 * the correct order in order to reconstruct the codewords bytes contained
 * within the QR Code.</p>
 *
 * @return bytes encoded within the QR Code
 */
ArrayRef<char> BitMatrixParser::readCodewords(ErrorHandler &err_handler) {
    Ref<FormatInformation> formatInfo = readFormatInformation(err_handler);
    if (err_handler.ErrCode()) return ArrayRef<char>();

    Version *version = readVersion(err_handler);
    if (err_handler.ErrCode()) return ArrayRef<char>();

    DataMask &dataMask = DataMask::forReference((int)formatInfo->getDataMask(), err_handler);
    if (err_handler.ErrCode()) return ArrayRef<char>();
    //	cout << (int)formatInfo->getDataMask() << endl;
    int dimension = bitMatrix_->getHeight();

    dataMask.unmaskBitMatrix(*bitMatrix_, dimension);

    //		cerr << *bitMatrix_ << endl;
    //	cerr << version->getTotalCodewords() << endl;

    Ref<BitMatrix> functionPattern = version->buildFunctionPattern(err_handler);
    if (err_handler.ErrCode()) return ArrayRef<char>();

    //	cout << *functionPattern << endl;

    bool readingUp = true;
    ArrayRef<char> result(version->getTotalCodewords());
    int resultOffset = 0;
    int currentByte = 0;
    int bitsRead = 0;
    // Read columns in pairs, from right to left
    for (int x = dimension - 1; x > 0; x -= 2) {
        if (x == 6) {
            // Skip whole column with vertical alignment pattern;
            // saves time and makes the other code proceed more cleanly
            x--;
        }
        // Read alternatingly from bottom to top then top to bottom
        for (int counter = 0; counter < dimension; counter++) {
            int y = readingUp ? dimension - 1 - counter : counter;
            for (int col = 0; col < 2; col++) {
                // Ignore bits covered by the function pattern
                if (!functionPattern->get(x - col, y)) {
                    // Read a bit
                    bitsRead++;
                    currentByte <<= 1;
                    if (bitMatrix_->get(x - col, y)) {
                        currentByte |= 1;
                    }
                    // If we've made a whole byte, save it off
                    if (bitsRead == 8) {
                        result[resultOffset++] = (char)currentByte;
                        bitsRead = 0;
                        currentByte = 0;
                    }
                }
            }
        }
        readingUp = !readingUp;  // switch directions
    }

    if (resultOffset != version->getTotalCodewords()) {
        err_handler = zxing::ReaderErrorHandler("Did not read all codewords");
        return ArrayRef<char>();
    }

    return result;
}

/**
 * Revert the mask removal done while reading the code words. The bit matrix
 * should revert to its original state.
 */
void BitMatrixParser::remask() {
    if (parsedFormatInfo_ == NULL) {
        return;  // We have no format information, and have no data mask
    }
    ErrorHandler err_handler;
    DataMask &dataMask = DataMask::forReference(parsedFormatInfo_->getDataMask(), err_handler);
    if (err_handler.ErrCode()) return;
    int dimension = bitMatrix_->getHeight();
    dataMask.unmaskBitMatrix(*bitMatrix_, dimension);
}

/**
 * Prepare the parser for a mirrored operation.
 * This flag has effect only on the {@link #readFormatInformation()} and the
 * {@link #readVersion()}. Before proceeding with {@link #readCodewords()} the
 * {@link #mirror()} method should be called.
 *
 * @param mirror Whether to read version and format information mirrored.
 */
void BitMatrixParser::setMirror(bool mirror) {
    parsedVersion_ = NULL;
    parsedFormatInfo_ = NULL;
    mirror_ = mirror;
}

/** Mirror the bit matrix in order to attempt a second reading. */
void BitMatrixParser::mirror() {
    for (int x = 0; x < bitMatrix_->getWidth(); x++) {
        for (int y = x + 1; y < bitMatrix_->getHeight(); y++) {
            if (bitMatrix_->get(x, y) != bitMatrix_->get(y, x)) {
                bitMatrix_->flip(y, x);
                bitMatrix_->flip(x, y);
            }
        }
    }
}

}  // namespace qrcode
}  // namespace zxing
