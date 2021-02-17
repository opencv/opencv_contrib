// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").
#include "../../precomp.hpp"
#include "qrcode_reader.hpp"
#include <ctime>
#include "../common/bitarray.hpp"
#include "detector/detector.hpp"


using zxing::ErrorHandler;

namespace zxing {
namespace qrcode {

QRCodeReader::QRCodeReader() : decoder_() {
    readerState_ = QRCodeReader::READER_START;
    detectedDimension_ = -1;
    lastDecodeTime_ = 0;
    lastDecodeID_ = 0;
    decodeID_ = 0;
    lastPossibleAPCount_ = 0;
    possibleAPCount_ = 0;
    lastSamePossibleAPCountTimes_ = 0;
    samePossibleAPCountTimes_ = 0;
    lastRecommendedImageSizeType_ = 0;
    recommendedImageSizeType_ = 0;
    smoothMaxMultiple_ = 40;
}

Ref<Result> QRCodeReader::decode(Ref<BinaryBitmap> image) { return decode(image, DecodeHints()); }

Ref<Result> QRCodeReader::decode(Ref<BinaryBitmap> image, DecodeHints hints) {
    // Binarize image using the Histogram Binarized method and be binarized
    ErrorHandler err_handler;
    Ref<BitMatrix> imageBitMatrix = image->getBlackMatrix(err_handler);
    if (err_handler.ErrCode() || imageBitMatrix == NULL) return Ref<Result>();

    Ref<Result> rst = decodeMore(image, imageBitMatrix, hints, err_handler);
    if (err_handler.ErrCode() || rst == NULL) {
        // black white mirro!!!
        Ref<BitMatrix> invertedMatrix = image->getInvertedMatrix(err_handler);
        if (err_handler.ErrCode() || invertedMatrix == NULL) return Ref<Result>();
        Ref<Result> tmp_rst = decodeMore(image, invertedMatrix, hints, err_handler);
        if (err_handler.ErrCode() || tmp_rst == NULL) return Ref<Result>();
        return tmp_rst;
    }

    return rst;
}

Ref<Result> QRCodeReader::decodeMore(Ref<BinaryBitmap> image, Ref<BitMatrix> imageBitMatrix,
                                     DecodeHints hints, ErrorHandler &err_handler) {
    nowHints_ = hints;
    std::string ept;

    if (imageBitMatrix == NULL) return Ref<Result>();
    image->m_poUnicomBlock->Init();
    image->m_poUnicomBlock->Reset(imageBitMatrix);

    for (int tryTimes = 0; tryTimes < 1; tryTimes++) {
        Ref<Detector> detector(new Detector(imageBitMatrix, image->m_poUnicomBlock));
        err_handler.Reset();

        detector->detect(hints, err_handler);
        if (err_handler.ErrCode()) {
            err_handler = zxing::ReaderErrorHandler("error detect");
            setReaderState(detector->getState());
            ept = err_handler.ErrMsg();
            continue;
        }

        setReaderState(detector->getState());

        int possiblePatternCount = detector->getPossiblePatternCount();

        if (possiblePatternCount <= 0) {
            continue;
        }
        for (int i = 0; i < possiblePatternCount; i++) {
            // filter and perserve the highest score.
            Ref<FinderPatternInfo> patternInfo = detector->getFinderPatternInfo(i);
            setPatternFix(patternInfo->getPossibleFix());
            if (patternInfo->getAnglePossibleFix() < 0.6 && i) continue;

            int possibleAlignmentCount = 0;
            possibleAlignmentCount = detector->getPossibleAlignmentCount(i);
            if (possibleAlignmentCount < 0) continue;

            detectedDimension_ = detector->getDimension(i);
            possibleModuleSize_ = detector->getPossibleModuleSize(i);
            setPossibleAPCountByVersion(detector->getPossibleVersion(i));

            vector<bool> needTryVariousDeimensions(possibleAlignmentCount, false);
            for (int j = 0; j < possibleAlignmentCount; j++) {
                ArrayRef<Ref<ResultPoint> > points;
                err_handler.Reset();
                Ref<DetectorResult> detectorResult =
                    detector->getResultViaAlignment(i, j, detectedDimension_, err_handler);
                if (err_handler.ErrCode()) {
                    ept = err_handler.ErrCode();
                    setDecoderFix(decoder_.getPossibleFix(), points);
                    setReaderState(decoder_.getState());

                    if ((patternInfo->getPossibleFix() > 0.9 && decoder_.getPossibleFix() < 0.1)) {
                        needTryVariousDeimensions[j] = true;
                    }
                    continue;
                }

                points = detectorResult->getPoints();
                Ref<DecoderResult> decoderResult(
                    decoder_.decode(detectorResult->getBits(), err_handler));
                if (err_handler.ErrCode()) {
                    ept = err_handler.ErrCode();
                    setDecoderFix(decoder_.getPossibleFix(), points);
                    setReaderState(decoder_.getState());

                    if ((patternInfo->getPossibleFix() > 0.9 && decoder_.getPossibleFix() < 0.1)) {
                        needTryVariousDeimensions[j] = true;
                    }
                    continue;
                }

                // If the code was mirrored: swap the bottom-left and the
                // top-right points.
                if (decoderResult->getOtherClassName() == "QRCodeDecoderMetaData") {
                    decoderResult->getOther()->applyMirroredCorrection(points);
                }

                setDecoderFix(decoder_.getPossibleFix(), points);
                setReaderState(decoder_.getState());

                Ref<Result> result(
                    new Result(decoderResult->getText(), decoderResult->getRawBytes(), points,
                               decoderResult->getCharset(), decoderResult->getQRCodeVersion(),
                               decoderResult->getEcLevel(), decoderResult->getCharsetMode()));
                setSuccFix(points);

                return result;
            }
            // try different dimentions
            for (int j = 0; j < possibleAlignmentCount; j++) {
                err_handler.Reset();
                ArrayRef<Ref<ResultPoint> > points;
                if (needTryVariousDeimensions[j]) {
                    vector<int> possibleDimensions = getPossibleDimentions(detectedDimension_);
                    for (size_t k = 1; k < possibleDimensions.size(); k++) {
                        err_handler.Reset();
                        int dimension = possibleDimensions[k];

                        Ref<DetectorResult> detectorResult =
                            detector->getResultViaAlignment(i, j, dimension, err_handler);
                        if (err_handler.ErrCode() || detectorResult == NULL) {
                            ept = err_handler.ErrMsg();
                            setDecoderFix(decoder_.getPossibleFix(), points);
                            setReaderState(decoder_.getState());
                            continue;
                        }

                        points = detectorResult->getPoints();
                        Ref<DecoderResult> decoderResult(
                            decoder_.decode(detectorResult->getBits(), err_handler));
                        if (err_handler.ErrCode() || decoderResult == NULL) {
                            ept = err_handler.ErrMsg();
                            setDecoderFix(decoder_.getPossibleFix(), points);
                            setReaderState(decoder_.getState());
                            continue;
                        }

                        if (decoderResult->getOtherClassName() == "QRCodeDecoderMetaData") {
                            decoderResult->getOther()->applyMirroredCorrection(points);
                        }

                        setDecoderFix(decoder_.getPossibleFix(), points);
                        setReaderState(decoder_.getState());

                        detectedDimension_ = possibleDimensions[k];
                        Ref<Result> result(new Result(
                            decoderResult->getText(), decoderResult->getRawBytes(), points,
                            decoderResult->getCharset(), decoderResult->getQRCodeVersion(),
                            decoderResult->getEcLevel(), decoderResult->getCharsetMode()));

                        setSuccFix(points);
                        return result;
                    }
                }
            }
        }
    }
    return Ref<Result>();
}

vector<int> QRCodeReader::getPossibleDimentions(int detectDimension) {
    vector<int> possibleDimentions;
    possibleDimentions.clear();

    if (detectDimension < 0) {
        return possibleDimentions;
    }

    possibleDimentions.push_back(detectDimension);

    if (detectDimension <= 169 && detectDimension >= 73) {
        possibleDimentions.push_back(detectDimension + 4);
        possibleDimentions.push_back(detectDimension - 4);
        possibleDimentions.push_back(detectDimension - 8);
        possibleDimentions.push_back(detectDimension + 8);
    } else if (detectDimension <= 69 && detectDimension >= 45) {
        possibleDimentions.push_back(detectDimension + 4);
        possibleDimentions.push_back(detectDimension - 4);
    }

    if (detectDimension == 19) {
        possibleDimentions.push_back(21);
    }

    return possibleDimentions;
}

void QRCodeReader::setPossibleAPCountByVersion(unsigned int version) {
    // cout<<"setPossibleAPCountByVersion"<<endl;
    if (version < 2)
        possibleAPCount_ = 0;
    else if (version < 7)
        possibleAPCount_ = 1;
    else if (version < 14)
        possibleAPCount_ = 2;
    else if (version < 21)
        possibleAPCount_ = 3;
    else if (version < 28)
        possibleAPCount_ = 4;
    else if (version < 35)
        possibleAPCount_ = 5;
    else
        possibleAPCount_ = 6;
}

float QRCodeReader::getPossibleFix() { return possibleQrcodeInfo_.possibleFix; }

int QRCodeReader::smooth(unsigned int *integral, Ref<BitMatrix> input, Ref<BitMatrix> output,
                         int window) {
    BitMatrix &imatrix = *input;
    BitMatrix &omatrix = *output;
    window >>= 1;
    int count = 0;
    int width = input->getWidth();
    int height = input->getHeight();
    int bitsize = imatrix.getRowBitsSize();

    bool *jrowtoset = new bool[bitsize];

    bool *jrow = NULL;

    jrow = NULL;

    unsigned int size = window * window;

    for (int j = (window + 1); j < (height - 1 - window); ++j) {
        int y1 = j - window - 1;
        int y2 = j + window;

        int offset1 = y1 * width;
        int offset2 = y2 * width;

        jrow = imatrix.getRowBoolPtr(j);

        memcpy(jrowtoset, jrow, bitsize * sizeof(bool));

        for (int i = (window + 1); i < (width - 1 - window); ++i) {
            int x1 = i - window - 1;
            int x2 = i + window;
            unsigned int sum = integral[offset2 + x2] - integral[offset2 + x1] +
                               integral[offset1 + x2] - integral[offset1 + x1];
            bool b = jrow[i];
            bool result;
            // the middle 1/3 contains informations of corner, these
            // informations is useful for finding the finder pattern
            int sum3 = 3 * sum;
            if ((unsigned int)sum3 <= size) {
                result = false;
            } else if ((unsigned int)sum3 >= size * 2) {
                result = true;
            } else {
                result = b;
            }

            if (result) {
                jrowtoset[i] = true;
            }
            count += (result ^ b) == 1 ? 1U : 0U;
        }
        omatrix.setRowBool(j, jrowtoset);
    }

    delete[] jrowtoset;
    return count;
}

void QRCodeReader::initIntegralOld(unsigned int *integral, Ref<BitMatrix> input) {
    BitMatrix &matrix = *input;
    int width = input->getWidth();
    int height = input->getHeight();

    bool *therow = NULL;

    therow = matrix.getRowBoolPtr(0);

    integral[0] = therow[0];

    int *s = new int[width];

    memset(s, 0, width * sizeof(int));

    integral[0] = therow[0];
    for (int j = 1; j < width; j++) {
        integral[j] = integral[j - 1] + therow[j];
        s[j] += therow[j];
    }

    int offset = width;
    unsigned int prevSum = 0;

    for (int i = 1; i < height; i++) {
        offset = i * width;
        therow = matrix.getRowBoolPtr(i);

        integral[offset] = integral[offset - width] + therow[0];
        offset++;

        for (int j = 1; j < width; j++) {
            s[j] += therow[j];
            integral[offset] = prevSum + s[j];
            prevSum = integral[offset];
            offset++;
        }
    }

    delete[] s;

    return;
}

void QRCodeReader::initIntegral(unsigned int *integral, Ref<BitMatrix> input) {
    BitMatrix &matrix = *input;
    int width = input->getWidth();
    int height = input->getHeight();

    bool *therow = NULL;

    therow = matrix.getRowBoolPtr(0);

    // first row only
    int rs = 0;
    for (int j = 0; j < width; j++) {
        rs += therow[j];
        integral[j] = rs;
    }

    // remaining cells are sum above and to the left
    int offset = 0;

    for (int i = 1; i < height; ++i) {
        therow = matrix.getRowBoolPtr(i);

        rs = 0;

        offset += width;

        for (int j = 0; j < width; ++j) {
            rs += therow[j];
            integral[offset + j] = rs + integral[offset - width + j];
        }
    }

    return;
}

int QRCodeReader::getRecommendedImageSizeTypeInteral() {
    if (time(0) - lastDecodeTime_ > 30) recommendedImageSizeType_ = 0;
    return recommendedImageSizeType_;
}

unsigned int QRCodeReader::getDecodeID() { return decodeID_; }

void QRCodeReader::setDecodeID(unsigned int id) {
    lastDecodeTime_ = time(0);

    decodeID_ = id;
    if (decodeID_ != lastDecodeID_) {
        lastDecodeID_ = decodeID_;
        lastPossibleAPCount_ = possibleAPCount_;
        lastSamePossibleAPCountTimes_ = samePossibleAPCountTimes_;
        lastRecommendedImageSizeType_ = getRecommendedImageSizeTypeInteral();
        possibleAPCount_ = 0;
        recommendedImageSizeType_ = 0;
    }
}

QRCodeReader::~QRCodeReader() {}
Decoder &QRCodeReader::getDecoder() { return decoder_; }

unsigned int QRCodeReader::getPossibleAPType() {
    int version = (detectedDimension_ - 21) / 4 + 1;
    setPossibleAPCountByVersion(version);
    return possibleAPCount_;
}
int QRCodeReader::getPossibleFixType() { return possibleQrcodeInfo_.possibleFix > 0.0 ? 1 : 0; }

void QRCodeReader::setPatternFix(float possibleFix) {
    possibleQrcodeInfo_.patternPossibleFix = possibleFix;
}

void QRCodeReader::setDecoderFix(float possibleFix, ArrayRef<Ref<ResultPoint> > border) {
    float realFix = possibleFix;
    if (possibleQrcodeInfo_.possibleFix < realFix) {
        possibleQrcodeInfo_.possibleFix = realFix;
        possibleQrcodeInfo_.qrcodeBorder.clear();
        possibleQrcodeInfo_.possibleModuleSize = possibleModuleSize_;
        if (border) {
            for (int i = 0; i < 4; ++i) {
                possibleQrcodeInfo_.qrcodeBorder.push_back(border[i]);
            }
        }
    }
}
void QRCodeReader::setSuccFix(ArrayRef<Ref<ResultPoint> > border) {
    possibleQrcodeInfo_.qrcodeBorder.clear();
    possibleQrcodeInfo_.possibleModuleSize = possibleModuleSize_;
    if (border) {
        for (int i = 0; i < 4; ++i) {
            possibleQrcodeInfo_.qrcodeBorder.push_back(border[i]);
        }
    }
}

void QRCodeReader::setReaderState(Detector::DetectorState state) {
    switch (state) {
        case Detector::START:
            this->readerState_ = QRCodeReader::DETECT_START;
            break;
        case Detector::FINDFINDERPATTERN:
            this->readerState_ = QRCodeReader::DETECT_FINDFINDERPATTERN;
            break;
        case Detector::FINDALIGNPATTERN:
            this->readerState_ = QRCodeReader::DETECT_FINDALIGNPATTERN;
            break;
    }
    return;
}
void QRCodeReader::setReaderState(Decoder::DecoderState state) {
    switch (state) {
        case Decoder::NOTSTART:
            this->readerState_ = QRCodeReader::DETECT_FAILD;
            break;
        case Decoder::START:
            if (this->readerState_ < QRCodeReader::DECODE_START) {
                this->readerState_ = QRCodeReader::DECODE_START;
            }
            break;
        case Decoder::READVERSION:
            if (this->readerState_ < QRCodeReader::DECODE_READVERSION) {
                this->readerState_ = QRCodeReader::DECODE_READVERSION;
            }
            break;
        case Decoder::READERRORCORRECTIONLEVEL:
            if (this->readerState_ < QRCodeReader::DECODE_READERRORCORRECTIONLEVEL) {
                this->readerState_ = QRCodeReader::DECODE_READERRORCORRECTIONLEVEL;
            }
            break;
        case Decoder::READCODEWORDSORRECTIONLEVEL:
            if (this->readerState_ < QRCodeReader::DECODE_READCODEWORDSORRECTIONLEVEL) {
                this->readerState_ = QRCodeReader::DECODE_READCODEWORDSORRECTIONLEVEL;
            }
            break;
        case Decoder::FINISH:
            if (this->readerState_ < QRCodeReader::DECODE_FINISH) {
                this->readerState_ = QRCodeReader::DECODE_FINISH;
            }
            break;
    }
    return;
}
}  // namespace qrcode
}  // namespace zxing
