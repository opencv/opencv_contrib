// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#include "imgsource.hpp"
#include "precomp.hpp"

#include "zxing/common/illegal_argument_exception.hpp"

using zxing::ArrayRef;
using zxing::ByteMatrix;
using zxing::ErrorHandler;
using zxing::LuminanceSource;
using zxing::Ref;
namespace cv {
namespace wechat_qrcode {
inline unsigned char ImgSource::convertPixel(unsigned char const* pixel,
                                             ErrorHandler& err_handler) const {
    if (_comps == 1 || _comps == 2) {
        // Gray or gray+alpha
        return pixel[0];
    }
    if (_comps == 3 || _comps == 4) {
        // Red, Green, Blue, (Alpha)
        // We assume 16 bit values here
        // 0x200 = 1<<9, half an lsb of the result to force rounding
        return (unsigned char)((306 * (int)pixel[0] + 601 * (int)pixel[1] + 117 * (int)pixel[2] +
                                0x200) >>
                               10);
    } else {
        err_handler = zxing::IllegalArgumentErrorHandler("Unexpected image depth");
        return 0;
    }
}

// Initialize the ImgSource
ImgSource::ImgSource(unsigned char* pixels, int width, int height, int comps_, int pixelStep_,
                     ErrorHandler& err_handler)
    : Super(width, height) {
    tvInter = -1;
    luminances = new unsigned char[width * height];
    memset(luminances, 0, width * height);

    rgbs = pixels;
    _comps = comps_;
    _pixelStep = pixelStep_;

    dataWidth = width;
    dataHeight = height;
    left = 0;
    top = 0;

    maxDataWidth = width;
    maxDataHeight = height;

    // Make gray luminances first
    makeGray(err_handler);
}

// Added for crop function
ImgSource::ImgSource(unsigned char* pixels, int width, int height, int left_, int top_,
                     int cropWidth, int cropHeight, int comps_, int pixelStep_,
                     ErrorHandler& err_handler)
    : Super(cropWidth, cropHeight) {
    rgbs = pixels;
    _comps = comps_;
    _pixelStep = pixelStep_;

    dataWidth = width;
    dataHeight = height;
    left = left_;
    top = top_;

    // super(width, height);
    if ((left_ + cropWidth) > dataWidth || (top_ + cropHeight) > dataHeight || top_ < 0 ||
        left_ < 0) {
        err_handler =
            zxing::IllegalArgumentErrorHandler("Crop rectangle does not fit within image data.");
        return;
    }

    luminances = new unsigned char[width * height];

    // Make gray luminances first
    makeGray(err_handler);
    if (err_handler.ErrCode()) return;
}

ImgSource::~ImgSource() {
    if (luminances != NULL) {
        delete[] luminances;
    }
}

Ref<ImgSource> ImgSource::create(unsigned char* pixels, int width, int height, int comps,
                                 int pixelStep, ErrorHandler& err_handler) {
    return Ref<ImgSource>(new ImgSource(pixels, width, height, comps, pixelStep, err_handler));
}

Ref<ImgSource> ImgSource::create(unsigned char* pixels, int width, int height, int left, int top,
                                 int cropWidth, int cropHeight, int comps, int pixelStep,
                                 zxing::ErrorHandler& err_handler) {
    return Ref<ImgSource>(new ImgSource(pixels, width, height, left, top, cropWidth, cropHeight,
                                        comps, pixelStep, err_handler));
}

void ImgSource::reset(unsigned char* pixels, int width, int height, int comps, int pixelStep,
                      ErrorHandler& err_handler) {
    rgbs = pixels;
    _comps = comps;
    _pixelStep = pixelStep;
    left = 0;
    top = 0;

    setWidth(width);
    setHeight(height);
    dataWidth = width;
    dataHeight = height;
    makeGrayReset(err_handler);
}

ArrayRef<char> ImgSource::getRow(int y, zxing::ArrayRef<char> row,
                                 zxing::ErrorHandler& err_handler) const {
    if (y < 0 || y >= getHeight()) {
        err_handler = zxing::IllegalArgumentErrorHandler("Requested row is outside the image");
        return ArrayRef<char>();
    }

    int width = getWidth();
    if (row->data() == NULL || row->empty() || row->size() < width) {
        row = zxing::ArrayRef<char>(width);
    }
    int offset = (y + top) * dataWidth + left;

    char* rowPtr = &row[0];
    arrayCopy(luminances, offset, rowPtr, 0, width);

    return row;
}

/** This is a more efficient implementation. */
ArrayRef<char> ImgSource::getMatrix() const {
    int width = getWidth();
    int height = getHeight();

    int area = width * height;

    if (tvInter > -1) tvDenoising();

    // If the caller asks for the entire underlying image, save the copy and
    // give them the original data. The docs specifically warn that
    // result.length must be ignored.
    if (width == dataWidth && height == dataHeight) {
        return _matrix;
    }

    zxing::ArrayRef<char> newMatrix = zxing::ArrayRef<char>(area);

    int inputOffset = top * dataWidth + left;

    // If the width matches the full width of the underlying data, perform a
    // single copy.
    if (width == dataWidth) {
        arrayCopy(luminances, inputOffset, &newMatrix[0], 0, area);
        return newMatrix;
    }

    // Otherwise copy one cropped row at a time.
    for (int y = 0; y < height; y++) {
        int outputOffset = y * width;
        arrayCopy(luminances, inputOffset, &newMatrix[0], outputOffset, width);
        inputOffset += dataWidth;
    }
    return newMatrix;
}

void ImgSource::makeGrayRow(int y, ErrorHandler& err_handler) {
    int offsetRGB = y * dataWidth * _pixelStep;

    const unsigned char* pixelRow = rgbs + offsetRGB;

    int offsetGRAY = y * dataWidth;

    for (int x = 0; x < dataWidth; x++) {
        luminances[offsetGRAY + x] = convertPixel(pixelRow + (x * _pixelStep), err_handler);
        if (err_handler.ErrCode()) return;
    }
}

void ImgSource::makeGray(ErrorHandler& err_handler) {
    int area = dataWidth * dataHeight;
    _matrix = zxing::ArrayRef<char>(area);

    if (_comps == 1) {
        arrayCopy(rgbs, 0, &_matrix[0], 0, area);
    } else {
        for (int y = 0; y < dataHeight; y++) {
            makeGrayRow(y, err_handler);
        }
        if (err_handler.ErrCode()) return;

        arrayCopy(luminances, 0, &_matrix[0], 0, area);
    }
}

void ImgSource::makeGrayReset(ErrorHandler& err_handler) {
    int area = dataWidth * dataHeight;
    if (_comps == 1) {
        arrayCopy(rgbs, 0, &_matrix[0], 0, area);
    } else {
        for (int y = 0; y < dataHeight; y++) {
            makeGrayRow(y, err_handler);
        }
        if (err_handler.ErrCode()) return;

        arrayCopy(luminances, 0, &_matrix[0], 0, area);
    }
}

void ImgSource::arrayCopy(unsigned char* src, int inputOffset, char* dst, int outputOffset,
                          int length) const {
    const unsigned char* srcPtr = src + inputOffset;
    char* dstPtr = dst + outputOffset;

    memcpy(dstPtr, srcPtr, length * sizeof(unsigned char));
}

bool ImgSource::isCropSupported() const { return true; }

Ref<LuminanceSource> ImgSource::crop(int left_, int top_, int width, int height,
                                     ErrorHandler& err_handler) const {
    return ImgSource::create(rgbs, dataWidth, dataHeight, left + left_, top + top_, width, height,
                             _comps, _pixelStep, err_handler);
}

bool ImgSource::isRotateSupported() const { return false; }

Ref<LuminanceSource> ImgSource::rotateCounterClockwise(ErrorHandler& err_handler) const {
    // Intentionally flip the left, top, width, and height arguments as
    // needed. dataWidth and dataHeight are always kept unrotated.
    int width = getWidth();
    int height = getHeight();

    return ImgSource::create(rgbs, dataWidth, dataHeight, top, left, height, width, _comps,
                             _pixelStep, err_handler);
}

zxing::ArrayRef<char> ImgSource::downSample(zxing::ArrayRef<char> image, int& width, int& height,
                                            int pixelStep) {
    zxing::ArrayRef<char> downSampleImage;
    width /= 2;
    height /= 2;
    char* buffer = new char[pixelStep * width * height];
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (pixelStep == 1)
                buffer[i * width + j] = image[i * 2 * width + j * 2];
            else if (pixelStep == 4) {
                for (int p = 0; p < 4; p++) {
                    buffer[i * width * pixelStep + j * pixelStep + p] =
                        image[i * width * 2 * pixelStep + j * 2 * pixelStep + p];
                }
            }
        }
    }
    downSampleImage = zxing::ArrayRef<char>(buffer, pixelStep * width * height);
    delete[] buffer;
    return downSampleImage;
}

// Total Variation denoising
void ImgSource::tvDenoising() const {
    int nx = getWidth();
    int ny = getHeight();
    // char * luminances_ = (char*)luminances;
    zxing::ArrayRef<char> pre_img = zxing::ArrayRef<char>(nx * ny);
    arrayCopy(luminances, 0, &pre_img[0], 0, nx * ny);
    int ep = 1;
    double dt = ep / 4.0, lam = 0.0;
    int ep2 = ep * ep;
    for (int t = 0; t < tvInter; t++) {
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                int tmp_i1 = (i + 1) < ny ? (i + 1) : (ny - 1);
                int tmp_j1 = (j + 1) < nx ? (j + 1) : (nx - 1);
                int tmp_i2 = (i - 1) > -1 ? (i - 1) : 0;
                int tmp_j2 = (j - 1) > -1 ? (j - 1) : 0;
                double tmp_x = (luminances[i * nx + tmp_j1] - luminances[i * nx + tmp_j2]) /
                               2;  // I_x  = (I(:,[2:nx nx])-I(:,[1 1:nx-1]))/2;
                double tmp_y = (luminances[tmp_i1 * nx + j] - luminances[tmp_i2 * nx + j]) /
                               2;  // I_y  = (I([2:ny ny],:)-I([1 1:ny-1],:))/2;
                double tmp_xx =
                    luminances[i * nx + tmp_j1] + luminances[i * nx + tmp_j2] -
                    luminances[i * nx + j] * 2;  // I_xx = I(:,[2:nx nx])+I(:,[1 1:nx-1])-2*I;
                double tmp_yy =
                    luminances[tmp_i1 * nx + j] + luminances[tmp_i2 * nx + j] -
                    luminances[i * nx + j] * 2;  // I_yy = I([2:ny ny],:)+I([1 1:ny-1],:)-2*I;
                double tmp_dp = luminances[tmp_i1 * nx + tmp_j1] +
                                luminances[tmp_i2 * nx + tmp_j2];  // Dp = I([2:ny ny],[2:nx
                                                                   // nx])+I([1 1:ny-1],[1 1:nx-1]);
                double tmp_dm = luminances[tmp_i2 * nx + tmp_j1] +
                                luminances[tmp_i1 * nx + tmp_j2];  // Dm = I([1 1:ny-1],[2:nx
                                                                   // nx])+I([2:ny ny],[1 1:nx-1]);
                double tmp_xy = (tmp_dp - tmp_dm) / 4;             // I_xy = (Dp-Dm)/4;
                double tmp_num =
                    tmp_xx * (tmp_y * tmp_y + ep2) - 2 * tmp_x * tmp_y * tmp_xy +
                    tmp_yy * (tmp_x * tmp_x +
                              ep2);  // Num =
                                     // I_xx.*(ep2+I_y.^2)-2*I_x.*I_y.*I_xy+I_yy.*(ep2+I_x.^2);
                double tmp_den = pow((tmp_x * tmp_x + tmp_y * tmp_y + ep2),
                                     1.5);  // Den = (ep2+I_x.^2+I_y.^2).^(3/2);
                luminances[i * nx + j] +=
                    dt * (tmp_num / tmp_den + lam * (pre_img[i * nx + j] - luminances[i * nx + j]));
            }
        }
    }
    return;
}

void ImgSource::denoseLuminanceSource(int inter) { tvInter = inter; }

Ref<ByteMatrix> ImgSource::getByteMatrix() const {
    return Ref<ByteMatrix>(new ByteMatrix(getWidth(), getHeight(), getMatrix()));
}
}  // namespace wechat_qrcode
}  // namespace cv