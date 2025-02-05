// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

namespace cv { namespace cuda { namespace device {
union BGR24 {
    uchar3 v;
    struct {
        uint8_t b, g, r;
    } c;
};

union RGB24 {
    uchar3 v;
    struct {
        uint8_t r, g, b;
    } c;
};

union BGRA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b, g, r, a;
    } c;
};

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

union BGR48 {
    ushort3 v;
    struct {
        uint16_t b, g, r;
    } c;
};

union RGB48 {
    ushort3 v;
    struct {
        uint16_t r, g, b;
    } c;
};

union BGRA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t b, g, r, a;
    } c;
};

union RGBA64 {
    uint64_t d;
    ushort4 v;
    struct {
        uint16_t r, g, b, a;
    } c;
};
}}}
