// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ColorSpace.h"
#include <opencv2/cudacodec.hpp>
#include <opencv2/core/cuda/common.hpp>

namespace cv { namespace cuda { namespace device {

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Gray, class YuvUnit>
__device__ inline Gray YToGrayForPixel(YuvUnit y, float lumaCoeff, bool videoFullRangeFlag) {
    const int low = videoFullRangeFlag ? 0 : 1 << (sizeof(YuvUnit) * 8 - 4);
    float fy = (int)y - low;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;

    YuvUnit g = (YuvUnit)Clamp(lumaCoeff * fy, 0.0f, maxf);
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(Gray)) * 8;
    Gray gray{};
    if (sizeof(YuvUnit) >= sizeof(Gray))
        gray = g >> nShift;
    else
        gray = g << nShift;
    return gray;
}

template<class Color, class YuvUnit>
__device__ inline Color YuvToColorForPixel(YuvUnit y, YuvUnit u, YuvUnit v, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    const int
        low = videoFullRangeFlag ? 0 : 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit
        r = (YuvUnit)Clamp(matYuv2Color.m[0][0] * fy + matYuv2Color.m[0][1] * fu + matYuv2Color.m[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Color.m[1][0] * fy + matYuv2Color.m[1][1] * fu + matYuv2Color.m[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Color.m[2][0] * fy + matYuv2Color.m[2][1] * fu + matYuv2Color.m[2][2] * fv, 0.0f, maxf);

    Color color{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(color.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(color.c.r)) {
        color.c.r = r >> nShift;
        color.c.g = g >> nShift;
        color.c.b = b >> nShift;
    }
    else {
        color.c.r = r << nShift;
        color.c.g = g << nShift;
        color.c.b = b << nShift;
    }
    return color;
}

template<class Color, class YuvUnit>
__device__ inline Color YuvToColoraForPixel(YuvUnit y, YuvUnit u, YuvUnit v, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    Color color = YuvToColorForPixel<Color>(y, u, v, matYuv2Color, videoFullRangeFlag);
    const float maxf = (1 << sizeof(color.c.r) * 8) - 1.0f;
    color.c.a = maxf;
    return color;
}

template<class Yuvx2, class Gray, class Grayx2>
__global__ static void YToGrayKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pGray, int nGrayPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(Yuvx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pGray + x * sizeof(Gray) + y * nGrayPitch;

    Yuvx2 l0 = *(Yuvx2*)pSrc;
    *(Grayx2*)pDst = Grayx2{
        YToGrayForPixel<Gray>(l0.x, matYuv2Color.m[0][0], videoFullRangeFlag),
        YToGrayForPixel<Gray>(l0.y, matYuv2Color.m[0][0], videoFullRangeFlag),
    };
}

template<class Yuvx2, class Color, class Colorx2>
__global__ static void YuvToColorKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(Yuvx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pColor + x * sizeof(Color) + y * nColorPitch;

    Yuvx2 l0 = *(Yuvx2*)pSrc;
    Yuvx2 l1 = *(Yuvx2*)(pSrc + nYuvPitch);
    Yuvx2 ch = *(Yuvx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    union ColorOutx2 {
        Colorx2 d;
        Color color[2];
    };
    ColorOutx2 l1Out;
    l1Out.color[0] = YuvToColorForPixel<Color>(l0.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);
    l1Out.color[1] = YuvToColorForPixel<Color>(l0.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);
    *(Colorx2*)pDst = l1Out.d;
    ColorOutx2 l2Out;
    l2Out.color[0] = YuvToColorForPixel<Color>(l1.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);
    l2Out.color[1] = YuvToColorForPixel<Color>(l1.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);
    *(Colorx2*)(pDst + nColorPitch) = l2Out.d;
}

template<class YuvUnitx2, class Color, class ColorIntx2>
__global__ static void YuvToColoraKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pColor + x * sizeof(Color) + y * nColorPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2*)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    *(ColorIntx2*)pDst = ColorIntx2{
        YuvToColoraForPixel<Color>(l0.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l0.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag).d,
    };
    *(ColorIntx2*)(pDst + nColorPitch) = ColorIntx2{
        YuvToColoraForPixel<Color>(l1.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l1.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag).d,
    };
}

template<class YuvUnitx2, class Color, class Colorx2>
__global__ static void Yuv444ToColorKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pColor + x * sizeof(Color) + y * nColorPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    union ColorOutx2 {
        Colorx2 d;
        Color color[2];
    };
    ColorOutx2 out;
    out.color[0] = YuvToColorForPixel<Color>(l0.x, ch1.x, ch2.x, matYuv2Color, videoFullRangeFlag);
    out.color[1] = YuvToColorForPixel<Color>(l0.y, ch1.y, ch2.y, matYuv2Color, videoFullRangeFlag);
    *(Colorx2*)pDst = out.d;
}

template<class YuvUnitx2, class Color, class ColorIntx2>
__global__ static void Yuv444ToColoraKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pColor + x * sizeof(Color) + y * nColorPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    *(ColorIntx2*)pDst = ColorIntx2{
        YuvToColoraForPixel<Color>(l0.x, ch1.x, ch2.x, matYuv2Color, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l0.y, ch1.y, ch2.y, matYuv2Color, videoFullRangeFlag).d,
    };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void YuvToColorPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2*)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Color color0 = YuvToColorForPixel<Color>(l0.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color1 = YuvToColorForPixel<Color>(l0.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color2 = YuvToColorForPixel<Color>(l1.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color3 = YuvToColorForPixel<Color>(l1.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);

    uint8_t* pDst = pColorp + x * sizeof(ColorUnitx2) / 2 + y * nColorpPitch;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.x, color1.v.x };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.x, color3.v.x };
    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.y, color1.v.y };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.y, color3.v.y };
    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.z, color1.v.z };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.z, color3.v.z };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void YuvToColoraPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2*)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Color color0 = YuvToColoraForPixel<Color>(l0.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color1 = YuvToColoraForPixel<Color>(l0.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color2 = YuvToColoraForPixel<Color>(l1.x, ch.x, ch.y, matYuv2Color, videoFullRangeFlag),
        color3 = YuvToColoraForPixel<Color>(l1.y, ch.x, ch.y, matYuv2Color, videoFullRangeFlag);

    uint8_t* pDst = pColorp + x * sizeof(ColorUnitx2) / 2 + y * nColorpPitch;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.x, color1.v.x };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.x, color3.v.x };
    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.y, color1.v.y };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.y, color3.v.y };
    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.z, color1.v.z };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.z, color3.v.z };
    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.w, color1.v.w };
    *(ColorUnitx2*)(pDst + nColorpPitch) = ColorUnitx2{ color2.v.w, color3.v.w };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void Yuv444ToColorPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    Color color0 = YuvToColorForPixel<Color>(l0.x, ch1.x, ch2.x, matYuv2Color, videoFullRangeFlag),
        color1 = YuvToColorForPixel<Color>(l0.y, ch1.y, ch2.y, matYuv2Color, videoFullRangeFlag);

    uint8_t* pDst = pColorp + x * sizeof(ColorUnitx2) / 2 + y * nColorpPitch;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.x, color1.v.x };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.y, color1.v.y };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.z, color1.v.z };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void Yuv444ToColoraPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    Color color0 = YuvToColoraForPixel<Color>(l0.x, ch1.x, ch2.x, matYuv2Color, videoFullRangeFlag),
        color1 = YuvToColoraForPixel<Color>(l0.y, ch1.y, ch2.y, matYuv2Color, videoFullRangeFlag);

    uint8_t* pDst = pColorp + x * sizeof(ColorUnitx2) / 2 + y * nColorpPitch;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.x, color1.v.x };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.y, color1.v.y };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.z, color1.v.z };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.w, color1.v.w };
}

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 8

void Y8ToGray8(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<uchar2, unsigned char, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY8, nY8Pitch, dpGray, nGrayPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y8ToGray16(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<uchar2, unsigned short, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY8, nY8Pitch, dpGray, nGrayPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y16ToGray8(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<ushort2, unsigned char, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY16, nY16Pitch, dpGray, nGrayPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y16ToGray16(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<ushort2, unsigned short, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY16, nY16Pitch, dpGray, nGrayPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void Nv12ToColor24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<uchar2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<uchar2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void Nv12ToColor48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<uchar2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void Nv12ToColor64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444ToColor24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<uchar2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<uchar2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444ToColor48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<uchar2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void P016ToColor24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<ushort2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void P016ToColor32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<ushort2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void P016ToColor48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<ushort2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444P16ToColor24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<ushort2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444P16ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<ushort2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444P16ToColor48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<ushort2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444P16ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void Nv12ToColorPlanar24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void Nv12ToColorPlanar32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void Nv12ToColorPlanar48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<uchar2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void Nv12ToColorPlanar64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<uchar2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void P016ToColorPlanar24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void P016ToColorPlanar32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void P016ToColorPlanar48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<ushort2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void P016ToColorPlanar64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<ushort2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444ToColorPlanar24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444ToColorPlanar32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444ToColorPlanar48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<uchar2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444ToColorPlanar64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<uchar2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444P16ToColorPlanar24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444P16ToColorPlanar32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444P16ToColorPlanar48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<ushort2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444P16ToColorPlanar64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<ushort2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, matYuv2Color, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template void Nv12ToColor24<BGR24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor24<RGB24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor32<BGRA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor32<RGBA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor48<BGR48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor48<RGB48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor64<BGRA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor64<RGBA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void Nv12ToColorPlanar24<BGR24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar24<RGB24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar32<BGRA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar32<RGBA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar48<BGR48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar48<RGB48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar64<BGRA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar64<RGBA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void P016ToColor24<BGR24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor24<RGB24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor32<BGRA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor32<RGBA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor48<BGR48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor48<RGB48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor64<BGRA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor64<RGBA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void P016ToColorPlanar24<BGR24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar24<RGB24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar32<BGRA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar32<RGBA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar48<BGR48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar48<RGB48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar64<BGRA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar64<RGBA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444ToColor24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444ToColorPlanar24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444P16ToColor24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444P16ToColorPlanar24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, ColorMatrix matYuv2Color, bool videoFullRangeFlag, const cudaStream_t stream);
}}}
