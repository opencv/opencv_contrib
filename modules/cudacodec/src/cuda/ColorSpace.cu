// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "ColorSpace.h"
#include <opencv2/cudacodec.hpp>
#include <opencv2/core/cuda/common.hpp>

namespace cv { namespace cuda { namespace device {

__constant__ float matYuv2Color[3][3];

void inline GetConstants(int iMatrix, float& wr, float& wb, int& black, int& white, int& uvWhite, int& max, bool fullRange = false) {
    if (fullRange) {
        black = 0; white = 255; uvWhite = 255;
    }
    else {
        black = 16; white = 235; uvWhite = 240;
    }
    max = 255;

    switch (static_cast<cv::cudacodec::ColorSpaceStandard>(iMatrix))
    {
    case cv::cudacodec::ColorSpaceStandard::BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case cv::cudacodec::ColorSpaceStandard::FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case cv::cudacodec::ColorSpaceStandard::BT470:
    case cv::cudacodec::ColorSpaceStandard::BT601:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case cv::cudacodec::ColorSpaceStandard::SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case cv::cudacodec::ColorSpaceStandard::BT2020:
    case cv::cudacodec::ColorSpaceStandard::BT2020C:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

void SetMatYuv2Rgb(int iMatrix, bool fullRange = false) {
    float wr, wb;
    int black, white, max, uvWhite;
    GetConstants(iMatrix, wr, wb, black, white, uvWhite, max, fullRange);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (j == 0)
                mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
            else
                mat[i][j] = (float)(1.0 * max / (uvWhite - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matYuv2Color, mat, sizeof(mat));
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Gray, class YuvUnit>
__device__ inline Gray YToGrayForPixel(YuvUnit y, bool videoFullRangeFlag) {
    const int low = videoFullRangeFlag ? 0 : 1 << (sizeof(YuvUnit) * 8 - 4);
    float fy = (int)y - low;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;

    YuvUnit g = (YuvUnit)Clamp(matYuv2Color[0][0] * fy, 0.0f, maxf);
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(Gray)) * 8;
    Gray gray{};
    if (sizeof(YuvUnit) >= sizeof(Gray))
        gray = g >> nShift;
    else
        gray = g << nShift;
    return gray;
}

template<class Color, class YuvUnit>
__device__ inline Color YuvToColorForPixel(YuvUnit y, YuvUnit u, YuvUnit v, bool videoFullRangeFlag) {
    const int
        low = videoFullRangeFlag ? 0 : 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit
        r = (YuvUnit)Clamp(matYuv2Color[0][0] * fy + matYuv2Color[0][1] * fu + matYuv2Color[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Color[1][0] * fy + matYuv2Color[1][1] * fu + matYuv2Color[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Color[2][0] * fy + matYuv2Color[2][1] * fu + matYuv2Color[2][2] * fv, 0.0f, maxf);

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
__device__ inline Color YuvToColoraForPixel(YuvUnit y, YuvUnit u, YuvUnit v, bool videoFullRangeFlag) {
    Color color = YuvToColorForPixel<Color>(y, u, v, videoFullRangeFlag);
    const float maxf = (1 << sizeof(color.c.r) * 8) - 1.0f;
    color.c.a = maxf;
    return color;
}

template<class Yuvx2, class Gray, class Grayx2>
__global__ static void YToGrayKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(Yuvx2) / 2 + y * nYuvPitch;
    uint8_t* pDst = pGray + x * sizeof(Gray) + y * nGrayPitch;

    Yuvx2 l0 = *(Yuvx2*)pSrc;
    *(Grayx2*)pDst = Grayx2{
        YToGrayForPixel<Gray>(l0.x, videoFullRangeFlag),
        YToGrayForPixel<Gray>(l0.y, videoFullRangeFlag),
    };
}

template<class Yuvx2, class Color, class Colorx2>
__global__ static void YuvToColorKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
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
        Color Color[2];
    };
    ColorOutx2 l1Out;
    l1Out.Color[0] = YuvToColorForPixel<Color>(l0.x, ch.x, ch.y, videoFullRangeFlag);
    l1Out.Color[1] = YuvToColorForPixel<Color>(l0.y, ch.x, ch.y, videoFullRangeFlag);
    *(Colorx2*)pDst = l1Out.d;
    ColorOutx2 l2Out;
    l2Out.Color[0] = YuvToColorForPixel<Color>(l1.x, ch.x, ch.y, videoFullRangeFlag);
    l2Out.Color[1] = YuvToColorForPixel<Color>(l1.y, ch.x, ch.y, videoFullRangeFlag);
    *(Colorx2*)(pDst + nColorPitch) = l2Out.d;
}

template<class YuvUnitx2, class Color, class ColorIntx2>
__global__ static void YuvToColoraKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
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
        YuvToColoraForPixel<Color>(l0.x, ch.x, ch.y, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l0.y, ch.x, ch.y, videoFullRangeFlag).d,
    };
    *(ColorIntx2*)(pDst + nColorPitch) = ColorIntx2{
        YuvToColoraForPixel<Color>(l1.x, ch.x, ch.y, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l1.y, ch.x, ch.y, videoFullRangeFlag).d,
    };
}

template<class YuvUnitx2, class Color, class Colorx2>
__global__ static void Yuv444ToColorKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
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
        Color Color[2];
    };
    ColorOutx2 out;
    out.Color[0] = YuvToColorForPixel<Color>(l0.x, ch1.x, ch2.x, videoFullRangeFlag);
    out.Color[1] = YuvToColorForPixel<Color>(l0.y, ch1.y, ch2.y, videoFullRangeFlag);
    *(Colorx2*)pDst = out.d;
}

template<class YuvUnitx2, class Color, class ColorIntx2>
__global__ static void Yuv444ToColoraKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
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
        YuvToColoraForPixel<Color>(l0.x, ch1.x, ch2.x, videoFullRangeFlag).d,
        YuvToColoraForPixel<Color>(l0.y, ch1.y, ch2.y, videoFullRangeFlag).d,
    };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void YuvToColorPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2*)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Color color0 = YuvToColorForPixel<Color>(l0.x, ch.x, ch.y, videoFullRangeFlag),
        color1 = YuvToColorForPixel<Color>(l0.y, ch.x, ch.y, videoFullRangeFlag),
        color2 = YuvToColorForPixel<Color>(l1.x, ch.x, ch.y, videoFullRangeFlag),
        color3 = YuvToColorForPixel<Color>(l1.y, ch.x, ch.y, videoFullRangeFlag);

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
__global__ static void YuvToColoraPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 l1 = *(YuvUnitx2*)(pSrc + nYuvPitch);
    YuvUnitx2 ch = *(YuvUnitx2*)(pSrc + (nHeight - y / 2) * nYuvPitch);

    Color color0 = YuvToColoraForPixel<Color>(l0.x, ch.x, ch.y, videoFullRangeFlag),
        color1 = YuvToColoraForPixel<Color>(l0.y, ch.x, ch.y, videoFullRangeFlag),
        color2 = YuvToColoraForPixel<Color>(l1.x, ch.x, ch.y, videoFullRangeFlag),
        color3 = YuvToColoraForPixel<Color>(l1.y, ch.x, ch.y, videoFullRangeFlag);

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
__global__ static void Yuv444ToColorPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    Color color0 = YuvToColorForPixel<Color>(l0.x, ch1.x, ch2.x, videoFullRangeFlag),
        color1 = YuvToColorForPixel<Color>(l0.y, ch1.y, ch2.y, videoFullRangeFlag);


    uint8_t* pDst = pColorp + x * sizeof(ColorUnitx2) / 2 + y * nColorpPitch;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.x, color1.v.x };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.y, color1.v.y };

    pDst += nColorpPitch * nHeight;
    *(ColorUnitx2*)pDst = ColorUnitx2{ color0.v.z, color1.v.z };
}

template<class YuvUnitx2, class Color, class ColorUnitx2>
__global__ static void Yuv444ToColoraPlanarKernel(uint8_t* pYuv, int nYuvPitch, uint8_t* pColorp, int nColorpPitch, int nWidth, int nHeight, bool videoFullRangeFlag) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y);
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }

    uint8_t* pSrc = pYuv + x * sizeof(YuvUnitx2) / 2 + y * nYuvPitch;

    YuvUnitx2 l0 = *(YuvUnitx2*)pSrc;
    YuvUnitx2 ch1 = *(YuvUnitx2*)(pSrc + (nHeight * nYuvPitch));
    YuvUnitx2 ch2 = *(YuvUnitx2*)(pSrc + (2 * nHeight * nYuvPitch));

    Color color0 = YuvToColoraForPixel<Color>(l0.x, ch1.x, ch2.x, videoFullRangeFlag),
        color1 = YuvToColoraForPixel<Color>(l0.y, ch1.y, ch2.y, videoFullRangeFlag);


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

void Y8ToGray8(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<uchar2, unsigned char, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY8, nY8Pitch, dpGray, nGrayPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y8ToGray16(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<uchar2, unsigned short, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY8, nY8Pitch, dpGray, nGrayPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y16ToGray8(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<ushort2, unsigned char, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY16, nY16Pitch, dpGray, nGrayPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

void Y16ToGray16(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YToGrayKernel<ushort2, unsigned short, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpY16, nY16Pitch, dpGray, nGrayPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void Nv12ToColor24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<uchar2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<uchar2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void Nv12ToColor48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<uchar2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void Nv12ToColor64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2* BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444ToColor24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<uchar2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<uchar2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444ToColor48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<uchar2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<uchar2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void P016ToColor24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<ushort2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void P016ToColor32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<ushort2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void P016ToColor48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorKernel<ushort2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444P16ToColor24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<ushort2, COLOR24, ushort3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444P16ToColor32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<ushort2, COLOR32, uint2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444P16ToColor48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorKernel<ushort2, COLOR48, uint3>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444P16ToColor64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraKernel<ushort2, COLOR64, ulonglong2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void Nv12ToColorPlanar24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void Nv12ToColorPlanar32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void Nv12ToColorPlanar48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<uchar2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void Nv12ToColorPlanar64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<uchar2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpNv12, nNv12Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void P016ToColorPlanar24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void P016ToColorPlanar32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void P016ToColorPlanar48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColorPlanarKernel<ushort2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void P016ToColorPlanar64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    YuvToColoraPlanarKernel<ushort2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, 2 * BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpP016, nP016Pitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444ToColorPlanar24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<uchar2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444ToColorPlanar32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<uchar2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444ToColorPlanar48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<uchar2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444ToColorPlanar64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<uchar2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR24>
void YUV444P16ToColorPlanar24(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<ushort2, COLOR24, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR32>
void YUV444P16ToColorPlanar32(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<ushort2, COLOR32, uchar2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR48>
void YUV444P16ToColorPlanar48(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColorPlanarKernel<ushort2, COLOR48, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template <class COLOR64>
void YUV444P16ToColorPlanar64(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream) {
    Yuv444ToColoraPlanarKernel<ushort2, COLOR64, ushort2>
        <<<dim3(divUp(nWidth, 2 * BLOCKSIZE_X), divUp(nHeight, BLOCKSIZE_Y)), dim3(BLOCKSIZE_X, BLOCKSIZE_Y), 0, stream>>>
        (dpYUV444, nPitch, dpColor, nColorPitch, nWidth, nHeight, videoFullRangeFlag);
    if (stream == 0)
        cudaSafeCall(cudaStreamSynchronize(stream));
}

template void Nv12ToColor24<BGR24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor24<RGB24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor32<BGRA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor32<RGBA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor48<BGR48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor48<RGB48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor64<BGRA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColor64<RGBA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void Nv12ToColorPlanar24<BGR24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar24<RGB24>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar32<BGRA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar32<RGBA32>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar48<BGR48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar48<RGB48>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar64<BGRA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void Nv12ToColorPlanar64<RGBA64>(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void P016ToColor24<BGR24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor24<RGB24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor32<BGRA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor32<RGBA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor48<BGR48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor48<RGB48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor64<BGRA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColor64<RGBA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void P016ToColorPlanar24<BGR24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar24<RGB24>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar32<BGRA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar32<RGBA32>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar48<BGR48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar48<RGB48>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar64<BGRA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void P016ToColorPlanar64<RGBA64>(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444ToColor24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColor64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444ToColorPlanar24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444ToColorPlanar64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444P16ToColor24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColor64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template void YUV444P16ToColorPlanar24<BGR24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar24<RGB24>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar32<BGRA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar32<RGBA32>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar48<BGR48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar48<RGB48>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar64<BGRA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template void YUV444P16ToColorPlanar64<RGBA64>(uint8_t* dpYUV444, int nPitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
}}}
