// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv { namespace fastcv {

static void fastcvColorWrapper(const Mat& src, Mat& dst, int code);

inline double heightFactor(int fmt /*420 / 422 / 444*/)
{
    switch (fmt)
    {
        case 420:           return 1.5;  // YUV420 has 1.5× rows
        case 422:           return 2.0;  // YUV422 have 2× rows
        case 444:           return 2.0;  // YUV444 have 3× rows
        default:            return 1.0;  // packed RGB565/RGB888 → no extra plane
    }
}

inline void getFormats(int code, int& srcFmt, int& dstFmt)
{
    switch (code)
    {
        case COLOR_YUV2YUV444sp_NV12: srcFmt=420; dstFmt=444; break;
        case COLOR_YUV2YUV422sp_NV12: srcFmt=420; dstFmt=422; break;
        case COLOR_YUV422sp2YUV444sp: srcFmt=422; dstFmt=444; break;
        case COLOR_YUV422sp2YUV_NV12: srcFmt=422; dstFmt=420; break;
        case COLOR_YUV444sp2YUV422sp: srcFmt=444; dstFmt=422; break;
        case COLOR_YUV444sp2YUV_NV12: srcFmt=444; dstFmt=420; break;
        case COLOR_YUV2RGB565_NV12:  srcFmt=420; dstFmt=565; break;
        case COLOR_YUV422sp2RGB565:  srcFmt=422; dstFmt=565; break;
        case COLOR_YUV422sp2RGB:  srcFmt=422; dstFmt=888; break;
        case COLOR_YUV422sp2RGBA:srcFmt=422; dstFmt=8888; break;
        case COLOR_YUV444sp2RGB565:  srcFmt=444; dstFmt=565; break;
        case COLOR_YUV444sp2RGB:  srcFmt=444; dstFmt=888; break;
        case COLOR_YUV444sp2RGBA:srcFmt=444; dstFmt=8888; break;
        case COLOR_RGB5652YUV444sp:  srcFmt=565; dstFmt=444; break;
        case COLOR_RGB5652YUV422sp:  srcFmt=565; dstFmt=422; break;
        case COLOR_RGB5652YUV_NV12:  srcFmt=565; dstFmt=420; break;
        case COLOR_RGB2YUV444sp:  srcFmt=888; dstFmt=444; break;
        case COLOR_RGB2YUV422sp:  srcFmt=888; dstFmt=422; break;
        case COLOR_RGB2YUV_NV12:  srcFmt=888; dstFmt=420; break;

        default:
            CV_Error(Error::StsBadArg, "Unknown FastCV color-code");
    }
}

void cvtColor( InputArray _src, OutputArray _dst, int code)
{
    switch( code )
    {
        case COLOR_YUV2YUV444sp_NV12:
        case COLOR_YUV2YUV422sp_NV12:
        case COLOR_YUV422sp2YUV444sp:
        case COLOR_YUV422sp2YUV_NV12:
        case COLOR_YUV444sp2YUV422sp:
        case COLOR_YUV444sp2YUV_NV12:
        case COLOR_YUV2RGB565_NV12:
        case COLOR_YUV422sp2RGB565:
        case COLOR_YUV422sp2RGB:
        case COLOR_YUV422sp2RGBA:
        case COLOR_YUV444sp2RGB565:
        case COLOR_YUV444sp2RGB:
        case COLOR_YUV444sp2RGBA:
        case COLOR_RGB5652YUV444sp:
        case COLOR_RGB5652YUV422sp:
        case COLOR_RGB5652YUV_NV12:
        case COLOR_RGB2YUV444sp:
        case COLOR_RGB2YUV422sp:
        case COLOR_RGB2YUV_NV12:
            fastcvColorWrapper(_src.getMat(), _dst.getMatRef(), code);
            break;

        default:
            CV_Error( cv::Error::StsBadFlag, "Unknown/unsupported color conversion code" );
    }
}

void fastcvColorWrapper(const Mat& src, Mat& dst, int code)
{
    CV_Assert(src.isContinuous());
    CV_Assert(reinterpret_cast<uintptr_t>(src.data) % 16 == 0);

    const uint32_t width  = static_cast<uint32_t>(src.cols);
    int srcFmt, dstFmt;
    getFormats(code, srcFmt, dstFmt);

    const double hFactorSrc = heightFactor(srcFmt);
    CV_Assert(std::fmod(src.rows, hFactorSrc) == 0.0);

    const uint32_t height = static_cast<uint32_t>(src.rows / hFactorSrc);  // Y-plane height we pass to FastCV

    const uint8_t* srcY      = src.data;
    const size_t   srcYBytes = static_cast<size_t>(src.step) * height;
    const uint8_t* srcC      = srcY + srcYBytes;
    const uint32_t srcStride = static_cast<uint32_t>(src.step);

    const int dstRows = static_cast<int>(height * heightFactor(dstFmt));   // 1.5·H  or  2·H

    int dstType = CV_8UC1; // default for planar/semi-planar YUV formats (1 byte per pixel)

    switch (dstFmt)
    {
        case 420: case 422: case 444: 
            dstType = CV_8UC1; 
            break;

        case 565:  // RGB565 – 16-bit packed RGB, 2 bytes per pixel
            dstType = CV_8UC2;
            break;

        case 888:  // RGB888 – 3 bytes per pixel
            dstType = CV_8UC3;
            break;

        case 8888: // RGBA8888 – 4 bytes per pixel
            dstType = CV_8UC4;
            break;

        default:
            CV_Error(cv::Error::StsBadArg, "Unsupported destination pixel format for FastCV");
    }

    dst.create(dstRows, width, dstType);

    CV_Assert(dst.isContinuous());
    CV_Assert(reinterpret_cast<uintptr_t>(dst.data) % 16 == 0);

    uint8_t* dstY      = dst.data;
    uint8_t* dstC      = dstY + static_cast<size_t>(dst.step) * height;    // offset by Y-plane bytes
    const uint32_t dstStride = static_cast<uint32_t>(dst.step);

    switch(code)
    {
        case COLOR_YUV2YUV444sp_NV12:
        {
            fcvColorYCbCr420PseudoPlanarToYCbCr444PseudoPlanaru8(
                srcY, srcC,
                width, height,
                srcStride, srcStride,
                dstY, dstC,
                dstStride, dstStride
            );
        }
        break;

        case COLOR_YUV2YUV422sp_NV12:
            {
                fcvColorYCbCr420PseudoPlanarToYCbCr422PseudoPlanaru8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );

            }
            break;

        case COLOR_YUV422sp2YUV444sp:
            {                
                fcvColorYCbCr422PseudoPlanarToYCbCr444PseudoPlanaru8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_YUV422sp2YUV_NV12:
            {                
                fcvColorYCbCr422PseudoPlanarToYCbCr420PseudoPlanaru8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_YUV444sp2YUV422sp:
            {
                fcvColorYCbCr444PseudoPlanarToYCbCr422PseudoPlanaru8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_YUV444sp2YUV_NV12:
            {                
                fcvColorYCbCr444PseudoPlanarToYCbCr420PseudoPlanaru8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB5652YUV444sp:
            {                
                fcvColorRGB565ToYCbCr444PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB5652YUV422sp:
            {
                fcvColorRGB565ToYCbCr422PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB5652YUV_NV12:
            {
                fcvColorRGB565ToYCbCr420PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB2YUV444sp:
            {                
                fcvColorRGB888ToYCbCr444PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB2YUV422sp:
            {
                fcvColorRGB888ToYCbCr422PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_RGB2YUV_NV12:
            {
                fcvColorRGB888ToYCbCr420PseudoPlanaru8(
                    srcY,
                    width, height,
                    srcStride,
                    dstY, dstC,
                    dstStride, dstStride
                );
            }
            break;

        case COLOR_YUV2RGB565_NV12:
            {
                fcvColorYCbCr420PseudoPlanarToRGB565u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV422sp2RGB565:
            {
                fcvColorYCbCr422PseudoPlanarToRGB565u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV422sp2RGB:
            {                
                fcvColorYCbCr422PseudoPlanarToRGB888u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV422sp2RGBA:
            {                
                fcvColorYCbCr422PseudoPlanarToRGBA8888u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV444sp2RGB565:
            {                
                fcvColorYCbCr444PseudoPlanarToRGB565u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV444sp2RGB:
            {
                fcvColorYCbCr444PseudoPlanarToRGB888u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        case COLOR_YUV444sp2RGBA:
            {
                fcvColorYCbCr444PseudoPlanarToRGBA8888u8(
                    srcY, srcC,
                    width, height,
                    srcStride, srcStride,
                    dstY,
                    dstStride
                );
            }
            break;

        default:
            CV_Error(cv::Error::StsBadArg, "Unsupported FastCV color code");
    }
}

}} // namespace cv::fastcv
