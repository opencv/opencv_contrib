// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudacodec;

#if !defined (HAVE_CUDA)
Ptr<NVSurfaceToColorConverter> cv::cudacodec::createNVSurfaceToColorConverter(const ColorSpaceStandard, const bool){ throw_no_cuda(); }
#else
#include "cuda/ColorSpace.h"
namespace cv { namespace cuda { namespace device {
template<class BGR24> void Nv12ToColor24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void Nv12ToColor24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void Nv12ToColor32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void Nv12ToColor48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void Nv12ToColor48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void Nv12ToColor64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void Nv12ToColor64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void Nv12ToColorPlanar24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void Nv12ToColorPlanar24(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void Nv12ToColorPlanar32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void Nv12ToColorPlanar32(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void Nv12ToColorPlanar48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void Nv12ToColorPlanar48(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void Nv12ToColorPlanar64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void Nv12ToColorPlanar64(uint8_t* dpNv12, int nNv12Pitch, uint8_t* dpBgrp, int nBgrpPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void P016ToColor24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void P016ToColor24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void P016ToColor32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void P016ToColor32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void P016ToColor48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void P016ToColor48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void P016ToColor64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void P016ToColorPlanar24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void P016ToColorPlanar24(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void P016ToColorPlanar32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void P016ToColorPlanar32(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag);
template<class BGR48> void P016ToColorPlanar48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void P016ToColorPlanar48(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void P016ToColorPlanar64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void P016ToColorPlanar64(uint8_t* dpP016, int nP016Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void YUV444ToColor24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void YUV444ToColor24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void YUV444ToColor32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void YUV444ToColor32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void YUV444ToColor48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void YUV444ToColor48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void YUV444ToColor64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void YUV444ToColor64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void YUV444ToColorPlanar24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void YUV444ToColorPlanar24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void YUV444ToColorPlanar32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void YUV444ToColorPlanar32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void YUV444ToColorPlanar48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void YUV444ToColorPlanar48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void YUV444ToColorPlanar64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void YUV444ToColorPlanar64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void YUV444P16ToColor24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void YUV444P16ToColor24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void YUV444P16ToColor32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void YUV444P16ToColor32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void YUV444P16ToColor48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void YUV444P16ToColor48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void YUV444P16ToColor64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void YUV444P16ToColor64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

template<class BGR24> void YUV444P16ToColorPlanar24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB24> void YUV444P16ToColorPlanar24(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA32> void YUV444P16ToColorPlanar32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA32> void YUV444P16ToColorPlanar32(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGR48> void YUV444P16ToColorPlanar48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGB48> void YUV444P16ToColorPlanar48(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class BGRA64> void YUV444P16ToColorPlanar64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
template<class RGBA64> void YUV444P16ToColorPlanar64(uint8_t* dpYuv444, int nYuv444Pitch, uint8_t* dpColor, int nColorPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

void Y8ToGray8(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
void Y8ToGray16(uint8_t* dpY8, int nY8Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
void Y16ToGray8(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);
void Y16ToGray16(uint8_t* dpY16, int nY16Pitch, uint8_t* dpGray, int nGrayPitch, int nWidth, int nHeight, bool videoFullRangeFlag, const cudaStream_t stream);

void SetMatYuv2Rgb(int iMatrix, bool);
}}}

using namespace cuda::device;
class NVSurfaceToColorConverterImpl : public NVSurfaceToColorConverter {
public:
    NVSurfaceToColorConverterImpl(ColorSpaceStandard colorSpace, bool fullColorRange = false) {
        SetMatYuv2Rgb(static_cast<int>(colorSpace), fullColorRange);
    }

    int OutputColorFormatIdx(const cudacodec::ColorFormat format) {
        switch (format) {
        case cudacodec::ColorFormat::BGR: return 0;
        case cudacodec::ColorFormat::RGB: return 1;
        case cudacodec::ColorFormat::BGRA: return 2;
        case cudacodec::ColorFormat::RGBA: return 3;
        case cudacodec::ColorFormat::GRAY: return 4;
        default: return -1;
        }
    }

    int NumChannels(const cudacodec::ColorFormat format) {
        switch (format) {
        case cudacodec::ColorFormat::BGR:
        case cudacodec::ColorFormat::RGB: return 3;
        case cudacodec::ColorFormat::BGRA:
        case cudacodec::ColorFormat::RGBA: return 4;
        case cudacodec::ColorFormat::GRAY: return 1;
        default: return -1;
        }
    }

    BitDepth GetBitDepthOut(const BitDepth bitDepth, const int nBitsIn) {
        switch (bitDepth) {
        case BitDepth::EIGHT:
        case BitDepth::SIXTEEN:
            return bitDepth;
        case BitDepth::UNCHANGED:
        default:
            if (nBitsIn == CV_8U)
                return BitDepth::EIGHT;
            else
                return BitDepth::SIXTEEN;
        }
    }

    bool convert(const InputArray yuv, const OutputArray out, const SurfaceFormat surfaceFormat, const ColorFormat outputFormat, const BitDepth bitDepth, const bool planar, const bool videoFullRangeFlag, cuda::Stream& stream) {
        CV_Assert(outputFormat == ColorFormat::BGR || outputFormat == ColorFormat::BGRA || outputFormat == ColorFormat::RGB || outputFormat == ColorFormat::RGBA || outputFormat == ColorFormat::GRAY);
        CV_Assert(yuv.depth() == CV_8U || yuv.depth() == CV_16U);
        const bool yuv420 = surfaceFormat == SurfaceFormat::SF_NV12 || surfaceFormat == SurfaceFormat::SF_P016;
        CV_Assert(yuv.cols() % 2 == 0);

        using func_t = void (*)(uint8_t* yuv, int yuvPitch, uint8_t* color, int colorPitch, int width, int height, bool videoFullRangeFlag, cudaStream_t stream);

        static const func_t funcsNV12[5][2][2] =
        {
            {
                { Nv12ToColor24<BGR24>, Nv12ToColorPlanar24<BGR24> },
                { Nv12ToColor48<BGR48>, Nv12ToColorPlanar48<BGR48> }
            },
            {
                { Nv12ToColor24<RGB24>, Nv12ToColorPlanar24<RGB24> },
                { Nv12ToColor48<RGB48>, Nv12ToColorPlanar48<RGB48> }
            },
            {
                { Nv12ToColor32<BGRA32>, Nv12ToColorPlanar32<BGRA32> },
                { Nv12ToColor64<BGRA64>, Nv12ToColorPlanar64<BGRA64> }
            },
            {
                { Nv12ToColor32<RGBA32>, Nv12ToColorPlanar32<RGBA32> },
                { Nv12ToColor64<RGBA64>, Nv12ToColorPlanar64<RGBA64> }
            },
            {
                { Y8ToGray8, Y8ToGray8 },
                { Y8ToGray16, Y8ToGray16 }
            }
        };

        static const func_t funcsP016[5][2][2] =
        {
            {
                { P016ToColor24<BGR24>, P016ToColorPlanar24<BGR24> },
                { P016ToColor48<BGR48>, P016ToColorPlanar48<BGR48> }
            },
            {
                { P016ToColor24<RGB24>, P016ToColorPlanar24<RGB24> },
                { P016ToColor48<RGB48>, P016ToColorPlanar48<RGB48> }
            },
            {
                { P016ToColor32<BGRA32>, P016ToColorPlanar32<BGRA32> },
                { P016ToColor64<BGRA64>, P016ToColorPlanar64<BGRA64> }
            },
            {
                { P016ToColor32<RGBA32>, P016ToColorPlanar32<RGBA32> },
                { P016ToColor64<RGBA64>, P016ToColorPlanar64<RGBA64> }
            },
            {
                { Y16ToGray8, Y16ToGray8 },
                { Y16ToGray16, Y16ToGray16 }
            }
        };

        static const func_t funcsYUV444[5][2][2] =
        {
            {
                { YUV444ToColor24<BGR24>, YUV444ToColorPlanar24<BGR24> },
                { YUV444ToColor48<BGR48>, YUV444ToColorPlanar48<BGR48> }
            },
            {
                { YUV444ToColor24<RGB24>, YUV444ToColorPlanar24<RGB24> },
                { YUV444ToColor48<RGB48>, YUV444ToColorPlanar48<RGB48> }
            },
            {
                { YUV444ToColor32<BGRA32>, YUV444ToColorPlanar32<BGRA32> },
                { YUV444ToColor64<BGRA64>, YUV444ToColorPlanar64<BGRA64> }
            },
            {
                { YUV444ToColor32<RGBA32>, YUV444ToColorPlanar32<RGBA32> },
                { YUV444ToColor64<RGBA64>, YUV444ToColorPlanar64<RGBA64> }
            },
            {
                { Y8ToGray8, Y8ToGray8 },
                { Y8ToGray16, Y8ToGray16 }
            }
        };

        static const func_t funcsYUV444P16[5][2][2] =
        {
            {
                { YUV444P16ToColor24<BGR24>, YUV444P16ToColorPlanar24<BGR24> },
                { YUV444P16ToColor48<BGR48>, YUV444P16ToColorPlanar48<BGR48> }
            },
            {
                { YUV444P16ToColor24<RGB24>, YUV444P16ToColorPlanar24<RGB24> },
                { YUV444P16ToColor48<RGB48>, YUV444P16ToColorPlanar48<RGB48> }
            },
            {
                { YUV444P16ToColor32<BGRA32>, YUV444P16ToColorPlanar32<BGRA32> },
                { YUV444P16ToColor64<BGRA64>, YUV444P16ToColorPlanar64<BGRA64> }
            },
            {
                { YUV444P16ToColor32<RGBA32>, YUV444P16ToColorPlanar32<RGBA32> },
                { YUV444P16ToColor64<RGBA64>, YUV444P16ToColorPlanar64<RGBA64> }
            },
            {
                { Y16ToGray8, Y16ToGray8 },
                { Y16ToGray16, Y16ToGray16 }
            }
        };

        GpuMat yuv_ = getInputMat(yuv, stream);
        CV_Assert(yuv_.step <= static_cast<size_t>(std::numeric_limits<int>::max()));

        const int nRows = static_cast<int>(yuv.rows() / (yuv420 ? 1.5f : 3.0f));
        CV_Assert(!yuv420 || nRows % 2 == 0);
        const int nChannels = NumChannels(outputFormat);
        const int nRowsOut = nRows * (planar ? nChannels : 1);
        const BitDepth bitDepth_ = GetBitDepthOut(bitDepth, yuv.depth());
        const int iBitDepth = bitDepth_ == BitDepth::EIGHT ? 0 : 1;
        const int typeOut = CV_MAKE_TYPE(bitDepth_ == BitDepth::EIGHT ? CV_8U : CV_16U, planar ? 1 : nChannels);
        GpuMat out_ = getOutputMat(out, nRowsOut, yuv.cols(), typeOut, stream);

        const int iSurfaceFormat = static_cast<int>(surfaceFormat);
        const int iPlanar = planar ? 1 : 0;
        const int iOutputFormat = OutputColorFormatIdx(outputFormat);
        func_t func = nullptr;

        switch (iSurfaceFormat)
        {
        case 0:
            func = funcsNV12[iOutputFormat][iBitDepth][iPlanar];
            break;
        case 1:
            func = funcsP016[iOutputFormat][iBitDepth][iPlanar];
            break;
        case 2:
            func = funcsYUV444[iOutputFormat][iBitDepth][iPlanar];
            break;
        case 3:
            func = funcsYUV444P16[iOutputFormat][iBitDepth][iPlanar];
            break;
        }

        if (!func)
            CV_Error(Error::StsUnsupportedFormat, "Unsupported combination of source and destination types");

        CV_Assert(out_.step <= static_cast<size_t>(std::numeric_limits<int>::max()));
        func((uint8_t*)yuv_.ptr(0), static_cast<int>(yuv_.step), (uint8_t*)out_.ptr(0), static_cast<int>(out_.step), out_.cols, nRows, videoFullRangeFlag, StreamAccessor::getStream(stream));
        return true;
    }


};

Ptr<NVSurfaceToColorConverter> cv::cudacodec::createNVSurfaceToColorConverter(const ColorSpaceStandard colorSpace, const bool videoFullRangeFlag) {
    return makePtr<NVSurfaceToColorConverterImpl>(colorSpace, videoFullRangeFlag);
}
#endif
