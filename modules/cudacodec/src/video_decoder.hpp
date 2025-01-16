/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __VIDEO_DECODER_HPP__
#define __VIDEO_DECODER_HPP__

namespace cv { namespace cudacodec { namespace detail {

class VideoDecoder
{
public:
    VideoDecoder(const Codec& codec, const int minNumDecodeSurfaces, cv::Size targetSz, cv::Rect srcRoi, cv::Rect targetRoi, const bool enableHistogram, CUcontext ctx, CUvideoctxlock lock) :
        ctx_(ctx), lock_(lock), decoder_(0)
    {
        videoFormat_.codec = codec;
        videoFormat_.ulNumDecodeSurfaces = minNumDecodeSurfaces;
        videoFormat_.enableHistogram = enableHistogram;
        // alignment enforced by nvcuvid, likely due to chroma subsampling
        videoFormat_.targetSz.width = targetSz.width - targetSz.width % 2; videoFormat_.targetSz.height = targetSz.height - targetSz.height % 2;
        videoFormat_.srcRoi.x = srcRoi.x - srcRoi.x % 4; videoFormat_.srcRoi.width = srcRoi.width - srcRoi.width % 4;
        videoFormat_.srcRoi.y = srcRoi.y - srcRoi.y % 2; videoFormat_.srcRoi.height = srcRoi.height - srcRoi.height % 2;
        videoFormat_.targetRoi.x = targetRoi.x - targetRoi.x % 4; videoFormat_.targetRoi.width = targetRoi.width - targetRoi.width % 4;
        videoFormat_.targetRoi.y = targetRoi.y - targetRoi.y % 2; videoFormat_.targetRoi.height = targetRoi.height - targetRoi.height % 2;
    }

    ~VideoDecoder()
    {
        release();
    }

    void create(const FormatInfo& videoFormat);
    int reconfigure(const FormatInfo& videoFormat);
    void release();
    bool inited() { AutoLock autoLock(mtx_); return decoder_; }

    // Get the codec-type currently used.
    cudaVideoCodec codec() const { return static_cast<cudaVideoCodec>(videoFormat_.codec); }
    int nDecodeSurfaces() const { return videoFormat_.ulNumDecodeSurfaces; }
    cv::Size getTargetSz() const { return videoFormat_.targetSz; }
    cv::Rect getSrcRoi() const { return videoFormat_.srcRoi; }
    cv::Rect getTargetRoi() const { return videoFormat_.targetRoi; }

    unsigned long frameWidth() const { return videoFormat_.ulWidth; }
    unsigned long frameHeight() const { return videoFormat_.ulHeight; }
    FormatInfo format() { AutoLock autoLock(mtx_); return videoFormat_;}

    unsigned long targetWidth() { return videoFormat_.width; }
    unsigned long targetHeight() { return videoFormat_.height; }

    cudaVideoChromaFormat chromaFormat() const { return static_cast<cudaVideoChromaFormat>(videoFormat_.chromaFormat); }
    int nBitDepthMinus8() const { return videoFormat_.nBitDepthMinus8; }
    bool enableHistogram() const { return videoFormat_.enableHistogram; }

    bool decodePicture(CUVIDPICPARAMS* picParams)
    {
        return cuvidDecodePicture(decoder_, picParams) == CUDA_SUCCESS;
    }

    cuda::GpuMat mapFrame(int picIdx, CUVIDPROCPARAMS& videoProcParams)
    {
        CUdeviceptr ptr;
        unsigned int pitch;

        cuSafeCall( cuvidMapVideoFrame(decoder_, picIdx, &ptr, &pitch, &videoProcParams) );

        const int height = (videoFormat_.surfaceFormat == cudaVideoSurfaceFormat_NV12 || videoFormat_.surfaceFormat == cudaVideoSurfaceFormat_P016) ? targetHeight() * 3 / 2 : targetHeight() * 3;
        const int type = (videoFormat_.surfaceFormat == cudaVideoSurfaceFormat_NV12 || videoFormat_.surfaceFormat == cudaVideoSurfaceFormat_YUV444) ? CV_8U : CV_16U;
        return cuda::GpuMat(height, targetWidth(), type, (void*) ptr, pitch);
    }

    void unmapFrame(cuda::GpuMat& frame)
    {
        cuSafeCall( cuvidUnmapVideoFrame(decoder_, (CUdeviceptr) frame.data) );
        frame.release();
    }

private:
    CUcontext ctx_ = 0;
    CUvideoctxlock lock_;
    CUvideodecoder decoder_ = 0;
    FormatInfo videoFormat_ = {};
    Mutex mtx_;
};

}}}

#endif // __VIDEO_DECODER_HPP__
