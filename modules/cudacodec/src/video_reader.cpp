/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudacodec;

#ifndef HAVE_NVCUVID

Ptr<VideoReader> cv::cudacodec::createVideoReader(const String&, const std::vector<int>&, const VideoReaderInitParams) { throw_no_cuda(); return Ptr<VideoReader>(); }
Ptr<VideoReader> cv::cudacodec::createVideoReader(const Ptr<RawVideoSource>&, const VideoReaderInitParams) { throw_no_cuda(); return Ptr<VideoReader>(); }
void cv::cudacodec::MapHist(const GpuMat&, Mat&) { throw_no_cuda(); }

#else // HAVE_NVCUVID

void nv12ToBgra(const GpuMat& decodedFrame, GpuMat& outFrame, int width, int height, const bool videoFullRangeFlag, cudaStream_t stream);

using namespace cv::cudacodec::detail;

namespace
{
    class VideoReaderImpl : public VideoReader
    {
    public:
        explicit VideoReaderImpl(const Ptr<VideoSource>& source, const int minNumDecodeSurfaces, const bool allowFrameDrop = false , const bool udpSource = false,
            const Size targetSz = Size(), const Rect srcRoi = Rect(), const Rect targetRoi = Rect(), const bool enableHistogram = false, const int firstFrameIdx = 0);
        ~VideoReaderImpl();

        bool nextFrame(GpuMat& frame, Stream& stream) CV_OVERRIDE;

        bool nextFrame(GpuMat& frame, GpuMat& histogram, Stream& stream) CV_OVERRIDE;

        FormatInfo format() const CV_OVERRIDE;

        bool grab(Stream& stream) CV_OVERRIDE;

        bool retrieve(OutputArray frame, const size_t idx) const CV_OVERRIDE;

        bool set(const VideoReaderProps propertyId, const double propertyVal) CV_OVERRIDE;

        bool set(const ColorFormat colorFormat, const BitDepth bitDepth = BitDepth::UNCHANGED, const bool planar = false) CV_OVERRIDE;

        bool get(const VideoReaderProps propertyId, double& propertyVal) const CV_OVERRIDE;
        bool getVideoReaderProps(const VideoReaderProps propertyId, double& propertyValOut, double propertyValIn) const CV_OVERRIDE;

        bool get(const int propertyId, double& propertyVal) const CV_OVERRIDE;

    private:
        bool skipFrame();
        bool aquireFrameInfo(std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS>& frameInfo, Stream& stream = Stream::Null());
        void releaseFrameInfo(const std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS>& frameInfo);
        bool internalGrab(GpuMat & frame, GpuMat & histogram, Stream & stream);
        void waitForDecoderInit();
        void cvtFromYuv(const GpuMat& decodedFrame, GpuMat& outFrame, const SurfaceFormat surfaceFormat, const bool videoFullRangeFlag, Stream& stream);

        Ptr<VideoSource> videoSource_;

        Ptr<FrameQueue> frameQueue_ = 0;
        Ptr<VideoDecoder> videoDecoder_ = 0;
        Ptr<VideoParser> videoParser_ = 0;

        CUvideoctxlock lock_;

        std::deque< std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> > frames_;
        std::vector<RawPacket> rawPackets;
        GpuMat lastFrame, lastHistogram;
        static const int decodedFrameIdx = 0;
        static const int extraDataIdx = 1;
        static const int rawPacketsBaseIdx = 2;
        Ptr<NVSurfaceToColorConverter> yuvConverter = 0;
        ColorFormat colorFormat = ColorFormat::BGRA;
        BitDepth bitDepth = BitDepth::UNCHANGED;
        bool planar = false;
        static const String errorMsg;
        int iFrame = 0;
    };

    const String VideoReaderImpl::errorMsg = "Parsing/Decoding video source failed, check GPU memory is available and GPU supports requested functionality.";

    FormatInfo VideoReaderImpl::format() const
    {
        return videoSource_->format();
    }

    void VideoReaderImpl::waitForDecoderInit() {
        for (;;) {
            if (videoDecoder_->inited()) break;
            if (videoParser_->hasError() || frameQueue_->isEndOfDecode())
                CV_Error(Error::StsError, errorMsg);
            Thread::sleep(1);
        }
    }

    VideoReaderImpl::VideoReaderImpl(const Ptr<VideoSource>& source, const int minNumDecodeSurfaces, const bool allowFrameDrop, const bool udpSource,
        const Size targetSz, const Rect srcRoi, const Rect targetRoi, const bool enableHistogram, const int firstFrameIdx) :
        videoSource_(source),
        lock_(0)
    {
        // init context
        GpuMat temp(1, 1, CV_8UC1);
        temp.release();

        CUcontext ctx;
        cuSafeCall( cuCtxGetCurrent(&ctx) );
        cuSafeCall( cuvidCtxLockCreate(&lock_, ctx) );
        frameQueue_.reset(new FrameQueue());
        videoDecoder_.reset(new VideoDecoder(videoSource_->format().codec, minNumDecodeSurfaces, targetSz, srcRoi, targetRoi, enableHistogram, ctx, lock_));
        videoParser_.reset(new VideoParser(videoDecoder_, frameQueue_, allowFrameDrop, udpSource));
        videoSource_->setVideoParser(videoParser_);
        videoSource_->start();
        waitForDecoderInit();
        FormatInfo format = videoDecoder_->format();
        if (format.colorSpaceStandard == ColorSpaceStandard::Unspecified) {
            if (format.width > 1280 || format.height > 720)
                format.colorSpaceStandard = ColorSpaceStandard::BT709;
            else
                format.colorSpaceStandard = ColorSpaceStandard::BT601;
        }
        yuvConverter = createNVSurfaceToColorConverter(format.colorSpaceStandard, format.videoFullRangeFlag);
        for(iFrame = videoSource_->getFirstFrameIdx(); iFrame < firstFrameIdx; iFrame++)
            CV_Assert(skipFrame());
        videoSource_->updateFormat(format);
    }

    VideoReaderImpl::~VideoReaderImpl()
    {
        frameQueue_->endDecode();
        videoSource_->stop();
    }

    class VideoCtxAutoLock
    {
    public:
        VideoCtxAutoLock(CUvideoctxlock lock) : m_lock(lock) { cuSafeCall( cuvidCtxLock(m_lock, 0) ); }
        ~VideoCtxAutoLock() { cuvidCtxUnlock(m_lock, 0); }

    private:
        CUvideoctxlock m_lock;
    };

    bool VideoReaderImpl::aquireFrameInfo(std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS>& frameInfo, Stream& stream) {
        if (frames_.empty())
        {
            CUVIDPARSERDISPINFO displayInfo;
            rawPackets.clear();
            for (;;)
            {
                if (frameQueue_->dequeue(displayInfo, rawPackets))
                    break;

                if (videoParser_->hasError())
                    CV_Error(Error::StsError, errorMsg);

                if (frameQueue_->isEndOfDecode())
                    return false;

                // Wait a bit
                Thread::sleep(1);
            }

            bool isProgressive = displayInfo.progressive_frame != 0;
            const int num_fields = isProgressive ? 1 : 2 + displayInfo.repeat_first_field;

            for (int active_field = 0; active_field < num_fields; ++active_field)
            {
                CUVIDPROCPARAMS videoProcParams;
                std::memset(&videoProcParams, 0, sizeof(CUVIDPROCPARAMS));

                videoProcParams.progressive_frame = displayInfo.progressive_frame;
                videoProcParams.second_field = active_field;
                videoProcParams.top_field_first = displayInfo.top_field_first;
                videoProcParams.unpaired_field = (num_fields == 1);
                videoProcParams.output_stream = StreamAccessor::getStream(stream);

                frames_.push_back(std::make_pair(displayInfo, videoProcParams));
            }
        }
        else {
            for (auto& frame : frames_)
                frame.second.output_stream = StreamAccessor::getStream(stream);
        }

        if (frames_.empty())
            return false;

        frameInfo = frames_.front();
        frames_.pop_front();
        return true;
    }

    void VideoReaderImpl::releaseFrameInfo(const std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS>& frameInfo) {
        // release the frame, so it can be re-used in decoder
        if (frames_.empty())
            frameQueue_->releaseFrame(frameInfo.first);
    }

    bool VideoReaderImpl::internalGrab(GpuMat& frame, GpuMat& histogram, Stream& stream) {
        if (videoParser_->hasError())
            CV_Error(Error::StsError, errorMsg);

        std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> frameInfo;
        if (!aquireFrameInfo(frameInfo, stream))
            return false;

        {
            VideoCtxAutoLock autoLock(lock_);

            unsigned long long cuHistogramPtr = 0;
            const cudacodec::FormatInfo fmt = videoDecoder_->format();
            if (fmt.enableHistogram)
                frameInfo.second.histogram_dptr = &cuHistogramPtr;

            // map decoded video frame to CUDA surface
            GpuMat decodedFrame = videoDecoder_->mapFrame(frameInfo.first.picture_index, frameInfo.second);

            if (fmt.enableHistogram) {
                const size_t histogramSz = 4 * fmt.nMaxHistogramBins;
                histogram.create(1, fmt.nMaxHistogramBins, CV_32S);
                cuSafeCall(cuMemcpyDtoDAsync((CUdeviceptr)(histogram.data), cuHistogramPtr, histogramSz, StreamAccessor::getStream(stream)));
            }

            cvtFromYuv(decodedFrame, frame, videoDecoder_->format().surfaceFormat, videoDecoder_->format().videoFullRangeFlag, stream);
            // unmap video frame
            // unmapFrame() synchronizes with the VideoDecode API (ensures the frame has finished decoding)
            videoDecoder_->unmapFrame(decodedFrame);
        }

        releaseFrameInfo(frameInfo);
        iFrame++;
        return true;
    }

    bool VideoReaderImpl::skipFrame() {
        std::pair<CUVIDPARSERDISPINFO, CUVIDPROCPARAMS> frameInfo;
        if (!aquireFrameInfo(frameInfo))
            return false;
        releaseFrameInfo(frameInfo);
        return true;
    }

    bool VideoReaderImpl::grab(Stream& stream) {
        return internalGrab(lastFrame, lastHistogram, stream);
    }

    bool VideoReaderImpl::retrieve(OutputArray frame, const size_t idx) const {
        if (idx == decodedFrameIdx) {
            if (!frame.isGpuMat())
                CV_Error(Error::StsUnsupportedFormat, "Decoded frame is stored on the device and must be retrieved using a cv::cuda::GpuMat");
            frame.getGpuMatRef() = lastFrame;
        }
        else if (idx == extraDataIdx) {
            if (!frame.isMat())
                CV_Error(Error::StsUnsupportedFormat, "Extra data  is stored on the host and must be retrieved using a cv::Mat");
            videoSource_->getExtraData(frame.getMatRef());
        }
        else{
            if (idx >= rawPacketsBaseIdx && idx < rawPacketsBaseIdx + rawPackets.size()) {
                if (!frame.isMat())
                    CV_Error(Error::StsUnsupportedFormat, "Raw data is stored on the host and must be retrieved using a cv::Mat");
                const size_t i = idx - rawPacketsBaseIdx;
                Mat tmp(1, rawPackets.at(i).Size(), CV_8UC1, const_cast<unsigned char*>(rawPackets.at(i).Data()), rawPackets.at(i).Size());
                frame.getMatRef() = tmp;
            }
        }
        return !frame.empty();
    }

    bool VideoReaderImpl::set(const VideoReaderProps propertyId, const double propertyVal) {
        switch (propertyId) {
        case VideoReaderProps::PROP_RAW_MODE :
            videoSource_->SetRawMode(static_cast<bool>(propertyVal));
            return true;
        }
        return false;
    }

    bool ValidColorFormat(const ColorFormat colorFormat) {
        if (colorFormat == ColorFormat::BGRA || colorFormat == ColorFormat::BGR || colorFormat == ColorFormat::RGB || colorFormat == ColorFormat::RGBA || colorFormat == ColorFormat::GRAY || colorFormat == ColorFormat::NV_YUV_SURFACE_FORMAT || colorFormat == ColorFormat::NV_YUV444)
            return true;
        return false;
    }

    bool VideoReaderImpl::set(const ColorFormat colorFormat_, const BitDepth bitDepth_, const bool planar_) {
        ColorFormat tmpFormat = colorFormat_;
        if (tmpFormat == ColorFormat::NV_NV12) {
            CV_LOG_WARNING(NULL, "ColorFormat::NV_NV12 is depreciated forcing ColorFormat::NV_YUV_SURFACE_FORMAT instead.");
            tmpFormat = ColorFormat::NV_YUV_SURFACE_FORMAT;
        }
        if (!ValidColorFormat(tmpFormat)) return false;
        colorFormat = tmpFormat;
        bitDepth = bitDepth_;
        planar = planar_;
        return true;
    }

    bool VideoReaderImpl::get(const VideoReaderProps propertyId, double& propertyVal) const {
        switch (propertyId)
        {
        case VideoReaderProps::PROP_DECODED_FRAME_IDX:
            propertyVal =  decodedFrameIdx;
            return true;
        case VideoReaderProps::PROP_EXTRA_DATA_INDEX:
            propertyVal = extraDataIdx;
            return true;
        case VideoReaderProps::PROP_RAW_PACKAGES_BASE_INDEX:
            if (videoSource_->RawModeEnabled()) {
                propertyVal = rawPacketsBaseIdx;
                return true;
            }
            else
                break;
        case VideoReaderProps::PROP_NUMBER_OF_RAW_PACKAGES_SINCE_LAST_GRAB:
            propertyVal = rawPackets.size();
            return true;
        case VideoReaderProps::PROP_RAW_MODE:
            propertyVal = videoSource_->RawModeEnabled();
            return true;
        case VideoReaderProps::PROP_LRF_HAS_KEY_FRAME: {
            const int iPacket = propertyVal - rawPacketsBaseIdx;
            if (videoSource_->RawModeEnabled() && iPacket >= 0 && iPacket < rawPackets.size()) {
                propertyVal = rawPackets.at(iPacket).ContainsKeyFrame();
                return true;
            }
            else
                break;
        }
        case VideoReaderProps::PROP_ALLOW_FRAME_DROP:
            propertyVal = videoParser_->allowFrameDrops();
            return true;
        case VideoReaderProps::PROP_UDP_SOURCE:
            propertyVal = videoParser_->udpSource();
            return true;
        case VideoReaderProps::PROP_COLOR_FORMAT:
            propertyVal = static_cast<double>(colorFormat);
            return true;
        case VideoReaderProps::PROP_BIT_DEPTH:
            propertyVal = static_cast<double>(bitDepth);
            return true;
        case VideoReaderProps::PROP_PLANAR:
            propertyVal = static_cast<double>(planar);
            return true;
        default:
            break;
        }
        return false;
    }

    bool VideoReaderImpl::getVideoReaderProps(const VideoReaderProps propertyId, double& propertyValOut, double propertyValIn) const {
        double propertyValInOut = propertyValIn;
        const bool ret = get(propertyId, propertyValInOut);
        propertyValOut = propertyValInOut;
        return ret;
    }

    bool VideoReaderImpl::get(const int propertyId, double& propertyVal) const {
        if (propertyId == cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES) {
            propertyVal = static_cast<double>(iFrame);
            return true;
        }
        return videoSource_->get(propertyId, propertyVal);
    }

    bool VideoReaderImpl::nextFrame(GpuMat& frame, Stream& stream)
    {
        GpuMat tmp;
        return nextFrame(frame, tmp, stream);
    }

    bool VideoReaderImpl::nextFrame(GpuMat& frame, GpuMat& histogram, Stream& stream)
    {
        if (!internalGrab(frame, histogram, stream))
            return false;
        return true;
    }

    void VideoReaderImpl::cvtFromYuv(const GpuMat& decodedFrame, GpuMat& outFrame, const SurfaceFormat surfaceFormat, const bool videoFullRangeFlag, Stream& stream)
    {
        if (colorFormat == ColorFormat::NV_YUV_SURFACE_FORMAT) {
            decodedFrame.copyTo(outFrame, stream);
            return;
        }
        yuvConverter->convert(decodedFrame, outFrame, surfaceFormat, colorFormat, bitDepth, planar, videoFullRangeFlag, stream);
    }
}

Ptr<VideoReader> cv::cudacodec::createVideoReader(const String& filename, const std::vector<int>& sourceParams, const VideoReaderInitParams params)
{
    CV_Assert(!filename.empty());

    Ptr<VideoSource> videoSource;
    try
    {
        // prefer ffmpeg to cuvidGetSourceVideoFormat() which doesn't always return the corrct raw pixel format
        Ptr<RawVideoSource> source(new FFmpegVideoSource(filename, sourceParams, params.firstFrameIdx));
        videoSource.reset(new RawVideoSourceWrapper(source, params.rawMode));
    }
    catch (...)
    {
        if (sourceParams.size()) throw;
        videoSource.reset(new CuvidVideoSource(filename));
    }
    return makePtr<VideoReaderImpl>(videoSource, params.minNumDecodeSurfaces, params.allowFrameDrop, params.udpSource, params.targetSz,
        params.srcRoi, params.targetRoi, params.enableHistogram, params.firstFrameIdx);
}

Ptr<VideoReader> cv::cudacodec::createVideoReader(const Ptr<RawVideoSource>& source, const VideoReaderInitParams params)
{
    Ptr<VideoSource> videoSource(new RawVideoSourceWrapper(source, params.rawMode));
    return makePtr<VideoReaderImpl>(videoSource, params.minNumDecodeSurfaces, params.allowFrameDrop, params.udpSource, params.targetSz,
        params.srcRoi, params.targetRoi, params.enableHistogram, params.firstFrameIdx);
}

void cv::cudacodec::MapHist(const GpuMat& hist, Mat& histFull) {
    Mat histHost; hist.download(histHost);
    histFull.create(histHost.size(), histHost.type());
    histFull = 0;
    const float scale = 255.0f / 219.0f;
    const int offset = 16;
    for (int iScaled = 0; iScaled < histHost.cols; iScaled++) {
        const int iHistFull = std::min(std::max(0, static_cast<int>(std::round((iScaled - offset) * scale))), static_cast<int>(histFull.total()) - 1);
        histFull.at<int>(iHistFull) += histHost.at<int>(iScaled);
    }
}

#endif // HAVE_NVCUVID
