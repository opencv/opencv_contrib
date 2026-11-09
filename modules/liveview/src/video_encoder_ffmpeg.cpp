#include "video_encoder.hpp"

#if defined(HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG)

#include "opencv2/imgproc.hpp"

#include <deque>
#include <mutex>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace cv {
namespace liveview {
namespace {

static String ffmpegError(int code)
{
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(code, buffer, sizeof(buffer));
    return String(buffer);
}

static std::vector<String> encoderPriority(VideoCodec codec)
{
    std::vector<String> out;
    if (codec == VideoCodec::H264)
    {
        out.push_back("h264_nvenc");
        out.push_back("h264_qsv");
        out.push_back("h264_amf");
        out.push_back("h264_videotoolbox");
        out.push_back("h264_mf");
        out.push_back("libx264");
        out.push_back("libopenh264");
    }
    else if (codec == VideoCodec::VP8)
    {
        out.push_back("libvpx-vp8");
        out.push_back("libvpx");
    }
    return out;
}

static bool isAutoName(const String& name)
{
    return name.empty() || name == "auto" || name == "AUTO";
}

static bool isKnownEncoderForCodec(VideoCodec codec, const String& name)
{
    const std::vector<String> names = encoderPriority(codec);
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (names[i] == name)
            return true;
    }
    return false;
}

static void normalizeToBgr(const Mat& src, Mat& bgr)
{
    if (src.empty())
        CV_Error(Error::StsBadArg, "LiveView video encoder cannot encode an empty frame");
    if (src.depth() != CV_8U)
        CV_Error(Error::StsBadArg, "LiveView video encoder accepts only 8-bit frames");
    if (src.channels() == 1)
        cvtColor(src, bgr, COLOR_GRAY2BGR);
    else if (src.channels() == 3)
        bgr = src.isContinuous() ? src : src.clone();
    else if (src.channels() == 4)
        cvtColor(src, bgr, COLOR_BGRA2BGR);
    else
        CV_Error(Error::StsBadArg, "LiveView video encoder accepts 1, 3, or 4 channel frames");
}

class FfmpegVideoEncoder CV_FINAL : public VideoEncoder
{
public:
    FfmpegVideoEncoder()
        : codec_(NULL), context_(NULL), sws_(NULL), frame_(NULL), packet_(NULL),
          opened_(false), nextPts_(0), lastSequence_(0), lastPtsUsec_(0)
    {
        av_log_set_level(AV_LOG_WARNING);
    }

    ~FfmpegVideoEncoder() CV_OVERRIDE
    {
        close();
    }

    void open(const VideoEncoderParams& params) CV_OVERRIDE
    {
        validateVideoEncoderParams(params);
        std::lock_guard<std::mutex> lock(mutex_);
        closeUnlocked();

        std::vector<String> candidates;
        if (isAutoName(params.encoderName))
        {
            candidates = encoderPriority(params.preferredCodec);
        }
        else
        {
            if (!isKnownEncoderForCodec(params.preferredCodec, params.encoderName))
                CV_Error(Error::StsBadArg, "LiveView explicit FFmpeg encoder is not in the supported " +
                         videoCodecName(params.preferredCodec) + " encoder set");
            candidates.push_back(params.encoderName);
        }

        String lastError;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (openCandidate(params, candidates[i], lastError))
                return;
        }

        if (isAutoName(params.encoderName))
        {
            CV_Error(Error::StsError, "LiveView could not open any FFmpeg " +
                     videoCodecName(params.preferredCodec) + " encoder: " + lastError);
        }
        CV_Error(Error::StsError, "LiveView could not open FFmpeg encoder '" +
                 params.encoderName + "': " + lastError);
    }

    void close() CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closeUnlocked();
    }

    bool isOpened() const CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return opened_;
    }

    void encode(InputArray input, int64 sequence, int64 ptsUsec) CV_OVERRIDE
    {
        Mat src = input.getMat();
        Mat bgr;
        normalizeToBgr(src, bgr);

        std::lock_guard<std::mutex> lock(mutex_);
        if (!opened_)
            CV_Error(Error::StsError, "LiveView video encoder is not open");
        if (bgr.cols != params_.width || bgr.rows != params_.height)
            CV_Error(Error::StsBadArg, "LiveView video encoder frame size does not match encoder parameters");

        int ret = av_frame_make_writable(frame_);
        if (ret < 0)
            CV_Error(Error::StsError, "FFmpeg frame is not writable: " + ffmpegError(ret));

        const uint8_t* inData[1] = { bgr.ptr<uint8_t>(0) };
        int inLinesize[1] = { static_cast<int>(bgr.step[0]) };
        sws_scale(sws_, inData, inLinesize, 0, bgr.rows, frame_->data, frame_->linesize);

        frame_->pts = nextPts_++;
        lastSequence_ = sequence;
        lastPtsUsec_ = ptsUsec;

        ret = avcodec_send_frame(context_, frame_);
        if (ret < 0)
            CV_Error(Error::StsError, "FFmpeg failed to accept frame: " + ffmpegError(ret));

        drainPackets();
    }

    bool tryPop(EncodedFrame& out) CV_OVERRIDE
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty())
            return false;
        out = queue_.front();
        queue_.pop_front();
        return true;
    }

private:
    bool openCandidate(const VideoEncoderParams& params, const String& name, String& lastError)
    {
        codec_ = avcodec_find_encoder_by_name(name.c_str());
        if (!codec_)
        {
            lastError = "encoder not found: " + name;
            return false;
        }

        context_ = avcodec_alloc_context3(codec_);
        if (!context_)
        {
            lastError = "could not allocate codec context";
            return false;
        }

        params_ = params;
        params_.encoderName = name;
        context_->bit_rate = params.bitrate;
        context_->width = params.width;
        context_->height = params.height;
        context_->time_base = AVRational{1, params.fps};
        context_->framerate = AVRational{params.fps, 1};
        context_->gop_size = params.gop;
        context_->max_b_frames = 0;
        context_->flags |= AV_CODEC_FLAG_LOW_DELAY;
        context_->pix_fmt = AV_PIX_FMT_YUV420P;

        AVDictionary* opts = NULL;
        av_dict_set(&opts, "bf", "0", 0);
        if (params.preferredCodec == VideoCodec::H264)
            av_dict_set(&opts, "annexb", "1", 0);
        if (name.find("libx264") != String::npos)
        {
            av_dict_set(&opts, "preset", "ultrafast", 0);
            av_dict_set(&opts, "tune", "zerolatency", 0);
            av_dict_set(&opts, "b-pyramid", "none", 0);
            av_dict_set(&opts, "x264-params", "rc-lookahead=0", 0);
        }
        else if (name.find("nvenc") != String::npos)
        {
            av_dict_set(&opts, "rc-lookahead", "0", 0);
        }

        const int openResult = avcodec_open2(context_, codec_, &opts);
        av_dict_free(&opts);
        if (openResult < 0)
        {
            lastError = name + ": " + ffmpegError(openResult);
            closeUnlocked();
            return false;
        }

        sws_ = sws_getContext(params.width, params.height, AV_PIX_FMT_BGR24,
                              params.width, params.height, context_->pix_fmt,
                              SWS_BICUBIC, NULL, NULL, NULL);
        if (!sws_)
        {
            lastError = name + ": could not create color converter";
            closeUnlocked();
            return false;
        }

        frame_ = av_frame_alloc();
        packet_ = av_packet_alloc();
        if (!frame_ || !packet_)
        {
            lastError = name + ": could not allocate frame or packet";
            closeUnlocked();
            return false;
        }
        frame_->format = context_->pix_fmt;
        frame_->width = context_->width;
        frame_->height = context_->height;
        const int frameBufferResult = av_frame_get_buffer(frame_, 32);
        if (frameBufferResult < 0)
        {
            lastError = name + ": could not allocate frame buffer: " + ffmpegError(frameBufferResult);
            closeUnlocked();
            return false;
        }

        queue_.clear();
        nextPts_ = 0;
        opened_ = true;
        return true;
    }

    void drainPackets()
    {
        for (;;)
        {
            const int ret = avcodec_receive_packet(context_, packet_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                return;
            if (ret < 0)
                CV_Error(Error::StsError, "FFmpeg failed to produce packet: " + ffmpegError(ret));

            EncodedFrame out;
            out.data.assign(packet_->data, packet_->data + packet_->size);
            out.sequence = lastSequence_;
            out.ptsUsec = lastPtsUsec_;
            out.dtsUsec = lastPtsUsec_;
            out.keyframe = (packet_->flags & AV_PKT_FLAG_KEY) != 0;
            out.codec = params_.preferredCodec;

            queue_.push_back(out);
            while (queue_.size() > 8)
                queue_.pop_front();

            av_packet_unref(packet_);
        }
    }

    void closeUnlocked()
    {
        if (context_ && opened_)
        {
            avcodec_send_frame(context_, NULL);
            try
            {
                drainPackets();
            }
            catch (...)
            {
            }
        }
        if (packet_)
            av_packet_free(&packet_);
        if (frame_)
            av_frame_free(&frame_);
        if (sws_)
            sws_freeContext(sws_);
        if (context_)
            avcodec_free_context(&context_);
        codec_ = NULL;
        context_ = NULL;
        sws_ = NULL;
        frame_ = NULL;
        packet_ = NULL;
        opened_ = false;
        nextPts_ = 0;
        queue_.clear();
    }

    const AVCodec* codec_;
    AVCodecContext* context_;
    SwsContext* sws_;
    AVFrame* frame_;
    AVPacket* packet_;
    VideoEncoderParams params_;
    bool opened_;
    int64 nextPts_;
    int64 lastSequence_;
    int64 lastPtsUsec_;
    std::deque<EncodedFrame> queue_;
    mutable std::mutex mutex_;
};

} // namespace

bool haveVideoEncoderBackend()
{
    return true;
}

std::vector<String> availableVideoEncoders(VideoCodec codec)
{
    std::vector<String> out;
    const std::vector<String> names = encoderPriority(codec);
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (avcodec_find_encoder_by_name(names[i].c_str()))
            out.push_back(names[i]);
    }
    return out;
}

Ptr<VideoEncoder> createVideoEncoder()
{
    return makePtr<FfmpegVideoEncoder>();
}

} // namespace liveview
} // namespace cv

#endif // HAVE_LIVEVIEW_VIDEO_ENCODER_FFMPEG
