// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../src/cap_ffmpeg_legacy_api.hpp"
//#include "cuvid_ffmpeg_api.hpp"
using namespace cv;

#if !(defined(_WIN32) || defined(WINCE))
# include <pthread.h>
#endif
#include <assert.h>
#include <algorithm>
#include <limits>

#ifndef __OPENCV_BUILD
#define CV_FOURCC(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))
#endif

#define CALC_FFMPEG_VERSION(a,b,c) ( a<<16 | b<<8 | c )

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4244 4510 4610 )
#endif

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#ifndef CV_UNUSED  // Required for standalone compilation mode (OpenCV defines this in base.hpp)
#define CV_UNUSED(name) (void)name
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "../src/ffmpeg_codecs.hpp"

#include <libavutil/mathematics.h>

#if LIBAVUTIL_BUILD > CALC_FFMPEG_VERSION(51,11,0)
#include <libavutil/opt.h>
#endif

#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
#include <libavutil/imgutils.h>
#endif

#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>

#ifdef __cplusplus
}
#endif

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( default: 4244 4510 4610 )
#endif

#ifdef NDEBUG
#define CV_WARN(message)
#else
#define CV_WARN(message) fprintf(stderr, "warning: %s (%s:%d)\n", message, __FILE__, __LINE__)
#endif

#if defined _WIN32
#include <windows.h>
#if defined _MSC_VER && _MSC_VER < 1900
struct timespec
{
    time_t tv_sec;
    long   tv_nsec;
};
#endif
#elif defined __linux__ || defined __APPLE__ || defined __HAIKU__
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/time.h>
#if defined __APPLE__
#include <sys/sysctl.h>
#include <mach/clock.h>
#include <mach/mach.h>
#endif
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(__APPLE__)
#define AV_NOPTS_VALUE_ ((int64_t)0x8000000000000000LL)
#else
#define AV_NOPTS_VALUE_ ((int64_t)AV_NOPTS_VALUE)
#endif

#ifndef AVERROR_EOF
#define AVERROR_EOF (-MKTAG( 'E','O','F',' '))
#endif

#if LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(54,25,0)
#  define CV_CODEC_ID AVCodecID
#  define CV_CODEC(name) AV_##name
#else
#  define CV_CODEC_ID CodecID
#  define CV_CODEC(name) name
#endif

#if LIBAVUTIL_BUILD < (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 74, 100) : CALC_FFMPEG_VERSION(51, 42, 0))
#define AVPixelFormat PixelFormat
#define AV_PIX_FMT_BGR24 PIX_FMT_BGR24
#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#define AV_PIX_FMT_GRAY8 PIX_FMT_GRAY8
#define AV_PIX_FMT_YUV422P PIX_FMT_YUV422P
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUV444P PIX_FMT_YUV444P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P
#define AV_PIX_FMT_GRAY16LE PIX_FMT_GRAY16LE
#define AV_PIX_FMT_GRAY16BE PIX_FMT_GRAY16BE
#endif

#ifndef PKT_FLAG_KEY
#define PKT_FLAG_KEY AV_PKT_FLAG_KEY
#endif

#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(52, 38, 100) : CALC_FFMPEG_VERSION(52, 13, 0))
#define USE_AV_FRAME_GET_BUFFER 1
#else
#define USE_AV_FRAME_GET_BUFFER 0
#ifndef AV_NUM_DATA_POINTERS // required for 0.7.x/0.8.x ffmpeg releases
#define AV_NUM_DATA_POINTERS 4
#endif
#endif


#ifndef USE_AV_INTERRUPT_CALLBACK
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 21, 0)
#define USE_AV_INTERRUPT_CALLBACK 1
#else
#define USE_AV_INTERRUPT_CALLBACK 0
#endif
#endif

#if USE_AV_INTERRUPT_CALLBACK
#define LIBAVFORMAT_INTERRUPT_OPEN_TIMEOUT_MS 30000
#define LIBAVFORMAT_INTERRUPT_READ_TIMEOUT_MS 30000

#ifdef _WIN32
// http://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows

static
inline LARGE_INTEGER get_filetime_offset()
{
    SYSTEMTIME s;
    FILETIME f;
    LARGE_INTEGER t;

    s.wYear = 1970;
    s.wMonth = 1;
    s.wDay = 1;
    s.wHour = 0;
    s.wMinute = 0;
    s.wSecond = 0;
    s.wMilliseconds = 0;
    SystemTimeToFileTime(&s, &f);
    t.QuadPart = f.dwHighDateTime;
    t.QuadPart <<= 32;
    t.QuadPart |= f.dwLowDateTime;
    return t;
}

static
inline void get_monotonic_time(timespec* tv)
{
    LARGE_INTEGER           t;
    FILETIME				f;
    double                  microseconds;
    static LARGE_INTEGER    offset;
    static double           frequencyToMicroseconds;
    static int              initialized = 0;
    static BOOL             usePerformanceCounter = 0;

    if (!initialized)
    {
        LARGE_INTEGER performanceFrequency;
        initialized = 1;
        usePerformanceCounter = QueryPerformanceFrequency(&performanceFrequency);
        if (usePerformanceCounter)
        {
            QueryPerformanceCounter(&offset);
            frequencyToMicroseconds = (double)performanceFrequency.QuadPart / 1000000.;
        }
        else
        {
            offset = get_filetime_offset();
            frequencyToMicroseconds = 10.;
        }
    }

    if (usePerformanceCounter)
    {
        QueryPerformanceCounter(&t);
    }
    else {
        GetSystemTimeAsFileTime(&f);
        t.QuadPart = f.dwHighDateTime;
        t.QuadPart <<= 32;
        t.QuadPart |= f.dwLowDateTime;
    }

    t.QuadPart -= offset.QuadPart;
    microseconds = (double)t.QuadPart / frequencyToMicroseconds;
    t.QuadPart = microseconds;
    tv->tv_sec = t.QuadPart / 1000000;
    tv->tv_nsec = (t.QuadPart % 1000000) * 1000;
}
#else
static
inline void get_monotonic_time(timespec* time)
{
#if defined(__APPLE__) && defined(__MACH__)
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    time->tv_sec = mts.tv_sec;
    time->tv_nsec = mts.tv_nsec;
#else
    clock_gettime(CLOCK_MONOTONIC, time);
#endif
}
#endif

static
inline timespec get_monotonic_time_diff(timespec start, timespec end)
{
    timespec temp;
    if (end.tv_nsec - start.tv_nsec < 0)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

static
inline double get_monotonic_time_diff_ms(timespec time1, timespec time2)
{
    timespec delta = get_monotonic_time_diff(time1, time2);
    double milliseconds = delta.tv_sec * 1000 + (double)delta.tv_nsec / 1000000.0;

    return milliseconds;
}
#endif // USE_AV_INTERRUPT_CALLBACK

static int get_number_of_cpus(void)
{
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(52, 111, 0)
    return 1;
#elif defined _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);

    return (int)sysinfo.dwNumberOfProcessors;
#elif defined __linux__ || defined __HAIKU__
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#elif defined __APPLE__
    int numCPU = 0;
    int mib[4];
    size_t len = sizeof(numCPU);

    // set the mib for hw.ncpu
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    // get the number of CPUs from the system
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if (numCPU < 1)
    {
        mib[1] = HW_NCPU;
        sysctl(mib, 2, &numCPU, &len, NULL, 0);

        if (numCPU < 1)
            numCPU = 1;
    }

    return (int)numCPU;
#else
    return 1;
#endif
}


struct Image_FFMPEG
{
    unsigned char* data;
    int step;
    int width;
    int height;
    int cn;
};


#if USE_AV_INTERRUPT_CALLBACK
struct AVInterruptCallbackMetadata
{
    timespec value;
    unsigned int timeout_after_ms;
    int timeout;
};

// https://github.com/opencv/opencv/pull/12693#issuecomment-426236731
static
inline const char* _opencv_avcodec_get_name(AVCodecID id)
{
#if LIBAVCODEC_VERSION_MICRO >= 100 \
    && LIBAVCODEC_BUILD >= CALC_FFMPEG_VERSION(53, 47, 100)
    return avcodec_get_name(id);
#else
    const AVCodecDescriptor* cd;
    AVCodec* codec;

    if (id == AV_CODEC_ID_NONE)
    {
        return "none";
    }
    cd = avcodec_descriptor_get(id);
    if (cd)
    {
        return cd->name;
    }
    codec = avcodec_find_decoder(id);
    if (codec)
    {
        return codec->name;
    }
    codec = avcodec_find_encoder(id);
    if (codec)
    {
        return codec->name;
    }

    return "unknown_codec";
#endif
}

static
inline void _opencv_ffmpeg_free(void** ptr)
{
    if (*ptr) free(*ptr);
    *ptr = 0;
}

static
inline int _opencv_ffmpeg_interrupt_callback(void* ptr)
{
    AVInterruptCallbackMetadata* metadata = (AVInterruptCallbackMetadata*)ptr;
    assert(metadata);

    if (metadata->timeout_after_ms == 0)
    {
        return 0; // timeout is disabled
    }

    timespec now;
    get_monotonic_time(&now);

    metadata->timeout = get_monotonic_time_diff_ms(metadata->value, now) > metadata->timeout_after_ms;

    return metadata->timeout ? -1 : 0;
}
#endif

static
inline void _opencv_ffmpeg_av_packet_unref(AVPacket* pkt)
{
#if LIBAVCODEC_BUILD >= (LIBAVCODEC_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(55, 25, 100) : CALC_FFMPEG_VERSION(55, 16, 0))
    av_packet_unref(pkt);
#else
    av_free_packet(pkt);
#endif
};

static
inline void _opencv_ffmpeg_av_image_fill_arrays(void* frame, uint8_t* ptr, enum AVPixelFormat pix_fmt, int width, int height)
{
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
    av_image_fill_arrays(((AVFrame*)frame)->data, ((AVFrame*)frame)->linesize, ptr, pix_fmt, width, height, 1);
#else
    avpicture_fill((AVPicture*)frame, ptr, pix_fmt, width, height);
#endif
};

static
inline int _opencv_ffmpeg_av_image_get_buffer_size(enum AVPixelFormat pix_fmt, int width, int height)
{
#if LIBAVUTIL_BUILD >= (LIBAVUTIL_VERSION_MICRO >= 100 \
    ? CALC_FFMPEG_VERSION(51, 63, 100) : CALC_FFMPEG_VERSION(54, 6, 0))
    return av_image_get_buffer_size(pix_fmt, width, height, 1);
#else
    return avpicture_get_size(pix_fmt, width, height);
#endif
};

static AVRational _opencv_ffmpeg_get_sample_aspect_ratio(AVStream* stream)
{
#if LIBAVUTIL_VERSION_MICRO >= 100 && LIBAVUTIL_BUILD >= CALC_FFMPEG_VERSION(54, 5, 100)
    return av_guess_sample_aspect_ratio(NULL, stream, NULL);
#else
    AVRational undef = { 0, 1 };

    // stream
    AVRational ratio = stream ? stream->sample_aspect_ratio : undef;
    av_reduce(&ratio.num, &ratio.den, ratio.num, ratio.den, INT_MAX);
    if (ratio.num > 0 && ratio.den > 0)
        return ratio;

    // codec
    ratio = stream && stream->codec ? stream->codec->sample_aspect_ratio : undef;
    av_reduce(&ratio.num, &ratio.den, ratio.num, ratio.den, INT_MAX);
    if (ratio.num > 0 && ratio.den > 0)
        return ratio;

    return undef;
#endif
}

/*
 * For CUDA encoder
 */

struct OutputMediaStream_FFMPEG
{
    bool open(const char* fileName, int width, int height, double fps);
    void close();

    void write(unsigned char* data, int size, int keyFrame);

    // add a video output stream to the container
    static AVStream* addVideoStream(AVFormatContext* oc, CV_CODEC_ID codec_id, int w, int h, int bitrate, double fps, AVPixelFormat pixel_format);

    AVOutputFormat* fmt_;
    AVFormatContext* oc_;
    AVStream* video_st_;
};

void OutputMediaStream_FFMPEG::close()
{
    // no more frame to compress. The codec has a latency of a few
    // frames if using B frames, so we get the last frames by
    // passing the same picture again

    // TODO -- do we need to account for latency here?

    if (oc_)
    {
        // write the trailer, if any
        av_write_trailer(oc_);

        // free the streams
        for (unsigned int i = 0; i < oc_->nb_streams; ++i)
        {
            av_freep(&oc_->streams[i]->codec);
            av_freep(&oc_->streams[i]);
        }

        if (!(fmt_->flags & AVFMT_NOFILE) && oc_->pb)
        {
            // close the output file

#if LIBAVCODEC_VERSION_INT < ((52<<16)+(123<<8)+0)
#if LIBAVCODEC_VERSION_INT >= ((51<<16)+(49<<8)+0)
            url_fclose(oc_->pb);
#else
            url_fclose(&oc_->pb);
#endif
#else
            avio_close(oc_->pb);
#endif
        }

        // free the stream
        av_free(oc_);
    }
}

AVStream* OutputMediaStream_FFMPEG::addVideoStream(AVFormatContext* oc, CV_CODEC_ID codec_id, int w, int h, int bitrate, double fps, AVPixelFormat pixel_format)
{
    AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec)
    {
        fprintf(stderr, "Could not find encoder for codec id %d\n", codec_id);
        return NULL;
    }

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 10, 0)
    AVStream * st = avformat_new_stream(oc, 0);
#else
    AVStream* st = av_new_stream(oc, 0);
#endif
    if (!st)
        return 0;

#if LIBAVFORMAT_BUILD > 4628
    AVCodecContext * c = st->codec;
#else
    AVCodecContext* c = &(st->codec);
#endif

    c->codec_id = codec_id;
    c->codec_type = AVMEDIA_TYPE_VIDEO;

    // put sample parameters
    c->bit_rate = bitrate;

    // took advice from
    // http://ffmpeg-users.933282.n4.nabble.com/warning-clipping-1-dct-coefficients-to-127-127-td934297.html
    c->qmin = 3;

    // resolution must be a multiple of two
    c->width = w;
    c->height = h;

    // time base: this is the fundamental unit of time (in seconds) in terms
    // of which frame timestamps are represented. for fixed-fps content,
    // timebase should be 1/framerate and timestamp increments should be
    // identically 1

    int frame_rate = static_cast<int>(fps + 0.5);
    int frame_rate_base = 1;
    while (fabs((static_cast<double>(frame_rate) / frame_rate_base) - fps) > 0.001)
    {
        frame_rate_base *= 10;
        frame_rate = static_cast<int>(fps * frame_rate_base + 0.5);
    }
    c->time_base.den = frame_rate;
    c->time_base.num = frame_rate_base;

#if LIBAVFORMAT_BUILD > 4752
    // adjust time base for supported framerates
    if (codec && codec->supported_framerates)
    {
        AVRational req = { frame_rate, frame_rate_base };
        const AVRational* best = NULL;
        AVRational best_error = { INT_MAX, 1 };

        for (const AVRational* p = codec->supported_framerates; p->den != 0; ++p)
        {
            AVRational error = av_sub_q(req, *p);

            if (error.num < 0)
                error.num *= -1;

            if (av_cmp_q(error, best_error) < 0)
            {
                best_error = error;
                best = p;
            }
        }

        if (best == NULL)
            return NULL;
        c->time_base.den = best->num;
        c->time_base.num = best->den;
    }
#endif

    c->gop_size = 12; // emit one intra frame every twelve frames at most
    c->pix_fmt = pixel_format;

    if (c->codec_id == CV_CODEC(CODEC_ID_MPEG2VIDEO))
        c->max_b_frames = 2;

    if (c->codec_id == CV_CODEC(CODEC_ID_MPEG1VIDEO) || c->codec_id == CV_CODEC(CODEC_ID_MSMPEG4V3))
    {
        // needed to avoid using macroblocks in which some coeffs overflow
        // this doesn't happen with normal video, it just happens here as the
        // motion of the chroma plane doesn't match the luma plane

        // avoid FFMPEG warning 'clipping 1 dct coefficients...'

        c->mb_decision = 2;
    }

#if LIBAVCODEC_VERSION_INT > 0x000409
    // some formats want stream headers to be separate
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
    {
#if LIBAVCODEC_BUILD > CALC_FFMPEG_VERSION(56, 35, 0)
        c->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
#else
        c->flags |= CODEC_FLAG_GLOBAL_HEADER;
#endif
    }
#endif

    return st;
}

bool OutputMediaStream_FFMPEG::open(const char* fileName, int width, int height, double fps)
{
    fmt_ = 0;
    oc_ = 0;
    video_st_ = 0;

    // auto detect the output format from the name and fourcc code
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
    fmt_ = av_guess_format(NULL, fileName, NULL);
#else
    fmt_ = guess_format(NULL, fileName, NULL);
#endif
    if (!fmt_)
        return false;

    CV_CODEC_ID codec_id = CV_CODEC(CODEC_ID_H264);

    // alloc memory for context
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 2, 0)
    oc_ = avformat_alloc_context();
#else
    oc_ = av_alloc_format_context();
#endif
    if (!oc_)
        return false;

    // set some options
    oc_->oformat = fmt_;
    snprintf(oc_->filename, sizeof(oc_->filename), "%s", fileName);

    oc_->max_delay = (int)(0.7 * AV_TIME_BASE); // This reduces buffer underrun warnings with MPEG

    // set a few optimal pixel formats for lossless codecs of interest..
    AVPixelFormat codec_pix_fmt = AV_PIX_FMT_YUV420P;
    int bitrate_scale = 64;

    // TODO -- safe to ignore output audio stream?
    video_st_ = addVideoStream(oc_, codec_id, width, height, width * height * bitrate_scale, fps, codec_pix_fmt);
    if (!video_st_)
        return false;

    // set the output parameters (must be done even if no parameters)
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
    if (av_set_parameters(oc_, NULL) < 0)
        return false;
#endif

    // now that all the parameters are set, we can open the audio and
    // video codecs and allocate the necessary encode buffers

#if LIBAVFORMAT_BUILD > 4628
    AVCodecContext * c = (video_st_->codec);
#else
    AVCodecContext* c = &(video_st_->codec);
#endif

    c->codec_tag = MKTAG('H', '2', '6', '4');
    c->bit_rate_tolerance = c->bit_rate;

    // open the output file, if needed
    if (!(fmt_->flags & AVFMT_NOFILE))
    {
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
        int err = url_fopen(&oc_->pb, fileName, URL_WRONLY);
#else
        int err = avio_open(&oc_->pb, fileName, AVIO_FLAG_WRITE);
#endif

        if (err != 0)
            return false;
    }

    // write the stream header, if any
    int header_err =
#if LIBAVFORMAT_BUILD < CALC_FFMPEG_VERSION(53, 2, 0)
        av_write_header(oc_);
#else
        avformat_write_header(oc_, NULL);
#endif
    if (header_err != 0)
        return false;

    return true;
}

void OutputMediaStream_FFMPEG::write(unsigned char* data, int size, int keyFrame)
{
    // if zero size, it means the image was buffered
    if (size > 0)
    {
        AVPacket pkt;
        av_init_packet(&pkt);

        if (keyFrame)
            pkt.flags |= PKT_FLAG_KEY;

        pkt.stream_index = video_st_->index;
        pkt.data = data;
        pkt.size = size;

        // write the compressed frame in the media file
        av_write_frame(oc_, &pkt);
    }
}

struct OutputMediaStream_FFMPEG* create_OutputMediaStream_FFMPEG(const char* fileName, int width, int height, double fps)
{
    OutputMediaStream_FFMPEG* stream = (OutputMediaStream_FFMPEG*)malloc(sizeof(OutputMediaStream_FFMPEG));
    if (!stream)
        return 0;

    if (stream->open(fileName, width, height, fps))
        return stream;

    stream->close();
    free(stream);

    return 0;
}

void release_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream)
{
    stream->close();
    free(stream);
}

void write_OutputMediaStream_FFMPEG(struct OutputMediaStream_FFMPEG* stream, unsigned char* data, int size, int keyFrame)
{
    stream->write(data, size, keyFrame);
}

/*
 * For CUDA decoder
 */

enum
{
    VideoCodec_MPEG1 = 0,
    VideoCodec_MPEG2,
    VideoCodec_MPEG4,
    VideoCodec_VC1,
    VideoCodec_H264,
    VideoCodec_JPEG,
    VideoCodec_H264_SVC,
    VideoCodec_H264_MVC,

    // Uncompressed YUV
    VideoCodec_YUV420 = (('I' << 24) | ('Y' << 16) | ('U' << 8) | ('V')),   // Y,U,V (4:2:0)
    VideoCodec_YV12 = (('Y' << 24) | ('V' << 16) | ('1' << 8) | ('2')),   // Y,V,U (4:2:0)
    VideoCodec_NV12 = (('N' << 24) | ('V' << 16) | ('1' << 8) | ('2')),   // Y,UV  (4:2:0)
    VideoCodec_YUYV = (('Y' << 24) | ('U' << 16) | ('Y' << 8) | ('V')),   // YUYV/YUY2 (4:2:2)
    VideoCodec_UYVY = (('U' << 24) | ('Y' << 16) | ('V' << 8) | ('Y'))    // UYVY (4:2:2)
};

enum
{
    VideoChromaFormat_Monochrome = 0,
    VideoChromaFormat_YUV420,
    VideoChromaFormat_YUV422,
    VideoChromaFormat_YUV444
};

struct InputMediaStream_FFMPEG
{
public:
    bool open(const char* fileName, int* codec, int* chroma_format, int* width, int* height);
    void close();

    bool read(unsigned char** data, int* size, int* endOfFile);

private:
    InputMediaStream_FFMPEG(const InputMediaStream_FFMPEG&);
    InputMediaStream_FFMPEG& operator =(const InputMediaStream_FFMPEG&);

    AVFormatContext* ctx_;
    int video_stream_id_;
    AVPacket pkt_;

#if USE_AV_INTERRUPT_CALLBACK
    AVInterruptCallbackMetadata interrupt_metadata;
#endif
};

bool InputMediaStream_FFMPEG::open(const char* fileName, int* codec, int* chroma_format, int* width, int* height)
{
    int err;

    ctx_ = 0;
    video_stream_id_ = -1;
    memset(&pkt_, 0, sizeof(AVPacket));

#if USE_AV_INTERRUPT_CALLBACK
    /* interrupt callback */
    interrupt_metadata.timeout_after_ms = LIBAVFORMAT_INTERRUPT_OPEN_TIMEOUT_MS;
    get_monotonic_time(&interrupt_metadata.value);

    ctx_ = avformat_alloc_context();
    ctx_->interrupt_callback.callback = _opencv_ffmpeg_interrupt_callback;
    ctx_->interrupt_callback.opaque = &interrupt_metadata;
#endif

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 13, 0)
    avformat_network_init();
#endif

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 6, 0)
    err = avformat_open_input(&ctx_, fileName, 0, 0);
#else
    err = av_open_input_file(&ctx_, fileName, 0, 0, 0);
#endif
    if (err < 0)
        return false;

#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 6, 0)
    err = avformat_find_stream_info(ctx_, 0);
#else
    err = av_find_stream_info(ctx_);
#endif
    if (err < 0)
        return false;

    for (unsigned int i = 0; i < ctx_->nb_streams; ++i)
    {
#if LIBAVFORMAT_BUILD > 4628
        AVCodecContext * enc = ctx_->streams[i]->codec;
#else
        AVCodecContext* enc = &ctx_->streams[i]->codec;
#endif

        if (enc->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_id_ = static_cast<int>(i);

            switch (enc->codec_id)
            {
            case CV_CODEC(CODEC_ID_MPEG1VIDEO):
                *codec = ::VideoCodec_MPEG1;
                break;

            case CV_CODEC(CODEC_ID_MPEG2VIDEO):
                *codec = ::VideoCodec_MPEG2;
                break;

            case CV_CODEC(CODEC_ID_MPEG4):
                *codec = ::VideoCodec_MPEG4;
                break;

            case CV_CODEC(CODEC_ID_VC1):
                *codec = ::VideoCodec_VC1;
                break;

            case CV_CODEC(CODEC_ID_H264):
                *codec = ::VideoCodec_H264;
                break;

            default:
                return false;
            };

            switch (enc->pix_fmt)
            {
                // fix from Video_Codec_SDK_9.0.20\Video_Codec_SDK_9.0.20\Samples\Utils\FFmpegDemuxer.h
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUVJ420P:
            case AV_PIX_FMT_YUVJ422P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
            case AV_PIX_FMT_YUVJ444P:   // jpeg decoder output is subsampled to NV12 for 422/444 so treat it as 420
                *chroma_format = ::VideoChromaFormat_YUV420;
                break;

            case AV_PIX_FMT_YUV422P:
                *chroma_format = ::VideoChromaFormat_YUV422;
                break;

            case AV_PIX_FMT_YUV444P:
                *chroma_format = ::VideoChromaFormat_YUV444;
                break;

            default:
                return false;
            }

            *width = enc->coded_width;
            *height = enc->coded_height;

            break;
        }
    }

    if (video_stream_id_ < 0)
        return false;

    av_init_packet(&pkt_);

#if USE_AV_INTERRUPT_CALLBACK
    // deactivate interrupt callback
    interrupt_metadata.timeout_after_ms = 0;
#endif

    return true;
}

void InputMediaStream_FFMPEG::close()
{
    if (ctx_)
    {
#if LIBAVFORMAT_BUILD >= CALC_FFMPEG_VERSION(53, 24, 2)
        avformat_close_input(&ctx_);
#else
        av_close_input_file(ctx_);
#endif
    }

    // free last packet if exist
    if (pkt_.data)
        _opencv_ffmpeg_av_packet_unref(&pkt_);
}


// we may be able to use the inbuilt funcs in cap_ffmpeg_impl.hpp to get the packet?? - call grab then ref the packet????
bool InputMediaStream_FFMPEG::read(unsigned char** data, int* size, int* endOfFile)
{
    bool result = false;

#if USE_AV_INTERRUPT_CALLBACK
    // activate interrupt callback
    get_monotonic_time(&interrupt_metadata.value);
    interrupt_metadata.timeout_after_ms = LIBAVFORMAT_INTERRUPT_READ_TIMEOUT_MS;
#endif

    // free last packet if exist
    if (pkt_.data)
        _opencv_ffmpeg_av_packet_unref(&pkt_);

    // get the next frame
    for (;;)
    {
#if USE_AV_INTERRUPT_CALLBACK
        if (interrupt_metadata.timeout)
        {
            break;
        }
#endif

        int ret = av_read_frame(ctx_, &pkt_);

        if (ret == AVERROR(EAGAIN))
            continue;

        if (ret < 0)
        {
            if (ret == (int)AVERROR_EOF)
                * endOfFile = true;
            break;
        }

        if (pkt_.stream_index != video_stream_id_)
        {
            _opencv_ffmpeg_av_packet_unref(&pkt_);
            continue;
        }

        result = true;
        break;
    }

#if USE_AV_INTERRUPT_CALLBACK
    // deactivate interrupt callback
    interrupt_metadata.timeout_after_ms = 0;
#endif

    if (result)
    {
        *data = pkt_.data;
        *size = pkt_.size;
        *endOfFile = false;
    }

    return result;
}

InputMediaStream_FFMPEG* create_InputMediaStream_FFMPEG(const char* fileName, int* codec, int* chroma_format, int* width, int* height)
{
    InputMediaStream_FFMPEG* stream = (InputMediaStream_FFMPEG*)malloc(sizeof(InputMediaStream_FFMPEG));
    if (!stream)
        return 0;

    if (stream && stream->open(fileName, codec, chroma_format, width, height))
        return stream;

    stream->close();
    free(stream);

    return 0;
}

void release_InputMediaStream_FFMPEG(InputMediaStream_FFMPEG* stream)
{
    stream->close();
    free(stream);
}

int read_InputMediaStream_FFMPEG(InputMediaStream_FFMPEG* stream, unsigned char** data, int* size, int* endOfFile)
{
    return stream->read(data, size, endOfFile);
}