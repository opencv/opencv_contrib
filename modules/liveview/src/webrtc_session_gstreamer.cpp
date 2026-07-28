#include "webrtc_session.hpp"

#if defined(HAVE_LIVEVIEW_WEBRTC_GSTREAMER)

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#ifndef GST_USE_UNSTABLE_API
#define GST_USE_UNSTABLE_API
#endif

extern "C" {
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/sdp/gstsdpmessage.h>
#include <gst/webrtc/webrtc.h>
}

namespace cv {
namespace liveview {
namespace {

static bool hasProperty(GObject* object, const char* name)
{
    return object && g_object_class_find_property(G_OBJECT_GET_CLASS(object), name) != NULL;
}

static GstWebRTCSDPType toGstSdpType(const String& type)
{
    if (type == "offer")
        return GST_WEBRTC_SDP_TYPE_OFFER;
    if (type == "answer")
        return GST_WEBRTC_SDP_TYPE_ANSWER;
    return GST_WEBRTC_SDP_TYPE_OFFER;
}

static bool hasVideoMedia(const GstSDPMessage* sdp)
{
    if (!sdp)
        return false;
    const guint count = gst_sdp_message_medias_len(sdp);
    for (guint i = 0; i < count; ++i)
    {
        const GstSDPMedia* media = gst_sdp_message_get_media(sdp, i);
        const gchar* type = gst_sdp_media_get_media(media);
        if (type && std::strcmp(type, "video") == 0)
            return true;
    }
    return false;
}

static GstCaps* encodedCaps(VideoCodec codec)
{
    if (codec == VideoCodec::H264)
        return gst_caps_new_simple("video/x-h264",
                                   "stream-format", G_TYPE_STRING, "byte-stream",
                                   "alignment", G_TYPE_STRING, "au",
                                   NULL);
    if (codec == VideoCodec::VP8)
        return gst_caps_new_empty_simple("video/x-vp8");
    return NULL;
}

static GstElement* parserFor(VideoCodec codec)
{
    if (codec == VideoCodec::H264)
    {
        GstElement* parse = gst_element_factory_make("h264parse", NULL);
        if (parse && hasProperty(G_OBJECT(parse), "config-interval"))
            g_object_set(parse, "config-interval", 1, NULL);
        return parse;
    }
    if (codec == VideoCodec::VP8)
        return gst_element_factory_make("vp8parse", NULL);
    return NULL;
}

static GstElement* payloaderFor(VideoCodec codec)
{
    GstElement* pay = NULL;
    if (codec == VideoCodec::H264)
        pay = gst_element_factory_make("rtph264pay", NULL);
    else if (codec == VideoCodec::VP8)
        pay = gst_element_factory_make("rtpvp8pay", NULL);
    if (pay && hasProperty(G_OBJECT(pay), "pt"))
        g_object_set(pay, "pt", 96, NULL);
    if (pay && hasProperty(G_OBJECT(pay), "mtu"))
        g_object_set(pay, "mtu", 1200, NULL);
    if (pay && codec == VideoCodec::H264 && hasProperty(G_OBJECT(pay), "config-interval"))
        g_object_set(pay, "config-interval", 1, NULL);
    return pay;
}

static GstCaps* rtpCaps(VideoCodec codec)
{
    if (codec == VideoCodec::H264)
        return gst_caps_new_simple("application/x-rtp",
                                   "media", G_TYPE_STRING, "video",
                                   "encoding-name", G_TYPE_STRING, "H264",
                                   "payload", G_TYPE_INT, 96,
                                   "clock-rate", G_TYPE_INT, 90000,
                                   NULL);
    if (codec == VideoCodec::VP8)
        return gst_caps_new_simple("application/x-rtp",
                                   "media", G_TYPE_STRING, "video",
                                   "encoding-name", G_TYPE_STRING, "VP8",
                                   "payload", G_TYPE_INT, 96,
                                   "clock-rate", G_TYPE_INT, 90000,
                                   NULL);
    return NULL;
}

struct AnswerState
{
    ~AnswerState()
    {
        if (description)
            gst_webrtc_session_description_free(description);
    }

    std::mutex mutex;
    std::condition_variable updated;
    bool ready = false;
    GstWebRTCSessionDescription* description = NULL;
};

class GStreamerWebRtcSession CV_FINAL : public WebRtcSession
{
public:
    GStreamerWebRtcSession()
        : context_(NULL), loop_(NULL), pipeline_(NULL), appsrc_(NULL), parser_(NULL),
          pay_(NULL), webrtc_(NULL), codec_(VideoCodec::Unknown), fps_(30),
          opened_(false), stopRequested_(false), ptsNs_(0)
    {
        static std::once_flag once;
        std::call_once(once, []() { gst_init(NULL, NULL); });
    }

    ~GStreamerWebRtcSession() CV_OVERRIDE
    {
        close();
    }

    bool open(VideoCodec codec, int fps) CV_OVERRIDE
    {
        close();
        codec_ = codec;
        fps_ = fps > 0 ? fps : 30;
        context_ = g_main_context_new();
        pipeline_ = gst_pipeline_new(NULL);
        appsrc_ = gst_element_factory_make("appsrc", "liveview_src");
        parser_ = parserFor(codec);
        pay_ = payloaderFor(codec);
        webrtc_ = gst_element_factory_make("webrtcbin", "liveview_webrtc");

        if (!pipeline_ || !appsrc_ || !pay_ || !webrtc_)
        {
            close();
            return false;
        }

        GstCaps* caps = encodedCaps(codec);
        if (!caps)
        {
            close();
            return false;
        }
        g_object_set(appsrc_,
                     "is-live", TRUE,
                     "format", GST_FORMAT_TIME,
                     "block", FALSE,
                     "max-buffers", 2,
                     "max-bytes", static_cast<guint64>(1u << 20),
                     "caps", caps,
                     NULL);
        gst_caps_unref(caps);

        if (hasProperty(G_OBJECT(webrtc_), "bundle-policy"))
            g_object_set(webrtc_, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);
        if (hasProperty(G_OBJECT(webrtc_), "latency"))
            g_object_set(webrtc_, "latency", 50, NULL);

        GstElement* queue = gst_element_factory_make("queue", "liveview_webrtc_q");
        if (!queue)
        {
            close();
            return false;
        }
        g_object_set(queue, "leaky", 2, "max-size-buffers", 1, "max-size-bytes", 0, "max-size-time", 0, NULL);

        gst_bin_add(GST_BIN(pipeline_), appsrc_);
        if (parser_)
            gst_bin_add(GST_BIN(pipeline_), parser_);
        gst_bin_add_many(GST_BIN(pipeline_), queue, pay_, webrtc_, NULL);

        gboolean linked = FALSE;
        if (parser_)
            linked = gst_element_link_many(appsrc_, parser_, queue, pay_, NULL);
        else
            linked = gst_element_link_many(appsrc_, queue, pay_, NULL);
        if (!linked)
        {
            close();
            return false;
        }

        GstCaps* capsRtp = rtpCaps(codec);
        linked = capsRtp && gst_element_link_pads_filtered(pay_, "src", webrtc_, "sink_%u", capsRtp);
        if (capsRtp)
            gst_caps_unref(capsRtp);
        if (!linked)
        {
            close();
            return false;
        }

        g_signal_connect(webrtc_, "on-ice-candidate",
                         G_CALLBACK(+[](GstElement*, guint, gchar*, gpointer) {}), this);

        opened_ = true;
        {
            std::lock_guard<std::mutex> lock(loopMutex_);
            stopRequested_ = false;
        }
        loopThread_ = std::thread(&GStreamerWebRtcSession::runLoop, this);
        {
            std::unique_lock<std::mutex> lock(loopMutex_);
            loopUpdated_.wait_for(lock, std::chrono::seconds(1), [this]() { return loop_ != NULL; });
        }
        return true;
    }

    bool createAnswer(const WebRtcOffer& offer, WebRtcAnswer& answer) CV_OVERRIDE
    {
        if (!opened_ || !webrtc_)
            return false;
        GstSDPMessage* sdp = NULL;
        if (gst_sdp_message_new(&sdp) != GST_SDP_OK)
            return false;
        if (gst_sdp_message_parse_buffer(reinterpret_cast<const guint8*>(offer.sdp.c_str()),
                                         offer.sdp.size(), sdp) != GST_SDP_OK)
        {
            gst_sdp_message_free(sdp);
            return false;
        }
        if (!hasVideoMedia(sdp))
        {
            gst_sdp_message_free(sdp);
            return false;
        }

        GstWebRTCSessionDescription* desc = gst_webrtc_session_description_new(toGstSdpType(offer.type), sdp);
        bool remoteSet = false;
        invokeOnContext([this, desc, &remoteSet]() {
            GstPromise* setPromise = gst_promise_new();
            g_signal_emit_by_name(webrtc_, "set-remote-description", desc, setPromise);
            gst_promise_wait(setPromise);
            gst_promise_unref(setPromise);
            remoteSet = true;
        });
        gst_webrtc_session_description_free(desc);
        if (!remoteSet)
            return false;

        std::shared_ptr<AnswerState> state = std::make_shared<AnswerState>();
        std::shared_ptr<AnswerState>* callbackState = new std::shared_ptr<AnswerState>(state);
        GstPromise* promise = gst_promise_new_with_change_func(
            +[](GstPromise* p, gpointer user) {
                std::shared_ptr<AnswerState> callbackAnswerState =
                    *static_cast<std::shared_ptr<AnswerState>*>(user);
                delete static_cast<std::shared_ptr<AnswerState>*>(user);
                const GstStructure* reply = gst_promise_get_reply(p);
                GstWebRTCSessionDescription* local = NULL;
                if (reply)
                    gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &local, NULL);
                {
                    std::lock_guard<std::mutex> lock(callbackAnswerState->mutex);
                    callbackAnswerState->description = local;
                    callbackAnswerState->ready = true;
                }
                callbackAnswerState->updated.notify_all();
                gst_promise_unref(p);
            }, callbackState, NULL);
        invokeOnContext([this, promise]() {
            g_signal_emit_by_name(webrtc_, "create-answer", NULL, promise);
        });

        GstWebRTCSessionDescription* local = NULL;
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->updated.wait_for(lock, std::chrono::seconds(2), [&state]() { return state->ready; });
            if (!state->ready || !state->description)
                return false;
            local = state->description;
            state->description = NULL;
        }

        bool localSet = false;
        invokeOnContext([this, local, &localSet]() {
            GstPromise* localPromise = gst_promise_new();
            g_signal_emit_by_name(webrtc_, "set-local-description", local, localPromise);
            gst_promise_wait(localPromise);
            gst_promise_unref(localPromise);
            localSet = true;
        });
        if (!localSet)
        {
            gst_webrtc_session_description_free(local);
            return false;
        }

        gchar* sdpText = gst_sdp_message_as_text(local->sdp);
        String sdpAnswer = sdpText ? sdpText : "";
        g_free(sdpText);
        gst_webrtc_session_description_free(local);
        if (sdpAnswer.empty())
            return false;
        answer.type = "answer";
        answer.sdp = sdpAnswer;
        return true;
    }

    bool addRemoteCandidate(const WebRtcIceCandidate& candidate) CV_OVERRIDE
    {
        if (!opened_ || !webrtc_)
            return false;
        const String candidateText = candidate.candidate;
        invokeOnContext([this, candidate, candidateText]() {
            if (candidateText.empty())
                g_signal_emit_by_name(webrtc_, "add-ice-candidate",
                                      candidate.sdpMLineIndex, static_cast<const gchar*>(NULL));
            else
                g_signal_emit_by_name(webrtc_, "add-ice-candidate",
                                      candidate.sdpMLineIndex, candidateText.c_str());
        });
        return true;
    }

    bool pushEncoded(const EncodedFrame& frame) CV_OVERRIDE
    {
        if (!opened_ || !appsrc_ || frame.data.empty() || frame.codec != codec_)
            return false;

        GstBuffer* buffer = gst_buffer_new_allocate(NULL, frame.data.size(), NULL);
        if (!buffer)
            return false;
        GstMapInfo map;
        gst_buffer_map(buffer, &map, GST_MAP_WRITE);
        std::memcpy(map.data, &frame.data[0], frame.data.size());
        gst_buffer_unmap(buffer, &map);

        const guint64 duration = static_cast<guint64>(1000000000ull / static_cast<guint64>(fps_));
        GST_BUFFER_PTS(buffer) = ptsNs_;
        GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
        GST_BUFFER_DURATION(buffer) = duration;
        ptsNs_ += duration;
        if (!frame.keyframe)
            GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DELTA_UNIT);

        GstFlowReturn ret = GST_FLOW_ERROR;
        g_signal_emit_by_name(appsrc_, "push-buffer", buffer, &ret);
        gst_buffer_unref(buffer);
        return ret == GST_FLOW_OK;
    }

    void close() CV_OVERRIDE
    {
        opened_ = false;
        if (appsrc_)
            gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
        {
            std::unique_lock<std::mutex> lock(loopMutex_);
            stopRequested_ = true;
            if (loopThread_.joinable() && !loop_)
                loopUpdated_.wait_for(lock, std::chrono::seconds(1), [this]() { return loop_ != NULL; });
            if (loop_)
                g_main_loop_quit(loop_);
        }
        if (loopThread_.joinable())
            loopThread_.join();
        if (pipeline_)
            gst_element_set_state(pipeline_, GST_STATE_NULL);
        if (pipeline_)
            gst_object_unref(pipeline_);
        pipeline_ = NULL;
        appsrc_ = NULL;
        parser_ = NULL;
        pay_ = NULL;
        webrtc_ = NULL;
        if (context_)
            g_main_context_unref(context_);
        context_ = NULL;
        {
            std::lock_guard<std::mutex> lock(loopMutex_);
            loop_ = NULL;
        }
        ptsNs_ = 0;
    }

private:
    void invokeOnContext(const std::function<void()>& fn)
    {
        if (!context_)
            return;
        struct Task
        {
            explicit Task(const std::function<void()>& f) : fn(f), done(false) {}
            std::function<void()> fn;
            std::mutex mutex;
            std::condition_variable updated;
            bool done;
        };
        std::shared_ptr<Task> task = std::make_shared<Task>(fn);
        std::shared_ptr<Task>* user = new std::shared_ptr<Task>(task);
        g_main_context_invoke(context_,
                              +[](gpointer data) -> gboolean {
                                  std::shared_ptr<Task> callbackTask =
                                      *static_cast<std::shared_ptr<Task>*>(data);
                                  delete static_cast<std::shared_ptr<Task>*>(data);
                                  callbackTask->fn();
                                  {
                                      std::lock_guard<std::mutex> lock(callbackTask->mutex);
                                      callbackTask->done = true;
                                  }
                                  callbackTask->updated.notify_all();
                                  return G_SOURCE_REMOVE;
                              },
                              user);
        std::unique_lock<std::mutex> lock(task->mutex);
        task->updated.wait_for(lock, std::chrono::seconds(3), [&task]() { return task->done; });
    }

    void runLoop()
    {
        g_main_context_push_thread_default(context_);
        GMainLoop* loop = g_main_loop_new(context_, FALSE);
        bool stopRequested = false;
        {
            std::lock_guard<std::mutex> lock(loopMutex_);
            loop_ = loop;
            stopRequested = stopRequested_;
        }
        loopUpdated_.notify_all();
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (!stopRequested)
            g_main_loop_run(loop);
        {
            std::lock_guard<std::mutex> lock(loopMutex_);
            if (loop_ == loop)
                loop_ = NULL;
        }
        loopUpdated_.notify_all();
        g_main_loop_unref(loop);
        g_main_context_pop_thread_default(context_);
    }

    GMainContext* context_;
    GMainLoop* loop_;
    GstElement* pipeline_;
    GstElement* appsrc_;
    GstElement* parser_;
    GstElement* pay_;
    GstElement* webrtc_;
    VideoCodec codec_;
    int fps_;
    bool opened_;
    bool stopRequested_;
    guint64 ptsNs_;
    std::thread loopThread_;
    std::mutex loopMutex_;
    std::condition_variable loopUpdated_;
};

} // namespace

bool haveWebRtcBackend()
{
    static std::once_flag once;
    static bool available = false;
    std::call_once(once, []() {
        gst_init(NULL, NULL);
        GstElementFactory* webrtc = gst_element_factory_find("webrtcbin");
        GstElementFactory* appsrc = gst_element_factory_find("appsrc");
        GstElementFactory* pay = gst_element_factory_find("rtph264pay");
        GstElementFactory* parse = gst_element_factory_find("h264parse");
        available = webrtc && appsrc && pay && parse;
        if (webrtc)
            gst_object_unref(webrtc);
        if (appsrc)
            gst_object_unref(appsrc);
        if (pay)
            gst_object_unref(pay);
        if (parse)
            gst_object_unref(parse);
    });
    return available;
}

Ptr<WebRtcSession> createWebRtcSession()
{
    if (!haveWebRtcBackend())
        CV_Error(Error::StsNotImplemented, "GStreamer WebRTC runtime elements are not available");
    return makePtr<GStreamerWebRtcSession>();
}

} // namespace liveview
} // namespace cv

#endif // HAVE_LIVEVIEW_WEBRTC_GSTREAMER
