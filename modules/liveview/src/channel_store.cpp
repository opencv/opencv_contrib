#include "channel_store.hpp"
#include "frame_encoder.hpp"
#include "route_matcher.hpp"

#include <chrono>

namespace cv {
namespace liveview {

struct ChannelStore::ChannelState
{
    ChannelInfo info;
    Mat pending;
    int64 pendingSequence = 0;
    std::vector<uchar> jpeg;
    int64 encodedSequence = 0;
    bool dirty = false;
    mutable std::mutex mutex;
    mutable std::condition_variable updated;
    mutable std::condition_variable rawUpdated;
};

ChannelStore::ChannelStore(int jpegQuality)
    : jpegQuality_(jpegQuality), stopping_(false)
{
    encoderThread_ = std::thread(&ChannelStore::encoderLoop, this);
}

ChannelStore::~ChannelStore()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    dirty_.notify_all();
    if (encoderThread_.joinable())
        encoderThread_.join();
}

std::shared_ptr<ChannelStore::ChannelState> ChannelStore::getOrCreateLocked(const String& name)
{
    std::map<String, std::shared_ptr<ChannelState> >::iterator it = channels_.find(name);
    if (it != channels_.end())
        return it->second;

    std::shared_ptr<ChannelState> state(new ChannelState);
    state->info.name = name;
    channels_[name] = state;
    return state;
}

std::shared_ptr<ChannelStore::ChannelState> ChannelStore::find(const String& name) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::map<String, std::shared_ptr<ChannelState> >::const_iterator it = channels_.find(name);
    return it == channels_.end() ? std::shared_ptr<ChannelState>() : it->second;
}

void ChannelStore::publish(const String& name, InputArray frame)
{
    if (!isValidChannelName(name))
        CV_Error(Error::StsBadArg, "Invalid LiveView channel name");

    Mat src = frame.getMat();
    if (src.empty())
        CV_Error(Error::StsBadArg, "LiveView cannot publish an empty frame");

    FrameEncoder encoder(jpegQuality_);
    encoder.validate(src);

    std::shared_ptr<ChannelState> state;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state = getOrCreateLocked(name);
    }

    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->info.width = src.cols;
        state->info.height = src.rows;
        state->info.type = src.type();
        state->info.sequence++;
        state->pending = src.clone();
        state->pendingSequence = state->info.sequence;
        state->dirty = true;
    }
    state->rawUpdated.notify_all();
    dirty_.notify_one();
}

std::vector<ChannelInfo> ChannelStore::channels() const
{
    std::vector<std::shared_ptr<ChannelState> > states;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::map<String, std::shared_ptr<ChannelState> >::const_iterator it = channels_.begin();
             it != channels_.end(); ++it)
            states.push_back(it->second);
    }

    std::vector<ChannelInfo> out;
    out.reserve(states.size());
    for (size_t i = 0; i < states.size(); ++i)
    {
        std::lock_guard<std::mutex> lock(states[i]->mutex);
        out.push_back(states[i]->info);
    }
    return out;
}

bool ChannelStore::getRawFrame(const String& name, RawFrameSnapshot& out) const
{
    std::shared_ptr<ChannelState> state = find(name);
    if (!state)
        return false;

    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->pending.empty())
        return false;
    out.info = state->info;
    out.frame = state->pending.clone();
    return true;
}

bool ChannelStore::waitForRawFrame(const String& name, int64 afterSequence, int timeoutMs, RawFrameSnapshot& out) const
{
    std::shared_ptr<ChannelState> state = find(name);
    if (!state)
        return false;

    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->info.sequence <= afterSequence)
    {
        state->rawUpdated.wait_for(lock, std::chrono::milliseconds(timeoutMs));
    }

    if (state->info.sequence <= afterSequence || state->pending.empty())
        return false;

    out.info = state->info;
    out.frame = state->pending.clone();
    return true;
}

bool ChannelStore::getSnapshot(const String& name, ChannelSnapshot& out) const
{
    std::shared_ptr<ChannelState> state = find(name);
    if (!state)
        return false;

    std::lock_guard<std::mutex> lock(state->mutex);
    out.info = state->info;
    out.encodedSequence = state->encodedSequence;
    out.jpeg = state->jpeg;
    return !out.jpeg.empty();
}

bool ChannelStore::waitForJpeg(const String& name, int64 afterSequence, int timeoutMs, ChannelSnapshot& out) const
{
    std::shared_ptr<ChannelState> state = find(name);
    if (!state)
        return false;

    std::unique_lock<std::mutex> lock(state->mutex);
    if (state->encodedSequence <= afterSequence)
    {
        state->updated.wait_for(lock, std::chrono::milliseconds(timeoutMs));
    }

    if (state->encodedSequence <= afterSequence || state->jpeg.empty())
        return false;

    out.info = state->info;
    out.encodedSequence = state->encodedSequence;
    out.jpeg = state->jpeg;
    return true;
}

std::shared_ptr<ChannelStore::ChannelState> ChannelStore::nextDirty()
{
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::map<String, std::shared_ptr<ChannelState> >::const_iterator it = channels_.begin();
         it != channels_.end(); ++it)
    {
        std::lock_guard<std::mutex> stateLock(it->second->mutex);
        if (it->second->dirty)
            return it->second;
    }
    return std::shared_ptr<ChannelState>();
}

void ChannelStore::encoderLoop()
{
    FrameEncoder encoder(jpegQuality_);
    for (;;)
    {
        std::shared_ptr<ChannelState> state = nextDirty();
        if (!state)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stopping_)
                return;
            dirty_.wait_for(lock, std::chrono::milliseconds(50));
            if (stopping_)
                return;
            continue;
        }

        {
            Mat frame;
            int64 seq = 0;
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                frame = state->pending;
                seq = state->pendingSequence;
                state->dirty = false;
            }
            std::vector<uchar> jpeg = encoder.encodeJpeg(frame);
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (seq >= state->encodedSequence)
                {
                    state->jpeg.swap(jpeg);
                    state->encodedSequence = seq;
                }
            }
            state->updated.notify_all();
        }
    }
}

} // namespace liveview
} // namespace cv
