#include "midiplayback.hpp"

#include <thread>
#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std::chrono;

std::vector<MidiEvent>::iterator findClosestEvent(std::vector<MidiEvent> & data, uint64_t key)
{
    if (data.size() == 0) {
        throw std::out_of_range("Received empty vector.");
    }

    std::vector<MidiEvent>::iterator lower = std::lower_bound(data.begin(), data.end(), key, [](const MidiEvent& lhs, const uint64_t& ts) {
    	return lhs.timestamp_ < ts;
	});

    if (lower == data.end()) // If none found, return the last one.
        return data.end()-1;

    if (lower == data.begin())
        return lower;

    // Check which one is closest.
    auto previous = std::prev(lower);
    if ((key - (*previous).timestamp_) < ((*lower).timestamp_ - key))
        return previous;

    return lower;
}


MidiPlayback::MidiPlayback(int32_t inport) : recv_(inport, false) {
}

MidiPlayback::~MidiPlayback(){
}

void MidiPlayback::record() {
	std::unique_lock<std::mutex> lock(bufferMtx_);
	if(running_)
		return;

	firstTimestamp_ = 0;
	recordBuffer_.clear();
	running_ = true;
	recv_.start();

	std::thread t([&]() {
		while(running_) {
			std::this_thread::sleep_for(10ms);
			std::vector<MidiEvent> events = recv_.receive();
			std::unique_lock<std::mutex> lock(bufferMtx_);
			if(events.empty())
				continue;

			recordBuffer_.insert(recordBuffer_.end(), events.begin(), events.end());
		}
	});
	t.detach();
}

void MidiPlayback::stop() {
	std::unique_lock<std::mutex> lock(bufferMtx_);
	recv_.stop();
	running_ = false;
}

std::vector<MidiEvent> MidiPlayback::get_until_epoch(uint64_t sinceEpoch) {
	std::unique_lock<std::mutex> lock(bufferMtx_);

	if(recordBuffer_.empty()) {
		return {};
	}

	if(firstTimestamp_ == 0) {
		firstTimestamp_ = recordBuffer_.front().timestamp_;
	}

	uint64_t timestamp = sinceEpoch;

    std::vector<MidiEvent>::iterator it = std::lower_bound(recordBuffer_.begin(), recordBuffer_.end(), timestamp, [](const MidiEvent& lhs, const uint64_t& ts) {
    	return lhs.timestamp_ < ts;
	});


	std::vector<MidiEvent> queuedEvents;
	queuedEvents.insert(queuedEvents.end(), recordBuffer_.begin(), it);
	recordBuffer_.erase(recordBuffer_.begin(), it);

	return queuedEvents;
}


std::vector<MidiEvent> MidiPlayback::get_until_tick(uint64_t tick) {
	std::unique_lock<std::mutex> lock(bufferMtx_);

	if(recordBuffer_.empty()) {
		return {};
	}

	if(firstTimestamp_ == 0) {
		firstTimestamp_ = recordBuffer_.front().timestamp_;
	}

	uint64_t timestamp = firstTimestamp_ + tick;

    std::vector<MidiEvent>::iterator it = std::lower_bound(recordBuffer_.begin(), recordBuffer_.end(), timestamp, [](const MidiEvent& lhs, const uint64_t& ts) {
    	return lhs.timestamp_ < ts;
	});


	std::vector<MidiEvent> queuedEvents;
	queuedEvents.insert(queuedEvents.end(), recordBuffer_.begin(), it);
	recordBuffer_.erase(recordBuffer_.begin(), it);

	return queuedEvents;
}
