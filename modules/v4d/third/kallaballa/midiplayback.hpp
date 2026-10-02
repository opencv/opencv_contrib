#ifndef SRC_LIB_MIDIPLAYBACK_HPP_
#define SRC_LIB_MIDIPLAYBACK_HPP_

#include "midireceiver.hpp"
#include <algorithm>

#include <mutex>

class MidiPlayback {
	MidiReceiver recv_;
	bool running_ = false;
	uint64_t firstTimestamp_ = 0;
	std::vector<MidiEvent> recordBuffer_;
	std::mutex bufferMtx_;
public:
	MidiPlayback(int32_t inport);
	virtual ~MidiPlayback();
	void record();
	void stop();
	std::vector<MidiEvent> get_until_epoch(uint64_t epoch);
	std::vector<MidiEvent> get_until_tick(uint64_t tick);
};



#endif /* SRC_LIB_MIDIPLAYBACK_HPP_ */
