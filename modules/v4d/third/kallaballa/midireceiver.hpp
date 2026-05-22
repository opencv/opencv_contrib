#ifndef SRC_SLIDE_MIDI_HPP_
#define SRC_SLIDE_MIDI_HPP_

#include <rtmidi/RtMidi.h>
#include <iostream>
#include <mutex>
#include <chrono>

struct MidiEvent {
	bool on_ = false;
	bool cc_ = false;
	bool clock_ = false;
	uint16_t note_ = 0;
	uint16_t velocity_ = 0;
	uint16_t channel_ = 0;
	uint64_t timestamp_ = 0;
	uint16_t controller_ = 0;
	uint16_t value_ = 0;
};

std::ostream& operator<<(std::ostream &out, const MidiEvent& ev);
class MidiReceiver {
private:
	RtMidiIn *midiin_ = new RtMidiIn();
	int32_t inport_;
public:
	static std::vector<MidiEvent>* queue_;
	static std::mutex* evMtx_;
	MidiReceiver(int32_t inport, bool autostart = true);
	virtual ~MidiReceiver();
	void start();
	void stop();
	void clear();

	std::vector<MidiEvent> receive();
};

#endif /* SRC_SLIDE_MIDI_HPP_ */
