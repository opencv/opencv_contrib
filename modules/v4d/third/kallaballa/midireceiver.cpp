#include "midireceiver.hpp"

//#include <iostream>

std::vector<MidiEvent>* MidiReceiver::queue_ = new std::vector<MidiEvent>();
std::mutex* MidiReceiver::evMtx_ = new std::mutex();

void midiCallback(double deltatime, std::vector<unsigned char>* msg, void* userData) {
	std::unique_lock<std::mutex> lock(*MidiReceiver::evMtx_);
	int nBytes;
	nBytes = (*msg).size();
	MidiEvent ev;

	if (nBytes == 3) {
		int mask = ((*msg)[0] & 240);
		if(mask == 144 || mask == 128) {
			ev.on_ = (mask == 144);
			ev.channel_ = (*msg)[0] & 15;
			ev.note_ = (*msg)[1];
			ev.velocity_ = (*msg)[2];
			if(ev.velocity_ == 0)
				ev.on_ = false;
		} else if(mask == 176) {
			ev.cc_ = true;
			ev.channel_ = (*msg)[0] & 15;
			ev.controller_ = (*msg)[1];
			ev.value_ = (*msg)[2];
		} else
			return;
	} else if(nBytes == 1) {
		ev.clock_ = true;
	}
	ev.timestamp_ = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//	std::cerr << "\tmr ts: " << ev.timestamp_ << std::endl;
	MidiReceiver::queue_->push_back(ev);
}

MidiReceiver::MidiReceiver(int32_t inport, bool autostart) : inport_(inport) {
	midiin_->ignoreTypes(true, false, true);
	if(autostart)
		start();
}

MidiReceiver::~MidiReceiver() {
	stop();
}

void MidiReceiver::start() {
	midiin_->setCallback(midiCallback);
	midiin_->openPort(inport_);
}

void MidiReceiver::stop() {
	midiin_->cancelCallback();
	midiin_->closePort();
}

void MidiReceiver::clear() {
	std::unique_lock<std::mutex> lock(*evMtx_);
	queue_->clear();
}


std::vector<MidiEvent> MidiReceiver::receive() {
	std::unique_lock<std::mutex> lock(*evMtx_);
	std::vector<MidiEvent> ret = *queue_;
	queue_->clear();
	return ret;
}

std::ostream& operator<<(std::ostream &out, const MidiEvent& ev) {
	out << ev.on_ << '\t' << ev.channel_ << '\t' << ev.note_ << '\t' << ev.velocity_ ;
	return out;
}

