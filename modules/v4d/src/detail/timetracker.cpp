/*
 * time_tracker.cpp
 *
 *  Created on: Mar 22, 2014
 *      Author: elchaschab
 */

#include "timetracker.hpp"

TimeTracker* TimeTracker::instance_;

TimeTracker::TimeTracker() : enabled_(false) {
}

TimeTracker::~TimeTracker() {
}
