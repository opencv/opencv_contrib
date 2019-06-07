/* Copyright (C) 2013-2016, The Regents of The University of Michigan.
All rights reserved.
This software was developed in the APRIL Robotics Lab under the
direction of Edwin Olson, ebolson@umich.edu. This software may be
available under alternative licensing terms; contact the address above.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the Regents of The University of Michigan.
*/

#include <stdlib.h>
#include <math.h>
#include "time_util.h"

struct timeutil_rest
{
    int64_t acc_time;
    int64_t start_time;
};

timeutil_rest_t *timeutil_rest_create()
{
    timeutil_rest_t *rest = calloc(1, sizeof(timeutil_rest_t));
    return rest;
}

void timeutil_rest_destroy(timeutil_rest_t *rest)
{
    free(rest);
}

int64_t utime_now() // blacklist-ignore
{
    struct timeval tv;
    gettimeofday (&tv, NULL); // blacklist-ignore
    return (int64_t) tv.tv_sec * 1000000 + tv.tv_usec;
}

int64_t utime_get_seconds(int64_t v)
{
    return v/1000000;
}

int64_t utime_get_useconds(int64_t v)
{
    return v%1000000;
}

void utime_to_timeval(int64_t v, struct timeval *tv)
{
    tv->tv_sec  = (time_t) utime_get_seconds(v);
    tv->tv_usec = (suseconds_t) utime_get_useconds(v);
}

void utime_to_timespec(int64_t v, struct timespec *ts)
{
    ts->tv_sec  = (time_t) utime_get_seconds(v);
    ts->tv_nsec = (suseconds_t) utime_get_useconds(v)*1000;
}

int32_t timeutil_usleep(int64_t useconds)
{
#ifdef _WIN32
    Sleep(useconds/1000);
    return 0;
#else
    // unistd.h function, but usleep is obsoleted in POSIX.1-2008.
    // TODO: Eventually, rewrite this to use nanosleep
    return usleep(useconds);
#endif
}

uint32_t timeutil_sleep(unsigned int seconds)
{
#ifdef _WIN32
    Sleep(seconds*1000);
    return 0;
#else
    // unistd.h function
    return sleep(seconds);
#endif
}

int32_t timeutil_sleep_hz(timeutil_rest_t *rest, double hz)
{
    int64_t max_delay = 1000000L/hz;
    int64_t curr_time = utime_now();
    int64_t diff = curr_time - rest->start_time;
    int64_t delay = max_delay - diff;
    if (delay < 0) delay = 0;

    int32_t ret = timeutil_usleep(delay);
    rest->start_time = utime_now();

    return ret;
}

void timeutil_timer_reset(timeutil_rest_t *rest)
{
    rest->start_time = utime_now();
    rest->acc_time = 0;
}

void timeutil_timer_start(timeutil_rest_t *rest)
{
    rest->start_time = utime_now();
}

void timeutil_timer_stop(timeutil_rest_t *rest)
{
    int64_t curr_time = utime_now();
    int64_t diff = curr_time - rest->start_time;

    rest->acc_time += diff;
}

bool timeutil_timer_timeout(timeutil_rest_t *rest, double timeout_s)
{
    int64_t timeout_us = (int64_t)(1000000L*timeout_s);
    return rest->acc_time > timeout_us;
}

int64_t time_util_hhmmss_ss_to_utime(double time)
{
    int64_t utime = 0;

    int itime = ((int) time);

    double seconds = fmod(time, 100.0);
    uint8_t minutes = (itime % 10000) / 100;
    uint8_t hours =  itime / 10000;

    utime += seconds *   100;
    utime += minutes *  6000;
    utime += hours   *360000;

    utime *= 10000;

    return utime;
}

int64_t timeutil_ms_to_us(int32_t ms)
{
    return ((int64_t) ms) * 1000;
}
