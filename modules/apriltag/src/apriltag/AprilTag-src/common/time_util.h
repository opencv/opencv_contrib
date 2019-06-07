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

#pragma once

#include <stdbool.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
typedef long long suseconds_t;
#endif
#ifdef _MSC_VER

inline int gettimeofday(struct timeval* tp, void* tzp)
{
  unsigned long t;
  t = timeGetTime();
  tp->tv_sec = t / 1000;
  tp->tv_usec = t % 1000;
  return 0;
}
#else
#include <sys/time.h>
#include <unistd.h>
#endif
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct timeutil_rest timeutil_rest_t;
timeutil_rest_t *timeutil_rest_create();
void timeutil_rest_destroy(timeutil_rest_t * rest);

int64_t utime_now(); // blacklist-ignore
int64_t utime_get_seconds(int64_t v);
int64_t utime_get_useconds(int64_t v);
void    utime_to_timeval(int64_t v, struct timeval *tv);
void    utime_to_timespec(int64_t v, struct timespec *ts);

int32_t  timeutil_usleep(int64_t useconds);
uint32_t timeutil_sleep(unsigned int seconds);
int32_t  timeutil_sleep_hz(timeutil_rest_t *rest, double hz);

void timeutil_timer_reset(timeutil_rest_t *rest);
void timeutil_timer_start(timeutil_rest_t *rest);
void timeutil_timer_stop(timeutil_rest_t *rest);
bool timeutil_timer_timeout(timeutil_rest_t *rest, double timeout_s);

int64_t time_util_hhmmss_ss_to_utime(double time);

int64_t timeutil_ms_to_us(int32_t ms);

#ifdef __cplusplus
}
#endif
