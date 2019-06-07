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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "time_util.h"
#include "zarray.h"

struct timeprofile_entry
{
    char name[32];
    int64_t utime;
};

typedef struct timeprofile timeprofile_t;
struct timeprofile
{
    int64_t utime;
    zarray_t *stamps;
};

static inline timeprofile_t *timeprofile_create()
{
    timeprofile_t *tp = (timeprofile_t*) calloc(1, sizeof(timeprofile_t));
    tp->stamps = zarray_create(sizeof(struct timeprofile_entry));

    tp->utime = utime_now();

    return tp;
}

static inline void timeprofile_destroy(timeprofile_t *tp)
{
    zarray_destroy(tp->stamps);
    free(tp);
}

static inline void timeprofile_clear(timeprofile_t *tp)
{
    zarray_clear(tp->stamps);
    tp->utime = utime_now();
}

static inline void timeprofile_stamp(timeprofile_t *tp, const char *name)
{
    struct timeprofile_entry tpe;

    strncpy(tpe.name, name, sizeof(tpe.name));
    tpe.name[sizeof(tpe.name)-1] = 0;
    tpe.utime = utime_now();

    zarray_add(tp->stamps, &tpe);
}

static inline void timeprofile_display(timeprofile_t *tp)
{
    int64_t lastutime = tp->utime;

    for (int i = 0; i < zarray_size(tp->stamps); i++) {
        struct timeprofile_entry *stamp;

        zarray_get_volatile(tp->stamps, i, &stamp);

        double cumtime = (stamp->utime - tp->utime)/1000000.0;

        double parttime = (stamp->utime - lastutime)/1000000.0;

        printf("%2d %32s %15f ms %15f ms\n", i, stamp->name, parttime*1000, cumtime*1000);

        lastutime = stamp->utime;
    }
}

static inline uint64_t timeprofile_total_utime(timeprofile_t *tp)
{
    if (zarray_size(tp->stamps) == 0)
        return 0;

    struct timeprofile_entry *first, *last;
    zarray_get_volatile(tp->stamps, 0, &first);
    zarray_get_volatile(tp->stamps, zarray_size(tp->stamps) - 1, &last);

    return last->utime - first->utime;
}

#ifdef __cplusplus
}
#endif
