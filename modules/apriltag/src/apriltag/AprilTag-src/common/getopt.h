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

#include "zarray.h"
#include "string_util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct getopt getopt_t;

getopt_t *getopt_create();
void getopt_destroy(getopt_t *gopt);

// Parse args. Returns 1 on success
int getopt_parse(getopt_t *gopt, int argc, char *argv[], int showErrors);
void getopt_do_usage(getopt_t *gopt);

// Returns a string containing the usage. Must be freed by caller
char * getopt_get_usage(getopt_t *gopt);

void getopt_add_spacer(getopt_t *gopt, const char *s);
void getopt_add_bool(getopt_t *gopt, char sopt, const char *lname, int def, const char *help);
void getopt_add_int(getopt_t *gopt, char sopt, const char *lname, const char *def, const char *help);
void getopt_add_string(getopt_t *gopt, char sopt, const char *lname, const char *def, const char *help);
void getopt_add_double(getopt_t *gopt, char sopt, const char *lname, const char *def, const char *help);

const char *getopt_get_string(getopt_t *gopt, const char *lname);
int getopt_get_int(getopt_t *getopt, const char *lname);
int getopt_get_bool(getopt_t *getopt, const char *lname);
double getopt_get_double(getopt_t *getopt, const char *lname);
int getopt_was_specified(getopt_t *gopt, const char *lname);
const zarray_t *getopt_get_extra_args(getopt_t *gopt);

#ifdef __cplusplus
}
#endif
