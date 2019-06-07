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

#include "image_u8.h"
#include "image_u8x3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pjpeg_component pjpeg_component_t;
struct pjpeg_component
{
    // resolution of this component (which is smaller than the
    // dimensions of the image if the channel has been sub-sampled.)
    uint32_t width, height;

    // number of bytes per row. May be larger than width for alignment
    // reasons.
    uint32_t stride;

    // data[y*stride + x]
    uint8_t *data;

    ////////////////////////////////////////////////////////////////
    // These items probably not of great interest to most
    // applications.
    uint8_t id; // the identifier associated with this component
    uint8_t hv; // horiz scale (high 4 bits) / vert scale (low 4 bits)
    uint8_t scalex, scaley; // derived from hv above
    uint8_t tq; // quantization table index

    // this filled in at the last moment by SOS
    uint8_t tda; // which huff tables will we use for DC (high 4 bits) and AC (low 4 bits)
};

typedef struct pjpeg pjpeg_t;
struct pjpeg
{
    // status of the decode is put here. Non-zero means error.
    int error;

    uint32_t width, height; // pixel dimensions

    int ncomponents;
    pjpeg_component_t *components;
};

enum PJPEG_FLAGS {
    PJPEG_STRICT = 1,  // Don't try to recover from errors.
    PJPEG_MJPEG = 2,   // Support JPGs with missing DHT segments.
};

enum PJPEG_ERROR {
    PJPEG_OKAY = 0,
    PJPEG_ERR_FILE, // something wrong reading file
    PJPEG_ERR_DQT, // something wrong with DQT marker
    PJPEG_ERR_SOF, // something wrong with SOF marker
    PJPEG_ERR_DHT, // something wrong with DHT marker
    PJPEG_ERR_SOS, // something wrong with SOS marker
    PJPEG_ERR_MISSING_DHT, // missing a necessary huffman table
    PJPEG_ERR_DRI, // something wrong with DRI marker
    PJPEG_ERR_RESET, // didn't get a reset marker where we expected. Corruption?
    PJPEG_ERR_EOF, // ran out of bytes while decoding
    PJEPG_ERR_UNSUPPORTED, // an unsupported format
};

pjpeg_t *pjpeg_create_from_file(const char *path, uint32_t flags, int *error);
pjpeg_t *pjpeg_create_from_buffer(uint8_t *buf, int buflen, uint32_t flags, int *error);
void pjpeg_destroy(pjpeg_t *pj);

image_u8_t *pjpeg_to_u8_baseline(pjpeg_t *pj);
image_u8x3_t *pjpeg_to_u8x3_baseline(pjpeg_t *pj);

#ifdef __cplusplus
}
#endif
