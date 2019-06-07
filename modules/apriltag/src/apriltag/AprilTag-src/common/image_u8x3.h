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

#include <stdint.h>
#include "image_types.h"

#ifdef __cplusplus
extern "C" {
#endif


/////////////////////////////////////
// IMPORTANT NOTE ON BYTE ORDER
//
// Format conversion routines will (unless otherwise specified) assume
// R, G, B, ordering of bytes. This is consistent with GTK, PNM, etc.
//
/////////////////////////////////////

// Create or load an image. returns NULL on failure
image_u8x3_t *image_u8x3_create(unsigned int width, unsigned int height);
image_u8x3_t *image_u8x3_create_alignment(unsigned int width, unsigned int height, unsigned int alignment);
image_u8x3_t *image_u8x3_create_from_pnm(const char *path);

image_u8x3_t *image_u8x3_copy(const image_u8x3_t *in);

void image_u8x3_gaussian_blur(image_u8x3_t *im, double sigma, int ksz);

void image_u8x3_destroy(image_u8x3_t *im);

int image_u8x3_write_pnm(const image_u8x3_t *im, const char *path);

// only width 1 supported
void image_u8x3_draw_line(image_u8x3_t *im, float x0, float y0, float x1, float y1, uint8_t rgb[3], int width);

#ifdef __cplusplus
}
#endif
