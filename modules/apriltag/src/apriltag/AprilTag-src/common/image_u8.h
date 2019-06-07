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

typedef struct image_u8_lut image_u8_lut_t;
struct image_u8_lut
{
    // When drawing, we compute the squared distance between a given pixel and a filled region.
    // int idx = squared_distance * scale;
    // We then index into values[idx] to obtain the color. (If we must index beyond nvalues,
    // no drawing is performed.)
    float    scale;

    int      nvalues;
    uint8_t *values;
};


// Create or load an image. returns NULL on failure. Uses default
// stride alignment.
image_u8_t *image_u8_create_stride(unsigned int width, unsigned int height, unsigned int stride);
image_u8_t *image_u8_create(unsigned int width, unsigned int height);
image_u8_t *image_u8_create_alignment(unsigned int width, unsigned int height, unsigned int alignment);
image_u8_t *image_u8_create_from_f32(image_f32_t *fim);

image_u8_t *image_u8_create_from_pnm(const char *path);
    image_u8_t *image_u8_create_from_pnm_alignment(const char *path, int alignment);

image_u8_t *image_u8_copy(const image_u8_t *in);
void image_u8_draw_line(image_u8_t *im, float x0, float y0, float x1, float y1, int v, int width);
void image_u8_draw_circle(image_u8_t *im, float x0, float y0, float r, int v);
void image_u8_draw_annulus(image_u8_t *im, float x0, float y0, float r0, float r1, int v);

void image_u8_fill_line_max(image_u8_t *im, const image_u8_lut_t *lut, const float *xy0, const float *xy1);

void image_u8_clear(image_u8_t *im);
void image_u8_darken(image_u8_t *im);
void image_u8_convolve_2D(image_u8_t *im, const uint8_t *k, int ksz);
void image_u8_gaussian_blur(image_u8_t *im, double sigma, int k);

// 1.5, 2, 3, 4, ... supported
image_u8_t *image_u8_decimate(image_u8_t *im, float factor);

void image_u8_destroy(image_u8_t *im);

// Write a pnm. Returns 0 on success
// Currently only supports GRAY and RGBA. Does not write out alpha for RGBA
int image_u8_write_pnm(const image_u8_t *im, const char *path);

// rotate the image by 'rad' radians. (Rotated in the "intuitive
// sense", i.e., if Y were up. When input values are unavailable, the
// value 'pad' is inserted instead. The geometric center of the output
// image corresponds to the geometric center of the input image.
image_u8_t *image_u8_rotate(const image_u8_t *in, double rad, uint8_t pad);

#ifdef __cplusplus
}
#endif
