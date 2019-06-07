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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "math_util.h"
#include "pnm.h"

#include "image_u8x3.h"

// least common multiple of 64 (sandy bridge cache line) and 48 (stride needed
// for 16byte-wide RGB processing). (It's possible that 48 would be enough).
#define DEFAULT_ALIGNMENT_U8X3 192

image_u8x3_t *image_u8x3_create(unsigned int width, unsigned int height)
{
    return image_u8x3_create_alignment(width, height, DEFAULT_ALIGNMENT_U8X3);
}

image_u8x3_t *image_u8x3_create_alignment(unsigned int width, unsigned int height, unsigned int alignment)
{
    int stride = 3*width;

    if ((stride % alignment) != 0)
        stride += alignment - (stride % alignment);

    uint8_t *buf = calloc(height*stride, sizeof(uint8_t));

    // const initializer
    image_u8x3_t tmp = { .width = width, .height = height, .stride = stride, .buf = buf };

    image_u8x3_t *im = calloc(1, sizeof(image_u8x3_t));
    memcpy(im, &tmp, sizeof(image_u8x3_t));
    return im;
}

image_u8x3_t *image_u8x3_copy(const image_u8x3_t *in)
{
    uint8_t *buf = malloc(in->height*in->stride*sizeof(uint8_t));
    memcpy(buf, in->buf, in->height*in->stride*sizeof(uint8_t));

    // const initializer
    image_u8x3_t tmp = { .width = in->width, .height = in->height, .stride = in->stride, .buf = buf };

    image_u8x3_t *copy = calloc(1, sizeof(image_u8x3_t));
    memcpy(copy, &tmp, sizeof(image_u8x3_t));
    return copy;
}

void image_u8x3_destroy(image_u8x3_t *im)
{
    if (!im)
        return;

    free(im->buf);
    free(im);
}

////////////////////////////////////////////////////////////
// PNM file i/o

// Create an RGB image from PNM
image_u8x3_t *image_u8x3_create_from_pnm(const char *path)
{
    pnm_t *pnm = pnm_create_from_file(path);
    if (pnm == NULL)
        return NULL;

    image_u8x3_t *im = NULL;

    switch (pnm->format) {
        case PNM_FORMAT_GRAY: {
            im = image_u8x3_create(pnm->width, pnm->height);

            for (int y = 0; y < im->height; y++) {
                for (int x = 0; x < im->width; x++) {
                    uint8_t gray = pnm->buf[y*im->width + x];
                    im->buf[y*im->stride + x*3 + 0] = gray;
                    im->buf[y*im->stride + x*3 + 1] = gray;
                    im->buf[y*im->stride + x*3 + 2] = gray;
                }
            }

            break;
        }

        case PNM_FORMAT_RGB: {
            im = image_u8x3_create(pnm->width, pnm->height);

            for (int y = 0; y < im->height; y++) {
                for (int x = 0; x < im->width; x++) {
                    uint8_t r = pnm->buf[y*im->width*3 + 3*x];
                    uint8_t g = pnm->buf[y*im->width*3 + 3*x+1];
                    uint8_t b = pnm->buf[y*im->width*3 + 3*x+2];

                    im->buf[y*im->stride + x*3 + 0] = r;
                    im->buf[y*im->stride + x*3 + 1] = g;
                    im->buf[y*im->stride + x*3 + 2] = b;
                }
            }

            break;
        }
    }

    pnm_destroy(pnm);
    return im;
}

int image_u8x3_write_pnm(const image_u8x3_t *im, const char *path)
{
    FILE *f = fopen(path, "wb");
    int res = 0;

    if (f == NULL) {
        res = -1;
        goto finish;
    }

    // Only outputs to RGB
    fprintf(f, "P6\n%d %d\n255\n", im->width, im->height);
    int linesz = im->width * 3;
    for (int y = 0; y < im->height; y++) {
        if (linesz != fwrite(&im->buf[y*im->stride], 1, linesz, f)) {
            res = -1;
            goto finish;
        }
    }

finish:
    if (f != NULL)
        fclose(f);

    return res;
}

// only width 1 supported
void image_u8x3_draw_line(image_u8x3_t *im, float x0, float y0, float x1, float y1, uint8_t rgb[3], int width)
{
    double dist = sqrtf((y1-y0)*(y1-y0) + (x1-x0)*(x1-x0));
    double delta = 0.5 / dist;

    // terrible line drawing code
    for (float f = 0; f <= 1; f += delta) {
        int x = ((int) (x1 + (x0 - x1) * f));
        int y = ((int) (y1 + (y0 - y1) * f));

        if (x < 0 || y < 0 || x >= im->width || y >= im->height)
            continue;

        int idx = y*im->stride + 3*x;
        for (int i = 0; i < 3; i++)
            im->buf[idx + i] = rgb[i];
    }
}

static void convolve(const uint8_t *x, uint8_t *y, int sz, const uint8_t *k, int ksz)
{
    assert((ksz&1)==1);

    for (int i = 0; i < ksz/2 && i < sz; i++)
        y[i] = x[i];

    for (int i = 0; i < sz - ksz; i++) {
        uint32_t acc = 0;

        for (int j = 0; j < ksz; j++)
            acc += k[j]*x[i+j];

        y[ksz/2 + i] = acc >> 8;
    }

    for (int i = sz - ksz + ksz/2; i < sz; i++)
        y[i] = x[i];
}

void image_u8x3_gaussian_blur(image_u8x3_t *im, double sigma, int ksz)
{
    if (sigma == 0)
        return;

    assert((ksz & 1) == 1); // ksz must be odd.

    // build the kernel.
    double *dk = malloc(sizeof(double)*ksz);

    // for kernel of length 5:
    // dk[0] = f(-2), dk[1] = f(-1), dk[2] = f(0), dk[3] = f(1), dk[4] = f(2)
    for (int i = 0; i < ksz; i++) {
        int x = -ksz/2 + i;
        double v = exp(-.5*sq(x / sigma));
        dk[i] = v;
    }

    // normalize
    double acc = 0;
    for (int i = 0; i < ksz; i++)
        acc += dk[i];

    for (int i = 0; i < ksz; i++)
        dk[i] /= acc;

    uint8_t *k = malloc(sizeof(uint8_t)*ksz);
    for (int i = 0; i < ksz; i++)
        k[i] = dk[i]*255;

    if (0) {
        for (int i = 0; i < ksz; i++)
            printf("%d %15f %5d\n", i, dk[i], k[i]);
    }
    free(dk);

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < im->height; y++) {

            uint8_t *in = malloc(sizeof(uint8_t)*im->stride);
            uint8_t *out = malloc(sizeof(uint8_t)*im->stride);

            for (int x = 0; x < im->width; x++)
                in[x] = im->buf[y*im->stride + 3 * x + c];

            convolve(in, out, im->width, k, ksz);
            free(in);

            for (int x = 0; x < im->width; x++)
                im->buf[y*im->stride + 3 * x + c] = out[x];
            free(out);
        }

        for (int x = 0; x < im->width; x++) {
            uint8_t *in = malloc(sizeof(uint8_t)*im->height);
            uint8_t *out = malloc(sizeof(uint8_t)*im->height);

            for (int y = 0; y < im->height; y++)
                in[y] = im->buf[y*im->stride + 3*x + c];

            convolve(in, out, im->height, k, ksz);
            free(in);

            for (int y = 0; y < im->height; y++)
                im->buf[y*im->stride + 3*x + c] = out[y];
            free(out);
        }
    }
    free(k);
}
