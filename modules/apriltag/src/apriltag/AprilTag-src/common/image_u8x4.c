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

#include "pam.h"
#include "pnm.h"
#include "image_u8x4.h"

// least common multiple of 64 (sandy bridge cache line) and 64 (stride needed
// for 16byte-wide RGBA processing).
#define DEFAULT_ALIGNMENT_U8X4 64

image_u8x4_t *image_u8x4_create(unsigned int width, unsigned int height)
{
    return image_u8x4_create_alignment(width, height, DEFAULT_ALIGNMENT_U8X4);
}

image_u8x4_t *image_u8x4_create_alignment(unsigned int width, unsigned int height, unsigned int alignment)
{
    int stride = 4*width;

    if ((stride % alignment) != 0)
        stride += alignment - (stride % alignment);

    uint8_t *buf = calloc(height*stride, sizeof(uint8_t));

    // const initializer
    image_u8x4_t tmp = { .width = width, .height = height, .stride = stride, .buf = buf };

    image_u8x4_t *im = calloc(1, sizeof(image_u8x4_t));
    memcpy(im, &tmp, sizeof(image_u8x4_t));
    return im;
}

image_u8x4_t *image_u8x4_copy(const image_u8x4_t *in)
{
    uint8_t *buf = malloc(in->height*in->stride*sizeof(uint8_t));
    memcpy(buf, in->buf, in->height*in->stride*sizeof(uint8_t));

    // const initializer
    image_u8x4_t tmp = { .width = in->width, .height = in->height, .stride = in->stride, .buf = buf };

    image_u8x4_t *copy = calloc(1, sizeof(image_u8x4_t));
    memcpy(copy, &tmp, sizeof(image_u8x4_t));
    return copy;
}

void image_u8x4_destroy(image_u8x4_t *im)
{
    if (!im)
        return;

    free(im->buf);
    free(im);
}

////////////////////////////////////////////////////////////
image_u8x4_t *image_u8x4_create_from_pam(const char *inpath)
{
    pam_t *pam = pam_create_from_file(inpath);
    if (!pam)
        return NULL;

    image_u8x4_t *im = image_u8x4_create(pam->width, pam->height);

    for (int y = 0; y < pam->height; y++) {
        if (pam->depth == 1) {
            for (int x = 0; x < pam->width; x++) {
                im->buf[y*im->stride + 4*x + 0] = pam->data[pam->width*y + x + 0];
                im->buf[y*im->stride + 4*x + 1] = pam->data[pam->width*y + x + 0];
                im->buf[y*im->stride + 4*x + 2] = pam->data[pam->width*y + x + 0];
                im->buf[y*im->stride + 4*x + 3] = 255;
            }
        } else if (pam->depth == 3) {
            for (int x = 0; x < pam->width; x++) {
                im->buf[y*im->stride + 4*x + 0] = pam->data[3*pam->width*y + 3*x + 0];
                im->buf[y*im->stride + 4*x + 1] = pam->data[3*pam->width*y + 3*x + 1];
                im->buf[y*im->stride + 4*x + 2] = pam->data[3*pam->width*y + 3*x + 2];
                im->buf[y*im->stride + 4*x + 3] = 255;
            }
        } else if (pam->depth == 4) {
            memcpy(&im->buf[y*im->stride], &pam->data[4*pam->width*y], 4*pam->width);
        } else {
            assert(0); // not implemented
        }
    }

    pam_destroy(pam);
    return im;
}
////////////////////////////////////////////////////////////
// PNM file i/o

// Create an RGBA image from PNM
image_u8x4_t *image_u8x4_create_from_pnm(const char *path)
{
    pnm_t *pnmp = pnm_create_from_file(path);
    if (pnmp == NULL)
        return NULL;

    pnm_t pnm = *pnmp;
    image_u8x4_t *imp = NULL;

    switch (pnm.format) {
        case PNM_FORMAT_GRAY: {
            imp = image_u8x4_create(pnm.width, pnm.height);

            // copy struct by value for common subexpression elimination
            const image_u8x4_t im = *imp;

            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {
                    uint8_t gray = pnm.buf[y*pnm.width + x];
                    im.buf[y*im.stride + 4*x + 0] = gray;
                    im.buf[y*im.stride + 4*x + 1] = gray;
                    im.buf[y*im.stride + 4*x + 2] = gray;
                    im.buf[y*im.stride + 4*x + 3] = 0xff;
                }
            }

            break;
        }

        case PNM_FORMAT_RGB: {
            imp = image_u8x4_create(pnm.width, pnm.height);

            // copy struct by value for common subexpression elimination
            const image_u8x4_t im = *imp;

            // Gray conversion for RGB is gray = (r + g + g + b)/4
            for (int y = 0; y < im.height; y++) {
                for (int x = 0; x < im.width; x++) {

                    uint8_t r = pnm.buf[y*pnm.width*3 + 3*x + 0];
                    uint8_t g = pnm.buf[y*pnm.width*3 + 3*x + 1];
                    uint8_t b = pnm.buf[y*pnm.width*3 + 3*x + 2];

                    im.buf[y*im.stride + 4*x + 0] = r;
                    im.buf[y*im.stride + 4*x + 1] = g;
                    im.buf[y*im.stride + 4*x + 2] = b;
                    im.buf[y*im.stride + 4*x + 3] = 0xff;
                }
            }

            break;
        }
    }

    pnm_destroy(pnmp);
    return imp;
}

int image_u8x4_write_pnm(const image_u8x4_t *imp, const char *path)
{
    // copy struct by value to ensure common subexpression elimination occurs
    const image_u8x4_t im = *imp;

    FILE *f = fopen(path, "wb");
    int res = 0;

    if (f == NULL) {
        res = -1;
        goto finish;
    }

    // Only outputs to RGB
    fprintf(f, "P6\n%d %d\n255\n", im.width, im.height);

    for (int y = im.height-1; y >= 0; y--) {
        for (int x = 0; x < im.width; x++) {

            uint8_t r = im.buf[y*im.stride + 4*x + 0];
            uint8_t g = im.buf[y*im.stride + 4*x + 1];
            uint8_t b = im.buf[y*im.stride + 4*x + 2];

            fwrite(&r, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&b, 1, 1, f);
        }
    }

  finish:
    if (f != NULL)
        fclose(f);

    return res;
}

void image_u8x4_write_pam(const image_u8x4_t *im, const char *path)
{
    FILE *f = fopen(path, "w");
    fprintf(f, "P7\n");
    fprintf(f, "WIDTH %d\n", im->width);
    fprintf(f, "HEIGHT %d\n", im->height);
    fprintf(f, "DEPTH 4\n");
    fprintf(f, "MAXVAL 255\n");
    fprintf(f, "TUPLTYPE RGB_ALPHA\n");
    fprintf(f, "ENDHDR\n");

    for (int y = 0; y < im->height; y++)
        fwrite(&im->buf[y*im->stride], 1, 4*im->width, f);

    fclose(f);

}
