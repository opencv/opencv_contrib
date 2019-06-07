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

#include <math.h>
#include <stdint.h>

#ifndef M_PI
# define M_PI 3.141592653589793238462643383279502884196
#endif

// 8 bits of fixed-point output
//
// This implementation has a worst-case complexity of 22 multiplies
// and 64 adds. This makes it significantly worse (about 2x) than the
// best-known fast inverse cosine transform methods. HOWEVER, zero
// coefficients can be skipped over, and since that's common (often
// more than half the coefficients are zero).
//
// The output is scaled by a factor of 256 (due to our fixed-point
// integer arithmetic)..
static inline void idct_1D_u32(int32_t *in, int instride, int32_t *out, int outstride)
{
    for (int x = 0; x < 8; x++)
        out[x*outstride] = 0;

    int32_t c;

    c = in[0*instride];
    if (c) {
        // 181  181  181  181  181  181  181  181
        int32_t c181 = c * 181;
        out[0*outstride] += c181;
        out[1*outstride] += c181;
        out[2*outstride] += c181;
        out[3*outstride] += c181;
        out[4*outstride] += c181;
        out[5*outstride] += c181;
        out[6*outstride] += c181;
        out[7*outstride] += c181;
    }

    c = in[1*instride];
    if (c) {
        // 251  212  142   49  -49 -142 -212 -251
        int32_t c251 = c * 251;
        int32_t c212 = c * 212;
        int32_t c142 = c * 142;
        int32_t c49 = c * 49;
        out[0*outstride] += c251;
        out[1*outstride] += c212;
        out[2*outstride] += c142;
        out[3*outstride] += c49;
        out[4*outstride] -= c49;
        out[5*outstride] -= c142;
        out[6*outstride] -= c212;
        out[7*outstride] -= c251;
    }

    c = in[2*instride];
    if (c) {
        // 236   97  -97 -236 -236  -97   97  236
        int32_t c236 = c*236;
        int32_t c97 = c*97;
        out[0*outstride] += c236;
        out[1*outstride] += c97;
        out[2*outstride] -= c97;
        out[3*outstride] -= c236;
        out[4*outstride] -= c236;
        out[5*outstride] -= c97;
        out[6*outstride] += c97;
        out[7*outstride] += c236;
    }

    c = in[3*instride];
    if (c) {
        // 212  -49 -251 -142  142  251   49 -212
        int32_t c212 = c*212;
        int32_t c49 = c*49;
        int32_t c251 = c*251;
        int32_t c142 = c*142;
        out[0*outstride] += c212;
        out[1*outstride] -= c49;
        out[2*outstride] -= c251;
        out[3*outstride] -= c142;
        out[4*outstride] += c142;
        out[5*outstride] += c251;
        out[6*outstride] += c49;
        out[7*outstride] -= c212;
    }

    c = in[4*instride];
    if (c) {
        // 181 -181 -181  181  181 -181 -181  181
        int32_t c181 = c*181;
        out[0*outstride] += c181;
        out[1*outstride] -= c181;
        out[2*outstride] -= c181;
        out[3*outstride] += c181;
        out[4*outstride] += c181;
        out[5*outstride] -= c181;
        out[6*outstride] -= c181;
        out[7*outstride] += c181;
    }

    c = in[5*instride];
    if (c) {
        // 142 -251   49  212 -212  -49  251 -142
        int32_t c142 = c*142;
        int32_t c251 = c*251;
        int32_t c49 = c*49;
        int32_t c212 = c*212;
        out[0*outstride] += c142;
        out[1*outstride] -= c251;
        out[2*outstride] += c49;
        out[3*outstride] += c212;
        out[4*outstride] -= c212;
        out[5*outstride] -= c49;
        out[6*outstride] += c251;
        out[7*outstride] -= c142;
    }

    c = in[6*instride];
    if (c) {
        //  97 -236  236  -97  -97  236 -236   97
        int32_t c97 = c*97;
        int32_t c236 = c*236;
        out[0*outstride] += c97;
        out[1*outstride] -= c236;
        out[2*outstride] += c236;
        out[3*outstride] -= c97;
        out[4*outstride] -= c97;
        out[5*outstride] += c236;
        out[6*outstride] -= c236;
        out[7*outstride] += c97;
    }

    c = in[7*instride];
    if (c) {
        //  49 -142  212 -251  251 -212  142  -49
        int32_t c49 = c*49;
        int32_t c142 = c*142;
        int32_t c212 = c*212;
        int32_t c251 = c*251;
        out[0*outstride] += c49;
        out[1*outstride] -= c142;
        out[2*outstride] += c212;
        out[3*outstride] -= c251;
        out[4*outstride] += c251;
        out[5*outstride] -= c212;
        out[6*outstride] += c142;
        out[7*outstride] -= c49;
    }
}

void pjpeg_idct_2D_u32(int32_t in[64], uint8_t *out, uint32_t outstride)
{
    int32_t tmp[64];

    // idct on rows
    for (int y = 0; y < 8; y++)
        idct_1D_u32(&in[8*y], 1, &tmp[8*y], 1);

    int32_t tmp2[64];

    // idct on columns
    for (int x = 0; x < 8; x++)
        idct_1D_u32(&tmp[x], 8, &tmp2[x], 8);

    // scale, adjust bias, and clamp
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int i = 8*y + x;

            // Shift of 18: the divide by 4 as part of the idct, and a shift by 16
            // to undo the fixed-point arithmetic. (We accumulated 8 bits of
            // fractional precision during each of the row and column IDCTs)
            //
            // Originally:
            //            int32_t v = (tmp2[i] >> 18) + 128;
            //
            // Move the add before the shift and we can do rounding at
            // the same time.
            const int32_t offset = (128 << 18) + (1 << 17);
            int32_t v = (tmp2[i] + offset) >> 18;

            if (v < 0)
                v = 0;
            if (v > 255)
                v = 255;

            out[y*outstride + x] = v;
        }
    }
}

///////////////////////////////////////////////////////
// Below: a "as straight-forward as I can make" implementation.
static inline void idct_1D_double(double *in, int instride, double *out, int outstride)
{
    for (int x = 0; x < 8; x++)
        out[x*outstride] = 0;

    // iterate over IDCT coefficients
    double Cu = 1/sqrt(2);

    for (int u = 0; u < 8; u++, Cu = 1) {

        double coeff = in[u*instride];
        if (coeff == 0)
            continue;

        for (int x = 0; x < 8; x++)
            out[x*outstride] += Cu*cos((2*x+1)*u*M_PI/16) * coeff;
    }
}

void pjpeg_idct_2D_double(int32_t in[64], uint8_t *out, uint32_t outstride)
{
    double din[64], dout[64];
    for (int i = 0; i < 64; i++)
        din[i] = in[i];

    double tmp[64];

    // idct on rows
    for (int y = 0; y < 8; y++)
        idct_1D_double(&din[8*y], 1, &tmp[8*y], 1);

    // idct on columns
    for (int x = 0; x < 8; x++)
        idct_1D_double(&tmp[x], 8, &dout[x], 8);

    // scale, adjust bias, and clamp
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int i = 8*y + x;

            dout[i] = (dout[i] / 4) + 128;
            if (dout[i] < 0)
                dout[i] = 0;
            if (dout[i] > 255)
                dout[i] = 255;

            // XXX round by adding +.5?
            out[y*outstride + x] = dout[i];
        }
    }
}

//////////////////////////////////////////////
static inline unsigned char njClip(const int x) {
    return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}

#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565

static inline void njRowIDCT(int* blk) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[4] << 11)
        | (x2 = blk[6])
        | (x3 = blk[2])
        | (x4 = blk[1])
        | (x5 = blk[7])
        | (x6 = blk[5])
        | (x7 = blk[3])))
    {
        blk[0] = blk[1] = blk[2] = blk[3] = blk[4] = blk[5] = blk[6] = blk[7] = blk[0] << 3;
        return;
    }
    x0 = (blk[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    blk[0] = (x7 + x1) >> 8;
    blk[1] = (x3 + x2) >> 8;
    blk[2] = (x0 + x4) >> 8;
    blk[3] = (x8 + x6) >> 8;
    blk[4] = (x8 - x6) >> 8;
    blk[5] = (x0 - x4) >> 8;
    blk[6] = (x3 - x2) >> 8;
    blk[7] = (x7 - x1) >> 8;
}

static inline void njColIDCT(const int* blk, unsigned char *out, int stride) {
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;
    if (!((x1 = blk[8*4] << 8)
        | (x2 = blk[8*6])
        | (x3 = blk[8*2])
        | (x4 = blk[8*1])
        | (x5 = blk[8*7])
        | (x6 = blk[8*5])
        | (x7 = blk[8*3])))
    {
        x1 = njClip(((blk[0] + 32) >> 6) + 128);
        for (x0 = 8;  x0;  --x0) {
            *out = (unsigned char) x1;
            out += stride;
        }
        return;
    }
    x0 = (blk[0] << 8) + 8192;
    x8 = W7 * (x4 + x5) + 4;
    x4 = (x8 + (W1 - W7) * x4) >> 3;
    x5 = (x8 - (W1 + W7) * x5) >> 3;
    x8 = W3 * (x6 + x7) + 4;
    x6 = (x8 - (W3 - W5) * x6) >> 3;
    x7 = (x8 - (W3 + W5) * x7) >> 3;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2) + 4;
    x2 = (x1 - (W2 + W6) * x2) >> 3;
    x3 = (x1 + (W2 - W6) * x3) >> 3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    *out = njClip(((x7 + x1) >> 14) + 128);  out += stride;
    *out = njClip(((x3 + x2) >> 14) + 128);  out += stride;
    *out = njClip(((x0 + x4) >> 14) + 128);  out += stride;
    *out = njClip(((x8 + x6) >> 14) + 128);  out += stride;
    *out = njClip(((x8 - x6) >> 14) + 128);  out += stride;
    *out = njClip(((x0 - x4) >> 14) + 128);  out += stride;
    *out = njClip(((x3 - x2) >> 14) + 128);  out += stride;
    *out = njClip(((x7 - x1) >> 14) + 128);
}

void pjpeg_idct_2D_nanojpeg(int32_t in[64], uint8_t *out, uint32_t outstride)
{
    int coef;

    for (coef = 0;  coef < 64;  coef += 8)
        njRowIDCT(&in[coef]);
    for (coef = 0;  coef < 8;  ++coef)
        njColIDCT(&in[coef], &out[coef], outstride);
}
