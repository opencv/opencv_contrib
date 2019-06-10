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

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "pjpeg.h"

#include "image_u8.h"
#include "image_u8x3.h"

// https://www.w3.org/Graphics/JPEG/itu-t81.pdf

void pjpeg_idct_2D_double(int32_t in[64], uint8_t *out, uint32_t outstride);
void pjpeg_idct_2D_u32(int32_t in[64], uint8_t *out, uint32_t outstride);
void pjpeg_idct_2D_nanojpeg(int32_t in[64], uint8_t *out, uint32_t outstride);

struct pjpeg_huffman_code
{
    uint8_t nbits;  // how many bits should we actually consume?
    uint8_t code;   // what is the symbol that was encoded? (not actually a DCT coefficient; see encoding)
};

struct pjpeg_decode_state
{
    int error;

    uint32_t width, height;
    uint8_t *in;
    uint32_t inlen;

    uint32_t flags;

    // to decode, we load the next 16 bits of input (generally more
    // than we need). We then look up in our code book how many bits
    // we have actually consumed. For example, if there was a code
    // whose bit sequence was "0", the first 32768 entries would all
    // be copies of {.bits=1, .value=XX}; no matter what the following
    // 15 bits are, we would get the correct decode.
    //
    // Can be up to 8 tables; computed as (ACDC * 2 + htidx)
    struct pjpeg_huffman_code huff_codes[4][65536];
    int huff_codes_present[4];

    uint8_t  qtab[4][64];

    int ncomponents;
    pjpeg_component_t *components;

    int reset_interval;
    int reset_count;
    int reset_next; // What reset marker do we expect next? (add 0xd0)

    int debug;
};

// from K.3.3.1 (page 158)
static uint8_t mjpeg_dht[] = { // header
    0xFF,0xC4,0x01,0xA2,

    /////////////////////////////////////////////////////////////
    // luminance dc coefficients.
    // DC table 0
    0x00,
    // code lengths
    0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    // values
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,

    /////////////////////////////////////////////////////////////
    // chrominance DC coefficents
    // DC table 1
    0x01,
    // code lengths
    0x00,0x03,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,
    // values
    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,

    /////////////////////////////////////////////////////////////
    // luminance AC coefficients
    // AC table 0
    0x10,
    // code lengths
    0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
    // codes
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,
    0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,0x24,
    0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,0x29,0x2A,0x34,
    0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,
    0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,
    0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,
    0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,
    0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,
    0xDA,0xE1,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,
    0xF8,0xF9,0xFA,

    /////////////////////////////////////////////////////////////
    // chrominance DC coefficients
    // DC table 1
    0x11,
    // code lengths
    0x00,0x02,0x01,0x02,0x04,0x04,0x03,0x04,0x07,0x05,0x04,0x04,0x00,0x01,0x02,0x77,
    // values
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,0x15,0x62,
    0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26,0x27,0x28,0x29,0x2A,
    0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4A,0x53,0x54,0x55,0x56,
    0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,
    0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,
    0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,
    0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,
    0xD9,0xDA,0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
    0xF9,0xFA
};

static inline uint8_t max_u8(uint8_t a, uint8_t b)
{
    return a > b ? a : b;
}

// order of coefficients in each DC block
static const char ZZ[64] = { 0,   1,  8, 16,  9,  2,  3, 10,
                             17, 24, 32, 25, 18, 11,  4,  5,
                             12, 19, 26, 33, 40, 48, 41, 34,
                             27, 20, 13,  6,  7, 14, 21, 28,
                             35, 42, 49, 56, 57, 50, 43, 36,
                             29, 22, 15, 23, 30, 37, 44, 51,
                             58, 59, 52, 45, 38, 31, 39, 46,
                             53, 60, 61, 54, 47, 55, 62, 63 };



struct bit_decoder
{
    uint8_t *in;
    uint32_t inpos;
    uint32_t inlen;

    uint32_t bits; // the low order bits contain the next nbits_avail bits.

    int nbits_avail; // how many bits in 'bits' (left aligned) are valid?

    int error;
};

// ensure that at least 'nbits' of data is available in the bit decoder.
static inline void bd_ensure(struct bit_decoder *bd, int nbits)
{
    while (bd->nbits_avail < nbits) {

        if (bd->inpos >= bd->inlen) {
            printf("hallucinating 1s!\n");
            // we hit end of stream hallucinate an infinite stream of 1s
            bd->bits = (bd->bits << 8) | 0xff;
            bd->nbits_avail += 8;
            continue;
        }

        uint8_t nextbyte = bd->in[bd->inpos];
        bd->inpos++;

        if (nextbyte == 0xff && bd->inpos < bd->inlen && bd->in[bd->inpos] == 0x00) {
            // a stuffed byte
            nextbyte = 0xff;
            bd->inpos++;
        }

        // it's an ordinary byte
        bd->bits = (bd->bits << 8) | nextbyte;
        bd->nbits_avail += 8;
    }
}

static inline uint32_t bd_peek_bits(struct bit_decoder *bd, int nbits)
{
    bd_ensure(bd, nbits);

    return (bd->bits >> (bd->nbits_avail - nbits)) & ((1 << nbits) - 1);
}

static inline uint32_t bd_consume_bits(struct bit_decoder *bd, int nbits)
{
    assert(nbits < 32);

    bd_ensure(bd, nbits);

    uint32_t v = (bd->bits >> (bd->nbits_avail - nbits)) & ((1 << nbits) - 1);

    bd->nbits_avail -= nbits;

    return v;
}

// discard without regard for byte stuffing!
static inline void bd_discard_bytes(struct bit_decoder *bd, int nbytes)
{
    assert(bd->nbits_avail == 0);
    bd->inpos += nbytes;
}

static inline int bd_has_more(struct bit_decoder *bd)
{
    return bd->nbits_avail > 0 || bd->inpos < bd->inlen;
}

// throw away up to 7 bits of data so that the next data returned
// began on a byte boundary.
static inline void bd_discard_to_byte_boundary(struct bit_decoder *bd)
{
    bd->nbits_avail -= (bd->nbits_avail & 7);
}

static inline uint32_t bd_get_offset(struct bit_decoder *bd)
{
    return bd->inpos - bd->nbits_avail / 8;
}

static int pjpeg_decode_buffer(struct pjpeg_decode_state *pjd)
{
    // XXX TODO Include sanity check that this is actually a JPG

    struct bit_decoder bd;
    memset(&bd, 0, sizeof(struct bit_decoder));
    bd.in = pjd->in;
    bd.inpos = 0;
    bd.inlen = pjd->inlen;

    int marker_sync_skipped = 0;
    int marker_sync_skipped_from_offset = 0;

    while (bd_has_more(&bd)) {

        uint32_t marker_offset = bd_get_offset(&bd);

        // Look for the 0xff that signifies the beginning of a marker
        bd_discard_to_byte_boundary(&bd);

        while (bd_consume_bits(&bd, 8) != 0xff) {
            if (marker_sync_skipped == 0)
                marker_sync_skipped_from_offset = marker_offset;
            marker_sync_skipped++;
            continue;
        }

        if (marker_sync_skipped) {
            printf("%08x: skipped %04x bytes\n", marker_sync_skipped_from_offset, marker_sync_skipped);
            marker_sync_skipped = 0;
        }

        uint8_t marker = bd_consume_bits(&bd, 8);

//        printf("marker %08x : %02x\n", marker_offset, marker);

        switch (marker) {

            case 0xd8:  // start of image. Great, continue.
                continue;

                // below are the markers that A) we don't care about
                // that B) encode length as two bytes.
                //
                // Note: Other unknown fields should not be added since
                // we should be able to skip over them by looking for
                // the next marker byte.
            case 0xe0: // JFIF header.
            case 0xe1: // EXIF header (Yuck: Payload may contain 0xff 0xff!)
            case 0xe2: // ICC Profile. (Yuck: payload may contain 0xff 0xff!)
            case 0xe6: // some other common header
            case 0xfe: // Comment
            {
                uint16_t length = bd_consume_bits(&bd, 16);
                bd_discard_bytes(&bd, length - 2);
                continue;
            }

            case 0xdb: { // DQT Define Quantization Table
                uint16_t length = bd_consume_bits(&bd, 16);

                if (((length-2) % 65) != 0)
                    return PJPEG_ERR_DQT;

                // can contain multiple DQTs
                for (int offset = 0; offset < length - 2; offset += 65) {

                    // pq: quant table element precision. 0=8bit, 1=16bit.
                    // tq: quant table destination id.
                    uint8_t  pqtq = bd_consume_bits(&bd, 8);

                    if ((pqtq & 0xf0) != 0 || (pqtq & 0x0f) >= 4)
                        return PJPEG_ERR_DQT;

                    uint8_t id = pqtq & 3;

                    for (int i = 0; i < 64; i++)
                        pjd->qtab[id][i] = bd_consume_bits(&bd, 8);
                }

                break;
            }

            case 0xc0: { // SOF, non-differential, huffman, baseline
                uint16_t length = bd_consume_bits(&bd, 16);
                (void) length;

                uint8_t p = bd_consume_bits(&bd, 8); // precision
                if (p != 8)
                    return PJPEG_ERR_SOF;

                pjd->height = bd_consume_bits(&bd, 16);
                pjd->width = bd_consume_bits(&bd, 16);

//                printf("%d x %d\n", pjd->height, pjd->width);

                int nf = bd_consume_bits(&bd, 8); // # image components

                if (nf < 1 || nf > 3)
                    return PJPEG_ERR_SOF;

                pjd->ncomponents = nf;
                pjd->components = calloc(nf, sizeof(struct pjpeg_component));

                for (int i = 0; i < nf; i++) {
                    // comp. identifier
                    pjd->components[i].id = bd_consume_bits(&bd, 8);

                    // horiz/vert sampling
                    pjd->components[i].hv = bd_consume_bits(&bd, 8);
                    pjd->components[i].scaley = pjd->components[i].hv & 0x0f;
                    pjd->components[i].scalex = pjd->components[i].hv >> 4;

                    // which quant table?
                    pjd->components[i].tq = bd_consume_bits(&bd, 8);
                }
                break;
            }

            case 0xc1: // SOF, non-differential, huffman,    extended DCT
            case 0xc2: // SOF, non-differential, huffman,    progressive DCT
            case 0xc3: // SOF, non-differential, huffman,    lossless
            case 0xc5: // SOF, differential,     huffman,    baseline DCT
            case 0xc6: // SOF, differential,     huffman,    progressive
            case 0xc7: // SOF, differential,     huffman,    lossless
            case 0xc8: // reserved
            case 0xc9: // SOF, non-differential, arithmetic, extended
            case 0xca: // SOF, non-differential, arithmetic, progressive
            case 0xcb: // SOF, non-differential, arithmetic, lossless
            case 0xcd: // SOF, differential,     arithmetic, sequential
            case 0xce: // SOF, differential,     arithmetic, progressive
            case 0xcf: // SOF, differential,     arithmetic, lossless
            {
                printf("pjepg.c: unsupported JPEG type %02x\n", marker);
                return PJEPG_ERR_UNSUPPORTED;
            }

            case 0xc4: { // DHT Define Huffman Tables
                // [ED: the encoding of these tables is really quite
                // clever!]
                uint16_t length = bd_consume_bits(&bd, 16);
                length = length - 2;

                while (length > 0) {
                    uint8_t TcTh = bd_consume_bits(&bd, 8);
                    length--;
                    uint8_t Tc = (TcTh >> 4);
                    int Th = TcTh & 0x0f; // which index are we using?

                    if (Tc >= 2 || Th >= 2)
                        // Tc must be either AC=1 or DC=0.
                        // Th must be less than 2
                        return PJPEG_ERR_DHT;

                    int htidx = Tc*2 + Th;

                    uint8_t L[17]; // how many symbols of each bit length?
                    L[0] = 0;      // no 0 bit codes :)
                    for (int nbits = 1; nbits <= 16; nbits++) {
                        L[nbits] = bd_consume_bits(&bd, 8);
                        length -= L[nbits];
                    }
                    length -= 16;

                    uint32_t code_pos = 0;

                    for (int nbits = 1; nbits <= 16; nbits++) {
                        int nvalues = L[nbits];

                        // how many entries will we fill?
                        // (a 1 bit code will fill 32768, a 2 bit code 16384, ...)
                        uint32_t ncodes = (1 << (16 - nbits));

                        // consume the values...
                        for (int vi = 0; vi < nvalues; vi++) {
                            uint8_t code = bd_consume_bits(&bd, 8);

                            if (code_pos + ncodes > 0xffff)
                                return PJPEG_ERR_DHT;

                            for (int ci = 0; ci < ncodes; ci++) {
                                pjd->huff_codes[htidx][code_pos].nbits = nbits;
                                pjd->huff_codes[htidx][code_pos].code = code;
                                code_pos++;
                            }
                        }
                    }
                    pjd->huff_codes_present[htidx] = 1;
                }
                break;
            }

                // a sequentially-encoded JPG has one SOS segment. A
                // progressive JPG will have multiple SOS segments.
            case 0xda: { // Start Of Scan (SOS)

                // Note that this marker frame (and its encoded
                // length) does NOT include the bitstream that
                // follows.

                uint16_t length = bd_consume_bits(&bd, 16);
                (void) length;

                // number of components in this scan
                uint8_t ns = bd_consume_bits(&bd, 8);

                // for each component, what is the index into our pjd->components[] array?
                uint8_t *comp_idx = calloc(ns, sizeof(uint8_t));

                for (int i = 0; i < ns; i++) {
                    // component name
                    uint8_t cs = bd_consume_bits(&bd, 8);

                    int found = 0;
                    for (int j = 0; j < pjd->ncomponents; j++) {

                        if (cs == pjd->components[j].id) {
                            // which huff tables will we use for
                            // DC (high 4 bits) and AC (low 4 bits)
                            pjd->components[j].tda = bd_consume_bits(&bd, 8);
                            comp_idx[i] = j;
                            found = 1;
                            break;
                        }
                    }

                    if (!found)
                        return PJPEG_ERR_SOS;
                }

                // start of spectral selection. baseline == 0
                uint8_t ss = bd_consume_bits(&bd, 8);

                // end of spectral selection. baseline == 0x3f
                uint8_t se = bd_consume_bits(&bd, 8);

                // successive approximation bits. baseline == 0
                uint8_t Ahl = bd_consume_bits(&bd, 8);

                if (ss != 0 || se != 0x3f || Ahl != 0x00)
                    return PJPEG_ERR_SOS;

                // compute the dimensions of each MCU in pixels
                int maxmcux = 0, maxmcuy = 0;
                for (int i = 0; i < ns; i++) {
                    struct pjpeg_component *comp = &pjd->components[comp_idx[i]];

                    maxmcux = max_u8(maxmcux, comp->scalex * 8);
                    maxmcuy = max_u8(maxmcuy, comp->scaley * 8);
                }

                // how many MCU blocks are required to encode the whole image?
                int mcus_x = (pjd->width + maxmcux - 1) / maxmcux;
                int mcus_y = (pjd->height + maxmcuy - 1) / maxmcuy;

                if (0)
                    printf("Image has %d x %d MCU blocks, each %d x %d pixels\n",
                           mcus_x, mcus_y, maxmcux, maxmcuy);

                // allocate output storage
                for (int i = 0; i < ns; i++) {
                    struct pjpeg_component *comp = &pjd->components[comp_idx[i]];
                    comp->width = mcus_x * comp->scalex * 8;
                    comp->height = mcus_y * comp->scaley * 8;
                    comp->stride = comp->width;

                    int alignment = 32;
                    if ((comp->stride % alignment) != 0)
                        comp->stride += alignment - (comp->stride % alignment);

                    comp->data = calloc(comp->height * comp->stride, 1);
                }


                // each component has its own DC prediction
                int32_t *dcpred = calloc(ns, sizeof(int32_t));

                pjd->reset_count = 0;

                for (int mcu_y = 0; mcu_y < mcus_y; mcu_y++) {
                    for (int mcu_x = 0; mcu_x < mcus_x; mcu_x++) {

                        // the next two bytes in the input stream
                        // should be 0xff 0xdN, where N is the next
                        // reset counter.
                        //
                        // Our bit decoder may have already shifted
                        // these into the buffer.  Consequently, we
                        // want to use our bit decoding functions to
                        // check for the marker. But we must first
                        // discard any fractional bits left.
                        if (pjd->reset_interval > 0 && pjd->reset_count == pjd->reset_interval) {

                            // RST markers are byte-aligned, so force
                            // the bit-decoder to the next byte
                            // boundary.
                            bd_discard_to_byte_boundary(&bd);

                            while (1) {
                                int32_t value = bd_consume_bits(&bd, 8);
                                if (bd.inpos > bd.inlen)
                                    return PJPEG_ERR_EOF;
                                if (value == 0xff)
                                    break;
                                printf("RST SYNC\n");
                            }

                            int32_t marker_32 = bd_consume_bits(&bd, 8);

//                            printf("%04x: RESET? %02x\n", *bd.inpos,  marker_32);
                            if (marker_32 != (0xd0 + pjd->reset_next))
                                return PJPEG_ERR_RESET;

                            pjd->reset_count = 0;
                            pjd->reset_next = (pjd->reset_next + 1) & 0x7;

                            memset(dcpred, 0, sizeof(*dcpred));
                        }

                        for (int nsidx = 0; nsidx < ns; nsidx++) {

                            struct pjpeg_component *comp = &pjd->components[comp_idx[nsidx]];

                            int32_t block[64];

                            int qtabidx = comp->tq; // which quant table?

                            for (int sby = 0; sby < comp->scaley; sby++) {
                                for (int sbx = 0; sbx < comp->scalex; sbx++) {
                                    // decode block for component nsidx
                                    memset(block, 0, sizeof(block));

                                    int dc_huff_table_idx = comp->tda >> 4;
                                    int ac_huff_table_idx = 2 + (comp->tda & 0x0f);

                                    if (!pjd->huff_codes_present[dc_huff_table_idx] ||
                                        !pjd->huff_codes_present[ac_huff_table_idx])
                                        return PJPEG_ERR_MISSING_DHT; // probably an MJPEG.


                                    if (1) {
                                        // do DC coefficient
                                        uint32_t next16 = bd_peek_bits(&bd, 16);
                                        struct pjpeg_huffman_code *huff_code = &pjd->huff_codes[dc_huff_table_idx][next16];
                                        bd_consume_bits(&bd, huff_code->nbits);

                                        int ssss = huff_code->code & 0x0f; // ssss == number of additional bits to read
                                        int32_t value = bd_consume_bits(&bd, ssss);

                                        // if high bit is clear, it's negative
                                        if ((value & (1 << (ssss-1))) == 0)
                                            value += ((-1) << ssss) + 1;

                                        dcpred[nsidx] += value;
                                        block[0] = dcpred[nsidx] * pjd->qtab[qtabidx][0];
                                    }

                                    if (1) {
                                        // do AC coefficients
                                        for (int coeff = 1; coeff < 64; coeff++) {

                                            uint32_t next16 = bd_peek_bits(&bd, 16);

                                            struct pjpeg_huffman_code *huff_code = &pjd->huff_codes[ac_huff_table_idx][next16];
                                            bd_consume_bits(&bd, huff_code->nbits);

                                            if (huff_code->code == 0) {
                                                break; // EOB
                                            }

                                            int rrrr = huff_code->code >> 4; // run length of zeros
                                            int ssss = huff_code->code & 0x0f;

                                            int32_t value = bd_consume_bits(&bd, ssss);

                                            // if high bit is clear, it's negative
                                            if ((value & (1 << (ssss-1))) == 0)
                                                value += ((-1) << ssss) + 1;

                                            coeff += rrrr;

                                            block[(int) ZZ[coeff]] = value * pjd->qtab[qtabidx][coeff];
                                        }
                                    }

                                    // do IDCT

                                    // output block's upper-left
                                    // coordinate (in pixels) is
                                    // (comp_x, comp_y).
                                    uint32_t comp_x = (mcu_x * comp->scalex + sbx) * 8;
                                    uint32_t comp_y = (mcu_y * comp->scaley + sby) * 8;
                                    uint32_t dataidx = comp_y * comp->stride + comp_x;

//                                    pjpeg_idct_2D_u32(block, &comp->data[dataidx], comp->stride);
                                    pjpeg_idct_2D_nanojpeg(block, &comp->data[dataidx], comp->stride);
                                }
                            }
                        }

                        pjd->reset_count++;
//                        printf("%04x: reset count %d / %d\n", pjd->inpos, pjd->reset_count, pjd->reset_interval);

                    }
                }

                free(dcpred);
                free(comp_idx);

                break;
            }

            case 0xd9: { // EOI End of Image
                goto got_end_of_image;
            }

            case 0xdd: { // Define Restart Interval
                uint16_t length = bd_consume_bits(&bd, 16);
                if (length != 4)
                    return PJPEG_ERR_DRI;

                // reset interval measured in the number of MCUs
                pjd->reset_interval = bd_consume_bits(&bd, 16);

                break;
            }

            default: {
                printf("pjepg: Unknown marker %02x at offset %04x\n", marker, marker_offset);

                // try to skip it.
                uint16_t length = bd_consume_bits(&bd, 16);
                bd_discard_bytes(&bd, length - 2);
                continue;
            }
        } // switch (marker)
    } // while inpos < inlen

  got_end_of_image:

    return PJPEG_OKAY;
}

void pjpeg_destroy(pjpeg_t *pj)
{
    if (!pj)
        return;

    for (int i = 0; i < pj->ncomponents; i++)
        free(pj->components[i].data);
    free(pj->components);

    free(pj);
}


// just grab the first component.
image_u8_t *pjpeg_to_u8_baseline(pjpeg_t *pj)
{
    assert(pj->ncomponents > 0);

    pjpeg_component_t *comp = &pj->components[0];

    assert(comp->width >= pj->width && comp->height >= pj->height);

    image_u8_t *im = image_u8_create(pj->width, pj->height);
    for (int y = 0; y < im->height; y++)
        memcpy(&im->buf[y*im->stride], &comp->data[y*comp->stride], pj->width);

    return im;
}

static inline uint8_t clampd(double v)
{
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;

    return (uint8_t) v;
}

static inline uint8_t clamp_u8(int32_t v)
{
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return v;
}

// color conversion formulas taken from JFIF spec v 1.02
image_u8x3_t *pjpeg_to_u8x3_baseline(pjpeg_t *pj)
{
    assert(pj->ncomponents == 3);

    pjpeg_component_t *Y = &pj->components[0];
    pjpeg_component_t *Cb = &pj->components[1];
    pjpeg_component_t *Cr = &pj->components[2];

    int Cb_factor_y = Y->height / Cb->height;
    int Cb_factor_x = Y->width / Cb->width;

    int Cr_factor_y = Y->height / Cr->height;
    int Cr_factor_x = Y->width / Cr->width;

    image_u8x3_t *im = image_u8x3_create(pj->width, pj->height);

    if (Cr_factor_y == 1 && Cr_factor_x == 1 && Cb_factor_y == 1 && Cb_factor_x == 1) {

        for (int y = 0; y < pj->height; y++) {
            for (int x = 0; x < pj->width; x++) {
                int32_t y_val  = Y->data[y*Y->stride + x] * 65536;
                int32_t cb_val = Cb->data[y*Cb->stride + x] - 128;
                int32_t cr_val = Cr->data[y*Cr->stride + x] - 128;

                int32_t r_val = y_val +  91881 * cr_val;
                int32_t g_val = y_val + -22554 * cb_val - 46802 * cr_val;
                int32_t b_val = y_val + 116130 * cb_val;

                im->buf[y*im->stride + 3*x + 0 ] = clamp_u8(r_val >> 16);
                im->buf[y*im->stride + 3*x + 1 ] = clamp_u8(g_val >> 16);
                im->buf[y*im->stride + 3*x + 2 ] = clamp_u8(b_val >> 16);
            }
        }
    } else if (Cb_factor_y == Cr_factor_y && Cb_factor_x == Cr_factor_x) {
        for (int by = 0; by < pj->height / Cb_factor_y; by++) {
            for (int bx = 0; bx < pj->width / Cb_factor_x; bx++) {

                int32_t cb_val = Cb->data[by*Cb->stride + bx] - 128;
                int32_t cr_val = Cr->data[by*Cr->stride + bx] - 128;

                int32_t r0 =  91881 * cr_val;
                int32_t g0 = -22554 * cb_val - 46802 * cr_val;
                int32_t b0 = 116130 * cb_val;

                for (int dy = 0; dy < Cb_factor_y; dy++) {
                    int y = by*Cb_factor_y + dy;

                    for (int dx = 0; dx < Cb_factor_x; dx++) {
                        int x = bx*Cb_factor_x + dx;

                        int32_t y_val = Y->data[y*Y->stride + x] * 65536;

                        int32_t r_val = r0 + y_val;
                        int32_t g_val = g0 + y_val;
                        int32_t b_val = b0 + y_val;

                        im->buf[y*im->stride + 3*x + 0 ] = clamp_u8(r_val >> 16);
                        im->buf[y*im->stride + 3*x + 1 ] = clamp_u8(g_val >> 16);
                        im->buf[y*im->stride + 3*x + 2 ] = clamp_u8(b_val >> 16);
                    }
                }
            }
        }
    } else {

        for (int y = 0; y < pj->height; y++) {
            for (int x = 0; x < pj->width; x++) {
                int32_t y_val  = Y->data[y*Y->stride + x];
                int32_t cb_val = Cb->data[(y / Cb_factor_y)*Cb->stride + (x / Cb_factor_x)] - 128;
                int32_t cr_val = Cr->data[(y / Cr_factor_y)*Cr->stride + (x / Cr_factor_x)] - 128;

                uint8_t r_val = clampd(y_val + 1.402 * cr_val);
                uint8_t g_val = clampd(y_val - 0.34414 * cb_val - 0.71414 * cr_val);
                uint8_t b_val = clampd(y_val + 1.772 * cb_val);

                im->buf[y*im->stride + 3*x + 0 ] = r_val;
                im->buf[y*im->stride + 3*x + 1 ] = g_val;
                im->buf[y*im->stride + 3*x + 2 ] = b_val;
            }
        }
    }

    return im;
}

///////////////////////////////////////////////////////////////////
// returns NULL if file loading fails.
pjpeg_t *pjpeg_create_from_file(const char *path, uint32_t flags, int *error)
{
    FILE *f = fopen(path, "r");
    if (f == NULL)
        return NULL;

    fseek(f, 0, SEEK_END);
    long buflen = ftell(f);

    uint8_t *buf = malloc(buflen);
    fseek(f, 0, SEEK_SET);
    int res = fread(buf, 1, buflen, f);
    fclose(f);
    if (res != buflen) {
        free(buf);
        if (error)
            *error = PJPEG_ERR_FILE;
        return NULL;
    }

    pjpeg_t *pj = pjpeg_create_from_buffer(buf, buflen, flags, error);

    free(buf);
    return pj;
}

pjpeg_t *pjpeg_create_from_buffer(uint8_t *buf, int buflen, uint32_t flags, int *error)
{
    struct pjpeg_decode_state pjd;
    memset(&pjd, 0, sizeof(pjd));

    if (flags & PJPEG_MJPEG) {
        pjd.in = mjpeg_dht;
        pjd.inlen = sizeof(mjpeg_dht);
        int result = pjpeg_decode_buffer(&pjd);
        assert(result == 0);
    }

    pjd.in = buf;
    pjd.inlen = buflen;
    pjd.flags = flags;

    int result = pjpeg_decode_buffer(&pjd);
    if (error)
        *error = result;

    if (result) {
        for (int i = 0; i < pjd.ncomponents; i++)
            free(pjd.components[i].data);
        free(pjd.components);

        return NULL;
    }

    pjpeg_t *pj = calloc(1, sizeof(pjpeg_t));

    pj->width = pjd.width;
    pj->height = pjd.height;
    pj->ncomponents = pjd.ncomponents;
    pj->components = pjd.components;

    return pj;
}
