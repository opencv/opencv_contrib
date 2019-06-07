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
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "pam.h"

pam_t *pam_create_from_file(const char *inpath)
{
    FILE *infile = fopen(inpath, "r");
    if (infile == NULL) {
        printf("pam.c: couldn't open input file: %s\n", inpath);
        return NULL;
    }

    pam_t *pam = calloc(1, sizeof(pam_t));
    pam->width = -1;
    pam->height = -1;
    pam->depth = -1;
    pam->maxval = -1;
    pam->type = -1;

    int linenumber = 0;

    while (1) {
        char line[1024];
        if (!fgets(line, sizeof(line), infile)) {
            printf("pam.c: unexpected EOF\n");
            goto fail;
        }
        linenumber++;

        char *tok0 = line;
        char *tok1 = NULL;

        if (line[0] == '#') // comment
            continue;

        size_t linelen = strlen(line);
        for (int idx = 0; idx < linelen; idx++) {
            if (line[idx] == ' ') {
                line[idx] = 0;
                if (tok1) {
                    printf("pam.c: More than two tokens, %s:%d\n", inpath, linenumber);
                }

                tok1 = &line[idx+1];
            }
            if (line[idx] == '\n')
                line[idx] = 0;
        }

        if (!strcmp(tok0, "P7"))
            continue;

        if (!strcmp(tok0, "ENDHDR"))
            break;

        if (!strcmp(tok0, "WIDTH") && tok1) {
            pam->width = atoi(tok1);
            continue;
        }

        if (!strcmp(tok0, "HEIGHT") && tok1) {
            pam->height = atoi(tok1);
            continue;
        }

        if (!strcmp(tok0, "DEPTH") && tok1) {
            pam->depth = atoi(tok1);
            continue;
        }

        if (!strcmp(tok0, "MAXVAL") && tok1) {
            pam->maxval = atoi(tok1);
            continue;
        }

        if (!strcmp(tok0, "TUPLTYPE") && tok1) {
            if (!strcmp(tok1, "GRAYSCALE_ALPHA")) {
                pam->type = PAM_GRAYSCALE_ALPHA;
                continue;
            }

            if (!strcmp(tok1, "RGB_ALPHA")) {
                pam->type = PAM_RGB_ALPHA;
                continue;
            }

            if (!strcmp(tok1, "RGB")) {
                pam->type = PAM_RGB;
                continue;
            }

            if (!strcmp(tok1, "GRAYSCALE")) {
                pam->type = PAM_GRAYSCALE;
                continue;
            }

            printf("pam.c: unrecognized tupl type %s\n", tok1);
            continue;
        }

        printf("pam.c: unrecognized attribute %s\n", tok0);
    }

    if (pam->width < 0 || pam->height < 0 || pam->depth < 0 ||
        pam->maxval < 0 || pam->type < 0) {
        printf("pam.c: missing required metadata field\n");
        goto fail;
    }

    assert(pam->maxval == 255);

    pam->datalen = pam->width * pam->height * pam->depth;
    pam->data = malloc(pam->datalen);
    if (pam->datalen != fread(pam->data, 1, pam->datalen, infile)) {
        printf("pam.c: couldn't read body\n");
        goto fail;
    }

    fclose(infile);
    return pam;

  fail:
    free(pam);
    fclose(infile);
    return NULL;
}

int pam_write_file(pam_t *pam, const char *outpath)
{
    FILE *f = fopen(outpath, "w+");
    if (!f)
        return -1;

    const char *tupl = NULL;
    switch (pam->type) {
        case PAM_GRAYSCALE_ALPHA:
            tupl = "GRAYSCALE_ALPHA";
            break;
        case PAM_RGB_ALPHA:
            tupl = "RGB_ALPHA";
            break;
        case PAM_RGB:
            tupl = "RGB";
            break;
        case PAM_GRAYSCALE:
            tupl = "GRAYSCALE";
            break;
        default:
            assert(0);
    }

    fprintf(f, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n",
            pam->width, pam->height, pam->depth, pam->maxval, tupl);
    int len = pam->width * pam->height * pam->depth;
    if (len != fwrite(pam->data, 1, len, f)) {
        fclose(f);
        return -2;
    }

    fclose(f);

    return 0;
}

void pam_destroy(pam_t *pam)
{
    if (!pam)
        return;

    free(pam->data);
    free(pam);
}

pam_t *pam_copy(pam_t *pam)
{
    pam_t *copy = calloc(1, sizeof(pam_t));
    copy->width = pam->width;
    copy->height = pam->height;
    copy->depth = pam->depth;
    copy->maxval = pam->maxval;
    copy->type = pam->type;

    copy->datalen = pam->datalen;
    copy->data = malloc(pam->datalen);
    memcpy(copy->data, pam->data, pam->datalen);

    return copy;
}

pam_t *pam_convert(pam_t *in, int type)
{
    if (type == in->type)
        return pam_copy(in);

    assert(type == PAM_RGB_ALPHA); // we don't support a lot yet
    assert(in->maxval == 255);

    int w = in->width;
    int h = in->height;

    pam_t *out = calloc(1, sizeof(pam_t));
    out->type = type;
    out->width = w;
    out->height = h;
    out->maxval = in->maxval;
    out->depth = 4;
    out->datalen = 4 * w * h;
    out->data = malloc(out->datalen);

    if (in->type == PAM_RGB) {
        assert(in->depth == 3);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                out->data[y*4*w + 4*x + 0] = in->data[y*3*w + 3*x + 0];
                out->data[y*4*w + 4*x + 1] = in->data[y*3*w + 3*x + 1];
                out->data[y*4*w + 4*x + 2] = in->data[y*3*w + 3*x + 2];
                out->data[y*4*w + 4*x + 3] = 255;
            }
        }
    } else {
        printf("pam.c unsupported type %d\n", in->type);
        assert(0);
    }

    return out;
}
