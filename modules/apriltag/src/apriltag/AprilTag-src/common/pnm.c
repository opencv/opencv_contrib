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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "pnm.h"

pnm_t *pnm_create_from_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL)
        return NULL;

    pnm_t *pnm = calloc(1, sizeof(pnm_t));
    pnm->format = -1;

    char tmp[1024];
    int nparams = 0; // will be 3 when we're all done.
    int params[3];

    while (nparams < 3 && !(pnm->format == PNM_FORMAT_BINARY && nparams == 2)) {
        if (fgets(tmp, sizeof(tmp), f) == NULL)
            goto error;

        // skip comments
        if (tmp[0]=='#')
            continue;

        char *p = tmp;

        if (pnm->format == -1 && tmp[0]=='P') {
            pnm->format = tmp[1]-'0';
            assert(pnm->format == PNM_FORMAT_GRAY || pnm->format == PNM_FORMAT_RGB || pnm->format == PNM_FORMAT_BINARY);
            p = &tmp[2];
        }

        // pull integers out of this line until there are no more.
        while (nparams < 3 && *p!=0) {
            while (*p==' ')
                p++;

            // encounter rubbish? (End of line?)
            if (*p < '0' || *p > '9')
                break;

            int acc = 0;
            while (*p >= '0' && *p <= '9') {
                acc = acc*10 + *p - '0';
                p++;
            }

            params[nparams++] = acc;
            p++;
        }
    }

    pnm->width = params[0];
    pnm->height = params[1];
    pnm->max = params[2];

    switch (pnm->format) {
        case PNM_FORMAT_BINARY: {
            // files in the wild sometimes simply don't set max
            pnm->max = 1;

            pnm->buflen = pnm->height * ((pnm->width + 7)  / 8);
            pnm->buf = malloc(pnm->buflen);
            size_t len = fread(pnm->buf, 1, pnm->buflen, f);
            if (len != pnm->buflen)
                goto error;

            fclose(f);
            return pnm;
        }

        case PNM_FORMAT_GRAY: {
            if (pnm->max == 255)
                pnm->buflen = pnm->width * pnm->height;
            else if (pnm->max == 65535)
                pnm->buflen = 2 * pnm->width * pnm->height;
            else
                assert(0);

            pnm->buf = malloc(pnm->buflen);
            size_t len = fread(pnm->buf, 1, pnm->buflen, f);
            if (len != pnm->buflen)
                goto error;

            fclose(f);
            return pnm;
        }

        case PNM_FORMAT_RGB: {
            if (pnm->max == 255)
                pnm->buflen = pnm->width * pnm->height * 3;
            else if (pnm->max == 65535)
                pnm->buflen = 2 * pnm->width * pnm->height * 3;
            else
                assert(0);

            pnm->buf = malloc(pnm->buflen);
            size_t len = fread(pnm->buf, 1, pnm->buflen, f);
            if (len != pnm->buflen)
                goto error;
            fclose(f);
            return pnm;
        }
    }

error:
    fclose(f);

    if (pnm != NULL) {
        free(pnm->buf);
        free(pnm);
    }

    return NULL;
}

void pnm_destroy(pnm_t *pnm)
{
    if (pnm == NULL)
        return;

    free(pnm->buf);
    free(pnm);
}
