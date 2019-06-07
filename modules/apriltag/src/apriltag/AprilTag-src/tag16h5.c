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

#include <stdlib.h>
#include "apriltag.h"

apriltag_family_t
#ifndef _MSC_VER
__attribute__((optimize("O0")))
#endif
*tag16h5_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tag16h5");
   tf->h = 5;
   tf->ncodes = 30;
   tf->codes = calloc(30, sizeof(uint64_t));
   tf->codes[0] = 0x00000000000027c8UL;
   tf->codes[1] = 0x00000000000031b6UL;
   tf->codes[2] = 0x0000000000003859UL;
   tf->codes[3] = 0x000000000000569cUL;
   tf->codes[4] = 0x0000000000006c76UL;
   tf->codes[5] = 0x0000000000007ddbUL;
   tf->codes[6] = 0x000000000000af09UL;
   tf->codes[7] = 0x000000000000f5a1UL;
   tf->codes[8] = 0x000000000000fb8bUL;
   tf->codes[9] = 0x0000000000001cb9UL;
   tf->codes[10] = 0x00000000000028caUL;
   tf->codes[11] = 0x000000000000e8dcUL;
   tf->codes[12] = 0x0000000000001426UL;
   tf->codes[13] = 0x0000000000005770UL;
   tf->codes[14] = 0x0000000000009253UL;
   tf->codes[15] = 0x000000000000b702UL;
   tf->codes[16] = 0x000000000000063aUL;
   tf->codes[17] = 0x0000000000008f34UL;
   tf->codes[18] = 0x000000000000b4c0UL;
   tf->codes[19] = 0x00000000000051ecUL;
   tf->codes[20] = 0x000000000000e6f0UL;
   tf->codes[21] = 0x0000000000005fa4UL;
   tf->codes[22] = 0x000000000000dd43UL;
   tf->codes[23] = 0x0000000000001aaaUL;
   tf->codes[24] = 0x000000000000e62fUL;
   tf->codes[25] = 0x0000000000006dbcUL;
   tf->codes[26] = 0x000000000000b6ebUL;
   tf->codes[27] = 0x000000000000de10UL;
   tf->codes[28] = 0x000000000000154dUL;
   tf->codes[29] = 0x000000000000b57aUL;
   tf->nbits = 16;
   tf->bit_x = calloc(16, sizeof(uint32_t));
   tf->bit_y = calloc(16, sizeof(uint32_t));
   tf->bit_x[0] = 1;
   tf->bit_y[0] = 1;
   tf->bit_x[1] = 2;
   tf->bit_y[1] = 1;
   tf->bit_x[2] = 3;
   tf->bit_y[2] = 1;
   tf->bit_x[3] = 2;
   tf->bit_y[3] = 2;
   tf->bit_x[4] = 4;
   tf->bit_y[4] = 1;
   tf->bit_x[5] = 4;
   tf->bit_y[5] = 2;
   tf->bit_x[6] = 4;
   tf->bit_y[6] = 3;
   tf->bit_x[7] = 3;
   tf->bit_y[7] = 2;
   tf->bit_x[8] = 4;
   tf->bit_y[8] = 4;
   tf->bit_x[9] = 3;
   tf->bit_y[9] = 4;
   tf->bit_x[10] = 2;
   tf->bit_y[10] = 4;
   tf->bit_x[11] = 3;
   tf->bit_y[11] = 3;
   tf->bit_x[12] = 1;
   tf->bit_y[12] = 4;
   tf->bit_x[13] = 1;
   tf->bit_y[13] = 3;
   tf->bit_x[14] = 1;
   tf->bit_y[14] = 2;
   tf->bit_x[15] = 2;
   tf->bit_y[15] = 3;
   tf->width_at_border = 6;
   tf->total_width = 8;
   tf->reversed_border = false;
   return tf;
}

void tag16h5_destroy(apriltag_family_t *tf)
{
   free(tf->codes);
   free(tf->bit_x);
   free(tf->bit_y);
   free(tf->name);
   free(tf);
}
