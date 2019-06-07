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
*tagCircle21h7_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tagCircle21h7");
   tf->h = 7;
   tf->ncodes = 38;
   tf->codes = calloc(38, sizeof(uint64_t));
   tf->codes[0] = 0x0000000000157863UL;
   tf->codes[1] = 0x0000000000047e28UL;
   tf->codes[2] = 0x00000000001383edUL;
   tf->codes[3] = 0x000000000000953cUL;
   tf->codes[4] = 0x00000000000da68bUL;
   tf->codes[5] = 0x00000000001cac50UL;
   tf->codes[6] = 0x00000000000bb215UL;
   tf->codes[7] = 0x000000000016ceeeUL;
   tf->codes[8] = 0x000000000005d4b3UL;
   tf->codes[9] = 0x00000000001ff751UL;
   tf->codes[10] = 0x00000000000efd16UL;
   tf->codes[11] = 0x0000000000072b3eUL;
   tf->codes[12] = 0x0000000000163103UL;
   tf->codes[13] = 0x0000000000106e56UL;
   tf->codes[14] = 0x00000000001996b9UL;
   tf->codes[15] = 0x00000000000c0234UL;
   tf->codes[16] = 0x00000000000624d2UL;
   tf->codes[17] = 0x00000000001fa985UL;
   tf->codes[18] = 0x00000000000344a5UL;
   tf->codes[19] = 0x00000000000762fbUL;
   tf->codes[20] = 0x000000000019e92bUL;
   tf->codes[21] = 0x0000000000043755UL;
   tf->codes[22] = 0x000000000001a4f4UL;
   tf->codes[23] = 0x000000000010fad8UL;
   tf->codes[24] = 0x0000000000001b52UL;
   tf->codes[25] = 0x000000000017e59fUL;
   tf->codes[26] = 0x00000000000e6f70UL;
   tf->codes[27] = 0x00000000000ed47aUL;
   tf->codes[28] = 0x00000000000c9931UL;
   tf->codes[29] = 0x0000000000014df2UL;
   tf->codes[30] = 0x00000000000a06f1UL;
   tf->codes[31] = 0x00000000000e5041UL;
   tf->codes[32] = 0x000000000012ec03UL;
   tf->codes[33] = 0x000000000016724eUL;
   tf->codes[34] = 0x00000000000af1a5UL;
   tf->codes[35] = 0x000000000008a8acUL;
   tf->codes[36] = 0x0000000000015b39UL;
   tf->codes[37] = 0x00000000001ec1e3UL;
   tf->nbits = 21;
   tf->bit_x = calloc(21, sizeof(uint32_t));
   tf->bit_y = calloc(21, sizeof(uint32_t));
   tf->bit_x[0] = 1;
   tf->bit_y[0] = -2;
   tf->bit_x[1] = 2;
   tf->bit_y[1] = -2;
   tf->bit_x[2] = 3;
   tf->bit_y[2] = -2;
   tf->bit_x[3] = 1;
   tf->bit_y[3] = 1;
   tf->bit_x[4] = 2;
   tf->bit_y[4] = 1;
   tf->bit_x[5] = 6;
   tf->bit_y[5] = 1;
   tf->bit_x[6] = 6;
   tf->bit_y[6] = 2;
   tf->bit_x[7] = 6;
   tf->bit_y[7] = 3;
   tf->bit_x[8] = 3;
   tf->bit_y[8] = 1;
   tf->bit_x[9] = 3;
   tf->bit_y[9] = 2;
   tf->bit_x[10] = 3;
   tf->bit_y[10] = 6;
   tf->bit_x[11] = 2;
   tf->bit_y[11] = 6;
   tf->bit_x[12] = 1;
   tf->bit_y[12] = 6;
   tf->bit_x[13] = 3;
   tf->bit_y[13] = 3;
   tf->bit_x[14] = 2;
   tf->bit_y[14] = 3;
   tf->bit_x[15] = -2;
   tf->bit_y[15] = 3;
   tf->bit_x[16] = -2;
   tf->bit_y[16] = 2;
   tf->bit_x[17] = -2;
   tf->bit_y[17] = 1;
   tf->bit_x[18] = 1;
   tf->bit_y[18] = 3;
   tf->bit_x[19] = 1;
   tf->bit_y[19] = 2;
   tf->bit_x[20] = 2;
   tf->bit_y[20] = 2;
   tf->width_at_border = 5;
   tf->total_width = 9;
   tf->reversed_border = true;
   return tf;
}

void tagCircle21h7_destroy(apriltag_family_t *tf)
{
   free(tf->codes);
   free(tf->bit_x);
   free(tf->bit_y);
   free(tf->name);
   free(tf);
}
