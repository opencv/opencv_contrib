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
*tag25h9_create()
{
   apriltag_family_t *tf = calloc(1, sizeof(apriltag_family_t));
   tf->name = strdup("tag25h9");
   tf->h = 9;
   tf->ncodes = 35;
   tf->codes = calloc(35, sizeof(uint64_t));
   tf->codes[0] = 0x000000000156f1f4UL;
   tf->codes[1] = 0x0000000001f28cd5UL;
   tf->codes[2] = 0x00000000016ce32cUL;
   tf->codes[3] = 0x0000000001ea379cUL;
   tf->codes[4] = 0x0000000001390f89UL;
   tf->codes[5] = 0x000000000034fad0UL;
   tf->codes[6] = 0x00000000007dcdb5UL;
   tf->codes[7] = 0x000000000119ba95UL;
   tf->codes[8] = 0x0000000001ae9daaUL;
   tf->codes[9] = 0x0000000000df02aaUL;
   tf->codes[10] = 0x000000000082fc15UL;
   tf->codes[11] = 0x0000000000465123UL;
   tf->codes[12] = 0x0000000000ceee98UL;
   tf->codes[13] = 0x0000000001f17260UL;
   tf->codes[14] = 0x00000000014429cdUL;
   tf->codes[15] = 0x00000000017248a8UL;
   tf->codes[16] = 0x00000000016ad452UL;
   tf->codes[17] = 0x00000000009670adUL;
   tf->codes[18] = 0x00000000016f65b2UL;
   tf->codes[19] = 0x0000000000b8322bUL;
   tf->codes[20] = 0x00000000005d715bUL;
   tf->codes[21] = 0x0000000001a1c7e7UL;
   tf->codes[22] = 0x0000000000d7890dUL;
   tf->codes[23] = 0x0000000001813522UL;
   tf->codes[24] = 0x0000000001c9c611UL;
   tf->codes[25] = 0x000000000099e4a4UL;
   tf->codes[26] = 0x0000000000855234UL;
   tf->codes[27] = 0x00000000017b81c0UL;
   tf->codes[28] = 0x0000000000c294bbUL;
   tf->codes[29] = 0x000000000089fae3UL;
   tf->codes[30] = 0x000000000044df5fUL;
   tf->codes[31] = 0x0000000001360159UL;
   tf->codes[32] = 0x0000000000ec31e8UL;
   tf->codes[33] = 0x0000000001bcc0f6UL;
   tf->codes[34] = 0x0000000000a64f8dUL;
   tf->nbits = 25;
   tf->bit_x = calloc(25, sizeof(uint32_t));
   tf->bit_y = calloc(25, sizeof(uint32_t));
   tf->bit_x[0] = 1;
   tf->bit_y[0] = 1;
   tf->bit_x[1] = 2;
   tf->bit_y[1] = 1;
   tf->bit_x[2] = 3;
   tf->bit_y[2] = 1;
   tf->bit_x[3] = 4;
   tf->bit_y[3] = 1;
   tf->bit_x[4] = 2;
   tf->bit_y[4] = 2;
   tf->bit_x[5] = 3;
   tf->bit_y[5] = 2;
   tf->bit_x[6] = 5;
   tf->bit_y[6] = 1;
   tf->bit_x[7] = 5;
   tf->bit_y[7] = 2;
   tf->bit_x[8] = 5;
   tf->bit_y[8] = 3;
   tf->bit_x[9] = 5;
   tf->bit_y[9] = 4;
   tf->bit_x[10] = 4;
   tf->bit_y[10] = 2;
   tf->bit_x[11] = 4;
   tf->bit_y[11] = 3;
   tf->bit_x[12] = 5;
   tf->bit_y[12] = 5;
   tf->bit_x[13] = 4;
   tf->bit_y[13] = 5;
   tf->bit_x[14] = 3;
   tf->bit_y[14] = 5;
   tf->bit_x[15] = 2;
   tf->bit_y[15] = 5;
   tf->bit_x[16] = 4;
   tf->bit_y[16] = 4;
   tf->bit_x[17] = 3;
   tf->bit_y[17] = 4;
   tf->bit_x[18] = 1;
   tf->bit_y[18] = 5;
   tf->bit_x[19] = 1;
   tf->bit_y[19] = 4;
   tf->bit_x[20] = 1;
   tf->bit_y[20] = 3;
   tf->bit_x[21] = 1;
   tf->bit_y[21] = 2;
   tf->bit_x[22] = 2;
   tf->bit_y[22] = 4;
   tf->bit_x[23] = 2;
   tf->bit_y[23] = 3;
   tf->bit_x[24] = 3;
   tf->bit_y[24] = 3;
   tf->width_at_border = 7;
   tf->total_width = 9;
   tf->reversed_border = false;
   return tf;
}

void tag25h9_destroy(apriltag_family_t *tf)
{
   free(tf->codes);
   free(tf->bit_x);
   free(tf->bit_y);
   free(tf->name);
   free(tf);
}
