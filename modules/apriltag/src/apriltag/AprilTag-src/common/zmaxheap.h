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

#include <stdio.h>

typedef struct zmaxheap zmaxheap_t;

typedef struct zmaxheap_iterator zmaxheap_iterator_t;
struct zmaxheap_iterator {
    zmaxheap_t *heap;
    int in, out;
};

zmaxheap_t *zmaxheap_create(size_t el_sz);

void zmaxheap_vmap(zmaxheap_t *heap, void (*f)());

void zmaxheap_destroy(zmaxheap_t *heap);

void zmaxheap_add(zmaxheap_t *heap, void *p, float v);

int zmaxheap_size(zmaxheap_t *heap);

// returns 0 if the heap is empty, so you can do
// while (zmaxheap_remove_max(...)) { }
int zmaxheap_remove_max(zmaxheap_t *heap, void *p, float *v);

////////////////////////////////////////////
// This is a peculiar iterator intended to support very specific (and
// unusual) applications, and the heap is not necessarily in a valid
// state until zmaxheap_iterator_finish is called.  Consequently, do
// not call any other methods on the heap while iterating through.

// you must provide your own storage for the iterator, and pass in a
// pointer.
void zmaxheap_iterator_init(zmaxheap_t *heap, zmaxheap_iterator_t *it);

// Traverses the heap in top-down/left-right order. makes a copy of
// the content into memory (p) that you provide.
int zmaxheap_iterator_next(zmaxheap_iterator_t *it, void *p, float *v);

// will set p to be a pointer to the heap's internal copy of the dfata.
int zmaxheap_iterator_next_volatile(zmaxheap_iterator_t *it, void *p, float *v);

// remove the current element.
void zmaxheap_iterator_remove(zmaxheap_iterator_t *it);

// call after all iterator operations are done. After calling this,
// the iterator should no longer be used, but the heap methods can be.
void zmaxheap_iterator_finish(zmaxheap_iterator_t *it);
