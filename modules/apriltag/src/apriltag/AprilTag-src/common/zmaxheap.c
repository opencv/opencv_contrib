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
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

#include "zmaxheap.h"

#ifdef _WIN32
static inline long int random(void)
{
        return rand();
}
#endif

//                 0
//         1               2
//      3     4        5       6
//     7 8   9 10    11 12   13 14
//
// Children of node i:  2*i+1, 2*i+2
// Parent of node i: (i-1) / 2
//
// Heap property: a parent is greater than (or equal to) its children.

#define MIN_CAPACITY 16

struct zmaxheap
{
    size_t el_sz;

    int size;
    int alloc;

    float *values;
    char *data;

    void (*swap)(zmaxheap_t *heap, int a, int b);
};

static inline void swap_default(zmaxheap_t *heap, int a, int b)
{
    float t = heap->values[a];
    heap->values[a] = heap->values[b];
    heap->values[b] = t;

    char *tmp = malloc(sizeof(char)*heap->el_sz);
    memcpy(tmp, &heap->data[a*heap->el_sz], heap->el_sz);
    memcpy(&heap->data[a*heap->el_sz], &heap->data[b*heap->el_sz], heap->el_sz);
    memcpy(&heap->data[b*heap->el_sz], tmp, heap->el_sz);
    free(tmp);
}

static inline void swap_pointer(zmaxheap_t *heap, int a, int b)
{
    float t = heap->values[a];
    heap->values[a] = heap->values[b];
    heap->values[b] = t;

    void **pp = (void**) heap->data;
    void *tmp = pp[a];
    pp[a] = pp[b];
    pp[b] = tmp;
}


zmaxheap_t *zmaxheap_create(size_t el_sz)
{
    zmaxheap_t *heap = calloc(1, sizeof(zmaxheap_t));
    heap->el_sz = el_sz;

    heap->swap = swap_default;

    if (el_sz == sizeof(void*))
        heap->swap = swap_pointer;

    return heap;
}

void zmaxheap_destroy(zmaxheap_t *heap)
{
    free(heap->values);
    free(heap->data);
    memset(heap, 0, sizeof(zmaxheap_t));
    free(heap);
}

int zmaxheap_size(zmaxheap_t *heap)
{
    return heap->size;
}

void zmaxheap_ensure_capacity(zmaxheap_t *heap, int capacity)
{
    if (heap->alloc >= capacity)
        return;

    int newcap = heap->alloc;

    while (newcap < capacity) {
        if (newcap < MIN_CAPACITY) {
            newcap = MIN_CAPACITY;
            continue;
        }

        newcap *= 2;
    }

    heap->values = realloc(heap->values, newcap * sizeof(float));
    heap->data = realloc(heap->data, newcap * heap->el_sz);
    heap->alloc = newcap;
}

void zmaxheap_add(zmaxheap_t *heap, void *p, float v)
{

    assert (isfinite(v) && "zmaxheap_add: Trying to add non-finite number to heap.  NaN's prohibited, could allow INF with testing");
    zmaxheap_ensure_capacity(heap, heap->size + 1);

    int idx = heap->size;

    heap->values[idx] = v;
    memcpy(&heap->data[idx*heap->el_sz], p, heap->el_sz);

    heap->size++;

    while (idx > 0) {

        int parent = (idx - 1) / 2;

        // we're done!
        if (heap->values[parent] >= v)
            break;

        // else, swap and recurse upwards.
        heap->swap(heap, idx, parent);
        idx = parent;
    }
}

void zmaxheap_vmap(zmaxheap_t *heap, void (*f)())
{
    assert(heap != NULL);
    assert(f != NULL);
    assert(heap->el_sz == sizeof(void*));

    for (int idx = 0; idx < heap->size; idx++) {
        void *p = NULL;
        memcpy(&p, &heap->data[idx*heap->el_sz], heap->el_sz);
        if (p == NULL) {
            printf("Warning: zmaxheap_vmap item %d is NULL\n", idx);
            fflush(stdout);
        }
        f(p);
    }
}

// Removes the item in the heap at the given index.  Returns 1 if the
// item existed. 0 Indicates an invalid idx (heap is smaller than
// idx). This is mostly intended to be used by zmaxheap_remove_max.
int zmaxheap_remove_index(zmaxheap_t *heap, int idx, void *p, float *v)
{
    if (idx >= heap->size)
        return 0;

    // copy out the requested element from the heap.
    if (v != NULL)
        *v = heap->values[idx];
    if (p != NULL)
        memcpy(p, &heap->data[idx*heap->el_sz], heap->el_sz);

    heap->size--;

    // If this element is already the last one, then there's nothing
    // for us to do.
    if (idx == heap->size)
        return 1;

    // copy last element to first element. (which probably upsets
    // the heap property).
    heap->values[idx] = heap->values[heap->size];
    memcpy(&heap->data[idx*heap->el_sz], &heap->data[heap->el_sz * heap->size], heap->el_sz);

    // now fix the heap. Note, as we descend, we're "pushing down"
    // the same node the entire time. Thus, while the index of the
    // parent might change, the parent_score doesn't.
    int parent = idx;
    float parent_score = heap->values[idx];

    // descend, fixing the heap.
    while (parent < heap->size) {

        int left = 2*parent + 1;
        int right = left + 1;

//            assert(parent_score == heap->values[parent]);

        float left_score = (left < heap->size) ? heap->values[left] : -INFINITY;
        float right_score = (right < heap->size) ? heap->values[right] : -INFINITY;

        // put the biggest of (parent, left, right) as the parent.

        // already okay?
        if (parent_score >= left_score && parent_score >= right_score)
            break;

        // if we got here, then one of the children is bigger than the parent.
        if (left_score >= right_score) {
            assert(left < heap->size);
            heap->swap(heap, parent, left);
            parent = left;
        } else {
            // right_score can't be less than left_score if right_score is -INFINITY.
            assert(right < heap->size);
            heap->swap(heap, parent, right);
            parent = right;
        }
    }

    return 1;
}

int zmaxheap_remove_max(zmaxheap_t *heap, void *p, float *v)
{
    return zmaxheap_remove_index(heap, 0, p, v);
}

void zmaxheap_iterator_init(zmaxheap_t *heap, zmaxheap_iterator_t *it)
{
    memset(it, 0, sizeof(zmaxheap_iterator_t));
    it->heap = heap;
    it->in = 0;
    it->out = 0;
}

int zmaxheap_iterator_next(zmaxheap_iterator_t *it, void *p, float *v)
{
    zmaxheap_t *heap = it->heap;

    if (it->in >= zmaxheap_size(heap))
        return 0;

    *v = heap->values[it->in];
    memcpy(p, &heap->data[it->in*heap->el_sz], heap->el_sz);

    if (it->in != it->out) {
        heap->values[it->out] = heap->values[it->in];
        memcpy(&heap->data[it->out*heap->el_sz], &heap->data[it->in*heap->el_sz], heap->el_sz);
    }

    it->in++;
    it->out++;
    return 1;
}

int zmaxheap_iterator_next_volatile(zmaxheap_iterator_t *it, void *p, float *v)
{
    zmaxheap_t *heap = it->heap;

    if (it->in >= zmaxheap_size(heap))
        return 0;

    *v = heap->values[it->in];
    *((void**) p) = &heap->data[it->in*heap->el_sz];

    if (it->in != it->out) {
        heap->values[it->out] = heap->values[it->in];
        memcpy(&heap->data[it->out*heap->el_sz], &heap->data[it->in*heap->el_sz], heap->el_sz);
    }

    it->in++;
    it->out++;
    return 1;
}

void zmaxheap_iterator_remove(zmaxheap_iterator_t *it)
{
    it->out--;
}

static void maxheapify(zmaxheap_t *heap, int parent)
{
    int left = 2*parent + 1;
    int right = 2*parent + 2;

    int betterchild = parent;

    if (left < heap->size && heap->values[left] > heap->values[betterchild])
        betterchild = left;
    if (right < heap->size && heap->values[right] > heap->values[betterchild])
        betterchild = right;

    if (betterchild != parent) {
        heap->swap(heap, parent, betterchild);
        return maxheapify(heap, betterchild);
    }
}

#if 0 //won't compile if defined but not used
// test the heap property
static void validate(zmaxheap_t *heap)
{
    for (int parent = 0; parent < heap->size; parent++) {
        int left = 2*parent + 1;
        int right = 2*parent + 2;

        if (left < heap->size) {
            assert(heap->values[parent] > heap->values[left]);
        }

        if (right < heap->size) {
            assert(heap->values[parent] > heap->values[right]);
        }
    }
}
#endif
void zmaxheap_iterator_finish(zmaxheap_iterator_t *it)
{
    // if nothing was removed, no work to do.
    if (it->in == it->out)
        return;

    zmaxheap_t *heap = it->heap;

    heap->size = it->out;

    // restore heap property
    for (int i = heap->size/2 - 1; i >= 0; i--)
        maxheapify(heap, i);
}

void zmaxheap_test()
{
    int cap = 10000;
    int sz = 0;
    int32_t *vals = calloc(sizeof(int32_t), cap);

    zmaxheap_t *heap = zmaxheap_create(sizeof(int32_t));

    int maxsz = 0;
    int zcnt = 0;

    for (int iter = 0; iter < 5000000; iter++) {
        assert(sz == heap->size);

        if ((random() & 1) == 0 && sz < cap) {
            // add a value
            int32_t v = (int32_t) (random() / 1000);
            float fv = v;
            assert(v == fv);

            vals[sz] = v;
            zmaxheap_add(heap, &v, fv);
            sz++;

//            printf("add %d %f\n", v, fv);
        } else {
            // remove a value
            int maxv = -1, maxi = -1;

            for (int i = 0; i < sz; i++) {
                if (vals[i] > maxv) {
                    maxv = vals[i];
                    maxi = i;
                }
            }


            int32_t outv;
            float outfv;
            int res = zmaxheap_remove_max(heap, &outv, &outfv);
            if (sz == 0) {
                assert(res == 0);
            } else {
//                printf("%d %d %d %f\n", sz, maxv, outv, outfv);
                assert(outv == outfv);
                assert(maxv == outv);

                // shuffle erase the maximum from our list.
                vals[maxi] = vals[sz - 1];
                sz--;
            }
        }

        if (sz > maxsz)
            maxsz = sz;

        if (maxsz > 0 && sz == 0)
            zcnt++;
    }

    printf("max size: %d, zcount %d\n", maxsz, zcnt);
    free (vals);
}
