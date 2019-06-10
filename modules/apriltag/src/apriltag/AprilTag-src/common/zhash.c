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
#include <string.h>
#include <assert.h>

#include "zhash.h"

// force a rehash when our capacity is less than this many times the size
#define ZHASH_FACTOR_CRITICAL 2

// When resizing, how much bigger do we want to be? (should be greater than _CRITICAL)
#define ZHASH_FACTOR_REALLOC 4

struct zhash
{
    size_t keysz, valuesz;
    int    entrysz; // valid byte (1) + keysz + values

    uint32_t(*hash)(const void *a);

    // returns 1 if equal
    int(*equals)(const void *a, const void *b);

    int size; // # of items in hash table

    char *entries; // each entry of size entrysz;
    int  nentries; // how many entries are allocated? Never 0.
};

zhash_t *zhash_create_capacity(size_t keysz, size_t valuesz,
                               uint32_t(*hash)(const void *a), int(*equals)(const void *a, const void*b),
                               int capacity)
{
    assert(hash != NULL);
    assert(equals != NULL);

    // resize...
    int _nentries = ZHASH_FACTOR_REALLOC * capacity;
    if (_nentries < 8)
        _nentries = 8;

    // to a power of 2.
    int nentries = _nentries;
    if ((nentries & (nentries - 1)) != 0) {
        nentries = 8;
        while (nentries < _nentries)
            nentries *= 2;
    }

    zhash_t *zh = (zhash_t*) calloc(1, sizeof(zhash_t));
    zh->keysz = keysz;
    zh->valuesz = valuesz;
    zh->hash = hash;
    zh->equals = equals;
    zh->nentries = nentries;

    zh->entrysz = 1 + zh->keysz + zh->valuesz;

    zh->entries = calloc(zh->nentries, zh->entrysz);
    zh->nentries = nentries;

    return zh;
}

zhash_t *zhash_create(size_t keysz, size_t valuesz,
                      uint32_t(*hash)(const void *a), int(*equals)(const void *a, const void *b))
{
    return zhash_create_capacity(keysz, valuesz, hash, equals, 8);
}

void zhash_destroy(zhash_t *zh)
{
    if (zh == NULL)
        return;

    free(zh->entries);
    free(zh);
}

int zhash_size(const zhash_t *zh)
{
    return zh->size;
}

void zhash_clear(zhash_t *zh)
{
    memset(zh->entries, 0, zh->nentries * zh->entrysz);
    zh->size = 0;
}

int zhash_get_volatile(const zhash_t *zh, const void *key, void *out_value)
{
    uint32_t code = zh->hash(key);
    uint32_t entry_idx = code & (zh->nentries - 1);

    while (zh->entries[entry_idx * zh->entrysz]) {
        void *this_key = &zh->entries[entry_idx * zh->entrysz + 1];
        if (zh->equals(key, this_key)) {
            *((void**) out_value) = &zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz];
            return 1;
        }

        entry_idx = (entry_idx + 1) & (zh->nentries - 1);
    }

    return 0;
}

int zhash_get(const zhash_t *zh, const void *key, void *out_value)
{
    void *tmp;
    if (zhash_get_volatile(zh, key, &tmp)) {
        memcpy(out_value, tmp, zh->valuesz);
        return 1;
    }

    return 0;
}

int zhash_put(zhash_t *zh, const void *key, const void *value, void *oldkey, void *oldvalue)
{
    uint32_t code = zh->hash(key);
    uint32_t entry_idx = code & (zh->nentries - 1);

    while (zh->entries[entry_idx * zh->entrysz]) {
        void *this_key = &zh->entries[entry_idx * zh->entrysz + 1];
        void *this_value = &zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz];

        if (zh->equals(key, this_key)) {
            // replace
            if (oldkey)
                memcpy(oldkey, this_key, zh->keysz);
            if (oldvalue)
                memcpy(oldvalue, this_value, zh->valuesz);
            memcpy(this_key, key, zh->keysz);
            memcpy(this_value, value, zh->valuesz);
            zh->entries[entry_idx * zh->entrysz] = 1; // mark valid
            return 1;
        }

        entry_idx = (entry_idx + 1) & (zh->nentries - 1);
    }

    // add the entry
    zh->entries[entry_idx * zh->entrysz] = 1;
    memcpy(&zh->entries[entry_idx * zh->entrysz + 1], key, zh->keysz);
    memcpy(&zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz], value, zh->valuesz);
    zh->size++;

    if (zh->nentries < ZHASH_FACTOR_CRITICAL * zh->size) {
        zhash_t *newhash = zhash_create_capacity(zh->keysz, zh->valuesz,
                                                 zh->hash, zh->equals,
                                                 zh->size);

        for (int idx = 0; idx < zh->nentries; idx++) {

            if (zh->entries[idx * zh->entrysz]) {
                void *this_key = &zh->entries[idx * zh->entrysz + 1];
                void *this_value = &zh->entries[idx * zh->entrysz + 1 + zh->keysz];
                if (zhash_put(newhash, this_key, this_value, NULL, NULL))
                    assert(0); // shouldn't already be present.
            }
        }

        // play switch-a-roo
        zhash_t tmp;
        memcpy(&tmp, zh, sizeof(zhash_t));
        memcpy(zh, newhash, sizeof(zhash_t));
        memcpy(newhash, &tmp, sizeof(zhash_t));
        zhash_destroy(newhash);
    }

    return 0;
}

int zhash_remove(zhash_t *zh, const void *key, void *old_key, void *old_value)
{
    uint32_t code = zh->hash(key);
    uint32_t entry_idx = code & (zh->nentries - 1);

    while (zh->entries[entry_idx * zh->entrysz]) {
        void *this_key = &zh->entries[entry_idx * zh->entrysz + 1];
        void *this_value = &zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz];

        if (zh->equals(key, this_key)) {
            if (old_key)
                memcpy(old_key, this_key, zh->keysz);
            if (old_value)
                memcpy(old_value, this_value, zh->valuesz);

            // mark this entry as available
            zh->entries[entry_idx * zh->entrysz] = 0;
            zh->size--;

            // reinsert any consecutive entries that follow
            while (1) {
                entry_idx = (entry_idx + 1) & (zh->nentries - 1);

                if (zh->entries[entry_idx * zh->entrysz]) {
                    // completely remove this entry
                    char *tmp = malloc(sizeof(char)*zh->entrysz);
                    memcpy(tmp, &zh->entries[entry_idx * zh->entrysz], zh->entrysz);
                    zh->entries[entry_idx * zh->entrysz] = 0;
                    zh->size--;
                    // reinsert it
                    if (zhash_put(zh, &tmp[1], &tmp[1+zh->keysz], NULL, NULL))
                        assert(0);
                    free(tmp);
                } else {
                    break;
                }
            }
            return 1;
        }

        entry_idx = (entry_idx + 1) & (zh->nentries - 1);
    }

    return 0;
}

zhash_t *zhash_copy(const zhash_t *zh)
{
    zhash_t *newhash = zhash_create_capacity(zh->keysz, zh->valuesz,
                                             zh->hash, zh->equals,
                                             zh->size);

    for (int entry_idx = 0; entry_idx < zh->nentries; entry_idx++) {
        if (zh->entries[entry_idx * zh->entrysz]) {
            void *this_key = &zh->entries[entry_idx * zh->entrysz + 1];
            void *this_value = &zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz];
            if (zhash_put(newhash, this_key, this_value, NULL, NULL))
                assert(0); // shouldn't already be present.
        }
    }

    return newhash;
}

int zhash_contains(const zhash_t *zh, const void *key)
{
    void *tmp;
    return zhash_get_volatile(zh, key, &tmp);
}

void zhash_iterator_init(zhash_t *zh, zhash_iterator_t *zit)
{
    zit->zh = zh;
    zit->czh = zh;
    zit->last_entry = -1;
}

void zhash_iterator_init_const(const zhash_t *zh, zhash_iterator_t *zit)
{
    zit->zh = NULL;
    zit->czh = zh;
    zit->last_entry = -1;
}

int zhash_iterator_next_volatile(zhash_iterator_t *zit, void *outkey, void *outvalue)
{
    const zhash_t *zh = zit->czh;

    while (1) {
        if (zit->last_entry + 1 >= zh->nentries)
            return 0;

        zit->last_entry++;

        if (zh->entries[zit->last_entry * zh->entrysz]) {
            void *this_key = &zh->entries[zit->last_entry * zh->entrysz + 1];
            void *this_value = &zh->entries[zit->last_entry * zh->entrysz + 1 + zh->keysz];

            if (outkey != NULL)
                *((void**) outkey) = this_key;
            if (outvalue != NULL)
                *((void**) outvalue) = this_value;

            return 1;
        }
    }
}

int zhash_iterator_next(zhash_iterator_t *zit, void *outkey, void *outvalue)
{
    const zhash_t *zh = zit->czh;

    void *outkeyp, *outvaluep;

    if (!zhash_iterator_next_volatile(zit, &outkeyp, &outvaluep))
        return 0;

    if (outkey != NULL)
        memcpy(outkey, outkeyp, zh->keysz);
    if (outvalue != NULL)
        memcpy(outvalue, outvaluep, zh->valuesz);

    return 1;
}

void zhash_iterator_remove(zhash_iterator_t *zit)
{
    assert(zit->zh); // can't call _remove on a iterator with const zhash
    zhash_t *zh = zit->zh;

    zh->entries[zit->last_entry * zh->entrysz] = 0;
    zh->size--;

    // re-insert following entries
    int entry_idx = (zit->last_entry + 1) & (zh->nentries - 1);
    while (zh->entries[entry_idx *zh->entrysz]) {
        // completely remove this entry
        char *tmp = malloc(sizeof(char)*zh->entrysz);
        memcpy(tmp, &zh->entries[entry_idx * zh->entrysz], zh->entrysz);
        zh->entries[entry_idx * zh->entrysz] = 0;
        zh->size--;

        // reinsert it
        if (zhash_put(zh, &tmp[1], &tmp[1+zh->keysz], NULL, NULL))
            assert(0);
        free(tmp);

        entry_idx = (entry_idx + 1) & (zh->nentries - 1);
    }

    zit->last_entry--;
}

void zhash_map_keys(zhash_t *zh, void (*f)())
{
    assert(zh != NULL);
    if (f == NULL)
        return;

    zhash_iterator_t itr;
    zhash_iterator_init(zh, &itr);

    void *key, *value;

    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        f(key);
    }
}

void zhash_vmap_keys(zhash_t * zh, void (*f)())
{
    assert(zh != NULL);
    if (f == NULL)
        return;

    zhash_iterator_t itr;
    zhash_iterator_init(zh, &itr);

    void *key, *value;

    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        void *p = *(void**) key;
        f(p);
    }
}

void zhash_map_values(zhash_t * zh, void (*f)())
{
    assert(zh != NULL);
    if (f == NULL)
        return;

    zhash_iterator_t itr;
    zhash_iterator_init(zh, &itr);

    void *key, *value;
    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        f(value);
    }
}

void zhash_vmap_values(zhash_t * zh, void (*f)())
{
    assert(zh != NULL);
    if (f == NULL)
        return;

    zhash_iterator_t itr;
    zhash_iterator_init(zh, &itr);

    void *key, *value;
    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        void *p = *(void**) value;
        f(p);
    }
}

zarray_t *zhash_keys(const zhash_t *zh)
{
    assert(zh != NULL);

    zarray_t *za = zarray_create(zh->keysz);

    zhash_iterator_t itr;
    zhash_iterator_init_const(zh, &itr);

    void *key, *value;
    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        zarray_add(za, key);
    }

    return za;
}

zarray_t *zhash_values(const zhash_t *zh)
{
    assert(zh != NULL);

    zarray_t *za = zarray_create(zh->valuesz);

    zhash_iterator_t itr;
    zhash_iterator_init_const(zh, &itr);

    void *key, *value;
    while(zhash_iterator_next_volatile(&itr, &key, &value)) {
        zarray_add(za, value);
    }

    return za;
}


uint32_t zhash_uint32_hash(const void *_a)
{
    assert(_a != NULL);

    uint32_t a = *((uint32_t*) _a);
    return a;
}

int zhash_uint32_equals(const void *_a, const void *_b)
{
    assert(_a != NULL);
    assert(_b != NULL);

    uint32_t a = *((uint32_t*) _a);
    uint32_t b = *((uint32_t*) _b);

    return a==b;
}

uint32_t zhash_uint64_hash(const void *_a)
{
    assert(_a != NULL);

    uint64_t a = *((uint64_t*) _a);
    return (uint32_t) (a ^ (a >> 32));
}

int zhash_uint64_equals(const void *_a, const void *_b)
{
    assert(_a != NULL);
    assert(_b != NULL);

    uint64_t a = *((uint64_t*) _a);
    uint64_t b = *((uint64_t*) _b);

    return a==b;
}


union uintpointer
{
    const void *p;
    uint32_t i;
};

uint32_t zhash_ptr_hash(const void *a)
{
    assert(a != NULL);

    union uintpointer ip;
    ip.p = * (void**)a;

    // compute a hash from the lower 32 bits of the pointer (on LE systems)
    uint32_t hash = ip.i;
    hash ^= (hash >> 7);

    return hash;
}


int zhash_ptr_equals(const void *a, const void *b)
{
    assert(a != NULL);
    assert(b != NULL);

    const void * ptra = * (void**)a;
    const void * ptrb = * (void**)b;
    return  ptra == ptrb;
}


int zhash_str_equals(const void *_a, const void *_b)
{
    assert(_a != NULL);
    assert(_b != NULL);

    char *a = * (char**)_a;
    char *b = * (char**)_b;

    return !strcmp(a, b);
}

uint32_t zhash_str_hash(const void *_a)
{
    assert(_a != NULL);

    char *a = * (char**)_a;

    int32_t hash = 0;
    while (*a != 0) {
        hash = (hash << 7) + (hash >> 23);
        hash += *a;
        a++;
    }

    return (uint32_t) hash;
}


void zhash_debug(zhash_t *zh)
{
    for (int entry_idx = 0; entry_idx < zh->nentries; entry_idx++) {
        char *k, *v;
        memcpy(&k, &zh->entries[entry_idx * zh->entrysz + 1], sizeof(char*));
        memcpy(&v, &zh->entries[entry_idx * zh->entrysz + 1 + zh->keysz], sizeof(char*));
        printf("%d: %d, %s => %s\n", entry_idx, zh->entries[entry_idx * zh->entrysz], k, v);
    }
}
