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

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "zarray.h"


/**
 * A hash table for structs and primitive types that stores entries by value.
 *   - The size of the key/values must be known at instantiation time, and remain fixed.
 *     e.g. for pointers: zhash_create(sizeof(void*), sizeof(void*)....)
 *          for structs: zhash_create(sizeof(struct key_struct), sizeof(struct value_struct)...)
 *          for bytes: zhash_create(sizeof(uint8_t), sizeof(uint8_t)...)
 *   - Entries are copied by value. This means you must always pass a reference to the start
 *     of 'key_size' and 'value_size' bytes, which you have already malloc'd or stack allocated
 *   - This data structure can be used to store types of any size, from bytes & doubles to
 *     user defined structs
 *     Note: if zhash stores pointers, user must be careful to manually manage the lifetime
 *     of the memory they point to.
 *
 */

typedef struct zhash zhash_t;

// The contents of the iterator should be considered private. However,
// since our usage model prefers stack-based allocation of iterators,
// we must publicly declare them.
struct zhash_iterator
{
    zhash_t *zh;
    const zhash_t *czh;
    int last_entry; // points to the last entry returned by _next
};

typedef struct zhash_iterator zhash_iterator_t;

/**
 * Create, initializes, and returns an empty hash table structure. It is the
 * caller's responsibility to call zhash_destroy() on the returned array when it
 * is no longer needed.
 *
 * The size of values used in the hash and equals function must match 'keysz'.
 * I.e. if keysz = sizeof(uint64_t), then hash() and equals() should accept
 * parameters as *uint64_t.
 */
zhash_t *zhash_create(size_t keysz, size_t valuesz,
                      uint32_t(*hash)(const void *a),
                      int(*equals)(const void *a, const void *b));

/**
 * Frees all resources associated with the hash table structure which was
 * created by zhash_create(). After calling, 'zh' will no longer be valid for storage.
 *
 * If 'zh' contains pointer data, it is the caller's responsibility to manage
 * the resources pointed to by those pointers.
 */
void zhash_destroy(zhash_t *zh);

/**
 * Creates and returns a new identical copy of the zhash (i.e. a "shallow" copy).
 * If you're storing pointers, be sure not to double free their pointees!
 * It is the caller's responsibility to call zhash_destroy() on the returned array
 * when it is no longer needed (in addition to the zhash_destroy() call for the
 * original zhash).
 */
zhash_t * zhash_copy(const zhash_t* other);

/**
 * Determines whether the supplied key value exists as an entry in the zhash
 * table. If zhash stores pointer types as keys, this function can differentiate
 * between a non-existent key and a key mapped to NULL.
 * Returns 1 if the supplied key exists in the zhash table, else 0.
 */
int zhash_contains(const zhash_t *zh, const void *key);

/**
 * Retrieves the value for the given key, if it exists, by copying its contents
 * into the space pointed to by 'out_value', which must already be allocated.
 * Returns 1 if the supplied key exists in the table, else 0, in which case
 * the contents of 'out_value' will be unchanged.
 */
int zhash_get(const zhash_t *zh, const void *key, void *out_value);

/**
 * Similar to zhash_get(), but more dangerous. Provides a pointer to the zhash's
 * internal storage.  This can be used to make simple modifications to
 * the underlying data while avoiding the memcpys associated with
 * zhash_get and zhash_put. However, some zhash operations (that
 * resize the underlying storage, in particular) render this pointer
 * invalid. For maximum safety, call no other zhash functions for the
 * period during which you intend to use the pointer.
 * 'out_p' should be a pointer to the pointer which will be set to the internal
 * data address.
 */
int zhash_get_volatile(const zhash_t *zh, const void *key, void *out_p);

/**
 * Adds a key/value pair to the hash table, if the supplied key does not already
 * exist in the table, or overwrites the value for the supplied key if it does
 * already exist. In the latter case, the previous contents of the key and value
 * will be copied into the spaces pointed to by 'oldkey' and 'oldvalue', respectively,
 * if they are not NULL.
 *
 * The key/value is added to / updated in the hash table by copying 'keysz' bytes
 * from the data pointed to by 'key' and 'valuesz' bytes from the data pointed
 * to by 'value'. It is up to the caller to manage the memory allocation of the
 * passed-in values, zhash will store and manage a copy.
 *
 * NOTE: If the key is a pointer type (such as a string), the contents of the
 * data that it points to must not be modified after the call to zhash_put(),
 * or future zhash calls will not successfully locate the key (using either its
 * previous or new value).
 *
 * NOTE: When using array data as a key (such as a string), the array should not
 * be passed directly or it will cause a segmentation fault when it is dereferenced.
 * Instead, pass a pointer which points to the array location, i.e.:
 *   char key[strlen];
 *   char *keyptr = key;
 *   zhash_put(zh, &keyptr, ...)
 *
 * Example:
 *   char * key = ...;
 *   zarray_t * val = ...;
 *   char * old_key = NULL;
 *   zarray_t * old_val = NULL;
 *   if (zhash_put(zh, &key, &val, &old_key, &old_value))
 *       // manage resources for old_key and old_value
 *
 * Returns 1 if the supplied key previously existed in the table, else 0, in
 * which case the data pointed to by 'oldkey' and 'oldvalue' will be set to zero
 * if they are not NULL.
 */
int zhash_put(zhash_t *zh, const void *key, const void *value, void *oldkey, void *oldvalue);

/**
 * Removes from the zhash table the key/value pair for the supplied key, if
 * it exists. If it does, the contents of the key and value will be copied into
 * the spaces pointed to by 'oldkey' and 'oldvalue', respectively, if they are
 * not NULL. If the key does not exist, the data pointed to by 'oldkey' and
 * 'oldvalue' will be set to zero if they are not NULL.
 *
 * Returns 1 if the key existed and was removed, else 0, indicating that the
 * table contents were not changed.
 */
int zhash_remove(zhash_t *zh, const void *key, void *oldkey, void *oldvalue);

/**
 * Removes all entries in the has table to create the equivalent of starting from
 * a zhash_create(), using the same size parameters. If any elements need to be
 * freed manually, this will need to occur before calling clear.
 */
void zhash_clear(zhash_t *zh);

/**
 * Retrieves the current number of key/value pairs currently contained in the
 * zhash table, or 0 if the table is empty.
 */
int zhash_size(const zhash_t *zh);

/**
 * Initializes an iterator which can be used to traverse the key/value pairs of
 * the supplied zhash table via successive calls to zhash_iterator_next() or
 * zhash_iterator_next_volatile(). The iterator can also be used to remove elements
 * from the zhash with zhash_iterator_remove().
 *
 * Any modifications to the zhash table structure will invalidate the
 * iterator, with the exception of zhash_iterator_remove().
 */
void zhash_iterator_init(zhash_t *zh, zhash_iterator_t *zit);

/**
 * Initializes an iterator which can be used to traverse the key/value pairs of
 * the supplied zhash table via successive calls to zhash_iterator_next() or
 * zhash_iterator_next_volatile().
 *
 * An iterator initialized with this function cannot be used with
 * zhash_iterator_remove(). For that you must use zhash_iterator_init().
 *
 * Any modifications to the zhash table structure will invalidate the
 * iterator.
 */
void zhash_iterator_init_const(const zhash_t *zh, zhash_iterator_t *zit);

/**
 * Retrieves the next key/value pair from a zhash table via the (previously-
 * initialized) iterator. Copies the key and value data into the space
 * pointed to by outkey and outvalue, respectively, if they are not NULL.
 *
 * Returns 1 if the call retrieved the next available key/value pair, else 0
 * indicating that no entries remain, in which case the contents of outkey and
 * outvalue will remain unchanged.
 */
int zhash_iterator_next(zhash_iterator_t *zit, void *outkey, void *outvalue);

/**
 * Similar to zhash_iterator_next() except that it retrieves a pointer to zhash's
 * internal storage.  This can be used to avoid the memcpys associated with
 * zhash_iterator_next(). Call no other zhash functions for the
 * period during which you intend to use the pointer.
 * 'outkey' and 'outvalue' should be pointers to the pointers which will be set
 * to the internal data addresses.
 *
 * Example:
 *   key_t *outkey;
 *   value_t *outvalue;
 *   if (zhash_iterator_next_volatile(&zit, &outkey, &outvalue))
 *       // access internal key and value storage via outkey and outvalue
 *
 * Returns 1 if the call retrieved the next available key/value pair, else 0
 * indicating that no entries remain, in which case the pointers outkey and
 * outvalue will remain unchanged.
 */
int zhash_iterator_next_volatile(zhash_iterator_t *zit, void *outkey, void *outvalue);

/**
 * Removes from the zhash table the key/value pair most recently returned via
 * a call to zhash_iterator_next() or zhash_iterator_next_volatile() for the
 * supplied iterator.
 *
 * Requires that the iterator was initialized with zhash_iterator_init(),
 * not zhash_iterator_init_const().
 */
void zhash_iterator_remove(zhash_iterator_t *zit);

/**
 * Calls the supplied function with a pointer to every key in the hash table in
 * turn. The function will be passed a pointer to the table's internal storage
 * for the key, which the caller should not modify, as the hash table will not be
 * re-indexed. The function may be NULL, in which case no action is taken.
 */
void zhash_map_keys(zhash_t *zh, void (*f)());

/**
 * Calls the supplied function with a pointer to every value in the hash table in
 * turn. The function will be passed a pointer to the table's internal storage
 * for the value, which the caller may safely modify. The function may be NULL,
 * in which case no action is taken.
 */
void zhash_map_values(zhash_t *zh, void (*f)());

/**
 * Calls the supplied function with a copy of every key in the hash table in
 * turn. While zhash_map_keys() passes a pointer to internal storage, this function
 * passes a copy of the actual storage. If the zhash stores pointers to data,
 * functions like free() can be used directly with zhash_vmap_keys().
 * The function may be NULL, in which case no action is taken.
 *
 * NOTE: zhash_vmap_keys() can only be used with pointer-data keys.
 * Use with non-pointer keys (i.e. integer, double, etc.) will likely cause a
 * segmentation fault.
 */
void zhash_vmap_keys(zhash_t *vh, void (*f)());

/**
 * Calls the supplied function with a copy of every value in the hash table in
 * turn. While zhash_map_values() passes a pointer to internal storage, this function
 * passes a copy of the actual storage. If the zhash stores pointers to data,
 * functions like free() can be used directly with zhash_vmap_values().
 * The function may be NULL, in which case no action is taken.
 *
 * NOTE: zhash_vmap_values() can only be used with pointer-data values.
 * Use with non-pointer values (i.e. integer, double, etc.) will likely cause a
 * segmentation fault.
 */
void zhash_vmap_values(zhash_t *vh, void (*f)());

/**
 * Returns an array which contains copies of all of the hash table's keys, in no
 * particular order. It is the caller's responsibility to call zarray_destroy()
 * on the returned structure when it is no longer needed.
 */
zarray_t *zhash_keys(const zhash_t *zh);

/**
 * Returns an array which contains copies of all of the hash table's values, in no
 * particular order. It is the caller's responsibility to call zarray_destroy()
 * on the returned structure when it is no longer needed.
 */
zarray_t *zhash_values(const zhash_t *zh);

/**
 * Defines a hash function which will calculate a zhash value for uint32_t input
 * data. Can be used with zhash_create() for a key size of sizeof(uint32_t).
 */
uint32_t zhash_uint32_hash(const void *a);

/**
 * Defines a function to compare zhash values for uint32_t input data.
 * Can be used with zhash_create() for a key size of sizeof(uint32_t).
 */
int zhash_uint32_equals(const void *a, const void *b);

/**
 * Defines a hash function which will calculate a zhash value for uint64_t input
 * data. Can be used with zhash_create() for a key size of sizeof(uint64_t).
 */
uint32_t zhash_uint64_hash(const void *a);

/**
 * Defines a function to compare zhash values for uint64_t input data.
 * Can be used with zhash_create() for a key size of sizeof(uint64_t).
 */
int zhash_uint64_equals(const void *a, const void *b);

/////////////////////////////////////////////////////
// functions for keys that can be compared via their pointers.
/**
 * Defines a hash function which will calculate a zhash value for pointer input
 * data. Can be used with zhash_create() for a key size of sizeof(void*). Will
 * use only the pointer value itself for computing the hash value.
 */
uint32_t zhash_ptr_hash(const void *a);

/**
 * Defines a function to compare zhash values for pointer input data.
 * Can be used with zhash_create() for a key size of sizeof(void*).
 */
int zhash_ptr_equals(const void *a, const void *b);

/////////////////////////////////////////////////////
// Functions for string-typed keys
/**
 * Defines a hash function which will calculate a zhash value for string input
 * data. Can be used with zhash_create() for a key size of sizeof(char*). Will
 * use the contents of the string in computing the hash value.
 */
uint32_t zhash_str_hash(const void *a);

/**
 * Defines a function to compare zhash values for string input data.
 * Can be used with zhash_create() for a key size of sizeof(char*).
 */
int zhash_str_equals(const void *a, const void *b);

void zhash_debug(zhash_t *zh);

    static inline zhash_t *zhash_str_str_create(void)
    {
        return zhash_create(sizeof(char*), sizeof(char*),
                            zhash_str_hash, zhash_str_equals);
    }



// for zhashes that map strings to strings, this is a convenience
// function that allows easier retrieval of values. NULL is returned
// if the key is not found.
static inline char *zhash_str_str_get(zhash_t *zh, const char *key)
{
    char *value;
    if (zhash_get(zh, &key, &value))
        return value;
    return NULL;
}

    static inline void zhash_str_str_put(zhash_t *zh, char *key, char *value)
    {
        char *oldkey, *oldval;
        if (zhash_put(zh, &key, &value, &oldkey, &oldval)) {
            free(oldkey);
            free(oldval);
        }
    }

    static inline void zhash_str_str_destroy(zhash_t *zh)
    {
        zhash_iterator_t zit;
        zhash_iterator_init(zh, &zit);

        char *key, *value;
        while (zhash_iterator_next(&zit, &key, &value)) {
            free(key);
            free(value);
        }

        zhash_destroy(zh);
    }


static inline uint32_t zhash_int_hash(const void *_a)
{
    assert(_a != NULL);

    uint32_t a = *((int*) _a);
    return a;
}

static inline int zhash_int_equals(const void *_a, const void *_b)
{
    assert(_a != NULL);
    assert(_b != NULL);

    int a = *((int*) _a);
    int b = *((int*) _b);

    return a==b;
}

#ifdef __cplusplus
}
#endif
