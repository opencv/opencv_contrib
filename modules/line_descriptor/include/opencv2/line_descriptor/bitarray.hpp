#ifndef __OPENCV_BITARRAY_HPP
#define __OPENCV_BITARRAY_HPP

#include "types.hpp"
#include <stdio.h>
#include <math.h>
#include <string.h>

/* class defining a sequence of bits */
class bitarray {

 public:
    /* pointer to bits sequence and sequence's length */
    UINT32 *arr;
    UINT32 length;
	
    /* constructor setting default values */
    bitarray()
    {
        arr = NULL;
        length = 0;
    }
	
    /* constructor setting sequence's length */
    bitarray(UINT64 _bits) {
        init(_bits);
    }
	
    /* initializer of private fields */
    void init(UINT64 _bits)
    {
        length = (UINT32)ceil(_bits/32.00);
        arr = new UINT32[length];
        erase();
    }
	
    /* destructor */
    ~bitarray() {
        if (arr)
            delete[] arr;
    }
	
    inline void flip(UINT64 index)
    {
        arr[index >> 5] ^= ((UINT32)0x01) << (index % 32);
    }

    inline void set(UINT64 index)
    {
        arr[index >> 5] |= ((UINT32)0x01) << (index % 32);
    }
	
    inline UINT8 get(UINT64 index)
    {
        return (arr[index >> 5] & (((UINT32)0x01) << (index % 32))) != 0;
    }
	
    /* reserve menory for an UINT32 */
    inline void erase()
    {
        memset(arr, 0, sizeof(UINT32) * length);
    }

};

#endif
