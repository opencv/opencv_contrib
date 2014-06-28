/* dynamic array of 32-bit integers
 * arr[0]   : array size
 * arr[1]   : array capacity
 * arr[2..] : array content */

#ifndef __OPENCV_ARRAY32_HPP
#define __OPENCV_ARRAY32_HPP

#include "types.hpp"

class Array32 {

 private:
    static double ARRAY_RESIZE_FACTOR;
    static double ARRAY_RESIZE_ADD_FACTOR;

 public:
    /* set ARRAY_RESIZE_FACTOR */
    static void setArrayResizeFactor(double arf);

    /* constructor */
    Array32();

    /* destructor */
    ~Array32();

    /* cleaning function used in destructor */
    void cleanup();

    /* push data */
    void push(UINT32 data);

    /* insert data at given index */
    void insert(UINT32 index, UINT32 data);

    /* return data */
    UINT32* data();

    /* return data size */
    UINT32 size();

    /* return capacity */
    UINT32 capacity();

    /* definition of operator = */
    void operator= (const Array32&);

    /* print data */
    void print();

    /* initializer */
    void init(int size);

    /* data */
    UINT32 *arr;


};

#endif
