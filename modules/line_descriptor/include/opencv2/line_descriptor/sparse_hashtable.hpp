#ifndef __OPENCV_SPARSE_HASHTABLE_HPP
#define __OPENCV_SPARSE_HASHTABLE_HPP

#include "types.hpp"
#include "bucket_group.hpp"

class SparseHashtable
{

 private:

    /* Maximum bits per key before folding the table */
    static const int MAX_B;

    /* Bins (each bin is an Array object for duplicates of the same key) */
    BucketGroup *table;

 public:

    /* constructor */
    SparseHashtable();

    /* destructor */
    ~SparseHashtable();

    /* initializer */
    int init(int _b);
	
    /* insert data */
    void insert(UINT64 index, UINT32 data);

    /* query data */
    UINT32* query(UINT64 index, int* size);

    /* Bits per index */
    int b;

    /*  Number of bins */
    UINT64 size;

};

#endif
