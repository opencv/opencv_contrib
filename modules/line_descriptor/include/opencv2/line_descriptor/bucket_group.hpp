#ifndef __OPENCV_BUCKET_GROUP_HPP
#define __OPENCV_BUCKET_GROUP_HPP

#include "types.hpp"
#include "array32.hpp"
#include "bitarray.hpp"

class BucketGroup {

 public:
    /* constructor */
    BucketGroup();

    /* destructor */
    ~BucketGroup();

    /* insert data into the bucket */
    void insert(int subindex, UINT32 data);

    /* perform a query to the bucket */
    UINT32* query(int subindex, int *size);

    /* data fields */
    UINT32 empty;
    Array32 *group;

};

#endif
