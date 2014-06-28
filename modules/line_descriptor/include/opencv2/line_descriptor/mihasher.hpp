#ifndef __OPENCV_MIHASHER_HPP
#define __OPENCV_MIHASHER_HPP

#include "types.hpp"
#include "bitops.hpp"

#include "sparse_hashtable.hpp"
#include "bitarray.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

//#include <iostream>
//#include <math.h>


class Mihasher {
 private:

    /* Bits per code */
    int B;

    /* B/8 */
    int B_over_8;

    /* Bits per chunk (must be less than 64) */
    int b;

    /* Number of chunks */
    int m;

    /* Number of chunks with b bits (have 1 bit more than others) */
    int mplus;

    /* Maximum hamming search radius (we use B/2 by default) */
    int D;

    /* Maximum hamming search radius per substring */
    int d;

    /* Maximum results to return */
    int K;

    /* Number of codes */
    UINT64 N;
    
    /* Table of original full-length codes */
    cv::Mat codes;

    /* Counter for eliminating duplicate results (it is not thread safe) */
    bitarray *counter;
    
    /* Array of m hashtables */
    SparseHashtable *H;
        
    /* Volume of a b-bit Hamming ball with radius s (for s = 0 to d) */
    UINT32 *xornum;

    /* Used within generation of binary codes at a certain Hamming distance */
    int power[100];

 public:
    
    /* constructor */
    Mihasher();

    /* desctructor */
    ~Mihasher();

    /* constructor 2 */
    Mihasher(int B, int m);

    /* K setter */
    void setK(int K);

    /* populate tables */
    void populate(cv::Mat & codes, UINT32 N, int dim1codes);

    /* execute a batch query */
    void batchquery (UINT32 * results, UINT32 *numres/*, qstat *stats*/,const cv::Mat & q, UINT32 numq, int dim1queries);
    
 private:

    /* execute a single query */
    void query(UINT32 * results, UINT32* numres/*, qstat *stats*/, UINT8 *q, UINT64 * chunks, UINT32 * res, int query_i);
};

#endif
