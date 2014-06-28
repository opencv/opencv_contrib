#ifndef __OPENCV_BITOPTS_HPP
#define __OPENCV_BITOPTS_HPP

#define popcntll __builtin_popcountll
#define popcnt __builtin_popcount

#include "precomp.hpp"

/* LUT */
const int lookup [] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,
                       2,3,2,3,3,4,2,3,3,4,3,4,4,5,1,2,2,3,
                       2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,
                       4,5,3,4,4,5,4,5,5,6,1,2,2,3,2,3,3,4,
                       2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,
                       4,5,4,5,5,6,2,3,3,4,3,4,4,5,3,4,4,5,
                       4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,
                       6,7,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
                       2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,2,3,
                       3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,
                       4,5,5,6,4,5,5,6,5,6,6,7,2,3,3,4,3,4,
                       4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,
                       4,5,5,6,5,6,6,7,3,4,4,5,4,5,5,6,4,5,
                       5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,
                       6,7,7,8};

/*matching function */
inline int match(UINT8*P, UINT8*Q, int codelb)
{
    switch(codelb)
    {
        case 4: // 32 bit
            return popcnt(*(UINT32*)P ^ *(UINT32*)Q);
            break;
        case 8: // 64 bit
            return popcntll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]);
            break;
        case 16: // 128 bit
            return popcntll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]) \
                + popcntll(((UINT64*)P)[1] ^ ((UINT64*)Q)[1]);
            break;
        case 32: // 256 bit
            return popcntll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]) \
                + popcntll(((UINT64*)P)[1] ^ ((UINT64*)Q)[1]) \
                + popcntll(((UINT64*)P)[2] ^ ((UINT64*)Q)[2]) \
                + popcntll(((UINT64*)P)[3] ^ ((UINT64*)Q)[3]);
            break;
        case 64: // 512 bit
            return popcntll(((UINT64*)P)[0] ^ ((UINT64*)Q)[0]) \
                + popcntll(((UINT64*)P)[1] ^ ((UINT64*)Q)[1]) \
                + popcntll(((UINT64*)P)[2] ^ ((UINT64*)Q)[2]) \
                + popcntll(((UINT64*)P)[3] ^ ((UINT64*)Q)[3]) \
                + popcntll(((UINT64*)P)[4] ^ ((UINT64*)Q)[4]) \
                + popcntll(((UINT64*)P)[5] ^ ((UINT64*)Q)[5]) \
                + popcntll(((UINT64*)P)[6] ^ ((UINT64*)Q)[6]) \
                + popcntll(((UINT64*)P)[7] ^ ((UINT64*)Q)[7]);
            break;
        default:
            int output = 0;
            for (int i=0; i<codelb; i++)
                output+= lookup[P[i] ^ Q[i]];
            return output;
            break;
    }

    return -1;
}

/* splitting function (b <= 64) */
inline void split (UINT64 *chunks, UINT8 *code, int m, int mplus, int b)
{
    UINT64 temp = 0x0;
    int nbits = 0;
    int nbyte = 0;
    UINT64 mask = b==64 ? 0xFFFFFFFFFFFFFFFFLLU : ((UINT64_1 << b) - UINT64_1);

    for (int i=0; i<m; i++)
    {
        while (nbits < b)
        {
            temp |= ((UINT64)code[nbyte++] << nbits);
            nbits += 8;
        }

        chunks[i] = temp & mask;
        temp = b==64 ? 0x0 : temp >> b;
        nbits -= b;

        if (i == mplus-1)
        {
            b--; /* b <= 63 */
            mask = ((UINT64_1 << b) - UINT64_1);
        }
    }
}

/* generates the next binary code (in alphabetical order) with the
   same number of ones as the input x. Taken from
   http://www.geeksforgeeks.org/archives/10375 */
inline UINT64 next_set_of_n_elements(UINT64 x)
{
    UINT64 smallest, ripple, new_smallest;

    smallest     = x & -x;
    ripple       = x + smallest;
    new_smallest = x ^ ripple;
    new_smallest = new_smallest / smallest;
    new_smallest >>= 2;
    return ripple | new_smallest;
}

/* print code */
inline void print_code(UINT64 tmp, int b)
{
    for (int j=(b-1); j>=0; j--)
    {
        printf("%llu", (long long int) tmp/(1 << j));
        tmp = tmp - (tmp/(1 << j)) * (1 << j);
    }

    printf("\n");
}

inline UINT64 choose(int n, int r)
{
    UINT64 nchooser = 1;
    for (int k=0; k < r; k++)
    {
        nchooser *= n-k;
        nchooser /= k+1;
    }

    return nchooser;
}

#endif
