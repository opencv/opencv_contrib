/*-----------------------------------------------------------------------------
* MurmurHash3 was written by Austin Appleby, and is placed in the public
* domain.
*
* This implementation was written by Shane Day, and is also public domain.
*
* This is a portable ANSI C implementation of MurmurHash3_x86_32 (Murmur3A)
* with support for progressive processing.
*/

/* ------------------------------------------------------------------------- */
/* Determine what native type to use for uint32_t */

/* We can't use the name 'uint32_t' here because it will conflict with
* any version provided by the system headers or application. */

#if !defined(ulong)
#define ulong unsigned long
#endif

/* First look for special cases */
#if defined(_MSC_VER)
#define MH_UINT32 ulong
#endif

/* If the compiler says it's C99 then take its word for it */
#if !defined(MH_UINT32) && ( \
 defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L )
#include <stdint.h>
#define MH_UINT32 uint32_t
#endif

/* Otherwise try testing against max value macros from limit.h */
#if !defined(MH_UINT32)
#include  <limits.h>
#if   (USHRT_MAX == 0xffffffffUL)
#define MH_UINT32 ushort
#elif (UINT_MAX == 0xffffffffUL)
#define MH_UINT32 uint
#elif (ULONG_MAX == 0xffffffffUL)
#define MH_UINT32 ulong
#endif
#endif

#if !defined(MH_UINT32)
#error Unable to determine type name for u32-bit int
#endif

/* I'm yet to work on a platform where 'uchar' is not 8 bits */
#define MH_UINT8  uchar

void PMurHash32_Process(MH_UINT32 *ph1, MH_UINT32 *pcarry, const void *key, int len);
MH_UINT32 PMurHash32_Result(MH_UINT32 h1, MH_UINT32 carry, MH_UINT32 total_length);
MH_UINT32 PMurHash32(MH_UINT32 seed, const void *key, int len);
void hashMurmurx86 ( const void * key, const int len, const uint seed, void * out );

/* I used ugly type names in the header to avoid potential conflicts with
* application or system typedefs & defines. Since I'm not including any more
* headers below here I can rename these so that the code reads like C99 */
#undef uint32_t
#define uint32_t MH_UINT32
#undef uint8_t
#define uint8_t  MH_UINT8

/* MSVC warnings we choose to ignore */
#if defined(_MSC_VER)
#pragma warning(disable: 4127) /* conditional expression is constant */
#endif

/*-----------------------------------------------------------------------------
* Endianess, misalignment capabilities and util macros
*
* The following 3 macros are defined in this section. The other macros defined
* are only needed to help derive these 3.
*
* READ_UINT32(x)   Read a little endian u32-bit int
* UNALIGNED_SAFE   Defined if READ_UINT32 works on non-word boundaries
* ROTL32(x,r)      Rotate x left by r bits
*/

/* Convention is to define __BYTE_ORDER == to one of these values */
#if !defined(__BIG_ENDIAN)
#define __BIG_ENDIAN 4321
#endif
#if !defined(__LITTLE_ENDIAN)
#define __LITTLE_ENDIAN 1234
#endif

/* I386 */
#if defined(_M_IX86) || defined(__i386__) || defined(__i386) || defined(i386)
#define __BYTE_ORDER __LITTLE_ENDIAN
#define UNALIGNED_SAFE
#endif

/* gcc 'may' define __LITTLE_ENDIAN__ or __BIG_ENDIAN__ to 1 (Note the trailing __),
* or even _LITTLE_ENDIAN or _BIG_ENDIAN (Note the single _ prefix) */
#if !defined(__BYTE_ORDER)
#if defined(__LITTLE_ENDIAN__) && __LITTLE_ENDIAN__==1 || defined(_LITTLE_ENDIAN) && _LITTLE_ENDIAN==1
#define __BYTE_ORDER __LITTLE_ENDIAN
#elif defined(__BIG_ENDIAN__) && __BIG_ENDIAN__==1 || defined(_BIG_ENDIAN) && _BIG_ENDIAN==1
#define __BYTE_ORDER __BIG_ENDIAN
#endif
#endif

/* gcc (usually) defines xEL/EB macros for ARM and MIPS endianess */
#if !defined(__BYTE_ORDER)
#if defined(__ARMEL__) || defined(__MIPSEL__)
#define __BYTE_ORDER __LITTLE_ENDIAN
#endif
#if defined(__ARMEB__) || defined(__MIPSEB__)
#define __BYTE_ORDER __BIG_ENDIAN
#endif
#endif

/* Now find best way we can to READ_UINT32 */
#if __BYTE_ORDER==__LITTLE_ENDIAN
/* CPU endian matches murmurhash algorithm, so read 32-bit word directly */
#define READ_UINT32(ptr)   (*((uint32_t*)(ptr)))
#elif __BYTE_ORDER==__BIG_ENDIAN
/* TODO: Add additional cases below where a compiler provided bswap32 is available */
#if defined(__GNUC__) && (__GNUC__>4 || (__GNUC__==4 && __GNUC_MINOR__>=3))
#define READ_UINT32(ptr)   (__builtin_bswap32(*((uint32_t*)(ptr))))
#else
/* Without a known fast bswap32 we're just as well off doing this */
#define READ_UINT32(ptr)   (ptr[0]|ptr[1]<<8|ptr[2]<<16|ptr[3]<<24)
#define UNALIGNED_SAFE
#endif
#else
/* Unknown endianess so last resort is to read individual bytes */
#define READ_UINT32(ptr)   (ptr[0]|ptr[1]<<8|ptr[2]<<16|ptr[3]<<24)

/* Since we're not doing word-reads we can skip the messing about with realignment */
#define UNALIGNED_SAFE
#endif

/*-----------------------------------------------------------------------------
* Core murmurhash algorithm macros */

#define C1  (0xcc9e2d51)
#define C2  (0x1b873593)

/* This is the main processing body of the algorithm. It operates
* on each full 32-bits of input. */
#define DOBLOCK(h1, k1) do{ \
 k1 *= C1; \
      k1 = ROTL32(k1,15); \
           k1 *= C2; \
                 \
                 h1 ^= k1; \
                       h1 = ROTL32(h1,13); \
                            h1 = h1*5+0xe6546b64; \
                                 }while (0)


/* Append unaligned bytes to carry, forcing hash churn if we have 4 bytes */
/* cnt=bytes to process, h1=name of h1 var, c=carry, n=bytes in c, ptr/len=payload */
#define DOBYTES(cnt, h1, c, n, ptr, len) do{ \
                                  int _i = cnt; \
                                                                           while (_i--){    \
        c = c>>8 | *ptr++<<24;    \
        n++;    len--;    \
        if (n==4)        {            \
                        DOBLOCK(h1, c);            \
                        n = 0;            \
                    } \
            }}while (0)

/*---------------------------------------------------------------------------*/

/* Main hashing function. Initialise carry to 0 and h1 to 0 or an initial seed
* if wanted. Both ph1 and pcarry are required arguments. */
void PMurHash32_Process(uint32_t *ph1, uint32_t *pcarry, const void *key, int len)
{
    uint32_t h1 = *ph1;
    uint32_t c = *pcarry;

    const uint8_t *ptr = (uint8_t*)                         key;
    const uint8_t *end;

    /* Extract carry count from low 2 bits of c value */
    int n = c & 3;

#if defined(UNALIGNED_SAFE)
    /* This CPU handles unaligned word access */

    /* Consume any carry bytes */
    int i = (4-n) & 3;
    if (i && i <= len)
    {
        DOBYTES(i, h1, c, n, ptr, len);
    }

    /* Process 32-bit chunks */
    end = ptr + len/4*4;
    for ( ; ptr < end ; ptr+=4)
    {
        uint32_t k1 = READ_UINT32(ptr);
        DOBLOCK(h1, k1);
    }

#else /*UNALIGNED_SAFE*/
    /* This CPU does not handle unaligned word access */

    /* Consume enough so that the next data byte is word aligned */
    int i = -(long)ptr & 3;
    if (i && i <= len)
    {
        DOBYTES(i, h1, c, n, ptr, len);
    }

    /* We're now aligned. Process in aligned blocks. Specialise for each possible carry count */
    end = ptr + len/4*4;
    switch (n)
    {
            /* how many bytes in c */
        case 0: /* c=[----]  w=[3210]  b=[3210]=w            c'=[----] */
            for ( ; ptr < end ; ptr+=4)
            {
                uint32_t k1 = READ_UINT32(ptr);
                DOBLOCK(h1, k1);
            }
            break;
        case 1: /* c=[0---]  w=[4321]  b=[3210]=c>>24|w<<8   c'=[4---] */
            for ( ; ptr < end ; ptr+=4)
            {
                uint32_t k1 = c>>24;
                c = READ_UINT32(ptr);
                k1 |= c<<8;
                DOBLOCK(h1, k1);
            }
            break;
        case 2: /* c=[10--]  w=[5432]  b=[3210]=c>>16|w<<16  c'=[54--] */
            for ( ; ptr < end ; ptr+=4)
            {
                uint32_t k1 = c>>16;
                c = READ_UINT32(ptr);
                k1 |= c<<16;
                DOBLOCK(h1, k1);
            }
            break;
        case 3: /* c=[210-]  w=[6543]  b=[3210]=c>>8|w<<24   c'=[654-] */
            for ( ; ptr < end ; ptr+=4)
            {
                uint32_t k1 = c>>8;
                c = READ_UINT32(ptr);
                k1 |= c<<24;
                DOBLOCK(h1, k1);
            }
    }
#endif /*UNALIGNED_SAFE*/

    /* Advance over whole 32-bit chunks, possibly leaving 1..3 bytes */
    len -= len/4*4;

    /* Append any remaining bytes into carry */
    DOBYTES(len, h1, c, n, ptr, len);

    /* Copy out new running hash and carry */
    *ph1 = h1;
    *pcarry = (c & ~0xff) | n;
}

/*---------------------------------------------------------------------------*/

/* Finalize a hash. To match the original Murmur3A the total_length must be provided */
uint32_t PMurHash32_Result(uint32_t h, uint32_t carry, uint32_t total_length)
{
    uint32_t k1;
    int n = carry & 3;
    if (n)
    {
        k1 = carry >> (4-n)*8;
        k1 *= C1;
        k1 = ROTL32(k1,15);
        k1 *= C2;
        h ^= k1;
    }
    h ^= total_length;

    /* fmix */
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

/*---------------------------------------------------------------------------*/

/* Murmur3A compatable all-at-once */
uint32_t PMurHash32(uint32_t seed, const void *key, int len)
{
    uint32_t h1=seed, carry=0;
    PMurHash32_Process(&h1, &carry, key, len);
    return PMurHash32_Result(h1, carry, len);
}

void hashMurmurx86 ( const void * key, const int len, const uint seed, void * out )
{
    *(uint*)out = PMurHash32 (seed, key, len);
}

