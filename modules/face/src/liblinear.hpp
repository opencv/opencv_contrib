/*
Copyright (c) 2007-2015 The LIBLINEAR Project.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* blas.h  --  C header file for BLAS                         Ver 1.0 */
/* Jesse Bennett                                       March 23, 2000 */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

namespace liblinear
{

#ifndef BLAS_INCLUDE
#define BLAS_INCLUDE

/* Data types specific to BLAS implementation */
typedef struct { float r, i; } fcomplex;
typedef struct { double r, i; } dcomplex;
typedef int blasbool;

#ifdef __cplusplus
extern "C" {
#endif

#ifdef F2C_COMPAT

void cdotc_(fcomplex *dotval, int *n, fcomplex *cx, int *incx,
            fcomplex *cy, int *incy);

void cdotu_(fcomplex *dotval, int *n, fcomplex *cx, int *incx,
            fcomplex *cy, int *incy);

double sasum_(int *n, float *sx, int *incx);

double scasum_(int *n, fcomplex *cx, int *incx);

double scnrm2_(int *n, fcomplex *x, int *incx);

double sdot_(int *n, float *sx, int *incx, float *sy, int *incy);

double snrm2_(int *n, float *x, int *incx);

void zdotc_(dcomplex *dotval, int *n, dcomplex *cx, int *incx,
            dcomplex *cy, int *incy);

void zdotu_(dcomplex *dotval, int *n, dcomplex *cx, int *incx,
            dcomplex *cy, int *incy);

#else

fcomplex cdotc_(int *n, fcomplex *cx, int *incx, fcomplex *cy, int *incy);

fcomplex cdotu_(int *n, fcomplex *cx, int *incx, fcomplex *cy, int *incy);

float sasum_(int *n, float *sx, int *incx);

float scasum_(int *n, fcomplex *cx, int *incx);

float scnrm2_(int *n, fcomplex *x, int *incx);

float sdot_(int *n, float *sx, int *incx, float *sy, int *incy);

float snrm2_(int *n, float *x, int *incx);

dcomplex zdotc_(int *n, dcomplex *cx, int *incx, dcomplex *cy, int *incy);

dcomplex zdotu_(int *n, dcomplex *cx, int *incx, dcomplex *cy, int *incy);

#endif

/* Remaining functions listed in alphabetical order */

int caxpy_(int *n, fcomplex *ca, fcomplex *cx, int *incx, fcomplex *cy,
           int *incy);

int ccopy_(int *n, fcomplex *cx, int *incx, fcomplex *cy, int *incy);

int cgbmv_(char *trans, int *m, int *n, int *kl, int *ku,
           fcomplex *alpha, fcomplex *a, int *lda, fcomplex *x, int *incx,
           fcomplex *beta, fcomplex *y, int *incy);

int cgemm_(char *transa, char *transb, int *m, int *n, int *k,
           fcomplex *alpha, fcomplex *a, int *lda, fcomplex *b, int *ldb,
           fcomplex *beta, fcomplex *c, int *ldc);

int cgemv_(char *trans, int *m, int *n, fcomplex *alpha, fcomplex *a,
           int *lda, fcomplex *x, int *incx, fcomplex *beta, fcomplex *y,
           int *incy);

int cgerc_(int *m, int *n, fcomplex *alpha, fcomplex *x, int *incx,
           fcomplex *y, int *incy, fcomplex *a, int *lda);

int cgeru_(int *m, int *n, fcomplex *alpha, fcomplex *x, int *incx,
           fcomplex *y, int *incy, fcomplex *a, int *lda);

int chbmv_(char *uplo, int *n, int *k, fcomplex *alpha, fcomplex *a,
           int *lda, fcomplex *x, int *incx, fcomplex *beta, fcomplex *y,
           int *incy);

int chemm_(char *side, char *uplo, int *m, int *n, fcomplex *alpha,
           fcomplex *a, int *lda, fcomplex *b, int *ldb, fcomplex *beta,
           fcomplex *c, int *ldc);

int chemv_(char *uplo, int *n, fcomplex *alpha, fcomplex *a, int *lda,
           fcomplex *x, int *incx, fcomplex *beta, fcomplex *y, int *incy);

int cher_(char *uplo, int *n, float *alpha, fcomplex *x, int *incx,
          fcomplex *a, int *lda);

int cher2_(char *uplo, int *n, fcomplex *alpha, fcomplex *x, int *incx,
           fcomplex *y, int *incy, fcomplex *a, int *lda);

int cher2k_(char *uplo, char *trans, int *n, int *k, fcomplex *alpha,
            fcomplex *a, int *lda, fcomplex *b, int *ldb, float *beta,
            fcomplex *c, int *ldc);

int cherk_(char *uplo, char *trans, int *n, int *k, float *alpha,
           fcomplex *a, int *lda, float *beta, fcomplex *c, int *ldc);

int chpmv_(char *uplo, int *n, fcomplex *alpha, fcomplex *ap, fcomplex *x,
           int *incx, fcomplex *beta, fcomplex *y, int *incy);

int chpr_(char *uplo, int *n, float *alpha, fcomplex *x, int *incx,
          fcomplex *ap);

int chpr2_(char *uplo, int *n, fcomplex *alpha, fcomplex *x, int *incx,
           fcomplex *y, int *incy, fcomplex *ap);

int crotg_(fcomplex *ca, fcomplex *cb, float *c, fcomplex *s);

int cscal_(int *n, fcomplex *ca, fcomplex *cx, int *incx);

int csscal_(int *n, float *sa, fcomplex *cx, int *incx);

int cswap_(int *n, fcomplex *cx, int *incx, fcomplex *cy, int *incy);

int csymm_(char *side, char *uplo, int *m, int *n, fcomplex *alpha,
           fcomplex *a, int *lda, fcomplex *b, int *ldb, fcomplex *beta,
           fcomplex *c, int *ldc);

int csyr2k_(char *uplo, char *trans, int *n, int *k, fcomplex *alpha,
            fcomplex *a, int *lda, fcomplex *b, int *ldb, fcomplex *beta,
            fcomplex *c, int *ldc);

int csyrk_(char *uplo, char *trans, int *n, int *k, fcomplex *alpha,
           fcomplex *a, int *lda, fcomplex *beta, fcomplex *c, int *ldc);

int ctbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
           fcomplex *a, int *lda, fcomplex *x, int *incx);

int ctbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
           fcomplex *a, int *lda, fcomplex *x, int *incx);

int ctpmv_(char *uplo, char *trans, char *diag, int *n, fcomplex *ap,
           fcomplex *x, int *incx);

int ctpsv_(char *uplo, char *trans, char *diag, int *n, fcomplex *ap,
           fcomplex *x, int *incx);

int ctrmm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, fcomplex *alpha, fcomplex *a, int *lda, fcomplex *b,
           int *ldb);

int ctrmv_(char *uplo, char *trans, char *diag, int *n, fcomplex *a,
           int *lda, fcomplex *x, int *incx);

int ctrsm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, fcomplex *alpha, fcomplex *a, int *lda, fcomplex *b,
           int *ldb);

int ctrsv_(char *uplo, char *trans, char *diag, int *n, fcomplex *a,
           int *lda, fcomplex *x, int *incx);

int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
           int *incy);

int dcopy_(int *n, double *sx, int *incx, double *sy, int *incy);

int dgbmv_(char *trans, int *m, int *n, int *kl, int *ku,
           double *alpha, double *a, int *lda, double *x, int *incx,
           double *beta, double *y, int *incy);

int dgemm_(char *transa, char *transb, int *m, int *n, int *k,
           double *alpha, double *a, int *lda, double *b, int *ldb,
           double *beta, double *c, int *ldc);

int dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
           int *lda, double *x, int *incx, double *beta, double *y,
           int *incy);

int dger_(int *m, int *n, double *alpha, double *x, int *incx,
          double *y, int *incy, double *a, int *lda);

int drot_(int *n, double *sx, int *incx, double *sy, int *incy,
          double *c, double *s);

int drotg_(double *sa, double *sb, double *c, double *s);

int dsbmv_(char *uplo, int *n, int *k, double *alpha, double *a,
           int *lda, double *x, int *incx, double *beta, double *y,
           int *incy);

int dscal_(int *n, double *sa, double *sx, int *incx);

int dspmv_(char *uplo, int *n, double *alpha, double *ap, double *x,
           int *incx, double *beta, double *y, int *incy);

int dspr_(char *uplo, int *n, double *alpha, double *x, int *incx,
          double *ap);

int dspr2_(char *uplo, int *n, double *alpha, double *x, int *incx,
           double *y, int *incy, double *ap);

int dswap_(int *n, double *sx, int *incx, double *sy, int *incy);

int dsymm_(char *side, char *uplo, int *m, int *n, double *alpha,
           double *a, int *lda, double *b, int *ldb, double *beta,
           double *c, int *ldc);

int dsymv_(char *uplo, int *n, double *alpha, double *a, int *lda,
           double *x, int *incx, double *beta, double *y, int *incy);

int dsyr_(char *uplo, int *n, double *alpha, double *x, int *incx,
          double *a, int *lda);

int dsyr2_(char *uplo, int *n, double *alpha, double *x, int *incx,
           double *y, int *incy, double *a, int *lda);

int dsyr2k_(char *uplo, char *trans, int *n, int *k, double *alpha,
            double *a, int *lda, double *b, int *ldb, double *beta,
            double *c, int *ldc);

int dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha,
           double *a, int *lda, double *beta, double *c, int *ldc);

int dtbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
           double *a, int *lda, double *x, int *incx);

int dtbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
           double *a, int *lda, double *x, int *incx);

int dtpmv_(char *uplo, char *trans, char *diag, int *n, double *ap,
           double *x, int *incx);

int dtpsv_(char *uplo, char *trans, char *diag, int *n, double *ap,
           double *x, int *incx);

int dtrmm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, double *alpha, double *a, int *lda, double *b,
           int *ldb);

int dtrmv_(char *uplo, char *trans, char *diag, int *n, double *a,
           int *lda, double *x, int *incx);

int dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, double *alpha, double *a, int *lda, double *b,
           int *ldb);

int dtrsv_(char *uplo, char *trans, char *diag, int *n, double *a,
           int *lda, double *x, int *incx);


int saxpy_(int *n, float *sa, float *sx, int *incx, float *sy, int *incy);

int scopy_(int *n, float *sx, int *incx, float *sy, int *incy);

int sgbmv_(char *trans, int *m, int *n, int *kl, int *ku,
           float *alpha, float *a, int *lda, float *x, int *incx,
           float *beta, float *y, int *incy);

int sgemm_(char *transa, char *transb, int *m, int *n, int *k,
           float *alpha, float *a, int *lda, float *b, int *ldb,
           float *beta, float *c, int *ldc);

int sgemv_(char *trans, int *m, int *n, float *alpha, float *a,
           int *lda, float *x, int *incx, float *beta, float *y,
           int *incy);

int sger_(int *m, int *n, float *alpha, float *x, int *incx,
          float *y, int *incy, float *a, int *lda);

int srot_(int *n, float *sx, int *incx, float *sy, int *incy,
          float *c, float *s);

int srotg_(float *sa, float *sb, float *c, float *s);

int ssbmv_(char *uplo, int *n, int *k, float *alpha, float *a,
           int *lda, float *x, int *incx, float *beta, float *y,
           int *incy);

int sscal_(int *n, float *sa, float *sx, int *incx);

int sspmv_(char *uplo, int *n, float *alpha, float *ap, float *x,
           int *incx, float *beta, float *y, int *incy);

int sspr_(char *uplo, int *n, float *alpha, float *x, int *incx,
          float *ap);

int sspr2_(char *uplo, int *n, float *alpha, float *x, int *incx,
           float *y, int *incy, float *ap);

int sswap_(int *n, float *sx, int *incx, float *sy, int *incy);

int ssymm_(char *side, char *uplo, int *m, int *n, float *alpha,
           float *a, int *lda, float *b, int *ldb, float *beta,
           float *c, int *ldc);

int ssymv_(char *uplo, int *n, float *alpha, float *a, int *lda,
           float *x, int *incx, float *beta, float *y, int *incy);

int ssyr_(char *uplo, int *n, float *alpha, float *x, int *incx,
          float *a, int *lda);

int ssyr2_(char *uplo, int *n, float *alpha, float *x, int *incx,
           float *y, int *incy, float *a, int *lda);

int ssyr2k_(char *uplo, char *trans, int *n, int *k, float *alpha,
            float *a, int *lda, float *b, int *ldb, float *beta,
            float *c, int *ldc);

int ssyrk_(char *uplo, char *trans, int *n, int *k, float *alpha,
           float *a, int *lda, float *beta, float *c, int *ldc);

int stbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
           float *a, int *lda, float *x, int *incx);

int stbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
           float *a, int *lda, float *x, int *incx);

int stpmv_(char *uplo, char *trans, char *diag, int *n, float *ap,
           float *x, int *incx);

int stpsv_(char *uplo, char *trans, char *diag, int *n, float *ap,
           float *x, int *incx);

int strmm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, float *alpha, float *a, int *lda, float *b,
           int *ldb);

int strmv_(char *uplo, char *trans, char *diag, int *n, float *a,
           int *lda, float *x, int *incx);

int strsm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, float *alpha, float *a, int *lda, float *b,
           int *ldb);

int strsv_(char *uplo, char *trans, char *diag, int *n, float *a,
           int *lda, float *x, int *incx);

int zaxpy_(int *n, dcomplex *ca, dcomplex *cx, int *incx, dcomplex *cy,
           int *incy);

int zcopy_(int *n, dcomplex *cx, int *incx, dcomplex *cy, int *incy);

int zdscal_(int *n, double *sa, dcomplex *cx, int *incx);

int zgbmv_(char *trans, int *m, int *n, int *kl, int *ku,
           dcomplex *alpha, dcomplex *a, int *lda, dcomplex *x, int *incx,
           dcomplex *beta, dcomplex *y, int *incy);

int zgemm_(char *transa, char *transb, int *m, int *n, int *k,
           dcomplex *alpha, dcomplex *a, int *lda, dcomplex *b, int *ldb,
           dcomplex *beta, dcomplex *c, int *ldc);

int zgemv_(char *trans, int *m, int *n, dcomplex *alpha, dcomplex *a,
           int *lda, dcomplex *x, int *incx, dcomplex *beta, dcomplex *y,
           int *incy);

int zgerc_(int *m, int *n, dcomplex *alpha, dcomplex *x, int *incx,
           dcomplex *y, int *incy, dcomplex *a, int *lda);

int zgeru_(int *m, int *n, dcomplex *alpha, dcomplex *x, int *incx,
           dcomplex *y, int *incy, dcomplex *a, int *lda);

int zhbmv_(char *uplo, int *n, int *k, dcomplex *alpha, dcomplex *a,
           int *lda, dcomplex *x, int *incx, dcomplex *beta, dcomplex *y,
           int *incy);

int zhemm_(char *side, char *uplo, int *m, int *n, dcomplex *alpha,
           dcomplex *a, int *lda, dcomplex *b, int *ldb, dcomplex *beta,
           dcomplex *c, int *ldc);

int zhemv_(char *uplo, int *n, dcomplex *alpha, dcomplex *a, int *lda,
           dcomplex *x, int *incx, dcomplex *beta, dcomplex *y, int *incy);

int zher_(char *uplo, int *n, double *alpha, dcomplex *x, int *incx,
          dcomplex *a, int *lda);

int zher2_(char *uplo, int *n, dcomplex *alpha, dcomplex *x, int *incx,
           dcomplex *y, int *incy, dcomplex *a, int *lda);

int zher2k_(char *uplo, char *trans, int *n, int *k, dcomplex *alpha,
            dcomplex *a, int *lda, dcomplex *b, int *ldb, double *beta,
            dcomplex *c, int *ldc);

int zherk_(char *uplo, char *trans, int *n, int *k, double *alpha,
           dcomplex *a, int *lda, double *beta, dcomplex *c, int *ldc);

int zhpmv_(char *uplo, int *n, dcomplex *alpha, dcomplex *ap, dcomplex *x,
           int *incx, dcomplex *beta, dcomplex *y, int *incy);

int zhpr_(char *uplo, int *n, double *alpha, dcomplex *x, int *incx,
          dcomplex *ap);

int zhpr2_(char *uplo, int *n, dcomplex *alpha, dcomplex *x, int *incx,
           dcomplex *y, int *incy, dcomplex *ap);

int zrotg_(dcomplex *ca, dcomplex *cb, double *c, dcomplex *s);

int zscal_(int *n, dcomplex *ca, dcomplex *cx, int *incx);

int zswap_(int *n, dcomplex *cx, int *incx, dcomplex *cy, int *incy);

int zsymm_(char *side, char *uplo, int *m, int *n, dcomplex *alpha,
           dcomplex *a, int *lda, dcomplex *b, int *ldb, dcomplex *beta,
           dcomplex *c, int *ldc);

int zsyr2k_(char *uplo, char *trans, int *n, int *k, dcomplex *alpha,
            dcomplex *a, int *lda, dcomplex *b, int *ldb, dcomplex *beta,
            dcomplex *c, int *ldc);

int zsyrk_(char *uplo, char *trans, int *n, int *k, dcomplex *alpha,
           dcomplex *a, int *lda, dcomplex *beta, dcomplex *c, int *ldc);

int ztbmv_(char *uplo, char *trans, char *diag, int *n, int *k,
           dcomplex *a, int *lda, dcomplex *x, int *incx);

int ztbsv_(char *uplo, char *trans, char *diag, int *n, int *k,
           dcomplex *a, int *lda, dcomplex *x, int *incx);

int ztpmv_(char *uplo, char *trans, char *diag, int *n, dcomplex *ap,
           dcomplex *x, int *incx);

int ztpsv_(char *uplo, char *trans, char *diag, int *n, dcomplex *ap,
           dcomplex *x, int *incx);

int ztrmm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, dcomplex *alpha, dcomplex *a, int *lda, dcomplex *b,
           int *ldb);

int ztrmv_(char *uplo, char *trans, char *diag, int *n, dcomplex *a,
           int *lda, dcomplex *x, int *incx);

int ztrsm_(char *side, char *uplo, char *transa, char *diag, int *m,
           int *n, dcomplex *alpha, dcomplex *a, int *lda, dcomplex *b,
           int *ldb);

int ztrsv_(char *uplo, char *trans, char *diag, int *n, dcomplex *a,
           int *lda, dcomplex *x, int *incx);

#ifdef __cplusplus
}
#endif

#define FALSE 0
#define TRUE  1

/* Macro functions */
// #define MIN(a,b) ((a) <= (b) ? (a) : (b))
// #define MAX(a,b) ((a) >= (b) ? (a) : (b))

#endif

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
           int *incy)
{
  long int i, m, ix, iy, nn, iincx, iincy;
  register double ssa;

  /* constant times a vector plus a vector.
     uses unrolled loop for increments equal to one.
     jack dongarra, linpack, 3/11/78.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  ssa = *sa;
  iincx = *incx;
  iincy = *incy;

  if( nn > 0 && ssa != 0.0 )
  {
    if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
    {
      m = nn-3;
      for (i = 0; i < m; i += 4)
      {
        sy[i] += ssa * sx[i];
        sy[i+1] += ssa * sx[i+1];
        sy[i+2] += ssa * sx[i+2];
        sy[i+3] += ssa * sx[i+3];
      }
      for ( ; i < nn; ++i) /* clean-up loop */
        sy[i] += ssa * sx[i];
    }
    else /* code for unequal increments or equal increments not equal to 1 */
    {
      ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
      iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
      for (i = 0; i < nn; i++)
      {
        sy[iy] += ssa * sx[ix];
        ix += iincx;
        iy += iincy;
      }
    }
  }

  return 0;
} /* daxpy_ */

// #ifdef __cplusplus
// }
// #endif
//
// #ifdef __cplusplus
// extern "C" {
// #endif

double ddot_(int *n, double *sx, int *incx, double *sy, int *incy)
{
  long int i, m, nn, iincx, iincy;
  double stemp;
  long int ix, iy;

  /* forms the dot product of two vectors.
     uses unrolled loops for increments equal to one.
     jack dongarra, linpack, 3/11/78.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;
  iincy = *incy;

  stemp = 0.0;
  if (nn > 0)
  {
    if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
    {
      m = nn-4;
      for (i = 0; i < m; i += 5)
        stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                 sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

      for ( ; i < nn; i++)        /* clean-up loop */
        stemp += sx[i] * sy[i];
    }
    else /* code for unequal increments or equal increments not equal to 1 */
    {
      ix = 0;
      iy = 0;
      if (iincx < 0)
        ix = (1 - nn) * iincx;
      if (iincy < 0)
        iy = (1 - nn) * iincy;
      for (i = 0; i < nn; i++)
      {
        stemp += sx[ix] * sy[iy];
        ix += iincx;
        iy += iincy;
      }
    }
  }

  return stemp;
} /* ddot_ */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

double dnrm2_(int *n, double *x, int *incx)
{
  long int ix, nn, iincx;
  double norm, scale, absxi, ssq, temp;

/*  DNRM2 returns the euclidean norm of a vector via the function
    name, so that

       DNRM2 := sqrt( x'*x )

    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.   */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;

  if( nn > 0 && iincx > 0 )
  {
    if (nn == 1)
    {
      norm = fabs(x[0]);
    }
    else
    {
      scale = 0.0;
      ssq = 1.0;

      /* The following loop is equivalent to this call to the LAPACK
         auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

      for (ix=(nn-1)*iincx; ix>=0; ix-=iincx)
      {
        if (x[ix] != 0.0)
        {
          absxi = fabs(x[ix]);
          if (scale < absxi)
          {
            temp = scale / absxi;
            ssq = ssq * (temp * temp) + 1.0;
            scale = absxi;
          }
          else
          {
            temp = absxi / scale;
            ssq += temp * temp;
          }
        }
      }
      norm = scale * sqrt(ssq);
    }
  }
  else
    norm = 0.0;

  return norm;

} /* dnrm2_ */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

int dscal_(int *n, double *sa, double *sx, int *incx)
{
  long int i, m, nincx, nn, iincx;
  double ssa;

  /* scales a vector by a constant.
     uses unrolled loops for increment equal to 1.
     jack dongarra, linpack, 3/11/78.
     modified 3/93 to return if incx .le. 0.
     modified 12/3/93, array(1) declarations changed to array(*) */

  /* Dereference inputs */
  nn = *n;
  iincx = *incx;
  ssa = *sa;

  if (nn > 0 && iincx > 0)
  {
    if (iincx == 1) /* code for increment equal to 1 */
    {
      m = nn-4;
      for (i = 0; i < m; i += 5)
      {
        sx[i] = ssa * sx[i];
        sx[i+1] = ssa * sx[i+1];
        sx[i+2] = ssa * sx[i+2];
        sx[i+3] = ssa * sx[i+3];
        sx[i+4] = ssa * sx[i+4];
      }
      for ( ; i < nn; ++i) /* clean-up loop */
        sx[i] = ssa * sx[i];
    }
    else /* code for increment not equal to 1 */
    {
      nincx = nn * iincx;
      for (i = 0; i < nincx; i += iincx)
        sx[i] = ssa * sx[i];
    }
  }

  return 0;
} /* dscal_ */

#ifdef __cplusplus
}
#endif


#ifndef _TRON_H
#define _TRON_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

class function
{
public:
	virtual double fun(double *w) = 0 ;
	virtual void grad(double *w, double *g) = 0 ;
	virtual void Hv(double *s, double *Hs) = 0 ;

	virtual int get_nr_variable(void) = 0 ;
	virtual ~function(void){}
};

class TRON
{
public:
	TRON(const function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~TRON();

	void tron(double *w);
	void set_print_string(void (*i_print) (const char *buf));

private:
	int trcg(double delta, double *g, double *s, double *r);
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

// extern double dnrm2_(int *, double *, int *);
// extern double ddot_(int *, double *, int *, double *, int *);
// extern int daxpy_(int *, double *, double *, int *, double *, int *);
// extern int dscal_(int *, double *, double *, int *);

#ifdef __cplusplus
}
#endif

static void default_print(const char *buf)
{
	fputs(buf,stdout);
	fflush(stdout);
}

void TRON::info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*tron_print_string)(buf);
}

TRON::TRON(const function *_fun_obj, double _eps, int _max_iter)
{
	this->fun_obj=const_cast<function *>(_fun_obj);
	this->eps=_eps;
	this->max_iter=_max_iter;
	tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(double *w)
{
	// Parameters for updating the iterates.
	double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

	// Parameters for updating the trust region size delta.
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

	int n = fun_obj->get_nr_variable();
	int i, cg_iter;
	double delta, snorm, one=1.0;
	double alpha, f, fnew, prered, actred, gs;
	int search = 1, iter = 1, inc = 1;
	double *s = new double[n];
	double *r = new double[n];
	double *w_new = new double[n];
	double *g = new double[n];

	for (i=0; i<n; i++)
		w[i] = 0;

	f = fun_obj->fun(w);
	fun_obj->grad(w, g);
	delta = dnrm2_(&n, g, &inc);
	double gnorm1 = delta;
	double gnorm = gnorm1;

	if (gnorm <= eps*gnorm1)
		search = 0;

	iter = 1;

	while (iter <= max_iter && search)
	{
		cg_iter = trcg(delta, g, s, r);

		memcpy(w_new, w, sizeof(double)*n);
		daxpy_(&n, &one, s, &inc, w_new, &inc);

		gs = ddot_(&n, g, &inc, s, &inc);
		prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
		fnew = fun_obj->fun(w_new);

		// Compute the actual reduction.
		actred = f - fnew;

		// On the first iteration, adjust the initial step bound.
		snorm = dnrm2_(&n, s, &inc);
		if (iter == 1)
			delta = min(delta, snorm);

		// Compute prediction alpha*snorm of the step.
		if (fnew - f - gs <= 0)
			alpha = sigma3;
		else
			alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

		// Update the trust region bound according to the ratio of actual to predicted reduction.
		if (actred < eta0*prered)
			delta = min(max(alpha, sigma1)*snorm, sigma2*delta);
		else if (actred < eta1*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma2*delta));
		else if (actred < eta2*prered)
			delta = max(sigma1*delta, min(alpha*snorm, sigma3*delta));
		else
			delta = max(delta, min(alpha*snorm, sigma3*delta));

		info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);

		if (actred > eta0*prered)
		{
			iter++;
			memcpy(w, w_new, sizeof(double)*n);
			f = fnew;
			fun_obj->grad(w, g);

			gnorm = dnrm2_(&n, g, &inc);
			if (gnorm <= eps*gnorm1)
				break;
		}
		if (f < -1.0e+32)
		{
			info("WARNING: f < -1.0e+32\n");
			break;
		}
		if (fabs(actred) <= 0 && prered <= 0)
		{
			info("WARNING: actred and prered <= 0\n");
			break;
		}
		if (fabs(actred) <= 1.0e-12*fabs(f) &&
		    fabs(prered) <= 1.0e-12*fabs(f))
		{
			info("WARNING: actred and prered too small\n");
			break;
		}
	}

	delete[] g;
	delete[] r;
	delete[] w_new;
	delete[] s;
}

int TRON::trcg(double delta, double *g, double *s, double *r)
{
	int i, inc = 1;
	int n = fun_obj->get_nr_variable();
	double one = 1;
	double *d = new double[n];
	double *Hd = new double[n];
	double rTr, rnewTrnew, alpha, beta, cgtol;

	for (i=0; i<n; i++)
	{
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	cgtol = 0.1*dnrm2_(&n, g, &inc);

	int cg_iter = 0;
	rTr = ddot_(&n, r, &inc, r, &inc);
	while (1)
	{
		if (dnrm2_(&n, r, &inc) <= cgtol)
			break;
		cg_iter++;
		fun_obj->Hv(d, Hd);

		alpha = rTr/ddot_(&n, d, &inc, Hd, &inc);
		daxpy_(&n, &alpha, d, &inc, s, &inc);
		if (dnrm2_(&n, s, &inc) > delta)
		{
			info("cg reaches trust region boundary\n");
			alpha = -alpha;
			daxpy_(&n, &alpha, d, &inc, s, &inc);

			double std = ddot_(&n, s, &inc, d, &inc);
			double sts = ddot_(&n, s, &inc, s, &inc);
			double dtd = ddot_(&n, d, &inc, d, &inc);
			double dsq = delta*delta;
			double rad = sqrt(std*std + dtd*(dsq-sts));
			if (std >= 0)
				alpha = (dsq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;
			daxpy_(&n, &alpha, d, &inc, s, &inc);
			alpha = -alpha;
			daxpy_(&n, &alpha, Hd, &inc, r, &inc);
			break;
		}
		alpha = -alpha;
		daxpy_(&n, &alpha, Hd, &inc, r, &inc);
		rnewTrnew = ddot_(&n, r, &inc, r, &inc);
		beta = rnewTrnew/rTr;
		dscal_(&n, &beta, d, &inc);
		daxpy_(&n, &one, r, &inc, d, &inc);
		rTr = rnewTrnew;
	}

	delete[] d;
	delete[] Hd;

	return(cg_iter);
}

double TRON::norm_inf(int n, double *x)
{
	double dmax = fabs(x[0]);
	for (int i=1; i<n; i++)
		if (fabs(x[i]) >= dmax)
			dmax = fabs(x[i]);
	return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf))
{
	tron_print_string = print_string;
}
#endif

#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>



#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	struct feature_node **x;
	double bias;            /* < 0 if no bias term */
};

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL }; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_weight;
	int *weight_label;
	double* weight;
	double p;
};

struct model
{
	struct parameter param;
	int nr_class;		/* number of classes */
	int nr_feature;
	double *w;
	int *label;		/* label of each class */
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);
void cross_validation(const struct problem *prob, const struct parameter *param, int nr_fold, double *target);

double predict_values(const struct model *model_, const struct feature_node *x, double* dec_values);
double predict(const struct model *model_, const struct feature_node *x);
double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates);

int save_model(const char *model_file_name, const struct model *model_);
struct model *load_model(const char *model_file_name);

int get_nr_feature(const struct model *model_);
int get_nr_class(const struct model *model_);
void get_labels(const struct model *model_, int* label);
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx);
double get_decfun_bias(const struct model *model_, int label_idx);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
int check_probability_model(const struct model *model);
int check_regression_model(const struct model *model);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
// #ifndef min
// template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
// #endif
// #ifndef max
// template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
// #endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

class l2r_lr_fun: public function
{
public:
	l2r_lr_fun(const problem *prob, double *C);
	~l2r_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	const problem *prob;
};

l2r_lr_fun::l2r_lr_fun(const problem *_prob, double *_C)
{
	int l=_prob->l;

	this->prob = _prob;

	z = new double[l];
	D = new double[l];
	this->C = _C;
}

l2r_lr_fun::~l2r_lr_fun()
{
	delete[] z;
	delete[] D;
}


double l2r_lr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void l2r_lr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + g[i];
}

int l2r_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	double *wa = new double[l];

	Xv(s, wa);
	for(i=0;i<l;i++)
		wa[i] = C[i]*D[i]*wa[i];

	XTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];
	delete[] wa;
}

void l2r_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2r_l2_svc_fun: public function
{
public:
	l2r_l2_svc_fun(const problem *prob, double *C);
	~l2r_l2_svc_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

protected:
	void Xv(double *v, double *Xv);
	void subXv(double *v, double *Xv);
	void subXTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	int *I;
	int sizeI;
	const problem *prob;
};

l2r_l2_svc_fun::l2r_l2_svc_fun(const problem *_prob, double *_C)
{
	int l=_prob->l;

	this->prob = _prob;

	z = new double[l];
	D = new double[l];
	I = new int[l];
	this->C = _C;
}

l2r_l2_svc_fun::~l2r_l2_svc_fun()
{
	delete[] z;
	delete[] D;
	delete[] I;
}

double l2r_l2_svc_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2.0;
	for(i=0;i<l;i++)
	{
		z[i] = y[i]*z[i];
		double d = 1-z[i];
		if (d > 0)
			f += C[i]*d*d;
	}

	return(f);
}

void l2r_l2_svc_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();

	sizeI = 0;
	for (i=0;i<l;i++)
		if (z[i] < 1)
		{
			z[sizeI] = C[i]*y[i]*(z[i]-1);
			I[sizeI] = i;
			sizeI++;
		}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

int l2r_l2_svc_fun::get_nr_variable(void)
{
	return prob->n;
}

void l2r_l2_svc_fun::Hv(double *s, double *Hs)
{
	int i;
	int w_size=get_nr_variable();
	double *wa = new double[sizeI];

	subXv(s, wa);
	for(i=0;i<sizeI;i++)
		wa[i] = C[I[i]]*wa[i];

	subXTv(wa, Hs);
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + 2*Hs[i];
	delete[] wa;
}

void l2r_l2_svc_fun::Xv(double *v, double *Xv)
{
	int i;
	int l=prob->l;
	feature_node **x=prob->x;

	for(i=0;i<l;i++)
	{
		feature_node *s=x[i];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXv(double *v, double *Xv)
{
	int i;
	feature_node **x=prob->x;

	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		Xv[i]=0;
		while(s->index!=-1)
		{
			Xv[i]+=v[s->index-1]*s->value;
			s++;
		}
	}
}

void l2r_l2_svc_fun::subXTv(double *v, double *XTv)
{
	int i;
	int w_size=get_nr_variable();
	feature_node **x=prob->x;

	for(i=0;i<w_size;i++)
		XTv[i]=0;
	for(i=0;i<sizeI;i++)
	{
		feature_node *s=x[I[i]];
		while(s->index!=-1)
		{
			XTv[s->index-1]+=v[i]*s->value;
			s++;
		}
	}
}

class l2r_l2_svr_fun: public l2r_l2_svc_fun
{
public:
	l2r_l2_svr_fun(const problem *prob, double *C, double p);

	double fun(double *w);
	void grad(double *w, double *g);

private:
	double p;
};

l2r_l2_svr_fun::l2r_l2_svr_fun(const problem *_prob, double *_C, double _p):
	l2r_l2_svc_fun(_prob, _C)
{
	this->p = _p;
}

double l2r_l2_svr_fun::fun(double *w)
{
	int i;
	double f=0;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	Xv(w, z);

	for(i=0;i<w_size;i++)
		f += w[i]*w[i];
	f /= 2;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];
		if(d < -p)
			f += C[i]*(d+p)*(d+p);
		else if(d > p)
			f += C[i]*(d-p)*(d-p);
	}

	return(f);
}

void l2r_l2_svr_fun::grad(double *w, double *g)
{
	int i;
	double *y=prob->y;
	int l=prob->l;
	int w_size=get_nr_variable();
	double d;

	sizeI = 0;
	for(i=0;i<l;i++)
	{
		d = z[i] - y[i];

		// generate index set I
		if(d < -p)
		{
			z[sizeI] = C[i]*(d+p);
			I[sizeI] = i;
			sizeI++;
		}
		else if(d > p)
		{
			z[sizeI] = C[i]*(d-p);
			I[sizeI] = i;
			sizeI++;
		}

	}
	subXTv(z, g);

	for(i=0;i<w_size;i++)
		g[i] = w[i] + 2*g[i];
}

// A coordinate descent algorithm for
// multi-class support vector machines by Crammer and Singer
//
//  min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
//    s.t.     \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
//
//  where e^m_i = 0 if y_i  = m,
//        e^m_i = 1 if y_i != m,
//  C^m_i = C if m  = y_i,
//  C^m_i = 0 if m != y_i,
//  and w_m(\alpha) = \sum_i \alpha^m_i x_i
//
// Given:
// x, y, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Appendix of LIBLINEAR paper, Fan et al. (2008)

#define GETI(i) ((int) prob->y[i])
// To support weights for instances, use GETI(i) (i)

class Solver_MCSVM_CS
{
	public:
		Solver_MCSVM_CS(const problem *prob, int nr_class, double *C, double eps=0.1, int max_iter=100000);
		~Solver_MCSVM_CS();
		void Solve(double *w);
	private:
		void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new);
		bool be_shrunk(int i, int m, int yi, double alpha_i, double minG);
		double *B, *C, *G;
		int w_size, l;
		int nr_class;
		int max_iter;
		double eps;
		const problem *prob;
};

Solver_MCSVM_CS::Solver_MCSVM_CS(const problem *_prob, int _nr_class, double *weighted_C, double _eps, int _max_iter)
{
	this->w_size = _prob->n;
	this->l = _prob->l;
	this->nr_class = _nr_class;
	this->eps = _eps;
	this->max_iter = _max_iter;
	this->prob = _prob;
	this->B = new double[nr_class];
	this->G = new double[nr_class];
	this->C = weighted_C;
}

Solver_MCSVM_CS::~Solver_MCSVM_CS()
{
	delete[] B;
	delete[] G;
}

int compare_double(const void *a, const void *b)
{
	if(*(double *)a > *(double *)b)
		return -1;
	if(*(double *)a < *(double *)b)
		return 1;
	return 0;
}

void Solver_MCSVM_CS::solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double *alpha_new)
{
	int r;
	double *D;

	clone(D, B, active_i);
	if(yi < active_i)
		D[yi] += A_i*C_yi;
	qsort(D, active_i, sizeof(double), compare_double);

	double beta = D[0] - A_i*C_yi;
	for(r=1;r<active_i && beta<r*D[r];r++)
		beta += D[r];
	beta /= r;

	for(r=0;r<active_i;r++)
	{
		if(r == yi)
			alpha_new[r] = min(C_yi, (beta-B[r])/A_i);
		else
			alpha_new[r] = min((double)0, (beta - B[r])/A_i);
	}
	delete[] D;
}

bool Solver_MCSVM_CS::be_shrunk(int i, int m, int yi, double alpha_i, double minG)
{
	double bound = 0;
	if(m == yi)
		bound = C[GETI(i)];
	if(alpha_i == bound && G[m] < minG)
		return true;
	return false;
}

void Solver_MCSVM_CS::Solve(double *w)
{
	int i, m, s;
	int iter = 0;
	double *alpha =  new double[l*nr_class];
	double *alpha_new = new double[nr_class];
	int *index = new int[l];
	double *QD = new double[l];
	int *d_ind = new int[nr_class];
	double *d_val = new double[nr_class];
	int *alpha_index = new int[nr_class*l];
	int *y_index = new int[l];
	int active_size = l;
	int *active_size_i = new int[l];
	double eps_shrink = max(10.0*eps, 1.0); // stopping tolerance for shrinking
	bool start_from_all = true;

	// Initial alpha can be set here. Note that
	// sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
	// alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
	// alpha[i*nr_class+m] <= 0 if prob->y[i] != m
	// If initial alpha isn't zero, uncomment the for loop below to initialize w
	for(i=0;i<l*nr_class;i++)
		alpha[i] = 0;

	for(i=0;i<w_size*nr_class;i++)
		w[i] = 0;
	for(i=0;i<l;i++)
	{
		for(m=0;m<nr_class;m++)
			alpha_index[i*nr_class+m] = m;
		feature_node *xi = prob->x[i];
		QD[i] = 0;
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;

			// Uncomment the for loop if initial alpha isn't zero
			// for(m=0; m<nr_class; m++)
			//	w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
			xi++;
		}
		active_size_i[i] = nr_class;
		y_index[i] = (int)prob->y[i];
		index[i] = i;
	}

	while(iter < max_iter)
	{
		double stopping = -INF;
		for(i=0;i<active_size;i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}
		for(s=0;s<active_size;s++)
		{
			i = index[s];
			double Ai = QD[i];
			double *alpha_i = &alpha[i*nr_class];
			int *alpha_index_i = &alpha_index[i*nr_class];

			if(Ai > 0)
			{
				for(m=0;m<active_size_i[i];m++)
					G[m] = 1;
				if(y_index[i] < active_size_i[i])
					G[y_index[i]] = 0;

				feature_node *xi = prob->x[i];
				while(xi->index!= -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<active_size_i[i];m++)
						G[m] += w_i[alpha_index_i[m]]*(xi->value);
					xi++;
				}

				double minG = INF;
				double maxG = -INF;
				for(m=0;m<active_size_i[i];m++)
				{
					if(alpha_i[alpha_index_i[m]] < 0 && G[m] < minG)
						minG = G[m];
					if(G[m] > maxG)
						maxG = G[m];
				}
				if(y_index[i] < active_size_i[i])
					if(alpha_i[(int) prob->y[i]] < C[GETI(i)] && G[y_index[i]] < minG)
						minG = G[y_index[i]];

				for(m=0;m<active_size_i[i];m++)
				{
					if(be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG))
					{
						active_size_i[i]--;
						while(active_size_i[i]>m)
						{
							if(!be_shrunk(i, active_size_i[i], y_index[i],
											alpha_i[alpha_index_i[active_size_i[i]]], minG))
							{
								swap(alpha_index_i[m], alpha_index_i[active_size_i[i]]);
								swap(G[m], G[active_size_i[i]]);
								if(y_index[i] == active_size_i[i])
									y_index[i] = m;
								else if(y_index[i] == m)
									y_index[i] = active_size_i[i];
								break;
							}
							active_size_i[i]--;
						}
					}
				}

				if(active_size_i[i] <= 1)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}

				if(maxG-minG <= 1e-12)
					continue;
				else
					stopping = max(maxG - minG, stopping);

				for(m=0;m<active_size_i[i];m++)
					B[m] = G[m] - Ai*alpha_i[alpha_index_i[m]] ;

				solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
				int nz_d = 0;
				for(m=0;m<active_size_i[i];m++)
				{
					double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
					alpha_i[alpha_index_i[m]] = alpha_new[m];
					if(fabs(d) >= 1e-12)
					{
						d_ind[nz_d] = alpha_index_i[m];
						d_val[nz_d] = d;
						nz_d++;
					}
				}

				xi = prob->x[i];
				while(xi->index != -1)
				{
					double *w_i = &w[(xi->index-1)*nr_class];
					for(m=0;m<nz_d;m++)
						w_i[d_ind[m]] += d_val[m]*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
		{
			info(".");
		}

		if(stopping < eps_shrink)
		{
			if(stopping < eps && start_from_all == true)
				break;
			else
			{
				active_size = l;
				for(i=0;i<l;i++)
					active_size_i[i] = nr_class;
				info("*");
				eps_shrink = max(eps_shrink/2, eps);
				start_from_all = true;
			}
		}
		else
			start_from_all = false;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0;i<w_size*nr_class;i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0;i<l*nr_class;i++)
	{
		v += alpha[i];
		if(fabs(alpha[i]) > 0)
			nSV++;
	}
	for(i=0;i<l;i++)
		v -= alpha[i*nr_class+(int)prob->y[i]];
	info("Objective value = %lf\n",v);
	info("nSV = %d\n",nSV);

	delete [] alpha;
	delete [] alpha_new;
	delete [] index;
	delete [] QD;
	delete [] d_ind;
	delete [] d_val;
	delete [] alpha_index;
	delete [] y_index;
	delete [] active_size_i;
}

// A coordinate descent algorithm for
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svc(
	const problem *prob, double *w, double eps,
	double Cp, double Cn, int solver_type)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double C, d, G;
	double *QD = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[l];
	schar *y = new schar[l];
	int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 <= alpha[i] <= upper_bound[GETI(i)]
	for(i=0; i<l; i++)
		alpha[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = diag[GETI(i)];

		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += y[i]*alpha[i]*val;
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];

			feature_node *xi = prob->x[i];
			while(xi->index!= -1)
			{
				G += w[xi->index-1]*(xi->value);
				xi++;
			}
			G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				d = (alpha[i] - alpha_old)*yi;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;
}


// A coordinate descent algorithm for
// L1-loss and L2-loss epsilon-SVR dual problem
//
//  min_\beta  0.5\beta^T (Q + diag(lambda)) \beta - p \sum_{i=1}^l|\beta_i| + \sum_{i=1}^l yi\beta_i,
//    s.t.      -upper_bound_i <= \beta_i <= upper_bound_i,
//
//  where Qij = xi^T xj and
//  D is a diagonal matrix
//
// In L1-SVM case:
// 		upper_bound_i = C
// 		lambda_i = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		lambda_i = 1/(2*C)
//
// Given:
// x, y, p, C
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 4 of Ho and Lin, 2012

#undef GETI
#define GETI(i) (0)
// To support weights for instances, use GETI(i) (i)

static void solve_l2r_l1l2_svr(
	const problem *prob, double *w, const parameter *param,
	int solver_type)
{
	int l = prob->l;
	double C = param->C;
	double p = param->p;
	int w_size = prob->n;
	double eps = param->eps;
	int i, s, iter = 0;
	int max_iter = 1000;
	int active_size = l;
	int *index = new int[l];

	double d, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double *beta = new double[l];
	double *QD = new double[l];
	double *y = prob->y;

	// L2R_L2LOSS_SVR_DUAL
	double lambda[1], upper_bound[1];
	lambda[0] = 0.5/C;
	upper_bound[0] = INF;

	if(solver_type == L2R_L1LOSS_SVR_DUAL)
	{
		lambda[0] = 0;
		upper_bound[0] = C;
	}

	// Initial beta can be set here. Note that
	// -upper_bound <= beta[i] <= upper_bound
	for(i=0; i<l; i++)
		beta[i] = 0;

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		QD[i] = 0;
		feature_node *xi = prob->x[i];
		while(xi->index != -1)
		{
			double val = xi->value;
			QD[i] += val*val;
			w[xi->index-1] += beta[i]*val;
			xi++;
		}

		index[i] = i;
	}


	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			i = index[s];
			G = -y[i] + lambda[GETI(i)]*beta[i];
			H = QD[i] + lambda[GETI(i)];

			feature_node *xi = prob->x[i];
			while(xi->index != -1)
			{
				int ind = xi->index-1;
				double val = xi->value;
				G += val*w[ind];
				xi++;
			}

			double Gp = G+p;
			double Gn = G-p;
			double violation = 0;
			if(beta[i] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old && Gn<-Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] >= upper_bound[GETI(i)])
			{
				if(Gp > 0)
					violation = Gp;
				else if(Gp < -Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] <= -upper_bound[GETI(i)])
			{
				if(Gn < 0)
					violation = -Gn;
				else if(Gn > Gmax_old)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(beta[i] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*beta[i])
				d = -Gp/H;
			else if(Gn > H*beta[i])
				d = -Gn/H;
			else
				d = -beta[i];

			if(fabs(d) < 1.0e-12)
				continue;

			double beta_old = beta[i];
			beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
			d = beta[i]-beta_old;

			if(d != 0)
			{
				xi = prob->x[i];
				while(xi->index != -1)
				{
					w[xi->index-1] += d*xi->value;
					xi++;
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 11 may be faster\n\n");

	// calculate objective value
	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	v = 0.5*v;
	for(i=0; i<l; i++)
	{
		v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
		if(beta[i] != 0)
			nSV++;
	}

	info("Objective value = %lf\n", v);
	info("nSV = %d\n",nSV);

	delete [] beta;
	delete [] QD;
	delete [] index;
}


// A coordinate descent algorithm for
// the dual of L2-regularized logistic regression problems
//
//  min_\alpha  0.5(\alpha^T Q \alpha) + \sum \alpha_i log (\alpha_i) + (upper_bound_i - \alpha_i) log (upper_bound_i - \alpha_i),
//    s.t.      0 <= \alpha_i <= upper_bound_i,
//
//  where Qij = yi yj xi^T xj and
//  upper_bound_i = Cp if y_i = 1
//  upper_bound_i = Cn if y_i = -1
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Algorithm 5 of Yu et al., MLJ 2010

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
{
	int l = prob->l;
	int w_size = prob->n;
	int i, s, iter = 0;
	double *xTx = new double[l];
	int max_iter = 1000;
	int *index = new int[l];
	double *alpha = new double[2*l]; // store alpha and C - alpha
	schar *y = new schar[l];
	int max_inner_iter = 100; // for inner Newton
	double innereps = 1e-2;
	double innereps_min = min(1e-8, eps);
	double upper_bound[3] = {Cn, 0, Cp};

	for(i=0; i<l; i++)
	{
		if(prob->y[i] > 0)
		{
			y[i] = +1;
		}
		else
		{
			y[i] = -1;
		}
	}

	// Initial alpha can be set here. Note that
	// 0 < alpha[i] < upper_bound[GETI(i)]
	// alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
	for(i=0; i<l; i++)
	{
		alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
		alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
	}

	for(i=0; i<w_size; i++)
		w[i] = 0;
	for(i=0; i<l; i++)
	{
		xTx[i] = 0;
		feature_node *xi = prob->x[i];
		while (xi->index != -1)
		{
			double val = xi->value;
			xTx[i] += val*val;
			w[xi->index-1] += y[i]*alpha[2*i]*val;
			xi++;
		}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		for (i=0; i<l; i++)
		{
			int j = i+rand()%(l-i);
			swap(index[i], index[j]);
		}
		int newton_iter = 0;
		double Gmax = 0;
		for (s=0; s<l; s++)
		{
			i = index[s];
			schar yi = y[i];
			double C = upper_bound[GETI(i)];
			double ywTx = 0, xisq = xTx[i];
			feature_node *xi = prob->x[i];
			while (xi->index != -1)
			{
				ywTx += w[xi->index-1]*xi->value;
				xi++;
			}
			ywTx *= y[i];
			double a = xisq, b = ywTx;

			// Decide to minimize g_1(z) or g_2(z)
			int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
			if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
			{
				ind1 = 2*i+1;
				ind2 = 2*i;
				sign = -1;
			}

			//  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
			double alpha_old = alpha[ind1];
			double z = alpha_old;
			if(C - z < 0.5 * C)
				z = 0.1*z;
			double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
			Gmax = max(Gmax, fabs(gp));

			// Newton method on the sub-problem
			const double eta = 0.1; // xi in the paper
			int inner_iter = 0;
			while (inner_iter <= max_inner_iter)
			{
				if(fabs(gp) < innereps)
					break;
				double gpp = a + C/(C-z)/z;
				double tmpz = z - gp/gpp;
				if(tmpz <= 0)
					z *= eta;
				else // tmpz in (0, C)
					z = tmpz;
				gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
				newton_iter++;
				inner_iter++;
			}

			if(inner_iter > 0) // update w
			{
				alpha[ind1] = z;
				alpha[ind2] = C-z;
				xi = prob->x[i];
				while (xi->index != -1)
				{
					w[xi->index-1] += sign*(z-alpha_old)*yi*xi->value;
					xi++;
				}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gmax < eps)
			break;

		if(newton_iter <= l/10)
			innereps = max(innereps_min, 0.1*innereps);

	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	for(i=0; i<w_size; i++)
		v += w[i] * w[i];
	v *= 0.5;
	for(i=0; i<l; i++)
		v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
			- upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
	info("Objective value = %lf\n", v);

	delete [] xTx;
	delete [] alpha;
	delete [] y;
	delete [] index;
}

// A coordinate descent algorithm for
// L1-regularized L2-loss support vector classification
//
//  min_w \sum |wj| + C \sum max(0, 1-yi w^T xi)^2,
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2010) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_l2_svc(
	problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, iter = 0;
	int max_iter = 1000;
	int active_size = w_size;
	int max_num_linesearch = 20;

	double sigma = 0.01;
	double d, G_loss, G, H;
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double d_old, d_diff;
	double loss_old, loss_new;
	double appxcond, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *b = new double[l]; // b = 1-ywTx
	double *xj_sq = new double[w_size];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		b[j] = 1;
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;
	}
	for(j=0; j<w_size; j++)
	{
		index[j] = j;
		xj_sq[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x->value *= y[ind]; // x->value stores yi*xij
			double val = x->value;
			b[ind] -= w[j]*val;
			xj_sq[j] += C[GETI(ind)]*val*val;
			x++;
		}
	}

	while(iter < max_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;

		for(j=0; j<active_size; j++)
		{
			int i = j+rand()%(active_size-j);
			swap(index[i], index[j]);
		}

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			G_loss = 0;
			H = 0;

			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				if(b[ind] > 0)
				{
					double val = x->value;
					double tmp = C[GETI(ind)]*val;
					G_loss -= tmp*b[ind];
					H += tmp*val;
				}
				x++;
			}
			G_loss *= 2;

			G = G_loss;
			H *= 2;
			H = max(H, 1e-12);

			double Gp = G+1;
			double Gn = G-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;

			// obtain Newton direction d
			if(Gp < H*w[j])
				d = -Gp/H;
			else if(Gn > H*w[j])
				d = -Gn/H;
			else
				d = -w[j];

			if(fabs(d) < 1.0e-12)
				continue;

			double delta = fabs(w[j]+d)-fabs(w[j]) + G*d;
			d_old = 0;
			int num_linesearch;
			for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
			{
				d_diff = d_old - d;
				cond = fabs(w[j]+d)-fabs(w[j]) - sigma*delta;

				appxcond = xj_sq[j]*d*d + G_loss*d + cond;
				if(appxcond <= 0)
				{
					x = prob_col->x[j];
					while(x->index != -1)
					{
						b[x->index-1] += d_diff*x->value;
						x++;
					}
					break;
				}

				if(num_linesearch == 0)
				{
					loss_old = 0;
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						if(b[ind] > 0)
							loss_old += C[GETI(ind)]*b[ind]*b[ind];
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}
				else
				{
					loss_new = 0;
					x = prob_col->x[j];
					while(x->index != -1)
					{
						int ind = x->index-1;
						double b_new = b[ind] + d_diff*x->value;
						b[ind] = b_new;
						if(b_new > 0)
							loss_new += C[GETI(ind)]*b_new*b_new;
						x++;
					}
				}

				cond = cond + loss_new - loss_old;
				if(cond <= 0)
					break;
				else
				{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}

			w[j] += d;

			// recompute b[] if line search takes too many steps
			if(num_linesearch >= max_num_linesearch)
			{
				info("#");
				for(int i=0; i<l; i++)
					b[i] = 1;

				for(int i=0; i<w_size; i++)
				{
					if(w[i]==0) continue;
					x = prob_col->x[i];
					while(x->index != -1)
					{
						b[x->index-1] -= w[i]*x->value;
						x++;
					}
				}
			}
		}

		if(iter == 0)
			Gnorm1_init = Gnorm1_new;
		iter++;
		if(iter % 10 == 0)
			info(".");

		if(Gnorm1_new <= eps*Gnorm1_init)
		{
			if(active_size == w_size)
				break;
			else
			{
				active_size = w_size;
				info("*");
				Gmax_old = INF;
				continue;
			}
		}

		Gmax_old = Gmax_new;
	}

	info("\noptimization finished, #iter = %d\n", iter);
	if(iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
	{
		x = prob_col->x[j];
		while(x->index != -1)
		{
			x->value *= prob_col->y[x->index-1]; // restore x->value
			x++;
		}
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	}
	for(j=0; j<l; j++)
		if(b[j] > 0)
			v += C[GETI(j)]*b[j]*b[j];

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] b;
	delete [] xj_sq;
}

// A coordinate descent algorithm for
// L1-regularized logistic regression problems
//
//  min_w \sum |wj| + C \sum log(1+exp(-yi w^T xi)),
//
// Given:
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
//
// See Yuan et al. (2011) and appendix of LIBLINEAR paper, Fan et al. (2008)

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

static void solve_l1r_lr(
	const problem *prob_col, double *w, double eps,
	double Cp, double Cn)
{
	int l = prob_col->l;
	int w_size = prob_col->n;
	int j, s, newton_iter=0, iter=0;
	int max_newton_iter = 100;
	int max_iter = 1000;
	int max_num_linesearch = 20;
	int active_size;
	int QP_active_size;

	double nu = 1e-12;
	double inner_eps = 1;
	double sigma = 0.01;
	double w_norm, w_norm_new;
	double z, G, H;
	double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
	double Gmax_old = INF;
	double Gmax_new, Gnorm1_new;
	double QP_Gmax_old = INF;
	double QP_Gmax_new, QP_Gnorm1_new;
	double delta, negsum_xTd, cond;

	int *index = new int[w_size];
	schar *y = new schar[l];
	double *Hdiag = new double[w_size];
	double *Grad = new double[w_size];
	double *wpd = new double[w_size];
	double *xjneg_sum = new double[w_size];
	double *xTd = new double[l];
	double *exp_wTx = new double[l];
	double *exp_wTx_new = new double[l];
	double *tau = new double[l];
	double *D = new double[l];
	feature_node *x;

	double C[3] = {Cn,0,Cp};

	// Initial w can be set here.
	for(j=0; j<w_size; j++)
		w[j] = 0;

	for(j=0; j<l; j++)
	{
		if(prob_col->y[j] > 0)
			y[j] = 1;
		else
			y[j] = -1;

		exp_wTx[j] = 0;
	}

	w_norm = 0;
	for(j=0; j<w_size; j++)
	{
		w_norm += fabs(w[j]);
		wpd[j] = w[j];
		index[j] = j;
		xjneg_sum[j] = 0;
		x = prob_col->x[j];
		while(x->index != -1)
		{
			int ind = x->index-1;
			double val = x->value;
			exp_wTx[ind] += w[j]*val;
			if(y[ind] == -1)
				xjneg_sum[j] += C[GETI(ind)]*val;
			x++;
		}
	}
	for(j=0; j<l; j++)
	{
		exp_wTx[j] = exp(exp_wTx[j]);
		double tau_tmp = 1/(1+exp_wTx[j]);
		tau[j] = C[GETI(j)]*tau_tmp;
		D[j] = C[GETI(j)]*exp_wTx[j]*tau_tmp*tau_tmp;
	}

	while(newton_iter < max_newton_iter)
	{
		Gmax_new = 0;
		Gnorm1_new = 0;
		active_size = w_size;

		for(s=0; s<active_size; s++)
		{
			j = index[s];
			Hdiag[j] = nu;
			Grad[j] = 0;

			double tmp = 0;
			x = prob_col->x[j];
			while(x->index != -1)
			{
				int ind = x->index-1;
				Hdiag[j] += x->value*x->value*D[ind];
				tmp += x->value*tau[ind];
				x++;
			}
			Grad[j] = -tmp + xjneg_sum[j];

			double Gp = Grad[j]+1;
			double Gn = Grad[j]-1;
			double violation = 0;
			if(w[j] == 0)
			{
				if(Gp < 0)
					violation = -Gp;
				else if(Gn > 0)
					violation = Gn;
				//outer-level shrinking
				else if(Gp>Gmax_old/l && Gn<-Gmax_old/l)
				{
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
			}
			else if(w[j] > 0)
				violation = fabs(Gp);
			else
				violation = fabs(Gn);

			Gmax_new = max(Gmax_new, violation);
			Gnorm1_new += violation;
		}

		if(newton_iter == 0)
			Gnorm1_init = Gnorm1_new;

		if(Gnorm1_new <= eps*Gnorm1_init)
			break;

		iter = 0;
		QP_Gmax_old = INF;
		QP_active_size = active_size;

		for(int i=0; i<l; i++)
			xTd[i] = 0;

		// optimize QP over wpd
		while(iter < max_iter)
		{
			QP_Gmax_new = 0;
			QP_Gnorm1_new = 0;

			for(j=0; j<QP_active_size; j++)
			{
				int i = j+rand()%(QP_active_size-j);
				swap(index[i], index[j]);
			}

			for(s=0; s<QP_active_size; s++)
			{
				j = index[s];
				H = Hdiag[j];

				x = prob_col->x[j];
				G = Grad[j] + (wpd[j]-w[j])*nu;
				while(x->index != -1)
				{
					int ind = x->index-1;
					G += x->value*D[ind]*xTd[ind];
					x++;
				}

				double Gp = G+1;
				double Gn = G-1;
				double violation = 0;
				if(wpd[j] == 0)
				{
					if(Gp < 0)
						violation = -Gp;
					else if(Gn > 0)
						violation = Gn;
					//inner-level shrinking
					else if(Gp>QP_Gmax_old/l && Gn<-QP_Gmax_old/l)
					{
						QP_active_size--;
						swap(index[s], index[QP_active_size]);
						s--;
						continue;
					}
				}
				else if(wpd[j] > 0)
					violation = fabs(Gp);
				else
					violation = fabs(Gn);

				QP_Gmax_new = max(QP_Gmax_new, violation);
				QP_Gnorm1_new += violation;

				// obtain solution of one-variable problem
				if(Gp < H*wpd[j])
					z = -Gp/H;
				else if(Gn > H*wpd[j])
					z = -Gn/H;
				else
					z = -wpd[j];

				if(fabs(z) < 1.0e-12)
					continue;
				z = min(max(z,-10.0),10.0);

				wpd[j] += z;

				x = prob_col->x[j];
				while(x->index != -1)
				{
					int ind = x->index-1;
					xTd[ind] += x->value*z;
					x++;
				}
			}

			iter++;

			if(QP_Gnorm1_new <= inner_eps*Gnorm1_init)
			{
				//inner stopping
				if(QP_active_size == active_size)
					break;
				//active set reactivation
				else
				{
					QP_active_size = active_size;
					QP_Gmax_old = INF;
					continue;
				}
			}

			QP_Gmax_old = QP_Gmax_new;
		}

		if(iter >= max_iter)
			info("WARNING: reaching max number of inner iterations\n");

		delta = 0;
		w_norm_new = 0;
		for(j=0; j<w_size; j++)
		{
			delta += Grad[j]*(wpd[j]-w[j]);
			if(wpd[j] != 0)
				w_norm_new += fabs(wpd[j]);
		}
		delta += (w_norm_new-w_norm);

		negsum_xTd = 0;
		for(int i=0; i<l; i++)
			if(y[i] == -1)
				negsum_xTd += C[GETI(i)]*xTd[i];

		int num_linesearch;
		for(num_linesearch=0; num_linesearch < max_num_linesearch; num_linesearch++)
		{
			cond = w_norm_new - w_norm + negsum_xTd - sigma*delta;

			for(int i=0; i<l; i++)
			{
				double exp_xTd = exp(xTd[i]);
				exp_wTx_new[i] = exp_wTx[i]*exp_xTd;
				cond += C[GETI(i)]*log((1+exp_wTx_new[i])/(exp_xTd+exp_wTx_new[i]));
			}

			if(cond <= 0)
			{
				w_norm = w_norm_new;
				for(j=0; j<w_size; j++)
					w[j] = wpd[j];
				for(int i=0; i<l; i++)
				{
					exp_wTx[i] = exp_wTx_new[i];
					double tau_tmp = 1/(1+exp_wTx[i]);
					tau[i] = C[GETI(i)]*tau_tmp;
					D[i] = C[GETI(i)]*exp_wTx[i]*tau_tmp*tau_tmp;
				}
				break;
			}
			else
			{
				w_norm_new = 0;
				for(j=0; j<w_size; j++)
				{
					wpd[j] = (w[j]+wpd[j])*0.5;
					if(wpd[j] != 0)
						w_norm_new += fabs(wpd[j]);
				}
				delta *= 0.5;
				negsum_xTd *= 0.5;
				for(int i=0; i<l; i++)
					xTd[i] *= 0.5;
			}
		}

		// Recompute some info due to too many line search steps
		if(num_linesearch >= max_num_linesearch)
		{
			for(int i=0; i<l; i++)
				exp_wTx[i] = 0;

			for(int i=0; i<w_size; i++)
			{
				if(w[i]==0) continue;
				x = prob_col->x[i];
				while(x->index != -1)
				{
					exp_wTx[x->index-1] += w[i]*x->value;
					x++;
				}
			}

			for(int i=0; i<l; i++)
				exp_wTx[i] = exp(exp_wTx[i]);
		}

		if(iter == 1)
			inner_eps *= 0.25;

		newton_iter++;
		Gmax_old = Gmax_new;

		info("iter %3d  #CD cycles %d\n", newton_iter, iter);
	}

	info("=========================\n");
	info("optimization finished, #iter = %d\n", newton_iter);
	if(newton_iter >= max_newton_iter)
		info("WARNING: reaching max number of iterations\n");

	// calculate objective value

	double v = 0;
	int nnz = 0;
	for(j=0; j<w_size; j++)
		if(w[j] != 0)
		{
			v += fabs(w[j]);
			nnz++;
		}
	for(j=0; j<l; j++)
		if(y[j] == 1)
			v += C[GETI(j)]*log(1+1/exp_wTx[j]);
		else
			v += C[GETI(j)]*log(1+exp_wTx[j]);

	info("Objective value = %lf\n", v);
	info("#nonzeros/#features = %d/%d\n", nnz, w_size);

	delete [] index;
	delete [] y;
	delete [] Hdiag;
	delete [] Grad;
	delete [] wpd;
	delete [] xjneg_sum;
	delete [] xTd;
	delete [] exp_wTx;
	delete [] exp_wTx_new;
	delete [] tau;
	delete [] D;
}

// transpose matrix X from row format to column format
static void transpose(const problem *prob, feature_node **x_space_ret, problem *prob_col)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	size_t nnz = 0;
	size_t *col_ptr = new size_t [n+1];
	feature_node *x_space;
	prob_col->l = l;
	prob_col->n = n;
	prob_col->y = new double[l];
	prob_col->x = new feature_node*[n];

	for(i=0; i<l; i++)
		prob_col->y[i] = prob->y[i];

	for(i=0; i<n+1; i++)
		col_ptr[i] = 0;
	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			nnz++;
			col_ptr[x->index]++;
			x++;
		}
	}
	for(i=1; i<n+1; i++)
		col_ptr[i] += col_ptr[i-1] + 1;

	x_space = new feature_node[nnz+n];
	for(i=0; i<n; i++)
		prob_col->x[i] = &x_space[col_ptr[i]];

	for(i=0; i<l; i++)
	{
		feature_node *x = prob->x[i];
		while(x->index != -1)
		{
			int ind = x->index-1;
			x_space[col_ptr[ind]].index = i+1; // starts from 1
			x_space[col_ptr[ind]].value = x->value;
			col_ptr[ind]++;
			x++;
		}
	}
	for(i=0; i<n; i++)
		x_space[col_ptr[i]].index = -1;

	*x_space_ret = x_space;

	delete [] col_ptr;
}

// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void group_classes(const problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

static void train_one(const problem *prob, const parameter *param, double *w, double Cp, double Cn)
{
	double eps=param->eps;
	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;

	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case L2R_LR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
			{
				if(prob->y[i] > 0)
					C[i] = Cp;
				else
					C[i] = Cn;
			}
			fun_obj=new l2r_l2_svc_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case L2R_L2LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L2LOSS_SVC_DUAL);
			break;
		case L2R_L1LOSS_SVC_DUAL:
			solve_l2r_l1l2_svc(prob, w, eps, Cp, Cn, L2R_L1LOSS_SVC_DUAL);
			break;
		case L1R_L2LOSS_SVC:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_l2_svc(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L1R_LR:
		{
			problem prob_col;
			feature_node *x_space = NULL;
			transpose(prob, &x_space ,&prob_col);
			solve_l1r_lr(&prob_col, w, primal_solver_tol, Cp, Cn);
			delete [] prob_col.y;
			delete [] prob_col.x;
			delete [] x_space;
			break;
		}
		case L2R_LR_DUAL:
			solve_l2r_lr_dual(prob, w, eps, Cp, Cn);
			break;
		case L2R_L2LOSS_SVR:
		{
			double *C = new double[prob->l];
			for(int i = 0; i < prob->l; i++)
				C[i] = param->C;

			fun_obj=new l2r_l2_svr_fun(prob, C, param->p);
			TRON tron_obj(fun_obj, param->eps);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;

		}
		case L2R_L1LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L1LOSS_SVR_DUAL);
			break;
		case L2R_L2LOSS_SVR_DUAL:
			solve_l2r_l1l2_svr(prob, w, param, L2R_L2LOSS_SVR_DUAL);
			break;
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i,j;
	int l = prob->l;
	int n = prob->n;
	int w_size = prob->n;
	model *model_ = Malloc(model,1);

	if(prob->bias>=0)
		model_->nr_feature=n-1;
	else
		model_->nr_feature=n;
	model_->param = *param;
	model_->bias = prob->bias;

	if(check_regression_model(model_))
	{
		model_->w = Malloc(double, w_size);
		model_->nr_class = 2;
		model_->label = NULL;
		train_one(prob, param, &model_->w[0], 0, 0);
	}
	else
	{
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		group_classes(prob,&nr_class,&label,&start,&count,perm);

		model_->nr_class=nr_class;
		model_->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model_->label[i] = label[i];

		// calculate weighted C
		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"WARNING: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// constructing the subproblem
		feature_node **x = Malloc(feature_node *,l);
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		int k;
		problem sub_prob;
		sub_prob.l = l;
		sub_prob.n = n;
		sub_prob.x = Malloc(feature_node *,sub_prob.l);
		sub_prob.y = Malloc(double,sub_prob.l);

		for(k=0; k<sub_prob.l; k++)
			sub_prob.x[k] = x[k];

		// multi-class svm by Crammer and Singer
		if(param->solver_type == MCSVM_CS)
		{
			model_->w=Malloc(double, n*nr_class);
			for(i=0;i<nr_class;i++)
				for(j=start[i];j<start[i]+count[i];j++)
					sub_prob.y[j] = i;
			Solver_MCSVM_CS Solver(&sub_prob, nr_class, weighted_C, param->eps);
			Solver.Solve(model_->w);
		}
		else
		{
			if(nr_class == 2)
			{
				model_->w=Malloc(double, w_size);

				int e0 = start[0]+count[0];
				k=0;
				for(; k<e0; k++)
					sub_prob.y[k] = +1;
				for(; k<sub_prob.l; k++)
					sub_prob.y[k] = -1;

				train_one(&sub_prob, param, &model_->w[0], weighted_C[0], weighted_C[1]);
			}
			else
			{
				model_->w=Malloc(double, w_size*nr_class);
				double *w=Malloc(double, w_size);
				for(i=0;i<nr_class;i++)
				{
					int si = start[i];
					int ei = si+count[i];

					k=0;
					for(; k<si; k++)
						sub_prob.y[k] = -1;
					for(; k<ei; k++)
						sub_prob.y[k] = +1;
					for(; k<sub_prob.l; k++)
						sub_prob.y[k] = -1;

					train_one(&sub_prob, param, w, weighted_C[i], param->C);

					for(int jj=0;jj<w_size;jj++)
						model_->w[jj*nr_class+i] = w[jj];
				}
				free(w);
			}

		}

		free(x);
		free(label);
		free(start);
		free(count);
		free(perm);
		free(sub_prob.x);
		free(sub_prob.y);
		free(weighted_C);
	}
	return model_;
}

void cross_validation(const problem *prob, const parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	for(i=0;i<l;i++) perm[i]=i;
	for(i=0;i<l;i++)
	{
		int j = i+rand()%(l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<=nr_fold;i++)
		fold_start[i]=i*l/nr_fold;

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct problem subprob;

		subprob.bias = prob->bias;
		subprob.n = prob->n;
		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct feature_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct model *submodel = train(&subprob,param);
		for(j=begin;j<end;j++)
			target[perm[j]] = predict(submodel,prob->x[perm[j]]);
		free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);
}

double predict_values(const struct model *model_, const struct feature_node *x, double *dec_values)
{
	int idx;
	int n;
	if(model_->bias>=0)
		n=model_->nr_feature+1;
	else
		n=model_->nr_feature;
	double *w=model_->w;
	int nr_class=model_->nr_class;
	int i;
	int nr_w;
	if(nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	const feature_node *lx=x;
	for(i=0;i<nr_w;i++)
		dec_values[i] = 0;
	for(; (idx=lx->index)!=-1; lx++)
	{
		// the dimension of testing data may exceed that of training
		if(idx<=n)
			for(i=0;i<nr_w;i++)
				dec_values[i] += w[(idx-1)*nr_w+i]*lx->value;
	}

	if(nr_class==2)
	{
		if(check_regression_model(model_))
			return dec_values[0];
		else
			return (dec_values[0]>0)?model_->label[0]:model_->label[1];
	}
	else
	{
		int dec_max_idx = 0;
		for(i=1;i<nr_class;i++)
		{
			if(dec_values[i] > dec_values[dec_max_idx])
				dec_max_idx = i;
		}
		return model_->label[dec_max_idx];
	}
}

double predict(const model *model_, const feature_node *x)
{
	double *dec_values = Malloc(double, model_->nr_class);
	double label=predict_values(model_, x, dec_values);
	free(dec_values);
	return label;
}

double predict_probability(const struct model *model_, const struct feature_node *x, double* prob_estimates)
{
	if(check_probability_model(model_))
	{
		int i;
		int nr_class=model_->nr_class;
		int nr_w;
		if(nr_class==2)
			nr_w = 1;
		else
			nr_w = nr_class;

		double label=predict_values(model_, x, prob_estimates);
		for(i=0;i<nr_w;i++)
			prob_estimates[i]=1/(1+exp(-prob_estimates[i]));

		if(nr_class==2) // for binary classification
			prob_estimates[1]=1.-prob_estimates[0];
		else
		{
			double sum=0;
			for(i=0; i<nr_class; i++)
				sum+=prob_estimates[i];

			for(i=0; i<nr_class; i++)
				prob_estimates[i]=prob_estimates[i]/sum;
		}

		return label;
	}
	else
		return 0;
}

static const char *solver_type_table[]=
{
	"L2R_LR", "L2R_L2LOSS_SVC_DUAL", "L2R_L2LOSS_SVC", "L2R_L1LOSS_SVC_DUAL", "MCSVM_CS",
	"L1R_L2LOSS_SVC", "L1R_LR", "L2R_LR_DUAL",
	"", "", "",
	"L2R_L2LOSS_SVR", "L2R_L2LOSS_SVR_DUAL", "L2R_L1LOSS_SVR_DUAL", NULL
};

int save_model(const char *model_file_name, const struct model *model_)
{
	int i;
	int nr_feature=model_->nr_feature;
	int n;
	const parameter& param = model_->param;

	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	int nr_w;
	if(model_->nr_class==2 && model_->param.solver_type != MCSVM_CS)
		nr_w=1;
	else
		nr_w=model_->nr_class;

	fprintf(fp, "solver_type %s\n", solver_type_table[param.solver_type]);
	fprintf(fp, "nr_class %d\n", model_->nr_class);

	if(model_->label)
	{
		fprintf(fp, "label");
		for(i=0; i<model_->nr_class; i++)
			fprintf(fp, " %d", model_->label[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "nr_feature %d\n", nr_feature);

	fprintf(fp, "bias %.16g\n", model_->bias);

	fprintf(fp, "w\n");
	for(i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fprintf(fp, "%.16g ", model_->w[i*nr_w+j]);
		fprintf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

struct model *load_model(const char *model_file_name)
{
	FILE *fp = fopen(model_file_name,"r");
	if(fp==NULL) return NULL;

	// int i;
	int nr_feature;
	int n;
	int nr_class;
	double bias;
	model *model_ = Malloc(model,1);
	parameter& param = model_->param;

	model_->label = NULL;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	char cmd[81];
	while(1)
	{
		fscanf(fp,"%80s",cmd);
		if(strcmp(cmd,"solver_type")==0)
		{
			fscanf(fp,"%80s",cmd);
			int i;
			for(i=0;solver_type_table[i];i++)
			{
				if(strcmp(solver_type_table[i],cmd)==0)
				{
					param.solver_type=i;
					break;
				}
			}
			if(solver_type_table[i] == NULL)
			{
				fprintf(stderr,"unknown solver type.\n");

				setlocale(LC_ALL, old_locale);
				free(model_->label);
				free(model_);
				free(old_locale);
				return NULL;
			}
		}
		else if(strcmp(cmd,"nr_class")==0)
		{
			fscanf(fp,"%d",&nr_class);
			model_->nr_class=nr_class;
		}
		else if(strcmp(cmd,"nr_feature")==0)
		{
			fscanf(fp,"%d",&nr_feature);
			model_->nr_feature=nr_feature;
		}
		else if(strcmp(cmd,"bias")==0)
		{
			fscanf(fp,"%lf",&bias);
			model_->bias=bias;
		}
		else if(strcmp(cmd,"w")==0)
		{
			break;
		}
		else if(strcmp(cmd,"label")==0)
		{
			int _nr_class = model_->nr_class;
			model_->label = Malloc(int,_nr_class);
			for(int i=0;i<_nr_class;i++)
				fscanf(fp,"%d",&model_->label[i]);
		}
		else
		{
			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
			setlocale(LC_ALL, old_locale);
			free(model_->label);
			free(model_);
			free(old_locale);
			return NULL;
		}
	}

	nr_feature=model_->nr_feature;
	if(model_->bias>=0)
		n=nr_feature+1;
	else
		n=nr_feature;
	int w_size = n;
	int nr_w;
	if(nr_class==2 && param.solver_type != MCSVM_CS)
		nr_w = 1;
	else
		nr_w = nr_class;

	model_->w=Malloc(double, w_size*nr_w);
	for(int i=0; i<w_size; i++)
	{
		int j;
		for(j=0; j<nr_w; j++)
			fscanf(fp, "%lf ", &model_->w[i*nr_w+j]);
		fscanf(fp, "\n");
	}

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return NULL;

	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

int get_nr_class(const model *model_)
{
	return model_->nr_class;
}

void get_labels(const model *model_, int* label)
{
	if (model_->label != NULL)
		for(int i=0;i<model_->nr_class;i++)
			label[i] = model_->label[i];
}

// use inline here for better performance (around 20% faster than the non-inline one)
static inline double get_w_value(const struct model *model_, int idx, int label_idx)
{
	int nr_class = model_->nr_class;
	int solver_type = model_->param.solver_type;
	const double *w = model_->w;

	if(idx < 0 || idx > model_->nr_feature)
		return 0;
	if(check_regression_model(model_))
		return w[idx];
	else
	{
		if(label_idx < 0 || label_idx >= nr_class)
			return 0;
		if(nr_class == 2 && solver_type != MCSVM_CS)
		{
			if(label_idx == 0)
				return w[idx];
			else
				return -w[idx];
		}
		else
			return w[idx*nr_class+label_idx];
	}
}

// feat_idx: starting from 1 to nr_feature
// label_idx: starting from 0 to nr_class-1 for classification models;
//            for regression models, label_idx is ignored.
double get_decfun_coef(const struct model *model_, int feat_idx, int label_idx)
{
	if(feat_idx > model_->nr_feature)
		return 0;
	return get_w_value(model_, feat_idx-1, label_idx);
}

double get_decfun_bias(const struct model *model_, int label_idx)
{
	int bias_idx = model_->nr_feature;
	double bias = model_->bias;
	if(bias <= 0)
		return 0;
	else
		return bias*get_w_value(model_, bias_idx, label_idx);
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
	if(model_ptr->label != NULL)
		free(model_ptr->label);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

void destroy_param(parameter* param)
{
	if(param->weight_label != NULL)
		free(param->weight_label);
	if(param->weight != NULL)
		free(param->weight);
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->p < 0)
		return "p < 0";

	if(param->solver_type != L2R_LR
		&& param->solver_type != L2R_L2LOSS_SVC_DUAL
		&& param->solver_type != L2R_L2LOSS_SVC
		&& param->solver_type != L2R_L1LOSS_SVC_DUAL
		&& param->solver_type != MCSVM_CS
		&& param->solver_type != L1R_L2LOSS_SVC
		&& param->solver_type != L1R_LR
		&& param->solver_type != L2R_LR_DUAL
		&& param->solver_type != L2R_L2LOSS_SVR
		&& param->solver_type != L2R_L2LOSS_SVR_DUAL
		&& param->solver_type != L2R_L1LOSS_SVR_DUAL)
		return "unknown solver type";

	return NULL;
}

int check_probability_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_LR ||
			model_->param.solver_type==L2R_LR_DUAL ||
			model_->param.solver_type==L1R_LR);
}

int check_regression_model(const struct model *model_)
{
	return (model_->param.solver_type==L2R_L2LOSS_SVR ||
			model_->param.solver_type==L2R_L1LOSS_SVR_DUAL ||
			model_->param.solver_type==L2R_L2LOSS_SVR_DUAL);
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}



#endif /* _LIBLINEAR_H */

}// namespace liblinear
