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

#include <assert.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Defines a matrix structure for holding double-precision values with
 * data in row-major order (i.e. index = row*ncols + col).
 *
 * nrows and ncols are 1-based counts with the exception that a scalar (non-matrix)
 *   is represented with nrows=0 and/or ncols=0.
 */
typedef struct
{
    unsigned int nrows, ncols;
    double data[];
//    double *data;
} matd_t;

#define MATD_ALLOC(name, nrows, ncols) double name ## _storage [nrows*ncols]; matd_t name = { .nrows = nrows, .ncols = ncols, .data = &name ## _storage };

/**
 * Defines a small value which can be used in place of zero for approximating
 * calculations which are singular at zero values (i.e. inverting a matrix with
 * a zero or near-zero determinant).
 */
#define MATD_EPS 1e-8

/**
 * A macro to reference a specific matd_t data element given it's zero-based
 * row and column indexes. Suitable for both retrieval and assignment.
 */
#define MATD_EL(m, row, col) (m)->data[((row)*(m)->ncols + (col))]

/**
 * Creates a double matrix with the given number of rows and columns (or a scalar
 * in the case where rows=0 and/or cols=0). All data elements will be initialized
 * to zero. It is the caller's responsibility to call matd_destroy() on the
 * returned matrix.
 */
matd_t *matd_create(int rows, int cols);

/**
 * Creates a double matrix with the given number of rows and columns (or a scalar
 * in the case where rows=0 and/or cols=0). All data elements will be initialized
 * using the supplied array of data, which must contain at least rows*cols elements,
 * arranged in row-major order (i.e. index = row*ncols + col). It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 */
matd_t *matd_create_data(int rows, int cols, const double *data);

/**
 * Creates a double matrix with the given number of rows and columns (or a scalar
 * in the case where rows=0 and/or cols=0). All data elements will be initialized
 * using the supplied array of float data, which must contain at least rows*cols elements,
 * arranged in row-major order (i.e. index = row*ncols + col). It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 */
matd_t *matd_create_dataf(int rows, int cols, const float *data);

/**
 * Creates a square identity matrix with the given number of rows (and
 * therefore columns), or a scalar with value 1 in the case where dim=0.
 * It is the caller's responsibility to call matd_destroy() on the
 * returned matrix.
 */
matd_t *matd_identity(int dim);

/**
 * Creates a scalar with the supplied value 'v'. It is the caller's responsibility
 * to call matd_destroy() on the returned matrix.
 *
 * NOTE: Scalars are different than 1x1 matrices (implementation note:
 * they are encoded as 0x0 matrices). For example: for matrices A*B, A
 * and B must both have specific dimensions. However, if A is a
 * scalar, there are no restrictions on the size of B.
 */
matd_t *matd_create_scalar(double v);

/**
 * Retrieves the cell value for matrix 'm' at the given zero-based row and column index.
 * Performs more thorough validation checking than MATD_EL().
 */
double matd_get(const matd_t *m, int row, int col);

/**
 * Assigns the given value to the matrix cell at the given zero-based row and
 * column index. Performs more thorough validation checking than MATD_EL().
 */
void matd_put(matd_t *m, int row, int col, double value);

/**
 * Retrieves the scalar value of the given element ('m' must be a scalar).
 * Performs more thorough validation checking than MATD_EL().
 */
double matd_get_scalar(const matd_t *m);

/**
 * Assigns the given value to the supplied scalar element ('m' must be a scalar).
 * Performs more thorough validation checking than MATD_EL().
 */
void matd_put_scalar(matd_t *m, double value);

/**
 * Creates an exact copy of the supplied matrix 'm'. It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 */
matd_t *matd_copy(const matd_t *m);

/**
 * Creates a copy of a subset of the supplied matrix 'a'. The subset will include
 * rows 'r0' through 'r1', inclusive ('r1' >= 'r0'), and columns 'c0' through 'c1',
 * inclusive ('c1' >= 'c0'). All parameters are zero-based (i.e. matd_select(a, 0, 0, 0, 0)
 * will return only the first cell). Cannot be used on scalars or to extend
 * beyond the number of rows/columns of 'a'. It is the caller's  responsibility to
 * call matd_destroy() on the returned matrix.
 */
matd_t *matd_select(const matd_t *a, int r0, int r1, int c0, int c1);

/**
 * Prints the supplied matrix 'm' to standard output by applying the supplied
 * printf format specifier 'fmt' for each individual element. Each row will
 * be printed on a separate newline.
 */
void matd_print(const matd_t *m, const char *fmt);

/**
 * Prints the transpose of the supplied matrix 'm' to standard output by applying
 * the supplied printf format specifier 'fmt' for each individual element. Each
 * row will be printed on a separate newline.
 */
void matd_print_transpose(const matd_t *m, const char *fmt);

/**
 * Adds the two supplied matrices together, cell-by-cell, and returns the results
 * as a new matrix of the same dimensions. The supplied matrices must have
 * identical dimensions.  It is the caller's responsibility to call matd_destroy()
 * on the returned matrix.
 */
matd_t *matd_add(const matd_t *a, const matd_t *b);

/**
 * Adds the values of 'b' to matrix 'a', cell-by-cell, and overwrites the
 * contents of 'a' with the results. The supplied matrices must have
 * identical dimensions.
 */
void matd_add_inplace(matd_t *a, const matd_t *b);

/**
 * Subtracts matrix 'b' from matrix 'a', cell-by-cell, and returns the results
 * as a new matrix of the same dimensions. The supplied matrices must have
 * identical dimensions.  It is the caller's responsibility to call matd_destroy()
 * on the returned matrix.
 */
matd_t *matd_subtract(const matd_t *a, const matd_t *b);

/**
 * Subtracts the values of 'b' from matrix 'a', cell-by-cell, and overwrites the
 * contents of 'a' with the results. The supplied matrices must have
 * identical dimensions.
 */
void matd_subtract_inplace(matd_t *a, const matd_t *b);

/**
 * Scales all cell values of matrix 'a' by the given scale factor 's' and
 * returns the result as a new matrix of the same dimensions. It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 */
matd_t *matd_scale(const matd_t *a, double s);

/**
 * Scales all cell values of matrix 'a' by the given scale factor 's' and
 * overwrites the contents of 'a' with the results.
 */
void matd_scale_inplace(matd_t *a, double s);

/**
 * Multiplies the two supplied matrices together (matrix product), and returns the
 * results as a new matrix. The supplied matrices must have dimensions such that
 * columns(a) = rows(b). The returned matrix will have a row count of rows(a)
 * and a column count of columns(b). It is the caller's responsibility to call
 * matd_destroy() on the returned matrix.
 */
matd_t *matd_multiply(const matd_t *a, const matd_t *b);

/**
 * Creates a matrix which is the transpose of the supplied matrix 'a'. It is the
 * caller's responsibility to call matd_destroy() on the returned matrix.
 */
matd_t *matd_transpose(const matd_t *a);

/**
 * Calculates the determinant of the supplied matrix 'a'.
 */
double matd_det(const matd_t *a);

/**
 * Attempts to compute an inverse of the supplied matrix 'a' and return it as
 * a new matrix. This is strictly only possible if the determinant of 'a' is
 * non-zero (matd_det(a) != 0).
 *
 * If the determinant is zero, NULL is returned. It is otherwise the
 * caller's responsibility to cope with the results caused by poorly
 * conditioned matrices. (E.g.., if such a situation is likely to arise, compute
 * the pseudo-inverse from the SVD.)
 **/
matd_t *matd_inverse(const matd_t *a);

static inline void matd_set_data(matd_t *m, const double *data)
{
    memcpy(m->data, data, m->nrows * m->ncols * sizeof(double));
}

/**
 * Determines whether the supplied matrix 'a' is a scalar (positive return) or
 * not (zero return, indicating a matrix of dimensions at least 1x1).
 */
static inline int matd_is_scalar(const matd_t *a)
{
    assert(a != NULL);
    return a->ncols <= 1 && a->nrows <= 1;
}

/**
 * Determines whether the supplied matrix 'a' is a row or column vector
 * (positive return) or not (zero return, indicating either 'a' is a scalar or a
 * matrix with at least one dimension > 1).
 */
static inline int matd_is_vector(const matd_t *a)
{
    assert(a != NULL);
    return a->ncols == 1 || a->nrows == 1;
}

/**
 * Determines whether the supplied matrix 'a' is a row or column vector
 * with a dimension of 'len' (positive return) or not (zero return).
 */
static inline int matd_is_vector_len(const matd_t *a, int len)
{
    assert(a != NULL);
    return (a->ncols == 1 && a->nrows == (unsigned int)len) || (a->ncols == (unsigned int)len && a->nrows == 1);
}

/**
 * Calculates the magnitude of the supplied matrix 'a'.
 */
double matd_vec_mag(const matd_t *a);

/**
 * Calculates the magnitude of the distance between the points represented by
 * matrices 'a' and 'b'. Both 'a' and 'b' must be vectors and have the same
 * dimension (although one may be a row vector and one may be a column vector).
 */
double matd_vec_dist(const matd_t *a, const matd_t *b);


/**
 * Same as matd_vec_dist, but only uses the first 'n' terms to compute distance
 */
double matd_vec_dist_n(const matd_t *a, const matd_t *b, int n);

/**
 * Calculates the dot product of two vectors. Both 'a' and 'b' must be vectors
 * and have the same dimension (although one may be a row vector and one may be
 * a column vector).
 */
double matd_vec_dot_product(const matd_t *a, const matd_t *b);

/**
 * Calculates the normalization of the supplied vector 'a' (i.e. a unit vector
 * of the same dimension and orientation as 'a' with a magnitude of 1) and returns
 * it as a new vector. 'a' must be a vector of any dimension and must have a
 * non-zero magnitude. It is the caller's responsibility to call matd_destroy()
 * on the returned matrix.
 */
matd_t *matd_vec_normalize(const matd_t *a);

/**
 * Calculates the cross product of supplied matrices 'a' and 'b' (i.e. a x b)
 * and returns it as a new matrix. Both 'a' and 'b' must be vectors of dimension
 * 3, but can be either row or column vectors. It is the caller's responsibility
 * to call matd_destroy() on the returned matrix.
 */
matd_t *matd_crossproduct(const matd_t *a, const matd_t *b);

double matd_err_inf(const matd_t *a, const matd_t *b);

/**
 * Creates a new matrix by applying a series of matrix operations, as expressed
 * in 'expr', to the supplied list of matrices. Each matrix to be operated upon
 * must be represented in the expression by a separate matrix placeholder, 'M',
 * and there must be one matrix supplied as an argument for each matrix
 * placeholder in the expression. All rules and caveats of the corresponding
 * matrix operations apply to the operated-on matrices. It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 *
 * Available operators (in order of increasing precedence):
 *   M+M   add two matrices together
 *   M-M   subtract one matrix from another
 *   M*M   multiply two matrices together (matrix product)
 *   MM    multiply two matrices together (matrix product)
 *   -M    negate a matrix
 *   M^-1  take the inverse of a matrix
 *   M'    take the transpose of a matrix
 *
 * Expressions can be combined together and grouped by enclosing them in
 * parenthesis, i.e.:
 *   -M(M+M+M)-(M*M)^-1
 *
 * Scalar values can be generated on-the-fly, i.e.:
 *   M*2.2  scales M by 2.2
 *   -2+M   adds -2 to all elements of M
 *
 * All whitespace in the expression is ignored.
 */
matd_t *matd_op(const char *expr, ...);

/**
 * Frees the memory associated with matrix 'm', being the result of an earlier
 * call to a matd_*() function, after which 'm' will no longer be usable.
 */
void matd_destroy(matd_t *m);

typedef struct
{
    matd_t *U;
    matd_t *S;
    matd_t *V;
} matd_svd_t;

/** Compute a complete SVD of a matrix. The SVD exists for all
 * matrices. For a matrix MxN, we will have:
 *
 * A = U*S*V'
 *
 * where A is MxN, U is MxM (and is an orthonormal basis), S is MxN
 * (and is diagonal up to machine precision), and V is NxN (and is an
 * orthonormal basis).
 *
 * The caller is responsible for destroying U, S, and V.
 **/
matd_svd_t matd_svd(matd_t *A);

#define MATD_SVD_NO_WARNINGS 1
    matd_svd_t matd_svd_flags(matd_t *A, int flags);

////////////////////////////////
// PLU Decomposition

// All square matrices (even singular ones) have a partially-pivoted
// LU decomposition such that A = PLU, where P is a permutation
// matrix, L is a lower triangular matrix, and U is an upper
// triangular matrix.
//
typedef struct
{
    // was the input matrix singular? When a zero pivot is found, this
    // flag is set to indicate that this has happened.
    int singular;

    unsigned int *piv; // permutation indices
    int pivsign; // either +1 or -1

    // The matd_plu_t object returned "owns" the enclosed LU matrix. It
    // is not expected that the returned object is itself useful to
    // users: it contains the L and U information all smushed
    // together.
    matd_t *lu; // combined L and U matrices, permuted so they can be triangular.
} matd_plu_t;

matd_plu_t *matd_plu(const matd_t *a);
void matd_plu_destroy(matd_plu_t *mlu);
double matd_plu_det(const matd_plu_t *lu);
matd_t *matd_plu_p(const matd_plu_t *lu);
matd_t *matd_plu_l(const matd_plu_t *lu);
matd_t *matd_plu_u(const matd_plu_t *lu);
matd_t *matd_plu_solve(const matd_plu_t *mlu, const matd_t *b);

// uses LU decomposition internally.
matd_t *matd_solve(matd_t *A, matd_t *b);

////////////////////////////////
// Cholesky Factorization

/**
 * Creates a double matrix with the Cholesky lower triangular matrix
 * of A. A must be symmetric, positive definite. It is the caller's
 * responsibility to call matd_destroy() on the returned matrix.
 */
//matd_t *matd_cholesky(const matd_t *A);

typedef struct
{
    int is_spd;
    matd_t *u;
} matd_chol_t;

matd_chol_t *matd_chol(matd_t *A);
matd_t *matd_chol_solve(const matd_chol_t *chol, const matd_t *b);
void matd_chol_destroy(matd_chol_t *chol);
// only sensible on PSD matrices
matd_t *matd_chol_inverse(matd_t *a);

void matd_ltransposetriangle_solve(matd_t *u, const double *b, double *x);
void matd_ltriangle_solve(matd_t *u, const double *b, double *x);
void matd_utriangle_solve(matd_t *u, const double *b, double *x);


double matd_max(matd_t *m);

#ifdef __cplusplus
}
#endif
