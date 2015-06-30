UFconfig contains configuration settings for all many of the software packages
that I develop or co-author.  Note that older versions of some of these packages
do not require UFconfig.

  Package  Description
  -------  -----------
  AMD	   approximate minimum degree ordering
  CAMD	   constrained AMD
  COLAMD   column approximate minimum degree ordering
  CCOLAMD  constrained approximate minimum degree ordering
  UMFPACK  sparse LU factorization, with the BLAS
  CXSparse int/long/real/complex version of CSparse
  CHOLMOD  sparse Cholesky factorization, update/downdate
  KLU	   sparse LU factorization, BLAS-free
  BTF	   permutation to block triangular form
  LDL	   concise sparse LDL'
  LPDASA   LP Dual Active Set Algorithm
  SuiteSparseQR     sparse QR factorization

UFconfig is not required by:

  CSparse	a Concise Sparse matrix package
  RBio		read/write files in Rutherford/Boeing format
  UFcollection	tools for managing the UF Sparse Matrix Collection
  LINFACTOR     simple m-file to show how to use LU and CHOL to solve Ax=b
  MESHND        2D and 3D mesh generation and nested dissection ordering
  MATLAB_Tools  misc collection of m-files
  SSMULT        sparse matrix times sparse matrix, for use in MATLAB

In addition, the xerbla/ directory contains Fortan and C versions of the
BLAS/LAPACK xerbla routine, which is called when an invalid input is passed to
the BLAS or LAPACK.  The xerbla provided here does not print any message, so
the entire Fortran I/O library does not need to be linked into a C application.
Most versions of the BLAS contain xerbla, but those from K. Goto do not.  Use
this if you need too.
