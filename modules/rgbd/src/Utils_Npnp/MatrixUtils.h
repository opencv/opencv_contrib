#ifndef PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H
#define PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H
#ifdef HAVE_EIGEN
#include "Definitions.h"

namespace NPnP
{
  template <int Size>
  void symmetrize(RowMatrix<Size, Size> &matrix)
  {
    matrix = 0.5 * (matrix + matrix.transpose().eval());
  }
} // namespace NPnP
#endif
#endif // PNP_USING_EIGEN_LIBRARY_MATRIXUTILS_H
