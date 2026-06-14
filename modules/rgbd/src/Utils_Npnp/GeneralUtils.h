#ifndef PNP_USING_EIGEN_LIBRARY_GENERALUTILS_H
#define PNP_USING_EIGEN_LIBRARY_GENERALUTILS_H
#ifdef HAVE_EIGEN
#include "Definitions.h"
#include "functional"

namespace NPnP
{
  double find_zero_bin_search(const std::function<double(double)> &func,
                              double min, double max, int depth);

  template <typename T>
  inline T min2(T one, T two)
  {
    return one < two ? one : two;
  }
} // namespace NPnP

#endif
#endif // PNP_USING_EIGEN_LIBRARY_GENERALUTILS_H
