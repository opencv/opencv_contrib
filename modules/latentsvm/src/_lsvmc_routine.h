#ifndef _LSVM_ROUTINE_H_
#define _LSVM_ROUTINE_H_

#include "_lsvmc_types.h"
#include "_lsvmc_error.h"

namespace cv
{
namespace lsvmc
{


//////////////////////////////////////////////////////////////
// Memory management routines
// All paramaters names correspond to previous data structures description
// All "alloc" functions return allocated memory for 1 object
// with all fields including arrays
// Error status is return value
//////////////////////////////////////////////////////////////
int allocFilterObject(CvLSVMFilterObjectCaskade **obj, const int sizeX, const int sizeY, 
                      const int p);
int freeFilterObject (CvLSVMFilterObjectCaskade **obj);

int allocFeatureMapObject(CvLSVMFeatureMapCaskade **obj, const int sizeX, const int sizeY,
                          const int p);
int freeFeatureMapObject (CvLSVMFeatureMapCaskade **obj);

int allocFeaturePyramidObject(CvLSVMFeaturePyramidCaskade **obj, 
                              const int countLevel);

int freeFeaturePyramidObject (CvLSVMFeaturePyramidCaskade **obj);

}
}
#endif
