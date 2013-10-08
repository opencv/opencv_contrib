#ifndef FUNCTION_SC
#define FUNCTION_SC

#include "_lsvmc_types.h"

namespace cv
{
namespace lsvmc
{

float calcM         (int k,int di,int dj, const CvLSVMFeaturePyramidCaskade * H, const CvLSVMFilterObjectCaskade *filter);
float calcM_PCA     (int k,int di,int dj, const CvLSVMFeaturePyramidCaskade * H, const CvLSVMFilterObjectCaskade *filter);
float calcM_PCA_cash(int k,int di,int dj, const CvLSVMFeaturePyramidCaskade * H, const CvLSVMFilterObjectCaskade *filter, float * cashM, int * maskM, int step);
float calcFine (const CvLSVMFilterObjectCaskade *filter, int di, int dj);
}
}
#endif