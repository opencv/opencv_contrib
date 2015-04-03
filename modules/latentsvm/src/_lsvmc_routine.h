/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2013, University of Nizhny Novgorod, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef _LSVM_ROUTINE_H_
#define _LSVM_ROUTINE_H_

#include "_lsvmc_types.h"
#include "_lsvmc_error.h"

namespace cv
{
namespace lsvm
{


//////////////////////////////////////////////////////////////
// Memory management routines
// All paramaters names correspond to previous data structures description
// All "alloc" functions return allocated memory for 1 object
// with all fields including arrays
// Error status is return value
//////////////////////////////////////////////////////////////
int allocFilterObject(CvLSVMFilterObjectCascade **obj, const int sizeX, const int sizeY, 
                      const int p);
int freeFilterObject (CvLSVMFilterObjectCascade **obj);

int allocFeatureMapObject(CvLSVMFeatureMapCascade **obj, const int sizeX, const int sizeY,
                          const int p);
int freeFeatureMapObject (CvLSVMFeatureMapCascade **obj);

int allocFeaturePyramidObject(CvLSVMFeaturePyramidCascade **obj, 
                              const int countLevel);

int freeFeaturePyramidObject (CvLSVMFeaturePyramidCascade **obj);

}
}
#endif
