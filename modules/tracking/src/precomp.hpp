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
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include "opencv2/tracking.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include <typeinfo>
#include "opencv2/core/hal/hal.hpp"

namespace cv
{
	extern const double ColorNames[][10];

    namespace tracking {

    /* Cholesky decomposition
     The function performs Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>.
     A - the Hermitian, positive-definite matrix,
     astep - size of row in A,
     asize - number of cols and rows in A,
     L - the lower triangular matrix, A = L*Lt.
    */

    template<typename _Tp> bool
    inline callHalCholesky( _Tp* L, size_t lstep, int lsize );

    template<> bool
    inline callHalCholesky<float>( float* L, size_t lstep, int lsize )
    {
        return hal::Cholesky32f(L, lstep, lsize, NULL, 0, 0);
    }

    template<> bool
    inline callHalCholesky<double>( double* L, size_t lstep, int lsize)
    {
        return hal::Cholesky64f(L, lstep, lsize, NULL, 0, 0);
    }

    template<typename _Tp> bool
    inline choleskyDecomposition( const _Tp* A, size_t astep, int asize, _Tp* L, size_t lstep )
    {
        bool success = false;

        astep /= sizeof(_Tp);
        lstep /= sizeof(_Tp);

        for(int i = 0; i < asize; i++)
            for(int j = 0; j <= i; j++)
                L[i*lstep + j] = A[i*astep + j];

       success = callHalCholesky(L, lstep*sizeof(_Tp), asize);

       if(success)
       {
           for(int i = 0; i < asize; i++ )
               for(int j = i + 1; j < asize; j++ )
                   L[i*lstep + j] = 0.0;
       }

        return success;
    }

    } // tracking
} // cv

#endif
