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
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
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

#include "perf_precomp.hpp"

//write sanity: ./bin/opencv_perf_saliency --perf_write_sanity=true
//verify sanity: ./bin/opencv_perf_saliency

using namespace std;
using namespace cv;
using namespace perf;
using std::tr1::make_tuple;
using std::tr1::get;

typedef perf::TestBaseWithParam<std::string> sal;

#define STATIC_IMAGES \
    "cv/saliency/static_saliency/8.jpg",\
    "cv/saliency/static_saliency/39.jpg" ,\
    "cv/saliency/static_saliency/41.jpg" ,\
    "cv/saliency/static_saliency/62.jpg"

PERF_TEST_P(sal, statiSaliencySpectralResidual, testing::Values(STATIC_IMAGES))
{
  string filename = getDataPath(GetParam());
  Mat image = imread(filename);
  Mat saliencyMap;

  if (image.empty())
  FAIL() << "Unable to load source image " << filename;

  Ptr<saliency::Saliency> saliencyAlgorithm = saliency::Saliency::create( "SPECTRAL_RESIDUAL" );

  TEST_CYCLE_N(1)
  {

    if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
    {

    }
    else
    {
      FAIL()<< "***Error in the instantiation of the saliency algorithm...***\n" << endl;
      return;
    }
  } //end CYCLE

  SANITY_CHECK(saliencyMap);
}
