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
#include <fstream>
#include <opencv2/videoio.hpp>

//write sanity: ./bin/opencv_perf_saliency --perf_write_sanity=true
//verify sanity: ./bin/opencv_perf_saliency

using namespace std;
using namespace cv;
using namespace perf;

#define TESTSET_NAMES \
    "/cv/saliency/motion/blizzard.webm",\
    "/cv/saliency/motion/pedestrian.webm" ,\
    "/cv/saliency/motion/snowFall.webm"

const string SALIENCY_DIR = "cv/saliency";

typedef perf::TestBaseWithParam<std::string> sal;

PERF_TEST_P(sal, motionSaliencyBinWangApr2014, testing::Values(TESTSET_NAMES))
{
  string filename = getDataPath(GetParam());
  int startFrame=0;
  Mat frame;
  Mat saliencyMap;
  int videoSize=0;

  Ptr<saliency::Saliency> saliencyAlgorithm = saliency::Saliency::create( "BinWangApr2014" );

  TEST_CYCLE_N(1)
  {
    VideoCapture c;
    c.open( filename);
    videoSize=c.get( CAP_PROP_FRAME_COUNT);
    c.set( CAP_PROP_POS_FRAMES, startFrame );

    for ( int frameCounter = 0; frameCounter < videoSize; frameCounter++ )
    {
      c >> frame;

      if( frame.empty() )
      {
        break;
      }

      saliencyAlgorithm.dynamicCast<saliency::MotionSaliencyBinWangApr2014>()->setImagesize( frame.cols, frame.rows );
      saliencyAlgorithm.dynamicCast<saliency::MotionSaliencyBinWangApr2014>()->init();

      if( saliencyAlgorithm->computeSaliency( frame, saliencyMap ) )
      {

      }
      else
      {
        FAIL()<< "***Error in the instantiation of the saliency algorithm...***\n" << endl;
        return;
      }

    }
  }

  SANITY_CHECK(saliencyMap);

}
