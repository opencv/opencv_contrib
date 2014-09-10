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

#define BING_IMAGES \
    "cv/saliency/objectness/000021.jpg", \
"cv/saliency/objectness/000022.jpg"

void getMatOfRects( const vector<Vec4i>& saliencyMap, Mat& bbs_mat )
{
  for ( size_t b = 0; b < saliencyMap.size(); b++ )
  {
    bbs_mat.at<int>( b, 0 ) = saliencyMap[b].val[0];
    bbs_mat.at<int>( b, 1 ) = saliencyMap[b].val[1];
    bbs_mat.at<int>( b, 2 ) = saliencyMap[b].val[2];
    bbs_mat.at<int>( b, 3 ) = saliencyMap[b].val[3];
  }
}

PERF_TEST_P(sal, objectnessBING, testing::Values(BING_IMAGES))
{
  string filename = getDataPath(GetParam());
  cout<<endl<<endl<<"path "<<filename<<endl<<endl;
  Mat image = imread(filename);
  vector<Vec4i> saliencyMap;
  String training_path = "/home/puja/src/opencv_contrib/modules/saliency/samples/ObjectnessTrainedModel";

  if (image.empty())
  FAIL() << "Unable to load source image " << filename;

  Ptr<saliency::Saliency> saliencyAlgorithm = saliency::Saliency::create( "BING" );

  TEST_CYCLE_N(1)
  {
    if( training_path.empty() )
    {
      FAIL() << "Path of trained files missing! " << endl;
      return;
    }

    else
    {
      saliencyAlgorithm.dynamicCast<saliency::ObjectnessBING>()->setTrainingPath( training_path );
      saliencyAlgorithm.dynamicCast<saliency::ObjectnessBING>()->setBBResDir( training_path + "/Results" );

      if( saliencyAlgorithm->computeSaliency( image, saliencyMap ) )
      {

      }
    }
  }  //end CYCLE

  //save the bounding boxes in a Mat
   Mat bbs_mat( saliencyMap.size(), 4, CV_32F );
   getMatOfRects( saliencyMap, bbs_mat );

   SANITY_CHECK( bbs_mat);
}
