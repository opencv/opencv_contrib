/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


/*
This file contains implementation of the bio-inspired features (BIF) approach
for computing image descriptors, applicable for human age estimation. For more
details we refer to [1,2].

REFERENCES
  [1] Guo, Guodong, et al. "Human age estimation using bio-inspired features."
      Computer Vision and Pattern Recognition, 2009. CVPR 2009.
  [2] Spizhevoi, A. S., and A. V. Bovyrin. "Estimating human age using
      bio-inspired features and the ranking method." Pattern Recognition and
      Image Analysis 25.3 (2015): 547-552.
*/
#include "precomp.h"
#include "opencv2/face/face_alignment.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <vector>

namespace cv{ namespace face{

const string face_cascade_name = "lbpcascade_frontalface_improved.xml";

class ShapePredictor : public cv::face::FaceAlignment{
 public: 
    virtual int getNumLandmarks() const {return numlandmarks_;}

    virtual int getNumFaces() const {return numfaces_;}


    virtual std::vector<Point2f> getLandmarks(
            Mat img,
            const Rect face,
            std::vector<Point2f> > initial_feats
        ) const;
  private:
    int numlandmarks_;
    int numfaces_;
    std::vector<cv::Rect> faces;
    std::vector<cv::Point2f> landmarks;
    
};










}














}








}

