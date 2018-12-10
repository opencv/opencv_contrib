/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2015-2018, OpenCV Foundation, all rights reserved.
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
#ifndef __OPENCV_DEFAULTS_H__
#define __OPENCV_DEFAULTS_H__

// borders around the image
#define BORDER_X 15
#define BORDER_Y 15

#define CORR_WIN_SIZE_X 5		// corr window size
#define CORR_WIN_SIZE_Y 5

#define NEIGHBORHOOD_SIZE  5					// neighbors
#define CORR_THRESHOLD 0.5					// corr threshold for seeds
#define TEXTURE_THRESHOLD 200					// texture threshold for seeds
#define DISPARITY_GRADIENT 1					// disparity gradient


#define LK_FLOW_TEMPLAETE_SIZE 3
#define LK_FLOW_PYR_LVL 3
#define LK_FLOW_TERM_1 3
#define LK_FLOW_TERM_2 0.003

#define GFT_QUALITY_THRESHOLD 0.01
#define GFT_MIN_SEPERATION_DIST 10
#define GFT_MAX_NUM_FEATURES 500

#define DESPARITY_LVLS 50


#define NO_DISPARITY 0
#define NO_MATCH cv::Point(0,0)

#endif //__OPENCV_DEFAULTS_H__
