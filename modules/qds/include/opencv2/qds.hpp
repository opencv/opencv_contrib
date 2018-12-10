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





#ifndef __OPENCV_QDS_H__
#define __OPENCV_QDS_H__

#include <opencv2/qds/quasiDenseStereo.hpp>

/**
 * @defgroup qds Quasi Dense Stereo
 * This module contains the code to perform quasi dense stereo matching.
 * The method initially starts with a sparse 3D reconstruction based on feature matching across a
 * stereo image pair and subsequently propagates the structure into neighboring image regions.
 * To obtain initial seed correspondences, the algorithm locates Shi and Tomashi features in the
 * left image of the stereo pair and then tracks them using pyramidal Lucas-Kanade in the right image.
 * To densify the sparse correspondences, the algorithm computes the zero-mean normalized
 * cross-correlation (ZNCC) in small patches around every seed pair and uses it as a quality metric
 * for each match. In this code, we introduce a custom structure to store the location and ZNCC value
 * of correspondences called "Match". Seed Matches are stored in a priority queue sorted according to
 * their ZNCC value, allowing for the best quality Match to be readily available. The algorithm pops
 * Matches and uses them to extract new matches around them. This is done by considering a small
 * neighboring area around each Seed and retrieving correspondences above a certain texture threshold
 * that are not previously computed. New matches are stored in the seed priority queue and used as seeds.
 * The propagation process ends when no additional matches can be retrieved.
 *
 *
 * @sa This code represents the work presented in @cite Stoyanov2010.
 * If this code is useful for your work please cite @cite Stoyanov2010.
 *
 * Also the original growing scheme idea is described in @cite Lhuillier2000
 *
*/
#endif // __OPENCV_QDS_H__
