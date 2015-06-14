/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef __OPENCV_CNN_3DOBJ_HPP__
#define __OPENCV_CNN_3DOBJ_HPP__

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

/** @defgroup cnn_3dobj CNN based on Caffe aimming at 3D object recognition and pose estimation
*/
namespace cv
{ 
namespace cnn_3dobj
{

//! @addtogroup cnn_3dobj
//! @{

/** @brief Icosohedron based camera view generator.

The class create some sphere views of camera towards a 3D object meshed from .ply files @cite hinterstoisser2008panter .
 */
class CV_EXPORTS_W IcoSphere 
{


	private:
		float X;
		float Z;

	public:


		std::vector<float>* vertexNormalsList = new std::vector<float>;
		std::vector<float>* vertexList = new std::vector<float>;
		std::vector<cv::Point3d>* CameraPos = new std::vector<cv::Point3d>;
		float radius;
		IcoSphere(float radius_in, int depth_in);
		/** @brief Make all view points having the some distance from the focal point used by the camera view.
		*/
		CV_WRAP void norm(float v[]);
		/** @brief Add new view point between 2 point of the previous view point.
		*/
		CV_WRAP void add(float v[]);
		/** @brief Generating new view points from all triangles.
		*/
		CV_WRAP void subdivide(float v1[], float v2[], float v3[], int depth);

};
//! @}

}}



#endif /* CNN_3DOBJ_HPP_ */
