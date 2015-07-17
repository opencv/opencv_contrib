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

#ifndef __OPENCV_CNN_3DOBJ_HPP__
#define __OPENCV_CNN_3DOBJ_HPP__
#ifdef __cplusplus
#include <glog/logging.h>
#include <leveldb/db.h>
#include <caffe/proto/caffe.pb.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
//#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <string>
#include <set>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>
using std::string;
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
		std::vector<float> vertexNormalsList;
		std::vector<float> vertexList;
		std::vector<cv::Point3d> CameraPos;
		std::vector<cv::Point3d> CameraPos_temp;
		float radius;
		float diff;
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
		/** @brief Make all view points having the some distance from the focal point used by the camera view.
		*/
		CV_WRAP static uint32_t swap_endian(uint32_t val);
		/** @brief Suit the position of bytes in 4 byte data structure for particular system.
		*/
		CV_WRAP cv::Point3d getCenter(cv::Mat cloud);
		/** @brief Get the center of points on surface in .ply model.
		*/
		CV_WRAP float getRadius(cv::Mat cloud, cv::Point3d center);
		/** @brief Get the proper camera radius from the view point to the center of model.
		*/
		CV_WRAP static void createHeader(int num_item, int rows, int cols, const char* headerPath);
		/** @brief Create header in binary files collecting the image data and label.
		*/
		CV_WRAP static void writeBinaryfile(string filenameImg, const char* binaryPath, const char* headerPath, int num_item, int label_class);
		/** @brief Write binary files used for training in other open source project.
		*/

};

class CV_EXPORTS_W DataTrans
{
	private:
		std::set<string> all_class_name;
		std::map<string,int> class2id;
	public:
		DataTrans();
		CV_WRAP void list_dir(const char *path,std::vector<string>& files,bool r);
		CV_WRAP string get_classname(string path);
		CV_WRAP int get_labelid(string fileName);
		CV_WRAP void loadimg(string path,char* buffer,bool is_color);
		CV_WRAP void convert(string imgdir,string outputdb,string attachdir,int channel,int width,int height);	
};
//! @}
}}



#endif /* CNN_3DOBJ_HPP_ */
#endif
