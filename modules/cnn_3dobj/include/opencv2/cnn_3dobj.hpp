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

#include <string>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <set>
#include <string.h>
#include <stdlib.h>
#include <tr1/memory>
#include <dirent.h>
#define CPU_ONLY
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "opencv2/viz/vizcore.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc.hpp"
using std::string;
using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
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
		CV_WRAP static void writeBinaryfile(string filenameImg, const char* binaryPath, const char* headerPath, int num_item, int label_class, int x, int y, int z);
		/** @brief Write binary files used for training in other open source project.
		*/

};

class CV_EXPORTS_W Classification
{
	private:
		caffe::shared_ptr<caffe::Net<float> > net_;
		cv::Size input_geometry_;
		int num_channels_;
		cv::Mat mean_;
		std::vector<string> labels_;
		void SetMean(const string& mean_file);
		/** @brief Load the mean file in binaryproto format.
		*/
		void WrapInputLayer(std::vector<cv::Mat>* input_channels);
		/** @brief Wrap the input layer of the network in separate cv::Mat objects(one per channel). This way we save one memcpy operation and we don't need to rely on cudaMemcpy2D. The last preprocessing operation will write the separate channels directly to the input layer.
		*/
		void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels, bool mean_subtract);
		/** @brief Convert the input image to the input image format of the network.
		*/
	public:
		Classification();
		void list_dir(const char *path,std::vector<string>& files,bool r);
		/** @brief Get the file name from a root dictionary.
		*/
		void NetSetter(const string& model_file, const string& trained_file, const string& mean_file, const string& cpu_only, int device_id);
		/** @brief Initiate a classification structure.
		*/
		void GetLabellist(const std::vector<string>& name_gallery);
		/** @brief Get the label of the gallery images for result displaying in prediction.
		*/
		std::vector<std::pair<string, float> > Classify(const cv::Mat& reference, const cv::Mat& img, int N, bool mean_substract = false);
		/** @brief Make a classification.
		*/
		void FeatureExtract(InputArray inputimg, OutputArray feature, bool mean_subtract);
		/** @brief Extract a single featrue of one image.
		*/
		std::vector<int> Argmax(const std::vector<float>& v, int N);
		/** @brief Find the N largest number.
		*/
};
//! @}
}}



#endif /* CNN_3DOBJ_HPP_ */
#endif
