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
#include <dirent.h>
#define CPU_ONLY

#include <caffe/blob.hpp>
#include <caffe/common.hpp>
#include <caffe/net.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/io.hpp>

#include "opencv2/viz/vizcore.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc.hpp"

/** @defgroup cnn_3dobj 3D object recognition and pose estimation API

As CNN based learning algorithm shows better performance on the classification issues,
the rich labeled data could be more useful in the training stage. 3D object classification and pose estimation
is a jointed mission aiming at separate different posed apart in the descriptor form.

In the training stage, we prepare 2D training images generated from our module with their
class label and pose label. We fully exploit the information lies in their labels
by using a triplet and pair-wise jointed loss function in CNN training.

As CNN based learning algorithm shows better performance on the classification issues,
the rich labeled data could be more useful in the training stage. 3D object classification and pose estimation
is a jointed mission aiming at separate different posea apart in the descriptor form.

In the training stage, we prepare 2D training images generated from our module with their
class label and pose label. We fully exploit the information that lies in their labels
by using a triplet and pair-wise jointed loss function in CNN training.

Both class and pose label are in consideration in the triplet loss. The loss score
will be smaller when features from the same class and same pose is more similar
and features from different classes or different poses will lead to a much larger loss score.

This loss is also jointed with a pair wise component to make sure the loss is never be zero
and have a restriction on the model scale.

About the training and feature extraction process, it is a rough implementation by using OpenCV
and Caffe from the idea of Paul Wohlhart. The principal purpose of this API is constructing
a well labeled database from .ply models for CNN training with triplet loss and extracting features
with the constructed model for prediction or other purpose of pattern recognition, algorithms into two main Class:

**icoSphere: methods belonging to this class generates 2D images from a 3D model, together with their class and pose from camera view labels.

**descriptorExtractor: methods belonging to this class extract descriptors from 2D images which is
discriminant on category prediction and pose estimation.

@note This API need Caffe with triplet version which is designed for this module
<https://github.com/Wangyida/caffe/tree/cnn_triplet>.

*/
namespace cv
{
namespace cnn_3dobj
{

//! @addtogroup cnn_3dobj
//! @{

/** @brief Icosohedron based camera view data generator.
 The class create some sphere views of camera towards a 3D object meshed from .ply files @cite hinterstoisser2008panter .
 */

/************************************ Data Generation Class ************************************/
    class CV_EXPORTS_W icoSphere
    {
        private:
        /** @brief X position of one base point on the initial Icosohedron sphere,
          Y is set to be 0 as default.
         */
        float X;

        /** @brief Z position of one base point on the initial Icosohedron sphere.
         */
        float Z;

        /** @brief A threshold for the dupicated points elimination.
         */
        float diff;

        /** @brief Temp camera position for duplex position elimination.
         */
        std::vector<cv::Point3d> CameraPos_temp;

        /** @brief Make all view points having the same distance from the focal point used by the camera view.
         */
        CV_WRAP void norm(float v[]);

        /** @brief Add a new view point.
         */
        CV_WRAP void add(float v[]);

        /** @brief Generate new view points from all triangles.
         */
        CV_WRAP void subdivide(float v1[], float v2[], float v3[], int depth);

        public:
        /** @brief Camera position on the sphere after duplicated points elimination.
         */
        std::vector<cv::Point3d> CameraPos;

        /** @brief Generating a sphere by mean of a iteration based points selection process.
        @param radius_in Another radius used for adjusting the view distance.
        @param depth_in Number of interations for increasing the points on sphere.
         */
        icoSphere(float radius_in, int depth_in);

        /** @brief Get the center of points on surface in .ply model.
        @param cloud Point cloud used for computing the center point.
         */
        CV_WRAP cv::Point3d getCenter(cv::Mat cloud);

        /** @brief Get the proper camera radius from the view point to the center of model.
        @param cloud Point cloud used for computing the center point.
        @param center center point of the point cloud.
         */
        CV_WRAP float getRadius(cv::Mat cloud, cv::Point3d center);

        /** @brief Suit the position of bytes in 4 byte data structure for particular system.
         */
        CV_WRAP static int swapEndian(int val);

        /** @brief Create header in binary files collecting the image data and label.
        @param num_item Number of items.
        @param rows Rows of a single sample image.
        @param cols Columns of a single sample image.
        @param headerPath Path where the header will be stored.
         */
        CV_WRAP static void createHeader(int num_item, int rows, int cols, const char* headerPath);

        /** @brief Write binary files used for training in other open source project including Caffe.
        @param filenameImg Path which including a set of images.
        @param binaryPath Path which will output a binary file.
        @param headerPath Path which header belongs to.
        @param num_item Number of samples.
        @param label_class Class label of the sample.
        @param x Pose label of X.
        @param y Pose label of Y.
        @param z Pose label of Z.
        @param isrgb Option for choice of using RGB images or not.
         */
        CV_WRAP static void writeBinaryfile(String filenameImg, const char* binaryPath, const char* headerPath, int num_item, int label_class, int x, int y, int z, int isrgb);
    };

/** @brief Caffe based 3D images descriptor.
 A class to extract features from an image. The so obtained descriptors can be used for classification and pose estimation goals @cite wohlhart15.
 */

/************************************ Feature Extraction Class ************************************/
    class CV_EXPORTS_W descriptorExtractor
    {
        private:
        caffe::Net<float>* convnet;
        cv::Size input_geometry;
        int num_channels;
        bool net_set;
        int net_ready;
        cv::Mat mean_;
        String deviceType;
        int deviceId;

        /** @brief Load the mean file in binaryproto format if it is needed.
        @param mean_file Path of mean file which stores the mean of training images, it is usually generated by Caffe tool.
         */
        void setMean(const String& mean_file);

        /** @brief Wrap the input layer of the network in separate cv::Mat objects(one per channel).
         This way we save one memcpy operation and we don't need to rely on cudaMemcpy2D.
         The last preprocessing operation will write the separate channels directly to the input layer.
         */
        void wrapInput(std::vector<cv::Mat>* input_channels);

        /** @brief Convert the input image to the input image format of the network.
         */
        void preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);

        public:
        /** @brief Set the device for feature extraction, if the GPU is used, there should be a device_id.
        @param device_type CPU or GPU.
        @param device_id ID of GPU.
         */
        descriptorExtractor(const String& device_type, int device_id = 0);

        /** @brief Get device type information for feature extraction.
         */
        String getDeviceType();

        /** @brief Get device ID information for feature extraction.
         */
        int getDeviceId();

        /** @brief Set device type information for feature extraction.
         Useful to change device without the need to reload the net.
        @param device_type CPU or GPU.
         */
        void setDeviceType(const String& device_type);

        /** @brief Set device ID information for feature extraction.
         Useful to change device without the need to reload the net. Only used for GPU.
        @param device_id ID of GPU.
         */
        void setDeviceId(const int& device_id);

        /** @brief Initiate a classification structure, the net work parameter is stored in model_file,
         the network structure is stored in trained_file, you can decide whether to use mean images or not.
        @param model_file Path of caffemodel which including all parameters in CNN.
        @param trained_file Path of prototxt which defining the structure of CNN.
        @param mean_file Path of mean file(option).
         */
        void loadNet(const String& model_file, const String& trained_file, const String& mean_file = "");

        /** @brief Extract features from a single image or from a vector of images.
         If loadNet was not called before, this method invocation will fail.
        @param inputimg Input images.
        @param feature Output features.
        @param feature_blob Layer which the feature is extracted from.
         */
        void extract(InputArrayOfArrays inputimg, OutputArray feature, String feature_blob);
    };
    //! @}
}
}

#endif /* CNN_3DOBJ_HPP_ */
#endif
