/*M///////////////////////////////////////////////////////////////////////////////////////
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//                        (3-clause BSD License)
//
// Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
// Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2015, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * Neither the names of the copyright holders nor the names of the contributors
//     may be used to endorse or promote products derived from this software
//     without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// Author : Awabot SAS
// Copyright (C) 2015, Awabot SAS, all rights reserved.
//
//M*/

#ifndef __OPENCV_GESTURES_CLASSIFIER_HPP__
#define __OPENCV_GESTURES_CLASSIFIER_HPP__

#ifdef __cplusplus

#include <string>
#include <opencv2/core.hpp>

#include <opencv2/gestures/prediction.hpp>

/** @defgroup gestures Gestures Recognition module
 */

namespace cv
{
    namespace gestures
    {
        //! @addtogroup gestures
        //! @{
        /* @brief Base abstract class for Gestures Recognition algorithms.
         */
        class CV_EXPORTS GesturesClassifier : public Algorithm
        {
            public:
                /**
                 Set the labels of the classes output by the classifier.
                 @param labels Vector of string of size equal to the number of classes.
                 */
                CV_WRAP virtual void setLabels(std::vector<std::string> labels) = 0;
                /**
                 Read the labels of the classes output by the classifier from a text file.
                 The file should have one label per line, and as many lines as there are classes.
                 @param file The path to the file containing the list of labels.
                 */
                CV_WRAP virtual void setLabelsFromFile(std::string file) = 0;
                /**
                 Returns whether or not the classifier has valid labels.
                 This will be true if there are as many classes as there are labels.
                 */
                CV_WRAP virtual bool hasValidLabels() const = 0;
                /**
                 Returns a vector of string corresponding to the labels of every classes output by the classifier.
                 */
                CV_WRAP virtual std::vector<std::string> getLabels() const = 0;

                /**
                 Set the number of classes output by the classifier.
                 @param count The number of classes to output, assumed positive.
                 */
                CV_WRAP virtual void setClassesCount(int count) = 0;
                /**
                 Returns the number of classes output by the classifier.
                 */
                CV_WRAP virtual int getClassesCount() const = 0;

                /**
                 Ask the classifier to make a prediction based on its current internal state.
                 @param prediction Output array containing one probability per classes.
                 */
                CV_WRAP virtual bool getPrediction(OutputArray prediction) = 0;
                /** @overload
                 @param prediction Output vector of prediction, containing a probability per classes, with its label, sorted in decreasing order.
                 */
                CV_WRAP virtual bool getPrediction(std::vector<Prediction>& prediction) = 0;
        };
        //! @} gestures

        //! @addtogroup gestures
        //! @{
        /* @brief Class for Gestures Recognition based on Deep Neural Network (use external library: Caffe).
         */
        class CV_EXPORTS GesturesClassifierDNN : public GesturesClassifier
        {
            public:
                /**
                 Update internal buffer with new frames coming from each modality stream.
                 Frames will be automatically skipped to match the temporal stride of the input blocks.
                 It returns true if the internal state of the classifier was successfully updated.
                 @param videoFrame grayscale or RGB image from the color/video stream of a RGB-D sensor.
                 @param depthFrame grayscale image from the depth stream of a RGB-D sensor. It should contains values expressed in millimeters (uint16, max 10000).
                 @param mocapFrame array containing motion capture data generated by external library.
                 It should have 11 columns, one for each skeleton joint, in the following order:
                 HIP_CENTER, SHOULDER_CENTER, HEAD, SHOULDER_LEFT, ELBOW_LEFT, HAND_LEFT, SHOULDER_RIGHT, ELBOW_RIGHT, HAND_RIGHT, HIP_LEFT, HIP_RIGHT.
                 And 5 rows: The first 3 for the joint's world coordinates, expressed in meter in the sensor frame,
                 the last 2 for its image's coordinates, in pixels.
                 */
                CV_WRAP virtual bool feedNewFrames(InputArray videoFrame, InputArray depthFrame, InputArray mocapFrame) = 0;

                /**
                 Set the size of the bounding box cropped around each hand during preprocessing.
                 @param size The size of the square box, in pixels.
                 */
                CV_WRAP virtual void setHandSize(int size) = 0;
                /**
                 Returns the size of the square bounding box cropped around each hand during preprocessing.
                 */
                CV_WRAP virtual int getHandSize() const = 0;

                /**
                 Set the temporal size of a block fed to the network, ie the number of frames to buffer.
                 This shouldn't be called explicitely, as it must match the dimensions of the inputs of the network, declared in the Caffe prototxt file.
                 @param size The number of frames in a block, assumed positive.
                 */
                CV_WRAP virtual void setTemporalSize(int size) = 0;
                /**
                 Returns the temporal size of a block fed to the network, ie the number of frames to buffer.
                 */
                CV_WRAP virtual int getTemporalSize() const = 0;

                /**
                 Set the temporal stride of a block, ie the number of frame to skip in-between each frame added to the buffer.
                 @param stride The number of frames to skip, assumed positive.
                 */
                CV_WRAP virtual void setTemporalStride(int stride) = 0;
                /**
                 Returns the temporal stride of a block fed to the network, ie the number of frame to skip in-between each frame added to the buffer.
                 */
                CV_WRAP virtual int getTemporalStride() const = 0;

                /**
                 Set the .prototxt file describing the network architecture following the Caffe message format.
                 @param file A string containing the path to the .prototxt file.
                 */
                CV_WRAP virtual void setNeuralNetProtoFile(std::string file) = 0;
                /**
                 Returns a string containing the path to the .prototxt file used to initialize the network architecture.
                 */
                CV_WRAP virtual std::string getNeuralNetProtoFile() const = 0;

                /**
                 Set the binary file containing the optimized parameters of the network, trained using Caffe.
                 @param file A string containing the path to the .caffemodel file.
                 */
                CV_WRAP virtual void setNeuralNetWeightsFile(std::string file) = 0;
                /**
                 Returns a string containing the path to the .caffemodel file used to initialize the network parameters.
                 */
                CV_WRAP virtual std::string getNeuralNetWeightsFile() const = 0;

                /**
                 Initialize the Caffe DataTransformers handling mean subtraction and input scaling for each modality.
                 @param meanFolder A string containing the path to a folder where the three _mean.binaryproto files corresponding to each modality can be found.
                 */
                CV_WRAP virtual void setDataTransformers(std::string meanFolder) = 0;

                /**
                 Returns a pointer to an instance of gestures classifier initialize correctly.
                 */
                CV_WRAP static Ptr<GesturesClassifierDNN> create(
                        std::string nnetProto,
                        std::string nnetWeights,
                        std::string meanFolder = "",
                        std::string labelsFile = "",
                        int handSize = 72,
                        int stride = 4);
        };
        //! @} gestures
    } // namespace gestures
} // namespace cv

#endif // __cpluscplus
#endif // __OPENCV_GESTURES_CLASSIFIER_HPP__
