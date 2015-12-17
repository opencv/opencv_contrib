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

#include <opencv2/gestures/gestures_classifier.hpp>
#include <opencv2/gestures/skeleton_frame.hpp>

#include <opencv2/imgproc.hpp>

#include <glog/logging.h>
#include <caffe/net.hpp>
#include <caffe/data_transformer.hpp>

#include <deque>
#include <fstream>
#include <utility>
#include <sstream>
#include <cstring>

namespace cv
{
    namespace gestures
    {
        class GesturesClassifierDNNImpl : public GesturesClassifierDNN
        {
            public:
                enum InputName
                {
                    RIGHT_HAND_DEPTH = 0,
                    RIGHT_HAND_VIDEO = 1,
                    LEFT_HAND_DEPTH = 2,
                    LEFT_HAND_VIDEO = 3,
                    SKEL_DESCRIPTOR = 4,
                    INPUT_COUNT
                };

                GesturesClassifierDNNImpl(
                        std::string nnetProto,
                        std::string nnetWeights,
                        std::string meanFolder = "",
                        std::string labelsFile = "",
                        int handSize = 72,
                        int stride = 4)
                {
                    FLAGS_minloglevel = 3;

                    setNeuralNetProtoFile(nnetProto);
                    setNeuralNetWeightsFile(nnetWeights);

                    setLabelsFromFile(labelsFile);

                    setHandSize(handSize);
                    setTemporalStride(stride);
                    mBufferSize = 1 + (mTemporalSize - 1) * mTemporalStride;

                    setDataTransformers(meanFolder);
                }

                virtual void setLabels(std::vector<std::string> labels)
                {
                    if(labels == mLabels)
                    {
                        return;
                    }

                    mLabels = labels;
                    mHasValidLabels = (labels.size() == mClassesCount);
                }
                virtual void setLabelsFromFile(std::string file)
                {
                    mLabels.clear();
                    mHasValidLabels = false;
                    std::ifstream filestream(file.c_str());

                    if(!filestream.is_open())
                    {
                        return;
                    }

                    std::string elem;
                    while(std::getline(filestream, elem))
                    {
                        mLabels.push_back(elem);
                    }

                    if(mLabels.size() == mClassesCount)
                    {
                        mHasValidLabels = true;
                    }
                    else
                    {
                        mLabels.clear();
                    }
                }
                virtual bool hasValidLabels() const
                {
                    return mHasValidLabels;
                }
                virtual std::vector<std::string> getLabels() const
                {
                    return mLabels;
                }

                virtual void setClassesCount(int count)
                {
                    CV_Assert(count > 0);
                    mClassesCount = count;
                }
                virtual int getClassesCount() const
                {
                    return mClassesCount;
                }

                virtual bool getPrediction(OutputArray prediction)
                {
                    prediction.create(cv::Size(1, mClassesCount), CV_32F);
                    Mat pred_mat = prediction.getMat();

                    if(mInputBuffers[RIGHT_HAND_DEPTH].size() < mBufferSize)
                    {
                        return false;
                    }

                    fillInputBlobs();

                    mNNet->ForwardPrefilled();

                    const float* output = mNNet->output_blobs()[0]->cpu_data();
                    for(int c = 0; c < mClassesCount; ++c)
                    {
                        pred_mat.at<float>(c) = output[c];
                    }

                    return true;
                }
                virtual bool getPrediction(std::vector<Prediction>& prediction)
                {
                    if(mInputBuffers[RIGHT_HAND_DEPTH].size() < mBufferSize)
                    {
                        return false;
                    }

                    fillInputBlobs();

                    mNNet->ForwardPrefilled();

                    const float* output = mNNet->output_blobs()[0]->cpu_data();

                    prediction.clear();
                    for(int c = 0; c < mClassesCount; ++c)
                    {
                        if(mHasValidLabels)
                        {
                            prediction.push_back(Prediction(c, mLabels[c], output[c]));
                        }
                        else
                        {
                            std::stringstream sstream;
                            sstream << c;
                            prediction.push_back(Prediction(c, sstream.str(), output[c]));
                        }
                    }

                    std::sort(prediction.begin(), prediction.end(), std::greater<Prediction>());

                    return true;
                }

                virtual bool feedNewFrames(InputArray videoFrame, InputArray depthFrame, InputArray mocapFrame)
                {
                    SkeletonFrame skeletonFrame(mocapFrame);
                    mSkeletonBuffer.push_back(skeletonFrame);

                    SkeletonFrame previous2;
                    if(mSkeletonBuffer.size() > 2)
                    {
                        previous2 = mSkeletonBuffer.front();
                        mSkeletonBuffer.pop_front();
                    }
                    else
                    {
                        return false;
                    }

                    Mat video_mat = videoFrame.getMat();
                    CV_Assert(video_mat.channels() == 3 || video_mat.channels() == 1);
                    Mat fMat;
                    if(video_mat.channels() == 3)
                    {
                        Mat gray;
                        cvtColor(video_mat, gray, COLOR_RGB2GRAY);
                        gray.convertTo(fMat, CV_32F);
                    }
                    else
                    {
                        video_mat.convertTo(fMat, CV_32F);
                    }

                    Mat depth_mat = depthFrame.getMat();
                    CV_Assert(depth_mat.size() == video_mat.size() && depth_mat.type() == CV_32FC1);

                    if(skeletonFrame.isValid()
                            && checkPos(skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_LEFT), video_mat.size())
                            && checkPos(skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_RIGHT), video_mat.size()))
                    {

                        Mat descriptor;
                        skeletonFrame.createDescriptor(mSkeletonBuffer.front(), previous2, descriptor);
                        mInputBuffers[SKEL_DESCRIPTOR].push_back(descriptor);

                        mInputBuffers[RIGHT_HAND_DEPTH].push_back(cropBoundingBox(depth_mat, skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_RIGHT)));
                        mInputBuffers[LEFT_HAND_DEPTH].push_back(cropBoundingBox(depth_mat, skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_LEFT)));

                        mInputBuffers[RIGHT_HAND_VIDEO].push_back(cropBoundingBox(fMat, skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_RIGHT)));
                        mInputBuffers[LEFT_HAND_VIDEO].push_back(cropBoundingBox(fMat, skeletonFrame.getJointPixelCoords(SkeletonFrame::JOINT_HAND_LEFT)));

                        if(mInputBuffers[RIGHT_HAND_DEPTH].size() > mBufferSize)
                        {
                            for(int i = 0; i < INPUT_COUNT; ++i)
                            {
                                mInputBuffers[i].pop_front();
                            }
                        }

                        return true;
                    }
                    else
                    {
                        for(int i = 0; i < INPUT_COUNT; ++i)
                        {
                            mInputBuffers[i].clear();
                        }

                        return false;
                    }
                }

                virtual void setHandSize(int size)
                {
                    CV_Assert(size > 0);
                    mHandSize = size;

                    for(int i = 0; i < INPUT_COUNT; ++i)
                    {
                        mInputBuffers[i].clear();
                    }
                }
                virtual int getHandSize() const
                {
                    return mHandSize;
                }

                virtual void setTemporalSize(int size)
                {
                    CV_Assert(size > 0);
                    mTemporalSize = size;

                    for(int i = 0; i < INPUT_COUNT; ++i)
                    {
                        mInputBuffers[i].clear();
                    }
                }
                virtual int getTemporalSize() const
                {
                    return mTemporalSize;
                }

                virtual void setTemporalStride(int stride)
                {
                    CV_Assert(stride > 0);
                    mTemporalStride = stride;

                    for(int i = 0; i < INPUT_COUNT; ++i)
                    {
                        mInputBuffers[i].clear();
                    }
                }
                virtual int getTemporalStride() const
                {
                    return mTemporalStride;
                }

                virtual void setNeuralNetProtoFile(std::string file)
                {
                    mNNet = makePtr< caffe::Net<float> >(file, caffe::TEST);
                    mNNetProtoFile = file;

                    CV_Assert(mNNet->input_blobs().size() == 5);
                    CV_Assert(mNNet->output_blobs().size() == 1);

                    setTemporalSize(mNNet->input_blobs()[RIGHT_HAND_DEPTH]->shape(1));
                    setClassesCount(mNNet->output_blobs()[0]->shape(1));

                    mBlockSize = Size(mNNet->input_blobs()[RIGHT_HAND_DEPTH]->shape()[2], mNNet->input_blobs()[RIGHT_HAND_DEPTH]->shape()[3]);
                    mDescriptorLength = mNNet->input_blobs()[SKEL_DESCRIPTOR]->shape()[1] / mTemporalSize;

                    for(int i = 0; i < INPUT_COUNT; ++i)
                    {
                        mInputBuffers[i].clear();
                    }
                }
                virtual std::string getNeuralNetProtoFile() const
                {
                    return mNNetProtoFile;
                }

                virtual void setNeuralNetWeightsFile(std::string file)
                {
                    CV_Assert(!mNNet.empty());

                    mNNet->CopyTrainedLayersFrom(file);
                    mNNetWeightsFile = file;
                }
                virtual std::string getNeuralNetWeightsFile() const
                {
                    return mNNetWeightsFile;
                }

                virtual void setDataTransformers(std::string meanFolder)
                {
                    if(!meanFolder[meanFolder.size()-1] == '/')
                    {
                        meanFolder += '/';
                    }

                    caffe::TransformationParameter param;

                    // Depth
                    param.set_mean_file(meanFolder + "depth_mean.binaryproto");
                    mDataTransformers[RIGHT_HAND_DEPTH] = makePtr< caffe::DataTransformer<float> >(param, caffe::TEST);
                    mDataTransformers[LEFT_HAND_DEPTH] = mDataTransformers[RIGHT_HAND_DEPTH];

                    // Color
                    param.set_mean_file(meanFolder + "video_mean.binaryproto");
                    param.set_scale(0.00390625);
                    mDataTransformers[RIGHT_HAND_VIDEO] = makePtr< caffe::DataTransformer<float> >(param, caffe::TEST);
                    mDataTransformers[LEFT_HAND_VIDEO] = mDataTransformers[RIGHT_HAND_VIDEO];

                    // Skeleton
                    param.set_mean_file(meanFolder + "mocap_mean.binaryproto");
                    param.set_scale(0.25);
                    mDataTransformers[SKEL_DESCRIPTOR] = makePtr< caffe::DataTransformer<float> >(param, caffe::TEST);
                }

            private:
                bool checkPos(Point point, Size size)
                {
                    if(point.x - mHandSize/2 < 0 || point.y - mHandSize/2 < 0)
                    {
                        return false;
                    }

                    if(point.x + mHandSize/2 >= size.width || point.y + mHandSize/2 >= size.height)
                    {
                        return false;
                    }

                    return true;
                }


                Mat cropBoundingBox(Mat frame, Point center)
                {
                    Rect roi(center - Point(mHandSize/2, mHandSize/2), Size(mHandSize, mHandSize));
                    Mat ret;
                    resize(frame(roi), ret, mBlockSize);

                    return ret;
                }

                void fillInputBlobs()
                {
                    for(int i = 0; i < INPUT_COUNT; ++i)
                    {
                        caffe::Blob<float> src_blob;
                        src_blob.Reshape(mNNet->input_blobs()[i]->shape());
                        float* input_ptr = src_blob.mutable_cpu_data();

                        std::deque<Mat>::iterator it = mInputBuffers[i].begin();
                        for(int f = 0; f < mTemporalSize; ++f)
                        {
                            if(i < SKEL_DESCRIPTOR)
                            {
                                for(int h = 0; h < mBlockSize.height; ++h)
                                {
                                    for(int w = 0; w < mBlockSize.width; ++w)
                                    {
                                        input_ptr[(h*mBlockSize.width+w)*mTemporalSize+f] = it->at<float>(h,w);
                                    }
                                }
                            }
                            else
                            {
                                for(int d = 0; d < mDescriptorLength; ++d)
                                {
                                    input_ptr[f*mDescriptorLength+d] = it->at<float>(d);
                                }
                            }
                            it += mTemporalStride;
                        }
                        mDataTransformers[i]->Transform(&src_blob, mNNet->input_blobs()[i]);
                    }
                }


                bool mHasValidLabels;
                std::vector<std::string> mLabels;

                int mClassesCount;

                int mTemporalSize;
                int mTemporalStride;
                int mBufferSize;

                std::string mNNetProtoFile;
                std::string mNNetWeightsFile;
                Ptr< caffe::Net<float> > mNNet;
                Ptr< caffe::DataTransformer<float> > mDataTransformers[INPUT_COUNT];

                Size mBlockSize;
                int mDescriptorLength;

                int mHandSize;

                std::deque<Mat> mInputBuffers[INPUT_COUNT];

                std::deque<SkeletonFrame> mSkeletonBuffer;
        };

        Ptr<GesturesClassifierDNN> GesturesClassifierDNN::create(
                std::string nnetProto,
                std::string nnetWeights,
                std::string meanFolder,
                std::string labelsFile,
                int handSize,
                int stride)
        {
            return makePtr<GesturesClassifierDNNImpl>(
                    nnetProto,
                    nnetWeights,
                    meanFolder,
                    labelsFile,
                    handSize,
                    stride);
        }
    } // namespace gestures
} // namespace cv
