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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include <tuple>
#include "test_precomp.hpp"
#include <opencv2/dnnlegacy/dnnlegacy.hpp>
#include "npy_blob.hpp"
#include <opencv2/dnn/shape_utils.hpp>

using namespace cv;

namespace opencv_test {
    namespace {

        template<typename Tstring>
        static std::string _tf(Tstring filename)
        {
            return cv::samples::findFile(std::string("dnn/") + filename);
        }

        class Test_Caffe_nets : public DNNlegacyTestLayer
        {
        public:
            void testFaster(const std::string& proto, const std::string& model, const Mat& ref,
                            double scoreDiff = 0.0, double iouDiff = 0.0)
            {
                checkBackend();
                cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(cv::samples::findFile("dnn/" + proto),
                                           cv::samples::findFile("dnn/" + model, false));
                net.setPreferableBackend(backend);
                net.setPreferableTarget(target);

                if (target == cv::dnn::DNN_TARGET_CPU_FP16)
                    net.enableWinograd(false);

                Mat img = imread(cv::samples::findFile("dnn/dog416.png"));
                resize(img, img, Size(800, 600));
                Mat blob = cv::dnn::blobFromImage(img, 1.0, Size(), Scalar(102.9801, 115.9465, 122.7717), false, false);
                Mat imInfo = (Mat_<float>(1, 3) << img.rows, img.cols, 1.6f);

                net.setInput(blob, "data");
                net.setInput(imInfo, "im_info");
                // Output has shape 1x1xNx7 where N - number of detections.
                // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
                Mat out = net.forward();
                scoreDiff = scoreDiff ? scoreDiff : default_l1;
                iouDiff = iouDiff ? iouDiff : default_lInf;
                opencv_test::normAssertDetections(ref, out, ("model name: " + model).c_str(), 0.8, scoreDiff, iouDiff);
            }
        };

        TEST(Test_Caffe, memory_read)
        {
            const std::string proto = cv::samples::findFile("dnn/bvlc_googlenet.prototxt");
            const std::string model = cv::samples::findFile("dnn/bvlc_googlenet.caffemodel", false);

            std::vector<char> dataProto;
            readFileContent(proto, dataProto);

            std::vector<char> dataModel;
            readFileContent(model, dataModel);

            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(dataProto.data(), dataProto.size());
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            ASSERT_FALSE(net.empty());

            cv::dnn::Net net2 = cv::dnnlegacy::readNetFromCaffe(dataProto.data(), dataProto.size(),
                                        dataModel.data(), dataModel.size());
            ASSERT_FALSE(net2.empty());
        }

        TEST(Test_Caffe, read_gtsrb)
        {
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(_tf("gtsrb.prototxt"));
            ASSERT_FALSE(net.empty());
        }

        TEST(Test_Caffe, read_googlenet)
        {
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(_tf("bvlc_googlenet.prototxt"));
            ASSERT_FALSE(net.empty());
        }

        TEST_P(Test_Caffe_nets, Axpy)
        {
        #if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NN_BUILDER);
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH);
        #endif

            std::string proto = _tf("axpy.prototxt");
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto);

            checkBackend();
            net.setPreferableBackend(backend);
            net.setPreferableTarget(target);

            int size[] = {1, 2, 3, 4};
            int scale_size[] = {1, 2, 1, 1};
            Mat scale(4, &scale_size[0], CV_32F);
            Mat shift(4, &size[0], CV_32F);
            Mat inp(4, &size[0], CV_32F);
            randu(scale, -1.0f, 1.0f);
            randu(shift, -1.0f, 1.0f);
            randu(inp, -1.0f, 1.0f);

            net.setInput(scale, "scale");
            net.setInput(shift, "shift");
            net.setInput(inp, "data");

            Mat out = net.forward();

            Mat ref(4, &size[0], inp.type());
            for (int i = 0; i < inp.size[1]; i++) {
                for (int h = 0; h < inp.size[2]; h++) {
                    for (int w = 0; w < inp.size[3]; w++) {
                        int idx[] = {0, i, h, w};
                        int scale_idx[] = {0, i, 0, 0};
                        ref.at<float>(idx) = inp.at<float>(idx) * scale.at<float>(scale_idx) +
                                             shift.at<float>(idx);
                    }
                }
            }
            float l1 = 1e-5, lInf = 1e-4;
            if (target ==cv::dnn::DNN_TARGET_OPENCL_FP16 || target ==cv::dnn::DNN_TARGET_CPU_FP16)
            {
                l1 = 2e-4;
                lInf = 1e-3;
            }
            if (target ==cv::dnn::DNN_TARGET_MYRIAD)
            {
                l1 = 0.001;
                lInf = 0.001;
            }
            if(target ==cv::dnn::DNN_TARGET_CUDA_FP16)
            {
                l1 = 0.0002;
                lInf = 0.0007;
            }
            normAssert(ref, out, "", l1, lInf);
        }

        typedef testing::TestWithParam<std::tuple<bool, cv::dnn::Target> > Reproducibility_AlexNet;
        TEST_P(Reproducibility_AlexNet, Accuracy)
        {
            cv::dnn::Target targetId = get<1>(GetParam());
        #if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
            applyTestTag(CV_TEST_TAG_MEMORY_2GB);
        #else
            applyTestTag(targetId == cv::dnn::DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
        #endif
            ASSERT_TRUE(ocl::useOpenCL() || targetId == cv::dnn::DNN_TARGET_CPU || targetId == cv::dnn::DNN_TARGET_CPU_FP16);

            bool readFromMemory = get<0>(GetParam());
            cv::dnn::Net net;
            {
                const std::string proto = cv::samples::findFile("dnn/bvlc_alexnet.prototxt");
                const std::string model = cv::samples::findFile("dnn/bvlc_alexnet.caffemodel", false);
                if (readFromMemory)
                {
                    std::vector<char> dataProto;
                    readFileContent(proto, dataProto);
                    std::vector<char> dataModel;
                    readFileContent(model, dataModel);
                    net = cv::dnnlegacy::readNetFromCaffe(dataProto.data(), dataProto.size(),
                                           dataModel.data(), dataModel.size());
                }
                else
                    net = cv::dnnlegacy::readNetFromCaffe(proto, model);
                ASSERT_FALSE(net.empty());
            }

            // Test input layer size
            /* CHANGE in getLayerShapes param in OPENCV 5
            * TEST disable
            std::vector<cv::dnn::MatShape> inLayerShapes;
            std::vector<cv::dnn::MatShape> outLayerShapes;
            net.getLayerShapes(cv::dnn::MatShape(), 0, inLayerShapes, outLayerShapes);
            ASSERT_FALSE(inLayerShapes.empty());
            ASSERT_EQ(inLayerShapes[0].size(), 4);
            ASSERT_EQ(inLayerShapes[0][0], 1);
            ASSERT_EQ(inLayerShapes[0][1], 3);
            ASSERT_EQ(inLayerShapes[0][2], 227);
            ASSERT_EQ(inLayerShapes[0][3], 227);
            */
            const float l1 = 1e-5;
            const float lInf = (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16 || targetId == cv::dnn::DNN_TARGET_CPU_FP16) ? 4e-3 : 1e-4;

            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(targetId);

            if (targetId == cv::dnn::DNN_TARGET_CPU_FP16)
                net.enableWinograd(false);

            Mat sample = imread(_tf("grace_hopper_227.png"));
            ASSERT_TRUE(!sample.empty());

            net.setInput(cv::dnn::blobFromImage(sample, 1.0f, Size(227, 227), Scalar(), false), "data");
            Mat out = net.forward("prob");
            Mat ref = blobFromNPY(_tf("caffe_alexnet_prob.npy"));
            normAssert(ref, out, "", l1, lInf);
        }

        INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_AlexNet, Combine(testing::Bool(),
                                testing::ValuesIn(getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV))));

        TEST(Reproducibility_FCN, Accuracy)
        {
            applyTestTag(CV_TEST_TAG_LONG, CV_TEST_TAG_DEBUG_VERYLONG, CV_TEST_TAG_MEMORY_2GB);

            cv::dnn::Net net;
            {
                const std::string proto = cv::samples::findFile("dnn/fcn8s-heavy-pascal.prototxt");
                const std::string model = cv::samples::findFile("dnn/fcn8s-heavy-pascal.caffemodel", false);
                net = dnnlegacy::readNetFromCaffe(proto, model);
                ASSERT_FALSE(net.empty());
            }
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

            Mat sample = imread(_tf("street.png"));
            ASSERT_TRUE(!sample.empty());

            std::vector<int> layerIds;
            std::vector<size_t> weights, blobs;
            /* CHANGE in getMemoryConsumption param in OPENCV 5
            * TEST disable
            net.getMemoryConsumption(cv::dnn::shape(1,3,227,227), layerIds, weights, blobs);
        */
            net.setInput(cv::dnn::blobFromImage(sample, 1.0f, Size(500, 500), Scalar(), false), "data");
            Mat out = net.forward("score");

            Mat refData = imread(_tf("caffe_fcn8s_prob.png"), IMREAD_ANYDEPTH);
            int shape[] = {1, 21, 500, 500};
            Mat ref(4, shape, CV_32FC1, refData.data);

            normAssert(ref, out);
        }

        TEST(Reproducibility_SSD, Accuracy)
        {
            applyTestTag(
                CV_TEST_TAG_MEMORY_512MB,
                CV_TEST_TAG_DEBUG_VERYLONG
            );

            cv::dnn::Net net;
            {
                const std::string proto = cv::samples::findFile("dnn/ssd_vgg16.prototxt");
                const std::string model = cv::samples::findFile("dnn/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel", false);
                net = cv::dnnlegacy::readNetFromCaffe(proto, model);
                ASSERT_FALSE(net.empty());
            }
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

            Mat sample = imread(_tf("street.png"));
            ASSERT_TRUE(!sample.empty());

            if (sample.channels() == 4)
                cvtColor(sample, sample, COLOR_BGRA2BGR);

            Mat in_blob = cv::dnn::blobFromImage(sample, 1.0f, Size(300, 300), Scalar(), false);
            net.setInput(in_blob, "data");
            Mat out = net.forward("detection_out");

            Mat ref = blobFromNPY(_tf("ssd_out.npy"));
            normAssertDetections(ref, out, "", 0.06);
        }

        typedef testing::TestWithParam<std::tuple<cv::dnn::Backend, cv::dnn::Target> > Reproducibility_MobileNet_SSD;
        TEST_P(Reproducibility_MobileNet_SSD, Accuracy)
        {
            const std::string proto = cv::samples::findFile("dnn/MobileNetSSD_deploy_19e3ec3.prototxt", false);
            const std::string model = cv::samples::findFile("dnn/MobileNetSSD_deploy_19e3ec3.caffemodel", false);
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, model);
            int backendId = get<0>(GetParam());
            int targetId = get<1>(GetParam());

            net.setPreferableBackend(backendId);
            net.setPreferableTarget(targetId);

            Mat sample = imread(_tf("street.png"));

            Mat inp = cv::dnn::blobFromImage(sample, 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
            net.setInput(inp);
            Mat out = net.forward().clone();

            ASSERT_EQ(out.size[2], 100);

            float scores_diff = 1e-5, boxes_iou_diff = 1e-4;
            if (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16 || targetId == cv::dnn::DNN_TARGET_MYRIAD || targetId == cv::dnn::DNN_TARGET_CPU_FP16)
            {
                scores_diff = 1.5e-2;
                boxes_iou_diff = 6.3e-2;
            }
            else if (targetId == cv::dnn::DNN_TARGET_CUDA_FP16)
            {
                scores_diff = 0.015;
                boxes_iou_diff = 0.07;
            }
            Mat ref = blobFromNPY(_tf("mobilenet_ssd_caffe_out.npy"));
            normAssertDetections(ref, out, "", FLT_MIN, scores_diff, boxes_iou_diff);

            // Check that detections aren't preserved.
            inp.setTo(0.0f);
            net.setInput(inp);
            Mat zerosOut = net.forward();
            zerosOut = zerosOut.reshape(1, zerosOut.total() / 7);

            const int numDetections = zerosOut.rows;
            // TODO: fix it
            if (targetId != cv::dnn::DNN_TARGET_MYRIAD ||
                cv::dnn::getInferenceEngineVPUType() != CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
            {
                ASSERT_NE(numDetections, 0);
                for (int i = 0; i < numDetections; ++i)
                {
                    float confidence = zerosOut.ptr<float>(i)[2];
                    ASSERT_EQ(confidence, 0);
                }
            }

            // There is something wrong with Reshape layer in Myriad plugin.
            if (backendId == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019
                || backendId == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH
            )
            {
                if (targetId == cv::dnn::DNN_TARGET_MYRIAD || targetId == cv::dnn::DNN_TARGET_OPENCL_FP16)
                    return;
            }

            // Check batching mode.
            inp = cv::dnn::blobFromImages(std::vector<Mat>(2, sample), 1.0f / 127.5, Size(300, 300), Scalar(127.5, 127.5, 127.5), false);
            net.setInput(inp);
            Mat outBatch = net.forward();

            // Output blob has a shape 1x1x2Nx7 where N is a number of detection for
            // a single sample in batch. The first numbers of detection vectors are batch id.
            // For Inference Engine backend there is -1 delimiter which points the end of detections.
            const int numRealDetections = ref.size[2];
            EXPECT_EQ(outBatch.size[2], 2 * numDetections);
            out = out.reshape(1, numDetections).rowRange(0, numRealDetections);
            outBatch = outBatch.reshape(1, 2 * numDetections);
            for (int i = 0; i < 2; ++i)
            {
                Mat pred = outBatch.rowRange(i * numRealDetections, (i + 1) * numRealDetections);
                EXPECT_EQ(countNonZero(pred.col(0) != i), 0);
                normAssert(pred.colRange(1, 7), out.colRange(1, 7));
            }
        }
        INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_MobileNet_SSD, dnnBackendsAndTargets());

        typedef testing::TestWithParam<cv::dnn::Target> Reproducibility_ResNet50;
        TEST_P(Reproducibility_ResNet50, Accuracy)
        {
            cv::dnn::Target targetId = GetParam();
            applyTestTag(targetId == cv::dnn::DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB);
            ASSERT_TRUE(ocl::useOpenCL() || targetId ==cv::dnn::DNN_TARGET_CPU || targetId ==cv::dnn::DNN_TARGET_CPU_FP16);

            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(cv::samples::findFile("dnn/ResNet-50-deploy.prototxt"),
                                       cv::samples::findFile("dnn/ResNet-50-model.caffemodel", false));

            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(targetId);

            if (targetId == cv::dnn::DNN_TARGET_CPU_FP16)
                net.enableWinograd(false);

            float l1 = (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16 || targetId == cv::dnn::DNN_TARGET_CPU_FP16) ? 3e-5 : 1e-5;
            float lInf = (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16 || targetId == cv::dnn::DNN_TARGET_CPU_FP16) ? 6e-3 : 1e-4;

            Mat input = cv::dnn::blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(224,224), Scalar(), false);
            ASSERT_TRUE(!input.empty());

            net.setInput(input);
            Mat out = net.forward();

            Mat ref = blobFromNPY(_tf("resnet50_prob.npy"));
            normAssert(ref, out, "", l1, lInf);

            if (targetId == cv::dnn::DNN_TARGET_OPENCL || targetId == cv::dnn::DNN_TARGET_OPENCL_FP16)
            {
                UMat out_umat;
                net.forward(out_umat);
                normAssert(ref, out_umat, "out_umat", l1, lInf);

                std::vector<UMat> out_umats;
                net.forward(out_umats);
                normAssert(ref, out_umats[0], "out_umat_vector", l1, lInf);
            }
        }
        INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_ResNet50,
                                testing::ValuesIn(getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV)));

        typedef testing::TestWithParam<cv::dnn::Target> Reproducibility_SqueezeNet_v1_1;
        TEST_P(Reproducibility_SqueezeNet_v1_1, Accuracy)
        {
            int targetId = GetParam();
            if(targetId == cv::dnn::DNN_TARGET_OPENCL_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
            if(targetId == cv::dnn::DNN_TARGET_CPU_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
            cv::dnn::Net net= cv::dnnlegacy::readNetFromCaffe(cv::samples::findFile("dnn/squeezenet_v1.1.prototxt"),
                                       cv::samples::findFile("dnn/squeezenet_v1.1.caffemodel", false));
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(targetId);

            Mat input = cv::dnn::blobFromImage(imread(_tf("googlenet_0.png")), 1.0f, Size(227,227), Scalar(), false, true);
            ASSERT_TRUE(!input.empty());

            Mat out;
            if (targetId == cv::dnn::DNN_TARGET_OPENCL)
            {
                // Firstly set a wrong input blob and run the model to receive a wrong output.
                // Then set a correct input blob to check CPU->GPU synchronization is working well.
                net.setInput(input * 2.0f);
                out = net.forward();
            }
            net.setInput(input);
            out = net.forward();

            Mat ref = blobFromNPY(_tf("squeezenet_v1.1_prob.npy"));
            normAssert(ref, out);
        }
        INSTANTIATE_TEST_CASE_P(/**/, Reproducibility_SqueezeNet_v1_1,
            testing::ValuesIn(getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV)));

        TEST(Reproducibility_AlexNet_fp16, Accuracy)
        {
            applyTestTag(CV_TEST_TAG_MEMORY_512MB);
            const float l1 = 1e-5;
            const float lInf = 3e-3;

            const std::string proto = cv::samples::findFile("dnn/bvlc_alexnet.prototxt");
            const std::string model = cv::samples::findFile("dnn/bvlc_alexnet.caffemodel", false);

            cv::dnnlegacy::shrinkCaffeModel(model, "bvlc_alexnet.caffemodel_fp16");
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, "bvlc_alexnet.caffemodel_fp16");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

            Mat sample = imread(cv::samples::findFile("dnn/grace_hopper_227.png"));

            net.setInput(cv::dnn::blobFromImage(sample, 1.0f, Size(227, 227), Scalar()));
            Mat out = net.forward();
            Mat ref = blobFromNPY(cv::samples::findFile("dnn/caffe_alexnet_prob.npy"));
            normAssert(ref, out, "", l1, lInf);
        }

        TEST(Reproducibility_GoogLeNet_fp16, Accuracy)
        {
            const float l1 = 1e-5;
            const float lInf = 3e-3;

            const std::string proto = cv::samples::findFile("dnn/bvlc_googlenet.prototxt");
            const std::string model = cv::samples::findFile("dnn/bvlc_googlenet.caffemodel", false);

            shrinkCaffeModel(model, "bvlc_googlenet.caffemodel_fp16");
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, "bvlc_googlenet.caffemodel_fp16");
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

            std::vector<Mat> inpMats;
            inpMats.push_back( imread(_tf("googlenet_0.png")) );
            inpMats.push_back( imread(_tf("googlenet_1.png")) );
            ASSERT_TRUE(!inpMats[0].empty() && !inpMats[1].empty());

            net.setInput(cv::dnn::blobFromImages(inpMats, 1.0f, Size(), Scalar(), false), "data");
            Mat out = net.forward("prob");

            Mat ref = blobFromNPY(_tf("googlenet_prob.npy"));
            normAssert(out, ref, "", l1, lInf);
        }

        // https://github.com/richzhang/colorization
        TEST_P(Test_Caffe_nets, Colorization)
        {
            applyTestTag(
                target == cv::dnn::DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB,
                CV_TEST_TAG_DEBUG_VERYLONG
            );
            checkBackend();

            Mat inp = blobFromNPY(_tf("colorization_inp.npy"));
            Mat ref = blobFromNPY(_tf("colorization_out.npy"));
            Mat kernel = blobFromNPY(_tf("colorization_pts_in_hull.npy"));

            const std::string proto = cv::samples::findFile("dnn/colorization_deploy_v2.prototxt", false);
            const std::string model = cv::samples::findFile("dnn/colorization_release_v2.caffemodel", false);
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, model);
            net.setPreferableBackend(backend);
            net.setPreferableTarget(target);

            // This model has bad accuracy when the FP16 and Winograd are enable at same time.
            if (target == cv::dnn::DNN_TARGET_CPU_FP16)
                net.enableWinograd(false);

            net.getLayer(net.getLayerId("class8_ab"))->blobs.push_back(kernel);
            net.getLayer(net.getLayerId("conv8_313_rh"))->blobs.push_back(Mat(1, 313, CV_32F, 2.606));

            net.setInput(inp);
            Mat out = net.forward();

            // Reference output values are in range [-29.1, 69.5]
            double l1 = 4e-4, lInf = 3e-3;
            if (target == cv::dnn::DNN_TARGET_OPENCL_FP16 || target == cv::dnn::DNN_TARGET_CPU_FP16)
            {
                l1 = 0.25;
                lInf = 5.3;
            }
            else if (target == cv::dnn::DNN_TARGET_MYRIAD)
            {
                l1 = (cv::dnn::getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X) ? 0.5 : 0.25;
                lInf = (cv::dnn::getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X) ? 11 : 5.3;
            }
            else if(target == cv::dnn::DNN_TARGET_CUDA_FP16)
            {
                l1 = 0.21;
                lInf = 4.5;
            }
        #if defined(INF_ENGINE_RELEASE)
            if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target == cv::dnn::DNN_TARGET_OPENCL_FP16)
            {
                l1 = 0.3; lInf = 10;
            }
        #endif

            normAssert(out, ref, "", l1, lInf);
            expectNoFallbacksFromIE(net);
        }

        TEST_P(Test_Caffe_nets, DenseNet_121)
        {
            applyTestTag(CV_TEST_TAG_MEMORY_512MB);
            checkBackend();
            const std::string proto = cv::samples::findFile("dnn/DenseNet_121.prototxt", false);
            const std::string weights = cv::samples::findFile("dnn/DenseNet_121.caffemodel", false);

            Mat inp = imread(_tf("dog416.png"));
            cv::dnn::Model model(proto, weights);
            model.setInputScale(1.0 / 255).setInputSwapRB(true).setInputCrop(true);
            std::vector<Mat> outs;
            Mat ref = blobFromNPY(_tf("densenet_121_output.npy"));

            model.setPreferableBackend(backend);
            model.setPreferableTarget(target);
            model.predict(inp, outs);

            // Reference is an array of 1000 values from a range [-6.16, 7.9]
            float l1 = default_l1, lInf = default_lInf;
            if (target == cv::dnn::DNN_TARGET_OPENCL_FP16)
            {
        #if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_GE(2019020000)
                l1 = 0.05; lInf = 0.3;
        #else
                l1 = 0.017; lInf = 0.0795;
        #endif
            }
            else if (target == cv::dnn::DNN_TARGET_MYRIAD)
            {
                l1 = 0.11; lInf = 0.5;
            }
            else if (target == cv::dnn::DNN_TARGET_CUDA_FP16)
            {
                l1 = 0.04; lInf = 0.2;
            }
            else if (target == cv::dnn::DNN_TARGET_CPU_FP16)
            {
                l1 = 0.06; lInf = 0.3;
            }

            normAssert(outs[0], ref, "", l1, lInf);
            if (target != cv::dnn::DNN_TARGET_MYRIAD || cv::dnn::getInferenceEngineVPUType() != CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X)
                expectNoFallbacksFromIE(model.getNetwork_());
        }

        TEST(Test_Caffe, multiple_inputs)
        {
            const std::string proto = cv::samples::findFile("dnn/layers/net_input.prototxt");
            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

            Mat first_image(10, 11, CV_32FC3);
            Mat second_image(10, 11, CV_32FC3);
            randu(first_image, -1, 1);
            randu(second_image, -1, 1);

            first_image = cv::dnn::blobFromImage(first_image);
            second_image = cv::dnn::blobFromImage(second_image);

            Mat first_image_blue_green = cv::dnn::slice(first_image, Range::all(), Range(0, 2), Range::all(), Range::all());
            Mat first_image_red = cv::dnn::slice(first_image, Range::all(), Range(2, 3), Range::all(), Range::all());
            Mat second_image_blue_green = cv::dnn::slice(second_image, Range::all(), Range(0, 2), Range::all(), Range::all());
            Mat second_image_red = cv::dnn::slice(second_image, Range::all(), Range(2, 3), Range::all(), Range::all());

            net.setInput(first_image_blue_green, "old_style_input_blue_green");
            net.setInput(first_image_red, "different_name_for_red");
            net.setInput(second_image_blue_green, "input_layer_blue_green");
            net.setInput(second_image_red, "old_style_input_red");
            Mat out = net.forward();

            normAssert(out, first_image + second_image);
        }

        TEST(Test_Caffe, shared_weights)
        {
          const std::string proto = cv::samples::findFile("dnn/layers/shared_weights.prototxt");
          const std::string model = cv::samples::findFile("dnn/layers/shared_weights.caffemodel");

          cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, model);

          Mat input_1 = (Mat_<float>(2, 2) << 0., 2., 4., 6.);
          Mat input_2 = (Mat_<float>(2, 2) << 1., 3., 5., 7.);

          Mat blob_1 = cv::dnn::blobFromImage(input_1);
          Mat blob_2 = cv::dnn::blobFromImage(input_2);

          net.setInput(blob_1, "input_1");
          net.setInput(blob_2, "input_2");
          net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

          Mat sum = net.forward();

          EXPECT_EQ(sum.at<float>(0,0), 12.);
          EXPECT_EQ(sum.at<float>(0,1), 16.);
        }

        typedef testing::TestWithParam<std::tuple<std::string, cv::dnn::Target> > opencv_face_detector;
        TEST_P(opencv_face_detector, Accuracy)
        {
            std::string proto = cv::samples::findFile("dnn/opencv_face_detector.prototxt");
            std::string model = cv::samples::findFile(get<0>(GetParam()), false);
            cv::dnn::Target targetId = (cv::dnn::Target)(int)get<1>(GetParam());

            if (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
            if (targetId == cv::dnn::DNN_TARGET_CPU_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, model);
            Mat img = imread(cv::samples::findFile("gpu/lbpcascade/er.png"));
            Mat blob = cv::dnn::blobFromImage(img, 1.0, Size(), Scalar(104.0, 177.0, 123.0), false, false);

            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(targetId);

            net.setInput(blob);
            // Output has shape 1x1xNx7 where N - number of detections.
            // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
            Mat out = net.forward();
            Mat ref = (Mat_<float>(6, 7) << 0, 1, 0.99520785, 0.80997437, 0.16379407, 0.87996572, 0.26685631,
                                            0, 1, 0.9934696, 0.2831718, 0.50738752, 0.345781, 0.5985168,
                                            0, 1, 0.99096733, 0.13629119, 0.24892329, 0.19756334, 0.3310290,
                                            0, 1, 0.98977017, 0.23901358, 0.09084064, 0.29902688, 0.1769477,
                                            0, 1, 0.97203469, 0.67965847, 0.06876482, 0.73999709, 0.1513494,
                                            0, 1, 0.95097077, 0.51901293, 0.45863652, 0.5777427, 0.5347801);
            normAssertDetections(ref, out, "", 0.5, 1e-4, 2e-4);
        }

        // False positives bug for large faces: https://github.com/opencv/opencv/issues/15106
        TEST_P(opencv_face_detector, issue_15106)
        {
            std::string proto = cv::samples::findFile("dnn/opencv_face_detector.prototxt");
            std::string model = cv::samples::findFile(get<0>(GetParam()), false);
            cv::dnn::Target targetId = (cv::dnn::Target)(int)get<1>(GetParam());

            if (targetId == cv::dnn::DNN_TARGET_OPENCL_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_OPENCL_FP16);
            if (targetId == cv::dnn::DNN_TARGET_CPU_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);

            cv::dnn::Net net = cv::dnnlegacy::readNetFromCaffe(proto, model);
            Mat img = imread(cv::samples::findFile("cv/shared/lena.png"));
            img = img.rowRange(img.rows / 4, 3 * img.rows / 4).colRange(img.cols / 4, 3 * img.cols / 4);
            Mat blob = cv::dnn::blobFromImage(img, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);

            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(targetId);

            net.setInput(blob);
            // Output has shape 1x1xNx7 where N - number of detections.
            // An every detection is a vector of values [id, classId, confidence, left, top, right, bottom]
            Mat out = net.forward();
            Mat ref = (Mat_<float>(1, 7) << 0, 1, 0.9149431, 0.30424616, 0.26964942, 0.88733053, 0.99815309);
            normAssertDetections(ref, out, "", 0.89, 6e-5, 1e-4);
        }
        INSTANTIATE_TEST_CASE_P(Test_Caffe, opencv_face_detector,
            Combine(
                Values("dnn/opencv_face_detector.caffemodel",
                       "dnn/opencv_face_detector_fp16.caffemodel"),
                testing::ValuesIn(getAvailableTargets(cv::dnn::DNN_BACKEND_OPENCV))
            )
        );

        TEST_P(Test_Caffe_nets, FasterRCNN_vgg16)
        {
            applyTestTag(
        #if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
                CV_TEST_TAG_MEMORY_2GB,  // utilizes ~1Gb, but huge blobs may not be allocated on 32-bit systems due memory fragmentation
        #else
                CV_TEST_TAG_MEMORY_2GB,
        #endif
                CV_TEST_TAG_LONG,
                CV_TEST_TAG_DEBUG_VERYLONG
            );

        #if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_LT(2021040000)
            if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && (target ==cv::dnn::DNN_TARGET_OPENCL || target ==cv::dnn::DNN_TARGET_OPENCL_FP16))
                applyTestTag(target ==cv::dnn::DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);

            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);

            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
        #endif

        #if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
            // IE exception: Ngraph operation Reshape with name rpn_cls_score_reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target ==cv::dnn::DNN_TARGET_OPENCL || target ==cv::dnn::DNN_TARGET_OPENCL_FP16))
                applyTestTag(target ==cv::dnn::DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                    CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
                );
            // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:390:
            // While validating node 'v1::Reshape bbox_pred_reshape (bbox_pred[0]:f32{1,84}, Constant_241202[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
            // Requested output shape {1,6300,4,1} is incompatible with input shape Shape{1, 84}
            if (target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        #endif

            double scoreDiff = 0.0012, iouDiff = 0.03;
        #if defined(INF_ENGINE_RELEASE)
            if (target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
            if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
                iouDiff = 0.02;
                if (target ==cv::dnn::DNN_TARGET_OPENCL || target ==cv::dnn::DNN_TARGET_OPENCL_FP16) {
                    scoreDiff = 0.04;
                    iouDiff = 0.06;
                }
            }
        #endif

            static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.949398, 99.2454, 210.141, 601.205, 462.849,
                                                   0, 7, 0.997022, 481.841, 92.3218, 722.685, 175.953,
                                                   0, 12, 0.993028, 133.221, 189.377, 350.994, 563.166);
            testFaster("faster_rcnn_vgg16.prototxt", "VGG16_faster_rcnn_final.caffemodel", ref, scoreDiff, iouDiff);
        }

        TEST_P(Test_Caffe_nets, FasterRCNN_zf)
        {
            applyTestTag(
        #if defined(OPENCV_32BIT_CONFIGURATION) && defined(HAVE_OPENCL)
                CV_TEST_TAG_MEMORY_2GB,
        #else
                (target ==cv::dnn::DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_1GB),
        #endif
                CV_TEST_TAG_DEBUG_VERYLONG
            );
        #if defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
            // IE exception: Ngraph operation Reshape with name rpn_cls_score_reshape has dynamic output shape on 0 port, but CPU plug-in supports only static shape
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && (target ==cv::dnn::DNN_TARGET_OPENCL || target ==cv::dnn::DNN_TARGET_OPENCL_FP16))
                applyTestTag(target ==cv::dnn::DNN_TARGET_OPENCL ? CV_TEST_TAG_DNN_SKIP_IE_OPENCL : CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16,
                    CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION
                );
        #endif

            if ((backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
                 backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
            if (target ==cv::dnn::DNN_TARGET_CUDA_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_CUDA_FP16);
            if (target ==cv::dnn::DNN_TARGET_CPU_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_CPU_FP16);
            static Mat ref = (Mat_<float>(3, 7) << 0, 2, 0.90121, 120.407, 115.83, 570.586, 528.395,
                                                   0, 7, 0.988779, 469.849, 75.1756, 718.64, 186.762,
                                                   0, 12, 0.967198, 138.588, 206.843, 329.766, 553.176);

            double scoreDiff = 0.003, iouDiff = 0.07;
            if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) {
                scoreDiff = 0.02;
                iouDiff = 0.13;
            }

            testFaster("faster_rcnn_zf.prototxt", "ZF_faster_rcnn_final.caffemodel", ref, scoreDiff, iouDiff);
        }

        TEST_P(Test_Caffe_nets, RFCN)
        {
            applyTestTag(
                (target ==cv::dnn::DNN_TARGET_CPU ? CV_TEST_TAG_MEMORY_512MB : CV_TEST_TAG_MEMORY_2GB),
                CV_TEST_TAG_LONG,
                CV_TEST_TAG_DEBUG_VERYLONG
            );

            float scoreDiff = default_l1, iouDiff = default_lInf;
            if (backend == cv::dnn::DNN_BACKEND_OPENCV && (target ==cv::dnn::DNN_TARGET_OPENCL_FP16 || target ==cv::dnn::DNN_TARGET_CPU_FP16))
            {
                scoreDiff = 4e-3;
                iouDiff = 8e-2;
            }
            if (target ==cv::dnn::DNN_TARGET_CUDA_FP16)
            {
                scoreDiff = 0.0034;
                iouDiff = 0.12;
            }

        #if defined(INF_ENGINE_RELEASE)
            if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            {
                scoreDiff = 0.1f;
                iouDiff = 0.2f;
            }

            // Check 'backward_compatible_check || in_out_elements_equal' failed at core/src/op/reshape.cpp:427:
            // While validating node 'v1::Reshape bbox_pred_reshape (ave_bbox_pred_rois[0]:f32{1,8,1,1}, Constant_388[0]:i64{4}) -> (f32{?,?,?,?})' with friendly_name 'bbox_pred_reshape':
            // Requested output shape {1,300,8,1} is incompatible with input shape {1, 8, 1, 1}
            if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD, CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        #elif defined(INF_ENGINE_RELEASE) && INF_ENGINE_VER_MAJOR_EQ(2021040000)
            // Exception: Function contains several inputs and outputs with one friendly name! (HETERO bug?)
            if (backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && target !=cv::dnn::DNN_TARGET_CPU)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_NGRAPH, CV_TEST_TAG_DNN_SKIP_IE_VERSION);
        #elif defined(INF_ENGINE_RELEASE)
            if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
                 backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target ==cv::dnn::DNN_TARGET_OPENCL_FP16)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_OPENCL_FP16);
            if ((backend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 ||
                 backend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && target ==cv::dnn::DNN_TARGET_MYRIAD)
                applyTestTag(CV_TEST_TAG_DNN_SKIP_IE_MYRIAD);
        #endif

            static Mat ref = (Mat_<float>(2, 7) << 0, 7, 0.991359, 491.822, 81.1668, 702.573, 178.234,
                                                   0, 12, 0.94786, 132.093, 223.903, 338.077, 566.16);
            testFaster("rfcn_pascal_voc_resnet50.prototxt", "resnet50_rfcn_final.caffemodel", ref, scoreDiff, iouDiff);
        }

INSTANTIATE_TEST_CASE_P(/**/, Test_Caffe_nets, dnnBackendsAndTargets());

}} // namespace
