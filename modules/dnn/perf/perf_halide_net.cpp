// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

namespace cvtest
{

#ifdef HAVE_HALIDE
using namespace cv;
using namespace dnn;

static void loadNet(const std::string& weights, const std::string& proto,
                    const std::string& scheduler, int inWidth, int inHeight,
                    const std::string& outputLayer, const std::string& framework,
                    int targetId, Net* net, int* outputLayerId)
{
    Mat input(inHeight, inWidth, CV_32FC3);
    randu(input, 0.0f, 1.0f);

    if (framework == "caffe")
    {
        *net = cv::dnn::readNetFromCaffe(proto, weights);
    }
    else if (framework == "torch")
    {
        *net = cv::dnn::readNetFromTorch(weights);
    }
    else if (framework == "tensorflow")
    {
        *net = cv::dnn::readNetFromTensorflow(weights);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown framework " + framework);

    net->setBlob("", cv::dnn::blobFromImage(input, 1.0, false));
    net->setPreferableBackend(DNN_BACKEND_HALIDE);
    net->compileHalide(scheduler);
    *outputLayerId = net->getLayerId(outputLayer);
    net->forward(*outputLayerId);
}

PERF_TEST(GoogLeNet, HalidePerfTest)
{
    Net net;
    int outputLayerId;
    loadNet(findDataFile("dnn/bvlc_googlenet.caffemodel"),
            findDataFile("dnn/bvlc_googlenet.prototxt"),
            "", 227, 227, "prob", "caffe", DNN_TARGET_CPU, &net, &outputLayerId);

    TEST_CYCLE_N(10)
    {
        net.forward(outputLayerId);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(AlexNet, HalidePerfTest)
{
    Net net;
    int outputLayerId;
    loadNet(findDataFile("dnn/bvlc_alexnet.caffemodel"),
            findDataFile("dnn/bvlc_alexnet.prototxt"),
            findDataFile("dnn/halide_scheduler_alexnet.yml"),
            227, 227, "prob", "caffe", DNN_TARGET_CPU, &net, &outputLayerId);

    TEST_CYCLE_N(10)
    {
        net.forward(outputLayerId);
    }
    SANITY_CHECK_NOTHING();
}

// PERF_TEST(ResNet50, HalidePerfTest)
// {
//     Net net;
//     int outputLayerId;
//     loadNet(findDataFile("dnn/ResNet-50-model.caffemodel"),
//             findDataFile("dnn/ResNet-50-deploy.prototxt"),
//             findDataFile("dnn/halide_scheduler_resnet_50.yml"),
//             224, 224, "prob", "caffe", DNN_TARGET_CPU, &net, &outputLayerId);
//
//     TEST_CYCLE_N(10)
//     {
//         net.forward(outputLayerId);
//     }
//     SANITY_CHECK_NOTHING();
// }

// PERF_TEST(SqueezeNet_v1_1, HalidePerfTest)
// {
//     Net net;
//     int outputLayerId;
//     loadNet(findDataFile("dnn/squeezenet_v1_1.caffemodel"),
//             findDataFile("dnn/squeezenet_v1_1.prototxt"),
//             findDataFile("dnn/halide_scheduler_squeezenet_v1_1.yml"),
//             227, 227, "prob", "caffe", DNN_TARGET_CPU, &net, &outputLayerId);
//
//     TEST_CYCLE_N(10)
//     {
//         net.forward(outputLayerId);
//     }
//     SANITY_CHECK_NOTHING();
// }

PERF_TEST(Inception_5h, HalidePerfTest)
{
    Net net;
    int outputLayerId;
    loadNet(findDataFile("dnn/tensorflow_inception_graph.pb"), "",
            findDataFile("dnn/halide_scheduler_inception_5h.yml"),
            224, 224, "softmax2", "tensorflow", DNN_TARGET_CPU,
            &net, &outputLayerId);

    TEST_CYCLE_N(10)
    {
        net.forward(outputLayerId);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST(ENet, HalidePerfTest)
{
    Net net;
    int outputLayerId;
    loadNet(findDataFile("dnn/Enet-model-best.net"), "",
            findDataFile("dnn/halide_scheduler_enet.yml"),
            512, 256, "l367_Deconvolution", "torch", DNN_TARGET_CPU,
            &net, &outputLayerId);

    TEST_CYCLE_N(10)
    {
        net.forward(outputLayerId);
    }
    SANITY_CHECK_NOTHING();
}
#endif  // HAVE_HALIDE

}  // namespace cvtest
