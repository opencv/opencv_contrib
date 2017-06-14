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
                    const std::string& framework, Net* net)
{
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
}

static void test(const std::string& weights, const std::string& proto,
                 const std::string& scheduler, int inWidth, int inHeight,
                 const std::string& outputLayer, const std::string& framework,
                 int targetId)
{
    Mat input(inHeight, inWidth, CV_32FC3), outputDefault, outputHalide;
    randu(input, 0.0f, 1.0f);

    Net netDefault, netHalide;
    loadNet(weights, proto, framework, &netDefault);
    loadNet(weights, proto, framework, &netHalide);

    netDefault.setBlob("", blobFromImage(input.clone(), 1.0f, false));
    netDefault.forward(netDefault.getLayerId(outputLayer));
    outputDefault = netDefault.getBlob(outputLayer).clone();

    netHalide.setBlob("", blobFromImage(input.clone(), 1.0f, false));
    netHalide.setPreferableBackend(DNN_BACKEND_HALIDE);
    netHalide.compileHalide(scheduler);
    netHalide.forward(netHalide.getLayerId(outputLayer));
    outputHalide = netHalide.getBlob(outputLayer).clone();

    normAssert(outputDefault, outputHalide);

    // An extra test: change input.
    input *= 0.1f;
    netDefault.setBlob("", blobFromImage(input.clone(), 1.0, false));
    netHalide.setBlob("", blobFromImage(input.clone(), 1.0, false));

    normAssert(outputDefault, outputHalide);

    // Swap backends.
    netHalide.setPreferableBackend(DNN_BACKEND_DEFAULT);
    netHalide.forward(netHalide.getLayerId(outputLayer));

    netDefault.setPreferableBackend(DNN_BACKEND_HALIDE);
    netDefault.compileHalide(scheduler);
    netDefault.forward(netDefault.getLayerId(outputLayer));

    outputDefault = netHalide.getBlob(outputLayer).clone();
    outputHalide = netDefault.getBlob(outputLayer).clone();
    normAssert(outputDefault, outputHalide);
}

TEST(Reproducibility_GoogLeNet_Halide, Accuracy)
{
    test(findDataFile("dnn/bvlc_googlenet.caffemodel"),
         findDataFile("dnn/bvlc_googlenet.prototxt"),
         "", 227, 227, "prob", "caffe", DNN_TARGET_CPU);
};

TEST(Reproducibility_AlexNet_Halide, Accuracy)
{
    test(getOpenCVExtraDir() + "/dnn/bvlc_alexnet.caffemodel",
         getOpenCVExtraDir() + "/dnn/bvlc_alexnet.prototxt",
         getOpenCVExtraDir() + "/dnn/halide_scheduler_alexnet.yml",
         227, 227, "prob", "caffe", DNN_TARGET_CPU);
};

// TEST(Reproducibility_ResNet_50_Halide, Accuracy)
// {
//     test(getOpenCVExtraDir() + "/dnn/ResNet-50-model.caffemodel",
//          getOpenCVExtraDir() + "/dnn/ResNet-50-deploy.prototxt",
//          getOpenCVExtraDir() + "/dnn/halide_scheduler_resnet_50.yml",
//          224, 224, "prob", "caffe", DNN_TARGET_CPU);
// };

// TEST(Reproducibility_SqueezeNet_v1_1_Halide, Accuracy)
// {
//     test(getOpenCVExtraDir() + "/dnn/squeezenet_v1_1.caffemodel",
//          getOpenCVExtraDir() + "/dnn/squeezenet_v1_1.prototxt",
//          getOpenCVExtraDir() + "/dnn/halide_scheduler_squeezenet_v1_1.yml",
//          227, 227, "prob", "caffe", DNN_TARGET_CPU);
// };

TEST(Reproducibility_Inception_5h_Halide, Accuracy)
{
    test(getOpenCVExtraDir() + "/dnn/tensorflow_inception_graph.pb", "",
         getOpenCVExtraDir() + "/dnn/halide_scheduler_inception_5h.yml",
         224, 224, "softmax2", "tensorflow", DNN_TARGET_CPU);
};

TEST(Reproducibility_ENet_Halide, Accuracy)
{
    test(getOpenCVExtraDir() + "/dnn/Enet-model-best.net", "",
         getOpenCVExtraDir() + "/dnn/halide_scheduler_enet.yml",
         512, 512, "l367_Deconvolution", "torch", DNN_TARGET_CPU);
};
#endif  // HAVE_HALIDE

}  // namespace cvtest
