// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

const std::string DNN_SUPERRES_DIR = "dnn_superres";
const std::string IMAGE_FILENAME = "butterfly.png";

/****************************************************************************************\
*                                Test single output models                               *
\****************************************************************************************/

void runSingleModel(std::string algorithm, int scale, std::string model_filename)
{
    SCOPED_TRACE(algorithm);

    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();

    std::string path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME);

    Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    std::string pb_path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + model_filename);

    dnn_sr->readModel(pb_path);

    dnn_sr->setModel(algorithm, scale);

    ASSERT_EQ(scale, dnn_sr->getScale());
    ASSERT_EQ(algorithm, dnn_sr->getAlgorithm());

    Mat result;
    dnn_sr->upsample(img, result);

    ASSERT_FALSE(result.empty()) << "Could not perform upsampling for scale algorithm " << algorithm << " and scale factor " << scale;

    int new_cols = img.cols * scale;
    int new_rows = img.rows * scale;
    ASSERT_EQ(new_cols, result.cols);
    ASSERT_EQ(new_rows, result.rows);
}

// Test model parameter validation
TEST(CV_DnnSuperResParameterValidationTest, validate_parameters)
{
    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();
    
    // Test invalid algorithm
    try {
        dnn_sr->setModel("invalid_algorithm", 2);
        FAIL() << "Expected exception for invalid algorithm not thrown";
    } catch (const cv::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("Unknown/unsupported superres algorithm") != std::string::npos);
    }

    // Test invalid scale (0 or negative)
    try {
        dnn_sr->setModel("espcn", 0);
        FAIL() << "Expected exception for invalid scale not thrown";
    } catch (const cv::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("Upscaling ratio must be positive") != std::string::npos);
    }
    
    // Test empty model
    Mat img = Mat::zeros(100, 100, CV_8UC3);
    Mat result;
    try {
        dnn_sr->upsample(img, result);
        FAIL() << "Expected exception for empty model not thrown";
    } catch (const cv::Exception& e) {
        EXPECT_TRUE(std::string(e.what()).find("Model not specified") != std::string::npos);
    }
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy_espcn_2)
{
    runSingleModel("espcn", 2, "ESPCN_x2.pb");
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy_fsrcnn_2)
{
    runSingleModel("fsrcnn", 2, "FSRCNN_x2.pb");
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy_fsrcnn_3)
{
    runSingleModel("fsrcnn", 3, "FSRCNN_x3.pb");
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy_srgan_4)
{
    runSingleModel("srgan", 4, "SRGAN_x4.pb");
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy_rdn_3)
{
    runSingleModel("rdn", 3, "RDN_x3.pb");
}

// Extended tests for SRGAN
TEST(CV_DnnSuperResSRGANTest, various_input_sizes)
{
    SCOPED_TRACE("srgan");

    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();
    std::string path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME);
    Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;
    
    std::string pb_path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/SRGAN_x4.pb");
    dnn_sr->readModel(pb_path);
    dnn_sr->setModel("srgan", 4);
    
    // Test with different input sizes
    std::vector<Size> sizes = {Size(32, 32), Size(64, 64), Size(128, 96)};
    
    for (const auto& size : sizes) {
        Mat resized;
        resize(img, resized, size);
        
        Mat result;
        dnn_sr->upsample(resized, result);
        
        ASSERT_FALSE(result.empty()) << "Could not perform upsampling for input size " << size;
        ASSERT_EQ(size.width * 4, result.cols);
        ASSERT_EQ(size.height * 4, result.rows);
    }
}

// Extended tests for RDN
TEST(CV_DnnSuperResRDNTest, different_input_channels)
{
    SCOPED_TRACE("rdn");

    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();
    std::string path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME);
    Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;
    
    std::string pb_path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/RDN_x3.pb");
    dnn_sr->readModel(pb_path);
    dnn_sr->setModel("rdn", 3);
    
    // Test with color image
    Mat color_result;
    dnn_sr->upsample(img, color_result);
    ASSERT_FALSE(color_result.empty()) << "Could not perform upsampling for color image";
    ASSERT_EQ(img.cols * 3, color_result.cols);
    ASSERT_EQ(img.rows * 3, color_result.rows);
    ASSERT_EQ(img.channels(), color_result.channels());
    
    // Test with grayscale image
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat gray_result;
    dnn_sr->upsample(gray, gray_result);
    ASSERT_FALSE(gray_result.empty()) << "Could not perform upsampling for grayscale image";
    ASSERT_EQ(gray.cols * 3, gray_result.cols);
    ASSERT_EQ(gray.rows * 3, gray_result.rows);
    ASSERT_EQ(gray.channels(), gray_result.channels());
}

/****************************************************************************************\
*                                Test multi output models                               *
\****************************************************************************************/

void runMultiModel(std::string algorithm, int scale, std::string model_filename,
                std::vector<int> scales, std::vector<String> node_names)
{
    SCOPED_TRACE(algorithm);

    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();

    std::string path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME);

    Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    std::string pb_path = cvtest::findDataFile(DNN_SUPERRES_DIR + "/" + model_filename);

    dnn_sr->readModel(pb_path);

    dnn_sr->setModel(algorithm, scale);

    ASSERT_EQ(scale, dnn_sr->getScale());
    ASSERT_EQ(algorithm, dnn_sr->getAlgorithm());

    std::vector<Mat> outputs;
    dnn_sr->upsampleMultioutput(img, outputs, scales, node_names);

    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        SCOPED_TRACE(cv::format("i=%d scale[i]=%d", i, scales[i]));

        ASSERT_FALSE(outputs[i].empty());

        int new_cols = img.cols * scales[i];
        int new_rows = img.rows * scales[i];

        EXPECT_EQ(new_cols, outputs[i].cols);
        EXPECT_EQ(new_rows, outputs[i].rows);
    }
}

TEST(CV_DnnSuperResMultiOutputTest, accuracy)
{
    //LAPSRN
    //x4
    std::vector<String> names_4x {"NCHW_output_2x", "NCHW_output_4x"};
    std::vector<int> scales_4x {2, 4};
    runMultiModel("lapsrn", 4, "LapSRN_x4.pb", scales_4x, names_4x);
}

}}
