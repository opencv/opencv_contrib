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
