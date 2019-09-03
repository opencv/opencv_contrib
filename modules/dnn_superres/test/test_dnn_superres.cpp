// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test{ namespace {

const std::string DNN_SUPERRES_DIR = "dnn_superres";
const std::string IMAGE_FILENAME = "butterfly.png";

/****************************************************************************************\
*                                Test single output models                               *
\****************************************************************************************/

void runSingleModel(std::string algorithm, int scale, std::string model_filename)
{
Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();

std::string path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

Mat img = imread(path);
if (img.empty())
{
    TS::ptr()->printf(cvtest::TS::LOG, "Test image not found!\n");
    TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
    return;
    }

std::string pb_path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

dnn_sr->readModel(pb_path);

dnn_sr->setModel(algorithm, scale);

if (dnn_sr->getScale() != scale)
{
    TS::ptr()->printf(cvtest::TS::LOG,
                "Scale factor could not be set for scale algorithm %s and scale factor %d!\n",
                algorithm.c_str(), scale);
    TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
    return;
    }

    if (dnn_sr->getAlgorithm() != algorithm)
    {
        TS::ptr()->printf(cvtest::TS::LOG, "Algorithm could not be set for scale algorithm %s and scale factor %d!\n",
                    algorithm.c_str(), scale);
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }

    Mat img_new;
    dnn_sr->upsample(img, img_new);

    if (img_new.empty())
    {
        TS::ptr()->printf(cvtest::TS::LOG,
                "Could not perform upsampling for scale algorithm %s and scale factor %d!\n",
                algorithm.c_str(), scale);
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }

    int new_cols = img.cols * scale;
    int new_rows = img.rows * scale;
    if (img_new.cols != new_cols || img_new.rows != new_rows)
    {
        TS::ptr()->printf(cvtest::TS::LOG, "Dimensions are not correct for scale algorithm %s and scale factor %d!\n",
                algorithm.c_str(), scale);
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }
}

TEST(CV_DnnSuperResSingleOutputTest, accuracy)
{
    //x2
    runSingleModel("espcn", 2, "ESPCN_x2.pb");
}

/****************************************************************************************\
*                                Test multi output models                               *
\****************************************************************************************/

void runMultiModel(std::string algorithm, int scale, std::string model_filename,
                std::vector<int> scales, std::vector<String> node_names)
{
    Ptr <DnnSuperResImpl> dnn_sr = makePtr<DnnSuperResImpl>();

    std::string path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

    Mat img = imread(path);
    if ( img.empty() )
    {
        TS::ptr()->printf(cvtest::TS::LOG, "Test image not found!\n");
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    std::string pb_path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

    dnn_sr->readModel(pb_path);

    dnn_sr->setModel(algorithm, scale);

    if ( dnn_sr->getScale() != scale )
    {
        TS::ptr()->printf(cvtest::TS::LOG,
                    "Scale factor could not be set for scale algorithm %s and scale factor %d!\n",
                    algorithm.c_str(), scale);
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }

    if ( dnn_sr->getAlgorithm() != algorithm )
    {
        TS::ptr()->printf(cvtest::TS::LOG, "Algorithm could not be set for scale algorithm %s and scale factor %d!\n",
                    algorithm.c_str(), scale);
                    TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        return;
    }

    std::vector<Mat> outputs;
    dnn_sr->upsampleMultioutput(img, outputs, scales, node_names);

    for(unsigned int i = 0; i < outputs.size(); i++)
    {
        if( outputs[i].empty() )
        {
            TS::ptr()->printf(cvtest::TS::LOG,
                        "Could not perform upsampling for scale algorithm %s and scale factor %d!\n",
                        algorithm.c_str(), scale);
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        int new_cols = img.cols * scales[i];
        int new_rows = img.rows * scales[i];

        if ( outputs[i].cols != new_cols || outputs[i].rows != new_rows )
        {
            TS::ptr()->printf(cvtest::TS::LOG, "Dimensions are not correct for scale algorithm %s and scale factor %d!\n",
                        algorithm.c_str(), scale);
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }
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

/****************************************************************************************\
*                                Test benchmarking                                       *
\****************************************************************************************/

void runBenchmark(std::string algorithm, int scale, std::string model_filename)
{
    DnnSuperResImpl dnn_sr;

    std::string path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

    Mat img = imread(path);
    if ( img.empty() )
    {
        TS::ptr()->printf(cvtest::TS::LOG, "Test image not found!\n");
        TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    std::string pb_path = std::string(TS::ptr()->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

    dnn_sr.readModel(pb_path);

    dnn_sr.setModel(algorithm, scale);

    ASSERT_EQ(dnn_sr.getScale(), scale);

    ASSERT_EQ(dnn_sr.getAlgorithm(), algorithm);

    int width = img.cols - (img.cols % scale);
    int height = img.rows - (img.rows % scale);
    Mat cropped = img(Rect(0, 0, width, height));

    Mat img_downscaled;
    cv::resize(cropped, img_downscaled, cv::Size(), 1.0/scale, 1.0/scale);

    std::vector<double> psnrs, ssims, perfs;

    DnnSuperResQuality::benchmark(dnn_sr, cropped, psnrs, ssims, perfs, 0, 0);

    ASSERT_EQ(static_cast<int>(psnrs.size()), 4);
    ASSERT_EQ(static_cast<int>(ssims.size()), 4);
    ASSERT_EQ(static_cast<int>(perfs.size()), 4);

    ASSERT_EQ(psnrs.size(), ssims.size());
    ASSERT_EQ(psnrs.size(), perfs.size());

    for(unsigned int i = 0; i < 4; i++)
    {
        ASSERT_GT(psnrs[i], 0.0);

        ASSERT_GE(ssims[i], 0.0);
        ASSERT_LE(ssims[i], 1.0);

        ASSERT_GT(perfs[i], 0.0);
    }
}

TEST(CV_DnnSuperResBenchmarkingTest, accuracy)
{
    runBenchmark("espcn", 2, "ESPCN_x2.pb");
}

}}