// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test
{
    namespace
    {
        const std::string DNN_SUPERRES_DIR = "dnn_superres";
        const std::string IMAGE_FILENAME = "butterfly.png";

        /****************************************************************************************\
        *                                Test single output models                               *
        \****************************************************************************************/

        class CV_DnnSuperResSingleOutputTest : public cvtest::BaseTest
        {
            public:
                CV_DnnSuperResSingleOutputTest();

            protected:
                Ptr <DnnSuperResImpl> dnn_sr;

                virtual void run(int);

                void runOneModel(std::string algorithm, int scale, std::string model_filename);
        };

        void CV_DnnSuperResSingleOutputTest::runOneModel(std::string algorithm, int scale, std::string model_filename)
        {
            std::string path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

            Mat img = imread(path);
            if (img.empty())
            {
                ts->printf(cvtest::TS::LOG, "Test image not found!\n");
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
                return;
            }

            std::string pb_path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

            this->dnn_sr->readModel(pb_path);

            this->dnn_sr->setModel(algorithm, scale);

            if (this->dnn_sr->getScale() != scale)
            {
                ts->printf(cvtest::TS::LOG,
                           "Scale factor could not be set for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            if (this->dnn_sr->getAlgorithm() != algorithm)
            {
                ts->printf(cvtest::TS::LOG, "Algorithm could not be set for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            Mat img_new;
            this->dnn_sr->upsample(img, img_new);

            if (img_new.empty())
            {
                ts->printf(cvtest::TS::LOG,
                           "Could not perform upsampling for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            int new_cols = img.cols * scale;
            int new_rows = img.rows * scale;
            if (img_new.cols != new_cols || img_new.rows != new_rows)
            {
                ts->printf(cvtest::TS::LOG, "Dimensions are not correct for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }
        }

        CV_DnnSuperResSingleOutputTest::CV_DnnSuperResSingleOutputTest()
        {
            dnn_sr = makePtr<DnnSuperResImpl>();
        }

        void CV_DnnSuperResSingleOutputTest::run(int)
        {
            //EDSR
            //x2


            //x3


            //x4


            //FSRCNN
            //x2
            runOneModel("fsrcnn", 2, "FSRCNN_x2.pb");

            //x3
            runOneModel("fsrcnn", 3, "FSRCNN_x3.pb");

            //x4
            runOneModel("fsrcnn", 4, "FSRCNN_x4.pb");
        }

        TEST(CV_DnnSuperResSingleOutputTest, accuracy)
        {
            CV_DnnSuperResSingleOutputTest test;
            test.safe_run();
        }

        /****************************************************************************************\
        *                                Test multi output models                               *
        \****************************************************************************************/

        class CV_DnnSuperResMultiOutputTest : public cvtest::BaseTest
        {
            public:
                CV_DnnSuperResMultiOutputTest();

            protected:
                Ptr <DnnSuperResImpl> dnn_sr;

                virtual void run(int);

                void runOneModel(std::string algorithm, int scale, std::string model_filename,
                             std::vector<int> scales, std::vector<String> node_names);
        };

        void CV_DnnSuperResMultiOutputTest::runOneModel(std::string algorithm, int scale, std::string model_filename,
                std::vector<int> scales, std::vector<String> node_names)
        {
            std::string path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

            Mat img = imread(path);
            if ( img.empty() )
            {
                ts->printf(cvtest::TS::LOG, "Test image not found!\n");
                ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
                return;
            }

            std::string pb_path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

            this->dnn_sr->readModel(pb_path);

            this->dnn_sr->setModel(algorithm, scale);

            if ( this->dnn_sr->getScale() != scale )
            {
                ts->printf(cvtest::TS::LOG,
                           "Scale factor could not be set for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            if ( this->dnn_sr->getAlgorithm() != algorithm )
            {
                ts->printf(cvtest::TS::LOG, "Algorithm could not be set for scale algorithm %s and scale factor %d!\n",
                           algorithm.c_str(), scale);
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                return;
            }

            std::vector<Mat> outputs;
            this->dnn_sr->upsample_multioutput(img, outputs, scales, node_names);

            for(unsigned int i = 0; i < outputs.size(); i++)
            {
                if( outputs[i].empty() )
                {
                    ts->printf(cvtest::TS::LOG,
                               "Could not perform upsampling for scale algorithm %s and scale factor %d!\n",
                               algorithm.c_str(), scale);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }

                int new_cols = img.cols * scales[i];
                int new_rows = img.rows * scales[i];

                if ( outputs[i].cols != new_cols || outputs[i].rows != new_rows )
                {
                    ts->printf(cvtest::TS::LOG, "Dimensions are not correct for scale algorithm %s and scale factor %d!\n",
                               algorithm.c_str(), scale);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    return;
                }
            }
        }

        CV_DnnSuperResMultiOutputTest::CV_DnnSuperResMultiOutputTest()
        {
            dnn_sr = makePtr<DnnSuperResImpl>();
        }

        void CV_DnnSuperResMultiOutputTest::run(int)
        {
            //LAPSRN
            //x4
            std::vector<String> names_4x {"NCHW_output_0", "NCHW_output"};
            std::vector<int> scales_4x {2, 4};
            runOneModel("lapsrn", 4, "LapSRN_x4.pb", scales_4x, names_4x);

            //x8
            std::vector<String> names_8x {"NCHW_output_0", "NCHW_output_1", "NCHW_output"};
            std::vector<int> scales_8x  {2, 4, 8};
            runOneModel("lapsrn", 8, "LapSRN_x8.pb", scales_8x, names_8x);
        }

        TEST(CV_DnnSuperResMultiOutputTest, accuracy)
        {
            CV_DnnSuperResMultiOutputTest test;
            test.safe_run();
        }
    }
}