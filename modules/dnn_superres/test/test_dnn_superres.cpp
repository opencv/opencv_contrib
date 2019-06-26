// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

    const std::string DNN_SUPERRES_DIR = "dnn_superres";
    const std::string IMAGE_FILENAME = "butterfly.png";

    /****************************************************************************************\
    *                                ESPCN Test                                              *
    \****************************************************************************************/

    class CV_DnnSuperResESPCNTest : public cvtest::BaseTest
    {
        public:
            CV_DnnSuperResESPCNTest();

        protected:
            Ptr<DnnSuperResImpl> dnn_sr;
            virtual void run(int);
            void runOneModel(std::string algorithm, int scale, std::string model_filename);
    };

    CV_DnnSuperResESPCNTest::CV_DnnSuperResESPCNTest()
    {
        dnn_sr = makePtr<DnnSuperResImpl>();
    }

    void CV_DnnSuperResESPCNTest::runOneModel(std::string algorithm, int scale, std::string model_filename)
    {
        std::string path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + IMAGE_FILENAME;

        Mat img = imread(path);
        if( img.empty() )
        {
            ts->printf( cvtest::TS::LOG, "Test image not found!\n");
            ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
            return;
        }

        std::string pb_path = std::string(ts->get_data_path()) + DNN_SUPERRES_DIR + "/" + model_filename;

        this->dnn_sr->readModel(pb_path);

        this->dnn_sr->setModel(algorithm, scale);

        if( this->dnn_sr->getScale() != scale )
        {
            ts->printf( cvtest::TS::LOG, "Scale factor could not be set for scale algorithm %s and scale factor %d!\n", algorithm.c_str(), scale);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        if( this->dnn_sr->getAlgorithm() != algorithm )
        {
            ts->printf( cvtest::TS::LOG, "Algorithm could not be set for scale algorithm %s and scale factor %d!\n", algorithm.c_str(), scale);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        Mat img_new;
        this->dnn_sr->upsample(img, img_new);

        if( img_new.empty() )
        {
            ts->printf( cvtest::TS::LOG, "Could not perform upsampling for scale algorithm %s and scale factor %d!\n", algorithm.c_str(), scale);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }

        int new_cols = img.cols * scale;
        int new_rows = img.rows * scale;
        if( img_new.cols != new_cols || img_new.rows != new_rows )
        {
            ts->printf( cvtest::TS::LOG, "Dimensions are not correct for scale algorithm %s and scale factor %d!\n", algorithm.c_str(), scale);
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            return;
        }
    }

    void CV_DnnSuperResESPCNTest::run(int)
    {
        //x2
        runOneModel("espcn", 2, "ESPCN_x2.pb");

        //x3
        runOneModel("espcn", 3, "ESPCN_x3.pb");

        //x4
        runOneModel("espcn", 4, "ESPCN_x4.pb");
    }

    TEST(CV_DnnSuperResESPCNTest, accuracy)
    {
        CV_DnnSuperResESPCNTest test;
        test.safe_run();
    }

    /****************************************************************************************\
    *                                EDSR Test                                               *
     \****************************************************************************************/

    class CV_DnnSuperResEDSRTest : public cvtest::BaseTest
    {
        public:
            CV_DnnSuperResEDSRTest();

        protected:
            virtual void run( int );
    };

    CV_DnnSuperResEDSRTest::CV_DnnSuperResEDSRTest()
    {
    }

    void CV_DnnSuperResEDSRTest::run(int)
    {
    }

    TEST(CV_DnnSuperResEDSRTest, accuracy)
    {
        CV_DnnSuperResEDSRTest test;
        test.safe_run();
    }

    /****************************************************************************************\
    *                                LAPSRN Test                                             *
    \****************************************************************************************/

    class CV_DnnSuperResLAPSRNTest : public cvtest::BaseTest
    {
        public:
            CV_DnnSuperResLAPSRNTest();

        protected:
            virtual void run( int );
    };

    CV_DnnSuperResLAPSRNTest::CV_DnnSuperResLAPSRNTest()
    {
    }

    void CV_DnnSuperResLAPSRNTest::run(int)
    {
    }

    TEST(CV_DnnSuperResLAPSRNTest, accuracy)
    {
        CV_DnnSuperResLAPSRNTest test;
        test.safe_run();
    }

    /****************************************************************************************\
    *                                FSRCNN Test                                             *
    \****************************************************************************************/

    class CV_DnnSuperResFSRCNNTest : public cvtest::BaseTest
    {
        public:
            CV_DnnSuperResFSRCNNTest();

        protected:
            virtual void run( int );
    };

    CV_DnnSuperResFSRCNNTest::CV_DnnSuperResFSRCNNTest()
    {
    }

    void CV_DnnSuperResFSRCNNTest::run(int)
    {
    }

    TEST(CV_DnnSuperResFSRCNNTest, accuracy)
    {
        CV_DnnSuperResFSRCNNTest test;
        test.safe_run();
    }

}}
