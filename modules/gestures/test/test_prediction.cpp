#include <iostream>
#include <ctime>
#include <algorithm>

#include <opencv2/ts.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "chalearn_csv_readers.hpp"
#include <opencv2/gestures.hpp>

namespace cvtest
{
    class DNNPredictionTest : public BaseTest
    {
        public:
            DNNPredictionTest(int count, float accuracy, bool verbose);

        protected:
            void run (int);

            int mIterations;
            float mTargetAcc;
            bool mVerbose;
    };

    DNNPredictionTest::DNNPredictionTest(int count, float accuracy, bool verbose):
        mIterations(count),
        mTargetAcc(accuracy),
        mVerbose(verbose)
    {
    }

    void DNNPredictionTest::run(int)
    {
        ts->set_failed_test_info(TS::OK);

        // Initialization of the network.
        cv::Ptr<cv::gestures::GesturesClassifierDNN> classifier = cv::gestures::GesturesClassifierDNN::create(
            ts->get_data_path() + "deploy_chalearn.prototxt",
            ts->get_data_path() + "chalearn_trained.caffemodel",
            ts->get_data_path() + "input_mean/",
            ts->get_data_path() + "chalearn_labels.txt");

        cv::VideoCapture colorCap(ts->get_data_path() + "Sample0471/Sample0471_color.mp4");
        if(!colorCap.isOpened())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        cv::VideoCapture depthCap(ts->get_data_path() + "Sample0471/Sample0471_depth.mp4");
        if(!depthCap.isOpened())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        MocapCSVStreamer skelCap(ts->get_data_path() + "Sample0471/Sample0471_skeleton.csv");
        if(!skelCap.isOpened())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        MetaDataCSVReader dataReader(ts->get_data_path() + "Sample0471/Sample0471_data.csv");
        if(!dataReader.isValid())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        LabelsCSVReader labelsReader(ts->get_data_path() + "Sample0471/Sample0471_labels.csv", dataReader.getFrameCount());
        if(!labelsReader.isValid())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        cv::Mat colorFrame;
        cv::Mat depthFrame;
        cv::Mat skelFrame;

        cv::Mat grayDepth;
        cv::Mat scaledDepth;

        cv::Mat preds;

        bool valid_label = false;
        int gt_label;
        cv::Point maxpred;

        int frameCount = 0;
        clock_t tic = clock();

        float meanTime = 0.0;

        cv::Mat confusMat(classifier->getClassesCount(), classifier->getClassesCount(), CV_16U);
        confusMat.setTo(cv::Scalar(0));

        while(colorCap.read(colorFrame)
                && depthCap.read(depthFrame)
                && skelCap.read(skelFrame))
        {
            cv::cvtColor(depthFrame, grayDepth, cv::COLOR_BGR2GRAY);
            grayDepth.convertTo(scaledDepth, CV_32F, 1.0/255);

            classifier->feedNewFrames(colorFrame, scaledDepth, skelFrame);

            gt_label = labelsReader.getLabel(frameCount);

            for(int t = 1; t < classifier->getTemporalSize(); ++t)
            {
                int i = frameCount - t*classifier->getTemporalStride();
                valid_label = (i >= 0 && labelsReader.getLabel(i) == gt_label);
                if(!valid_label)
                {
                    break;
                }
            }

            ++frameCount;

            if(valid_label && classifier->getPrediction(preds))
            {
                cv::minMaxLoc(preds, NULL, NULL, NULL, &maxpred);
                confusMat.at<uint16_t>(gt_label, maxpred.y) += 1;
            }

            if(mIterations > 0 && frameCount >= mIterations)
            {
                break;
            }
        }

        clock_t tac = clock();
        meanTime += 1000.0*(tac-tic)/CLOCKS_PER_SEC;
        meanTime /= frameCount;
        std::cout << "Mean classification time on " << frameCount << " samples: " << ceil(meanTime) << "ms" << std::endl;

        if(mVerbose)
        {
            std::cout << "Confusion matrix:" << std::endl;
            for(int i = 0; i < classifier->getClassesCount(); ++i)
            {
                for(int j = 0; j < classifier->getClassesCount(); ++j)
                {
                    std::cout << std::setfill(' ') << std::setw(4) << confusMat.at<uint16_t>(i,j);
                    if(j < classifier->getClassesCount()-1)
                    {
                        std::cout << ",";
                    }
                }
                std::cout << std::endl;
            }
        }

        double successRate = 1.0*cv::trace(confusMat)[0] / cv::sum(confusMat)[0];
        std::cout << "Success rate: " << successRate << std::endl;

        if(successRate < mTargetAcc)
        {
            ts->set_failed_test_info(TS::FAIL_BAD_ACCURACY);
        }
        ASSERT_GE(successRate, mTargetAcc);
    }

    TEST(Gestures_DNNPredictionTest, accuracy) { DNNPredictionTest test(-1, 0.7, true); test.safe_run(); }
}
