#include <iostream>
#include <iomanip>
#include <deque>
#include <fstream>
#include <sstream>

#include <opencv2/ts.hpp>

#include "chalearn_csv_readers.hpp"
#include <opencv2/gestures.hpp>

#include <cmath>

namespace cvtest
{
    class DNNSkelDescTest : public BaseTest
    {
        public:
            DNNSkelDescTest(double tolerance, double epsilon, double maxFail, bool verbose);

        protected:
            void run (int);

            double mTolerance;
            double mMinEpsilon;
            double mMaxFailures;
            bool mVerbose;
    };

    DNNSkelDescTest::DNNSkelDescTest(double tolerance, double epsilon, double maxFail, bool verbose):
        mTolerance(tolerance),
        mMinEpsilon(epsilon),
        mMaxFailures(maxFail),
        mVerbose(verbose)
    {
    }

    void DNNSkelDescTest::run(int)
    {
        ts->set_failed_test_info(TS::OK);

        MocapCSVStreamer mocap(ts->get_data_path() + "Sample0471/Sample0471_skeleton.csv");
        if(!mocap.isOpened())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }

        cv::Mat mocapFrame;

        std::ifstream csvRef((ts->get_data_path() + "Sample0471/SkelDescriptors.csv").c_str());
        if(!csvRef.is_open())
        {
            ts->set_failed_test_info(TS::FAIL_MISSING_TEST_DATA);
        }
        std::string line;
        std::string elem;

        int frameCount = 1;
        int sampleCount = 0;
        int angleFailures = 0;
        int distanceFailures = 0;
        int motionFailures = 0;

        cv::Ptr<cv::gestures::SkeletonFrame> previous2;
        cv::Ptr<cv::gestures::SkeletonFrame> previous;
        cv::Ptr<cv::gestures::SkeletonFrame> current;

        while(std::getline(csvRef, line))
        {
            std::istringstream linestream(line);
            std::getline(linestream, elem, ',');
            int targetFrame = atoi(elem.c_str());
            CV_Assert(frameCount <= targetFrame);
            if(mVerbose)
            {
                std::cout << "==============\t     Frame " << std::setw(4) << targetFrame << "\t     ==============" << std::endl;
            }

            if(frameCount <= targetFrame - 2)
            {
                for(; frameCount < targetFrame - 2; ++frameCount)
                {
                    mocap.read(mocapFrame);
                }

                mocap.read(mocapFrame);
                previous2 = cv::makePtr<cv::gestures::SkeletonFrame>(mocapFrame);
                mocap.read(mocapFrame);
                previous = cv::makePtr<cv::gestures::SkeletonFrame>(mocapFrame);
                frameCount += 2;

                previous2->normalize();
                previous->normalize();
            }
            else if(frameCount == targetFrame - 1)
            {
                previous2 = previous;

                mocap.read(mocapFrame);
                previous = cv::makePtr<cv::gestures::SkeletonFrame>(mocapFrame);
                ++frameCount;

                previous->normalize();
            }

            mocap.read(mocapFrame);
            current = cv::makePtr<cv::gestures::SkeletonFrame>(mocapFrame);
            ++frameCount;

            cv::Mat testDescriptor;
            current->createDescriptor(*previous, *previous2, testDescriptor);

            float refDescriptor = 0.0;
            int i = 0;
            while(std::getline(linestream, elem, ','))
            {
                refDescriptor = atof(elem.c_str());

                if(fabs(testDescriptor.at<float>(i) - refDescriptor) > std::max(fabs(mTolerance * refDescriptor), mMinEpsilon))
                {
                    if(i < 28)
                    {
                        // Fail on angles
                        ++angleFailures;
                        if(mVerbose)
                        {
                            if(i < 10)
                            {
                                std::cout << "Bending  ";
                            }
                            else if(i%2 == 0)
                            {
                                std::cout << "Inclin.  ";
                            }
                            else
                            {
                                std::cout << "Azimuth  ";
                            }
                            std::cout << std::setw(2) << (i < 10 ? i%10 : (i-10)/2) << ":\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << testDescriptor.at<float>(i) << "\t|\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << refDescriptor << std::endl;
                        }
                    }
                    else if(i < 28 + 54)
                    {
                        // Fail on distances
                        ++distanceFailures;
                        if(mVerbose)
                        {
                            std::cout << "Distance ";
                            std::cout << std::setw(2) << i << ":\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << testDescriptor.at<float>(i) << "\t|\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << refDescriptor << std::endl;
                        }
                    }
                    else
                    {
                        // Fail on motion (pos, vel, acc)
                        ++motionFailures;
                        if(mVerbose)
                        {
                            if((i-28-54)%9 < 3)
                            {
                                std::cout << "Position ";
                            }
                            else if((i-28-54)%9 < 6)
                            {
                                std::cout << "Velocity ";
                            }
                            else
                            {
                                std::cout << "Acceler. ";
                            }
                            std::cout << std::setw(2) << (i-28-54)/9 << ":\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << testDescriptor.at<float>(i) << "\t|\t"
                                << std::setw(8) << std::setfill(' ') << std::setprecision(6) << refDescriptor << std::endl;
                        }
                    }
                }
                ++i;
            }
            ++sampleCount;

            previous2 = previous;
            previous = current;
        }

        std::cout << "==============\t     RESULTS\t     ==============" << std::endl;
        std::cout << "Failures on angles:\t " << angleFailures << " / " << sampleCount * 28 << std::endl;
        std::cout << "Failures on distances:\t " << distanceFailures << " / " << sampleCount * 54 << std::endl;
        std::cout << "Failures on motion:\t " << motionFailures << " / " << sampleCount * 90 << std::endl;

        if(angleFailures > mMaxFailures * sampleCount * 28
                || distanceFailures > mMaxFailures * sampleCount * 54
                || motionFailures > mMaxFailures * sampleCount * 90)
        {
            ts->set_failed_test_info(TS::FAIL_BAD_ACCURACY);
        }

        ASSERT_LE(angleFailures, mMaxFailures * sampleCount * 28);
        ASSERT_LE(distanceFailures, mMaxFailures * sampleCount * 54);
        ASSERT_LE(motionFailures, mMaxFailures * sampleCount * 90);
    }

    TEST(Gestures_DNNSkelDescTest, accuracy) { DNNSkelDescTest test(0.01, 1e-5, 0.1, true); test.safe_run(); }
}
