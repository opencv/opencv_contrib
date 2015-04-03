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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "test_precomp.hpp"

#include <string>

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

using namespace cv;

const float score_thr = 0.05f;

class LatentSVMDetectorCascadeTest : public cvtest::BaseTest
{
protected:
    void run(int);
};

static void writeDetections( FileStorage& fs, const std::string& nodeName, const std::vector<lsvm::LSVMDetector::ObjectDetection>& detections )
{
    fs << nodeName << "[";
    for( size_t i = 0; i < detections.size(); i++ ) //FIXME operator <<
    {
        lsvm::LSVMDetector::ObjectDetection const &d = detections[i];
        fs << d.rect.x << d.rect.y << d.rect.width << d.rect.height
           << d.score << d.classID;
    }
    fs << "]";
}

static void readDetections( FileStorage fs, const std::string& nodeName, 
                           std::vector<lsvm::LSVMDetector::ObjectDetection>& detections )
{
    detections.clear();

    FileNode fn = fs.root()[nodeName];
    FileNodeIterator fni = fn.begin();
    while( fni != fn.end() )
    {
        lsvm::LSVMDetector::ObjectDetection d;
        fni >> d.rect.x >> d.rect.y >> d.rect.width >> d.rect.height
            >> d.score >> d.classID;
        detections.push_back( d );
    }
}

static inline bool isEqualCaskad( const lsvm::LSVMDetector::ObjectDetection& d1,
                           const lsvm::LSVMDetector::ObjectDetection& d2, int eps, float threshold)
{
    return (
           std::abs(d1.rect.x - d2.rect.x) <= eps
           && std::abs(d1.rect.y - d2.rect.y) <= eps
           && std::abs(d1.rect.width - d2.rect.width) <= eps
           && std::abs(d1.rect.height - d2.rect.height) <= eps
           && (d1.classID == d2.classID)
           && std::abs(d1.score - d2.score) <= threshold
           );
}


bool compareResults( const std::vector<lsvm::LSVMDetector::ObjectDetection>& calc,
                    const std::vector<lsvm::LSVMDetector::ObjectDetection>& valid, int eps, float threshold)
{
    if( calc.size() != valid.size() )
        return false;

    for( size_t i = 0; i < calc.size(); i++ )
    {
        lsvm::LSVMDetector::ObjectDetection const &c = calc[i];
        lsvm::LSVMDetector::ObjectDetection const &v = valid[i];

        if( !isEqualCaskad(c, v, eps, threshold) )
        {
            std::cerr << "Expected: " << v.rect << " class=" << v.classID << " score=" << v.score << std::endl;
            std::cerr << "Actual:   " << c.rect << " class=" << c.classID << " score=" << c.score << std::endl;
            return false;
        }
    }
    return true;
}

void LatentSVMDetectorCascadeTest::run( int /* start_from */)
{
    std::string test_data_path = ts->get_data_path() + "latentsvmdetector/";
    std::string img_path_cat = test_data_path  + "cat.png";
    std::string img_path_cars = test_data_path + "cars.png";

    std::string model_path_cat = test_data_path + "models_VOC2007_cascade/cat.xml";
    std::string model_path_car = test_data_path + "models_VOC2007_cascade/car.xml";

    std::string true_res_path = test_data_path + "results_cascade.xml";


#ifdef HAVE_TBB
    int numThreads = 2;
#endif

    Mat image_cat = imread( img_path_cat );
    Mat image_cars = imread( img_path_cars );
    if( image_cat.empty() || image_cars.empty() )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    // We will test 2 cases:
    // detector1 - to test case of one class 'cat'
    // detector12 - to test case of two (several) classes 'cat' and car

    // Load detectors
    cv::Ptr<lsvm::LSVMDetector> detector1 = lsvm::LSVMDetector::create(std::vector<std::string>(1,model_path_cat));

    std::vector<std::string> models_pathes(2);
    models_pathes[0] = model_path_cat;
    models_pathes[1] = model_path_car;
    cv::Ptr<lsvm::LSVMDetector> detector12 = lsvm::LSVMDetector::create(models_pathes);

    if( detector1->isEmpty() || detector12->isEmpty() || detector12->getClassCount() != 2 )
    {
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_TEST_DATA );
        return;
    }

    // 1. Test method detect
    // Run detectors
    std::vector<lsvm::LSVMDetector::ObjectDetection> detections1_cat, detections12_cat, detections12_cars;
    detector1->detect( image_cat, detections1_cat, 0.5);
    detector12->detect( image_cat, detections12_cat, 0.5);
    detector12->detect( image_cars, detections12_cars, 0.5);

    // Load true results
    FileStorage fs( true_res_path, FileStorage::READ );
    if( fs.isOpened() )
    {
        std::vector<lsvm::LSVMDetector::ObjectDetection> true_detections1_cat, true_detections12_cat, true_detections12_cars;
        readDetections( fs, "detections1_cat", true_detections1_cat );
        readDetections( fs, "detections12_cat", true_detections12_cat );
        readDetections( fs, "detections12_cars", true_detections12_cars );

        if( !compareResults(detections1_cat, true_detections1_cat, 1, score_thr) )
        {
            std::cerr << "Results of detector1 are invalid on image cat.png" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
        if( !compareResults(detections12_cat, true_detections12_cat, 1, score_thr) )
        {
            std::cerr << "Results of detector12 are invalid on image cat.png" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
        if( !compareResults(detections12_cars, true_detections12_cars, 1, score_thr) )
        {
            std::cerr << "Results of detector12 are invalid on image cars.png" << std::endl;
            ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
        }
    }
    else
    {
        fs.open( true_res_path, FileStorage::WRITE );
        if( fs.isOpened() )
        {
            writeDetections( fs, "detections1_cat", detections1_cat );
            writeDetections( fs, "detections12_cat", detections12_cat );
            writeDetections( fs, "detections12_cars", detections12_cars );
        }
        else
            std::cerr << "File " << true_res_path << " cann't be opened to save test results" << std::endl;
    }

    ts->set_failed_test_info( cvtest::TS::OK);
}

TEST(Objdetect_LatentSVMDetectorCascade_cpp, regression) { LatentSVMDetectorCascadeTest test; test.safe_run(); }
