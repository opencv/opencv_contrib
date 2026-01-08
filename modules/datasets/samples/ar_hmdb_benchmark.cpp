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
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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
// In no event shall the Itseez Inc or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/datasets/ar_hmdb.hpp"
#include "opencv2/datasets/util.hpp"

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/ml.hpp>

#include <cstdio>

#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::datasets;
using namespace cv::flann;
using namespace cv::ml;

void fillData(const string &path, vector< Ptr<Object> > &curr, Index &flann_index, Mat1f &data, Mat1i &labels);
void fillData(const string &path, vector< Ptr<Object> > &curr, Index &flann_index, Mat1f &data, Mat1i &labels)
{
    const unsigned int descriptorNum = 162;
    Mat1f sample(1, descriptorNum);
    Mat1i nresps(1, 1);
    Mat1f dists(1, 1);

    unsigned int numFiles = 0;
    for (unsigned int i=0; i<curr.size(); ++i)
    {
        AR_hmdbObj *example = static_cast<AR_hmdbObj *>(curr[i].get());
        string featuresFullPath = path + "hmdb51_org_stips/" + example->name + "/" + example->videoName + ".txt";

        ifstream infile(featuresFullPath.c_str());
        string line;
        // skip header
        for (unsigned int j=0; j<3; ++j)
        {
            getline(infile, line);
        }
        while (getline(infile, line))
        {
            // 7 skip, hog+hof: 72+90 read
            vector<string> elems;
            split(line, elems, '\t');

            for (unsigned int j=0; j<descriptorNum; ++j)
            {
                sample(0, j) = (float)atof(elems[j+7].c_str());
            }

            flann_index.knnSearch(sample, nresps, dists, 1, SearchParams());
            data(numFiles, nresps(0, 0)) ++;
        }
        labels(numFiles, 0) = example->id;
        numFiles++;
    }
}

int main(int argc, char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset }";
    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }

    // loading dataset
    Ptr<AR_hmdb> dataset = AR_hmdb::create();
    dataset->load(path);

    int numSplits = dataset->getNumSplits();
    printf("splits number: %u\n", numSplits);


    const unsigned int descriptorNum = 162;
    const unsigned int clusterNum = 4000;
    const unsigned int sampleNum = 5613856; // max for all 3 splits

    vector<double> res;
    for (int currSplit=0; currSplit<numSplits; ++currSplit)
    {
        Mat1f samples(sampleNum, descriptorNum);
        unsigned int currSample = 0;
        vector< Ptr<Object> > &curr = dataset->getTrain(currSplit);
        unsigned int numFeatures = 0;
        for (unsigned int i=0; i<curr.size(); ++i)
        {
            AR_hmdbObj *example = static_cast<AR_hmdbObj *>(curr[i].get());
            string featuresFullPath = path + "hmdb51_org_stips/" + example->name + "/" + example->videoName + ".txt";
            ifstream infile(featuresFullPath.c_str());
            string line;
            // skip header
            for (unsigned int j=0; j<3; ++j)
            {
                getline(infile, line);
            }
            while (getline(infile, line))
            {
                numFeatures++;
                if (currSample < sampleNum)
                {
                    // 7 skip, hog+hof: 72+90 read
                    vector<string> elems;
                    split(line, elems, '\t');

                    for (unsigned int j=0; j<descriptorNum; ++j)
                    {
                        samples(currSample, j) = (float)atof(elems[j+7].c_str());
                    }
                    currSample++;
                }
            }
        }
        printf("split %u, train features number: %u, samples number: %u\n", currSplit, numFeatures, currSample);

        // clustering
        Mat1f centers(clusterNum, descriptorNum);
        ::cvflann::KMeansIndexParams kmean_params;
        unsigned int resultClusters = hierarchicalClustering< L2<float> >(samples, centers, kmean_params);
        if (resultClusters < clusterNum)
        {
            centers = centers.rowRange(Range(0, resultClusters));
        }
        Index flann_index(centers, KDTreeIndexParams());
        printf("resulted clusters number: %u\n", resultClusters);


        unsigned int numTrainFiles = curr.size();
        Mat1f trainData(numTrainFiles, resultClusters);
        Mat1i trainLabels(numTrainFiles, 1);

        for (unsigned int i=0; i<numTrainFiles; ++i)
        {
            for (unsigned int j=0; j<resultClusters; ++j)
            {
                trainData(i, j) = 0;
            }
        }

        printf("calculating train histograms\n");
        fillData(path, curr, flann_index, trainData, trainLabels);

        printf("train svm\n");
        Ptr<SVM> svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::POLY); //SVM::RBF;
        svm->setDegree(0.5);
        svm->setGamma(1);
        svm->setCoef0(1);
        svm->setC(1);
        svm->setNu(0.5);
        svm->setP(0);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, 0.01));
        svm->train(trainData, ROW_SAMPLE, trainLabels);

        // prepare to predict
        curr = dataset->getTest(currSplit);
        unsigned int numTestFiles = curr.size();
        Mat1f testData(numTestFiles, resultClusters);
        Mat1i testLabels(numTestFiles, 1); // ground true

        for (unsigned int i=0; i<numTestFiles; ++i)
        {
            for (unsigned int j=0; j<resultClusters; ++j)
            {
                testData(i, j) = 0;
            }
        }

        printf("calculating test histograms\n");
        fillData(path, curr, flann_index, testData, testLabels);

        printf("predicting\n");
        Mat1f testPredicted(numTestFiles, 1);
        svm->predict(testData, testPredicted);

        unsigned int correct = 0;
        for (unsigned int i=0; i<numTestFiles; ++i)
        {
            if ((int)testPredicted(i, 0) == testLabels(i, 0))
            {
                correct++;
            }
        }
        double accuracy = 1.0*correct/numTestFiles;
        printf("correctly recognized actions: %f\n", accuracy);
        res.push_back(accuracy);
    }

    double accuracy = 0.0;
    for (unsigned int i=0; i<res.size(); ++i)
    {
        accuracy += res[i];
    }
    printf("average: %f\n", accuracy/res.size());

    return 0;
}
