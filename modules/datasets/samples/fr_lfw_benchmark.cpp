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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/datasets/fr_lfw.hpp"

#include <iostream>
#include <cstdio>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;


int main(int argc, const char *argv[])
{
    const char *keys =
            "{ help h usage ? |    | show this message }"
            "{ path p         |true| path to dataset (lfw2 folder) }"
            "{ train t        |dev | train method: 'dev'(pairsDevTrain.txt) or 'split'(pairs.txt) }";

    CommandLineParser parser(argc, argv, keys);
    string path(parser.get<string>("path"));
    if (parser.has("help") || path=="true")
    {
        parser.printMessage();
        return -1;
    }
    string trainMethod(parser.get<string>("train"));

    // our trained threshold for "same":
    double threshold = 0;

    // load dataset
    Ptr<FR_lfw> dataset = FR_lfw::create();
    dataset->load(path);

    unsigned int numSplits = dataset->getNumSplits();
    printf("splits number: %u\n", numSplits);
    if (trainMethod == "dev")
        printf("train size: %u\n", (unsigned int)dataset->getTrain().size());
    else
        printf("train size: %u\n", (numSplits-1) * (unsigned int)dataset->getTest().size());
    printf("test size: %u\n", (unsigned int)dataset->getTest().size());


    if (trainMethod == "dev") // train on personsDevTrain.txt
    {
        // collect average same-distances:
        double avg = 0;
        int count = 0;
        for (unsigned int i=0; i<dataset->getTrain().size(); ++i)
        {
            FR_lfwObj *example = static_cast<FR_lfwObj *>(dataset->getTrain()[i].get());

            Mat a = imread(path+example->image1, IMREAD_GRAYSCALE);
            Mat b = imread(path+example->image2, IMREAD_GRAYSCALE);
            double dist = norm(a,b);
            if (example->same)
            {
                avg += dist;
                count ++;
            }
        }
        threshold = avg / count;
    }

    vector<double> p;
    for (unsigned int j=0; j<numSplits; ++j)
    {
        if (trainMethod == "split") // train on the remaining 9 splits from pairs.txt
        {
            double avg = 0;
            int count = 0;
            for (unsigned int j2=0; j2<numSplits; ++j2)
            {
                if (j==j2) continue; // skip test split for training

                vector < Ptr<Object> > &curr = dataset->getTest(j2);
                for (unsigned int i=0; i<curr.size(); ++i)
                {
                    FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[i].get());
                    Mat a = imread(path+example->image1, IMREAD_GRAYSCALE);
                    Mat b = imread(path+example->image2, IMREAD_GRAYSCALE);
                    double dist = norm(a,b);
                    if (example->same)
                    {
                        avg += dist;
                        count ++;
                    }
                }
            }
            threshold = avg / count;
        }

        unsigned int incorrect = 0, correct = 0;
        vector < Ptr<Object> > &curr = dataset->getTest(j);
        for (unsigned int i=0; i<curr.size(); ++i)
        {
            FR_lfwObj *example = static_cast<FR_lfwObj *>(curr[i].get());

            Mat a = imread(path+example->image1, IMREAD_GRAYSCALE);
            Mat b = imread(path+example->image2, IMREAD_GRAYSCALE);
            bool same = (norm(a,b) <= threshold);
            if (same == example->same)
                correct++;
            else
                incorrect++;
        }
        p.push_back(1.0*correct/(correct+incorrect));
        printf("correct: %u, from: %u -> %f\n", correct, correct+incorrect, p.back());
    }

    double mu = 0.0;
    for (vector<double>::iterator it=p.begin(); it!=p.end(); ++it)
    {
        mu += *it;
    }
    mu /= p.size();
    double sigma = 0.0;
    for (vector<double>::iterator it=p.begin(); it!=p.end(); ++it)
    {
        sigma += (*it - mu)*(*it - mu);
    }
    sigma = sqrt(sigma/p.size());
    double se = sigma/sqrt(double(p.size()));
    printf("estimated mean accuracy: %f and the standard error of the mean: %f\n", mu, se);

    return 0;
}
