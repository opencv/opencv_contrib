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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

/* It contains class number and probability of this class */
typedef std::pair<int, double> ClassProb;

/* Find best class for the blob (i. e. class with maximal probability) */
ClassProb getMaxClass(dnn::Blob &probBlob)
{
    Mat probMat = probBlob.matRefConst().reshape(1, 1);
    double classProb;
    Point classNumber;
    minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);

    return std::make_pair(classNumber.x, classProb);
}

std::vector<String> readClassNames(const char *filename = "synset_words.txt")
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found" << std::endl;
        std::cerr << "Check it: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back( name.substr(name.find(' ')+1) );
    }

    fp.close();
    return classNames;
}

/* Create batch from the image */
dnn::Blob makeInputBlob(const String &imagefile)
{
    Mat img = imread(imagefile);
    if (img.empty())
    {
        std::cerr << "Can't read image from file:" << std::endl;
        std::cerr << imagefile << std::endl;
        exit(-1);
    }

    cvtColor(img, img, COLOR_BGR2RGB);
    resize(img, img, Size(227, 227));

    return dnn::Blob(img); //construct 4-dim Blob (i. e. batch)
}

int main(int argc, char **argv)
{
    /* Initialize network */
    dnn::Net net;
    {
        String modelTxt = "bvlc_googlenet.prototxt";
        String modelBin = "bvlc_googlenet.caffemodel";

        Ptr<dnn::Importer> importer; //Try to import Caffe GoogleNet model
        try
        {
            importer = dnn::createCaffeImporter(modelTxt, modelBin);
        }
        catch(const cv::Exception &er) //importer can throw errors, we will catch them
        {
            std::cerr << er.msg << std::endl;
            importer = Ptr<Importer>(); //NULL
        }

        if (!importer)
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            std::cerr << "Please, check them." << std::endl;
            std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }

        importer->populateNet(net);
    }

    std::vector<String> classNames = readClassNames();

    String filename = (argc > 1) ? argv[1] : "space_shuttle.jpg";

    Blob inputBlob = makeInputBlob(filename);   //make batch
    net.setBlob(".data", inputBlob);            //set this blob to the network input
    net.forward();                              //compute output

    dnn::Blob prob = net.getBlob("prob");       //gather output of prob layer
    ClassProb bc = getMaxClass(prob);           //find best class

    String className = classNames.at(bc.first);

    std::cout << "Best class:";
    std::cout << " #" << bc.first;
    std::cout << " (from " << prob.total(1) << ")";
    std::cout << " \"" + className << "\"";
    std::cout <<  std::endl;
    std::cout << "Prob: " << bc.second * 100 << "%" << std::endl;

    return 0;
}
