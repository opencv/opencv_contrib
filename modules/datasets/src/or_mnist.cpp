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

#include "opencv2/datasets/or_mnist.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class OR_mnistImp CV_FINAL : public OR_mnist
{
public:
    OR_mnistImp() {}
    //OR_mnistImp(const string &path);
    virtual ~OR_mnistImp() {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void loadDatasetPart(const string &imagesFile, const string &labelsFile, unsigned int num, vector< Ptr<Object> > &dataset_);
};

/*OR_mnistImp::OR_mnistImp(const string &path)
{
    loadDataset(path);
}*/

void OR_mnistImp::load(const string &path)
{
    loadDataset(path);
}

void OR_mnistImp::loadDatasetPart(const string &imagesFile, const string &labelsFile, unsigned int num, vector< Ptr<Object> > &dataset_)
{
    FILE *f = fopen(imagesFile.c_str(), "rb");
    fseek(f, 16, SEEK_CUR);
    unsigned int imageSize = 28*28;
    char *images = new char[num*imageSize];
    size_t res = fread(images, 1, num*imageSize, f);
    fclose(f);
    if (num*imageSize != res)
    {
        delete[] images;
        return;
    }
    f = fopen(labelsFile.c_str(), "rb");
    fseek(f, 8, SEEK_CUR);
    char *labels = new char[num];
    res = fread(labels, 1, num, f);
    fclose(f);
    if (num != res)
    {
        delete[] images;
        delete[] labels;
        return;
    }

    for (unsigned int i=0; i<num; ++i)
    {
        Ptr<OR_mnistObj> curr(new OR_mnistObj);
        curr->label = labels[i];

        curr->image = Mat(28, 28, CV_8U);
        unsigned int imageIdx = i*imageSize;
        for (int j=0; j<curr->image.rows; ++j)
        {
            char *im = curr->image.ptr<char>(j);
            for (int k=0; k<curr->image.cols; ++k)
            {
                im[k] = images[imageIdx + j*28 + k];
            }
        }

        dataset_.push_back(curr);
    }
    delete[] images;
    delete[] labels;
}

void OR_mnistImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string trainImagesFile(path + "train-images-idx3-ubyte");
    string trainLabelsFile(path + "train-labels-idx1-ubyte");
    loadDatasetPart(trainImagesFile, trainLabelsFile, 60000, train.back());

    string testImagesFile(path + "t10k-images-idx3-ubyte");
    string testLabelsFile(path + "t10k-labels-idx1-ubyte");
    loadDatasetPart(testImagesFile, testLabelsFile, 10000, test.back());
}

Ptr<OR_mnist> OR_mnist::create()
{
    return Ptr<OR_mnistImp>(new OR_mnistImp);
}

}
}
