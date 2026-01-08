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

#include "opencv2/datasets/pd_caltech.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class PD_caltechImp CV_FINAL : public PD_caltech
{
public:
    PD_caltechImp() {}
    //PD_caltechImp(const string &path);
    virtual ~PD_caltechImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);
};

/*PD_caltechImp::PD_caltechImp(const string &path)
{
    loadDataset(path);
}*/

void PD_caltechImp::load(const string &path)
{
    loadDataset(path);
}

void PD_caltechImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    createDirectory((path + "../images/"));

    vector<string> objectNames;
    getDirList(path, objectNames);
    for (vector<string>::iterator it=objectNames.begin(); it!=objectNames.end(); ++it)
    {
        Ptr<PD_caltechObj> curr(new PD_caltechObj);
        curr->name = *it;

        string objectPath(path + "../images/" + curr->name + "/");
        createDirectory(objectPath);

        string seqImagesPath(path + curr->name + "/");
        vector<string> seqNames;
        getDirList(seqImagesPath, seqNames);
        for (vector<string>::iterator itSeq=seqNames.begin(); itSeq!=seqNames.end(); ++itSeq)
        {
            string &seqName = *itSeq;

            createDirectory((objectPath + seqName));

            FILE *f = fopen((seqImagesPath + seqName).c_str(), "rb");

            #define SKIP 28+8+512
            fseek(f, SKIP, SEEK_CUR);

            unsigned int header[9];
            size_t res = fread(header, 9, 4, f);
            double fps;
            res = fread(&fps, 1, 8, f);
            fseek(f, 432, SEEK_CUR);

            /*printf("width %u\n", header[0]);
            printf("height %u\n", header[1]);
            printf("imageBitDepth %u\n", header[2]);
            printf("imageBitDepthReal %u\n", header[3]);
            printf("imageSizeBytes %u\n", header[4]);
            printf("imageFormat %u\n", header[5]);
            printf("numFrames %u\n", numFrames);
            printf("fps %f\n", fps);
            printf("trueImageSize %u\n", header[8]);*/
            unsigned int numFrames = header[6];

            string ext;
            switch (header[5])
            {
            case 100:
            case 200:
                ext = "raw";
                break;
            case 101:
                ext = "brgb8";
                break;
            case 102:
            case 201:
                ext = "jpg";
                break;
            case 103:
                ext = "jbrgb";
                break;
            case 001:
            case 002:
                ext = "png";
                break;
            }

            for (unsigned int i=0; i<numFrames; ++i)
            {
                unsigned int size;
                res = fread(&size, 1, 4, f);

                char imgName[20];
                sprintf(imgName, "/%u.%s", i, ext.c_str());
                curr->imageNames.push_back(imgName);

                // comment fseek and uncomment next block to unpack all frames
                fseek(f, size, SEEK_CUR);
                /*char *img = new char[size];
                fread(img, size, 1, f);
                string imgPath(objectPath + seqName + imgName);
                FILE *fImg = fopen(imgPath.c_str(), "wb");
                fwrite(img, size, 1, fImg);
                fclose(fImg);
                delete[] img;*/

                fseek(f, 12, SEEK_CUR);
            }

            if (0 != res) // should fix unused variable warning
            {
                res = 0;
            }

            fclose(f);
        }

        train.back().push_back(curr);
    }
}

Ptr<PD_caltech> PD_caltech::create()
{
    return Ptr<PD_caltechImp>(new PD_caltechImp);
}

}
}
