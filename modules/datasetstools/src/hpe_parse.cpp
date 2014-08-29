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

#include "opencv2/util.hpp"
#include "opencv2/hpe_parse.hpp"

#include <cstdio>

#include <fstream>

namespace cv
{
namespace datasetstools
{

using namespace std;

hpe_parse::hpe_parse(std::string &path)
{
    loadDataset(path);
}

void hpe_parse::load(string &path, unsigned int number)
{
    if (number!=0)
    {
        return;
    }

    loadDataset(path);
}

void hpe_parse::loadDataset(string &path)
{
    unsigned int i=0;
    vector<string> fileNames;
    getDirList(path, fileNames);
    for (vector<string>::iterator it=fileNames.begin(); it!=fileNames.end(); ++it)
    {
        string &file = *it, ext;
        size_t len = file.length();
        if (len>4)
        {
            ext = file.substr(len-4, 4);
        }
        if (ext==".jpg")
        {
            if (i<100)
            {
                train.push_back(file);
            } else
            {
                test.push_back(file);
            }
            ++i;
        }
    }
}

}
}
