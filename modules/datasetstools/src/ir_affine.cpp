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

#include <opencv2/ir_affine.h>

#include <cstdio>
#include <cstdlib> // atoi

#include <fstream>

using namespace std;

ir_affine::ir_affine(std::string &path)
{
    loadDataset(path);
}

void ir_affine::load(string &path, unsigned int number)
{
    loadDataset(path);
}

void ir_affine::loadDataset(string &path)
{
    for (unsigned int i=1; i<=6; ++i)
    {
        imageParams curr;

        char tmp[2];
        sprintf(tmp, "%u", i);
        curr.imageName = path + "img" + tmp + ".ppm";

        if (i>1)
        {
            string matName(path + "H1to" + tmp + "p");
            ifstream infile(matName.c_str());
            for (int i=0; i<3; ++i)
            {
                for (int j=0; j<3; ++j)
                {
                    infile >> curr.mat[i][j];
                }
            }
        }

        train.push_back(curr);
    }
}
