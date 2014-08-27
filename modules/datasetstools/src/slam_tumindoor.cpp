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

#include <opencv2/util.h>
#include <opencv2/slam_tumindoor.h>

#include <cstdio>
#include <cstdlib> // atof
#include <cstring>

#include <fstream>

using namespace std;

slam_tumindoor::slam_tumindoor(std::string &path)
{
    loadDataset(path);
}

void slam_tumindoor::loadDataset(string &path)
{
    string infoPath(path + "info/2011-12-17_15.02.56-info.csv"); // TODO
    ifstream infile(infoPath.c_str());
    string line;
    while (getline(infile, line))
    {
        vector<string> elems;
        split(line, elems, ';');

        imageInfo curr;

        curr.name = elems[0];
        if (curr.name.substr(0, strlen("dslr_left")) == "dslr_left")
        {
            curr.type = LEFT;
        } else
        if (curr.name.substr(0, strlen("dslr_right")) == "dslr_right")
        {
            curr.type = RIGHT;
        } else
        {
            curr.type = LADYBUG;
        }

        for (unsigned int i=0; i<4; ++i)
        {
            for (unsigned int j=0; j<4; ++j)
            {
                curr.transformMat[i][j] = atof(elems[1 + j + i*4].c_str());
            }
        }

        train.push_back(curr);
    }
}
