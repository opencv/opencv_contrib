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

#include "opencv2/datasets/util.hpp"

#include <cstdlib>

#include <sstream>

#ifndef _WIN32
    #include <unistd.h>
    #include <dirent.h>
    #include <sys/stat.h>
#else
    #include <io.h>
    #include <direct.h>
#endif

namespace cv
{
namespace datasets
{

using namespace std;

void split(const string &s, vector<string> &elems, char delim)
{
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim))
    {
        elems.push_back(item);
    }
}

void createDirectory(const string &path)
{
#ifndef _WIN32
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#else
    mkdir(path.c_str());
#endif
}

void getDirList(const string &dirName, vector<string> &fileNames)
{
#ifndef _WIN32
    struct dirent **namelist;
    int n = scandir(dirName.c_str(), &namelist, NULL, alphasort);
    for (int i=0; i<n; ++i)
    {
        string fileName(namelist[i]->d_name);
        if ('.' != fileName[0])
        {
            fileNames.push_back(fileName);
        }
        free(namelist[i]);
    }
    free(namelist);
#else // for WIN32
    struct _finddata_t file;
    string filter(dirName);
    filter += "\\*.*";
    intptr_t hFile = _findfirst(filter.c_str(), &file);
    if (hFile==-1)
    {
        return;
    }
    do
    {
        string fileName(file.name);
        if ('.' != fileName[0])
        {
            fileNames.push_back(fileName);
        }
    } while (_findnext(hFile, &file)==0);
    _findclose(hFile);
#endif
}

}
}
