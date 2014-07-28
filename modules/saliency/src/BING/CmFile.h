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

#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#include "opencv2/saliency/kyheader.h"
#endif

struct CmFile
{

    static inline std::string GetFolder(CStr& path);
    static inline std::string GetName(CStr& path);
    static inline std::string GetNameNE(CStr& path);
    static inline std::string GetPathNE(CStr& path);

    // Get file names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
    static int GetNames(CStr &nameW, vecS &names, std::string _dir = std::string());
    static int GetNames(CStr& rootFolder, CStr &fileW, vecS &names);
    static int GetNamesNE(CStr& nameWC, vecS &names, std::string dir = std::string(), std::string ext = std::string());
    static int GetNamesNE(CStr& rootFolder, CStr &fileW, vecS &names);
    static inline std::string GetExtention(CStr name);

    static int GetSubFolders(CStr& folder, vecS& subFolders);

    static inline std::string GetWkDir();

    static bool MkDir(CStr&  path);
    static void loadStrList(CStr &fName, vecS &strs, bool flag=false);
    static bool writeStrList(CStr &fName, const vecS &strs);
};

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/
std::string CmFile::GetFolder(CStr& path)
{
    return path.substr(0, path.find_last_of("\\/")+1);
}

std::string CmFile::GetName(CStr& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_not_of(' ')+1;
    return path.substr(start, end - start);
}

std::string CmFile::GetNameNE(CStr& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(start, end - start);
    else
        return path.substr(start,  path.find_last_not_of(' ')+1 - start);
}

std::string CmFile::GetPathNE(CStr& path)
{
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(0, end);
    else
        return path.substr(0,  path.find_last_not_of(' ') + 1);
}

std::string CmFile::GetExtention(CStr name)
{
    return name.substr(name.find_last_of('.'));
}

/************************************************************************/
/*                   Implementations                                    */
/************************************************************************/

