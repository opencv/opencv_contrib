#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>
#include <dirent.h>
#endif

struct CmFile
{

    static inline string GetFolder(CStr& path);
    static inline string GetName(CStr& path);
    static inline string GetNameNE(CStr& path);
    static inline string GetPathNE(CStr& path);

    // Get file names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
    static int GetNames(CStr &nameW, vecS &names, std::string _dir = std::string());
    static int GetNames(CStr& rootFolder, CStr &fileW, vecS &names);
    static int GetNamesNE(CStr& nameWC, vecS &names, string dir = string(), string ext = string());
    static int GetNamesNE(CStr& rootFolder, CStr &fileW, vecS &names);
    static inline string GetExtention(CStr name);

    static int GetSubFolders(CStr& folder, vecS& subFolders);

    static inline string GetWkDir();

    static bool MkDir(CStr&  path);
    static void loadStrList(CStr &fName, vecS &strs, bool flag=false);
    static bool writeStrList(CStr &fName, const vecS &strs);
};

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/
string CmFile::GetFolder(CStr& path)
{
    return path.substr(0, path.find_last_of("\\/")+1);
}

string CmFile::GetName(CStr& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_not_of(' ')+1;
    return path.substr(start, end - start);
}

string CmFile::GetNameNE(CStr& path)
{
    int start = path.find_last_of("\\/")+1;
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(start, end - start);
    else
        return path.substr(start,  path.find_last_not_of(' ')+1 - start);
}

string CmFile::GetPathNE(CStr& path)
{
    int end = path.find_last_of('.');
    if (end >= 0)
        return path.substr(0, end);
    else
        return path.substr(0,  path.find_last_not_of(' ') + 1);
}

string CmFile::GetExtention(CStr name)
{
    return name.substr(name.find_last_of('.'));
}

/************************************************************************/
/*                   Implementations                                    */
/************************************************************************/

