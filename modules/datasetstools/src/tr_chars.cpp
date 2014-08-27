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
#include <opencv2/tr_chars.h>

#include <cstdio>
#include <cstdlib> // atoi

#include <fstream>

using namespace std;

void parseLine(string &line, vector<int> &currSet, unsigned int number)
{
    vector<string> elems;
    split(line, elems, ' ');
    if (number>=elems.size())
    {
        return;
    }

    unsigned int ind = atoi(elems[number].c_str());
    if (ind>0)
    {
        currSet.push_back(ind-1); // take first split
    }
}

tr_chars::tr_chars(std::string &path, unsigned int number)
{
    loadDataset(path, number);
}

void tr_chars::loadDataset(string &path, unsigned int number)
{
    vector<int> allLabels, trainSet, testSet;
    vector<string> allNames;

    ifstream infile((path + "list_English_Img.m").c_str());
    string line;
    bool labels = false, names = false, isTrain = false, isTest = false;
    while (getline(infile, line))
    {
        size_t pos = line.find("];");
        if (string::npos!=pos)
        {
            labels = false;
            names = false;
            isTrain = false;
            isTest = false;
        }

        string slabels("list.ALLlabels = [");
        pos = line.find(slabels);
        if (string::npos != pos)
        {
            labels = true;
            string s(line.substr(pos+slabels.length()));
            allLabels.push_back(atoi(s.c_str()));
        } else
        if (labels)
        {
            allLabels.push_back(atoi(line.c_str()));
        }

        string snames("list.ALLnames = [");
        pos = line.find(snames);
        if (string::npos!=pos)
        {
            names = true;
            size_t start = pos+snames.length();
            string s(line.substr(start+1, line.length()-start-2));
            allNames.push_back(s);
        } else
        if (names)
        {
            string s(line.substr(1, line.length()-2));
            allNames.push_back(s);
        }

        string strain("list.TRNind = [");
        pos = line.find(strain);
        if (string::npos!=pos)
        {
            isTrain = true;
            string s(line.substr(pos+strain.length()));
            parseLine(s, trainSet, number);
        } else
        if (isTrain)
        {
            parseLine(line, trainSet, number);
        }

        string stest("list.TSTind = [");
        pos = line.find(stest);
        if (string::npos!=pos)
        {
            isTest = true;
            string s(line.substr(pos+stest.length()));
            parseLine(s, testSet, number);
        } else
        if (isTest)
        {
            parseLine(line, testSet, number);
        }

        /*"list.classlabels = ["
        "list.classnames = ["
        "list.NUMclasses = 62;"
        "list.VALind = [" // TODO: load validation
        "list.TXNind = ["*/
    }

    for (vector<int>::iterator it=trainSet.begin(); it!=trainSet.end(); ++it)
    {
        if (*it>=allNames.size() || *it>=allLabels.size())
        {
            printf("incorrect train index: %u\n", *it);
            continue;
        }

        character curr;
        curr.imgName = allNames[*it];
        curr.label = allLabels[*it];
        train.push_back(curr);
    }

    for (vector<int>::iterator it=testSet.begin(); it!=testSet.end(); ++it)
    {
        if (*it>=allNames.size() || *it>=allLabels.size())
        {
            printf("incorrect test index: %u\n", *it);
            continue;
        }

        character curr;
        curr.imgName = allNames[*it];
        curr.label = allLabels[*it];
        test.push_back(curr);
    }
}
