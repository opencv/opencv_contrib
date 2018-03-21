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

#include "opencv2/datasets/tr_chars.hpp"
#include "opencv2/datasets/util.hpp"

namespace cv
{
namespace datasets
{

using namespace std;

class TR_charsImp CV_FINAL : public TR_chars
{
public:
    TR_charsImp() {}
    //TR_charsImp(const string &path, int number = 0);
    virtual ~TR_charsImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDatasetSplit(const string &path, int number);

    void loadDataset(const string &path);

    void parseLine(const string &line, vector<int> &currSet, int number);

    inline void convert(vector<int> &from, vector< Ptr<Object> > &to, vector<int> &allLabels, vector<string> &allNames);

    inline void parseSet(const string &line, const string &pattern, bool &flag, vector<int> &set, int number);
};

void TR_charsImp::parseLine(const string &line, vector<int> &currSet, int number)
{
    vector<string> elems;
    split(line, elems, ' ');
    if (number >= (int)elems.size())
    {
        return;
    }

    unsigned int ind = atoi(elems[number].c_str());
    if (ind > 0)
    {
        currSet.push_back(ind-1);
    }
}

inline void TR_charsImp::convert(vector<int> &from, vector< Ptr<Object> > &to, vector<int> &allLabels, vector<string> &allNames)
{
    for (vector<int>::iterator it=from.begin(); it!=from.end(); ++it)
    {
        if (*it>=(int)allNames.size() || *it>=(int)allLabels.size())
        {
            printf("incorrect index: %u\n", *it);
            continue;
        }

        Ptr<TR_charsObj> curr(new TR_charsObj);
        curr->imgName = allNames[*it];
        curr->label = allLabels[*it];
        to.push_back(curr);
    }
}

inline void TR_charsImp::parseSet(const string &line, const string &pattern, bool &flag, vector<int> &set, int number)
{
    size_t pos = line.find(pattern);
    if (string::npos != pos)
    {
        flag = true;
        string s(line.substr(pos + pattern.length()));
        parseLine(s, set, number);
    } else
    if (flag)
    {
        parseLine(line, set, number);
    }
}

/*TR_charsImp::TR_charsImp(const string &path, int number)
{
    loadDataset(path, number);
}*/

void TR_charsImp::load(const string &path)
{
    loadDataset(path);
}

void TR_charsImp::loadDataset(const string &path)
{
    int number = 0;
    do
    {
        loadDatasetSplit(path, number);
        number++;
    } while (train.back().size()>0);

    train.pop_back(); // remove last empty split
    test.pop_back(); // remove last empty split
    validation.pop_back(); // remove last empty split
}

void TR_charsImp::loadDatasetSplit(const string &path, int number)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    vector<int> allLabels, trainSet, testSet, validationSet;
    vector<string> allNames;

    ifstream infile((path + "list_English_Img.m").c_str());
    string line;
    bool labels = false, names = false, isTrain = false, isTest = false, isValidation = false;
    while (getline(infile, line))
    {
        size_t pos = line.find("];");
        if (string::npos != pos)
        {
            labels = false;
            names = false;
            isTrain = false;
            isTest = false;
            isValidation = false;
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
        if (string::npos != pos)
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

        string trainStr("list.TRNind = [");
        parseSet(line, trainStr, isTrain, trainSet, number);

        string testStr("list.TSTind = [");
        parseSet(line, testStr, isTest, testSet, number);

        string validationStr("list.VALind = [");
        parseSet(line, validationStr, isValidation, validationSet, number);

        /*"list.classlabels = ["
        "list.classnames = ["
        "list.NUMclasses = 62;"
        "list.TXNind = ["*/
    }

    convert(trainSet, train.back(), allLabels, allNames);
    convert(testSet, test.back(), allLabels, allNames);
    convert(validationSet, validation.back(), allLabels, allNames);
}

Ptr<TR_chars> TR_chars::create()
{
    return Ptr<TR_charsImp>(new TR_charsImp);
}

}
}
