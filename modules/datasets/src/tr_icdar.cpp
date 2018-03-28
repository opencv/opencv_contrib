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

#include "opencv2/datasets/tr_icdar.hpp"
#include "opencv2/datasets/util.hpp"

#include <sstream>
#include <fstream>


namespace cv
{
namespace datasets
{

using namespace std;

class TR_icdarImp CV_FINAL : public TR_icdar
{
public:
    TR_icdarImp() {}
    //TR_icdarImp(const string &path);
    virtual ~TR_icdarImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void objParseFiles(const string &path, int img_id, vector<Ptr <Object> > &out);
};

void TR_icdarImp::objParseFiles(const string &path, int img_id, vector<Ptr <Object> > &out)
{
    Ptr<TR_icdarObj> curr(new TR_icdarObj);

    stringstream fileName;
    fileName << "img_" << img_id << ".jpg";
    curr->fileName = fileName.str();

    stringstream gtFileName;
    gtFileName << path << "/gt_img_" << img_id << ".txt";
    ifstream infile(gtFileName.str().c_str());
    if (!infile.is_open()) CV_Error(Error::StsBadArg, gtFileName.str().c_str());
    string line;
    while (getline(infile, line))
    {
        //Ignore EOL characters
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        //Ignore byte-order marks (BOM first utf character in W$ files)
        if ( (line[0] == (char)0xEFu) && (line[1] == (char)0xBBu) && (line[2] == (char)0xBFu) )
            line.erase (line.begin(),line.begin()+3);
        vector<string> fields;
        split(line, fields, ',');
        word w;
        w.value  = fields[8];
        w.x      = atoi(fields[0].c_str());
        w.y      = atoi(fields[1].c_str());
        w.width  = atoi(fields[2].c_str()) - atoi(fields[0].c_str());
        w.height = atoi(fields[7].c_str()) - atoi(fields[1].c_str());
        curr->words.push_back(w);
    }
    infile.close();

    stringstream lex100FileName;
    lex100FileName << path << "/voc_img_" << img_id << ".txt";
    infile.open(lex100FileName.str().c_str());
    if (!infile.is_open()) CV_Error(Error::StsBadArg, lex100FileName.str().c_str());
    while (getline(infile, line))
    {
        //Ignore EOL characters
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        //Ignore byte-order marks (BOM first utf character in W$ files)
        if ( (line[0] == (char)0xEFu) && (line[1] == (char)0xBBu) && (line[2] == (char)0xBFu) )
            line.erase (line.begin(),line.begin()+3);
        curr->lex100.push_back(line);
    }
    infile.close();

    stringstream lexFullFileName;
    if (path.substr(path.size()-5,4) == string("test"))
        lexFullFileName << path << "ch2_test_vocabulary.txt";
    else
        lexFullFileName << path << "ch2_training_vocabulary.txt";
    infile.open(lexFullFileName.str().c_str());
    if (!infile.is_open()) CV_Error(Error::StsBadArg, lexFullFileName.str().c_str());
    while (getline(infile, line))
    {
        //Ignore EOL characters
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        //Ignore byte-order marks (BOM first utf character in W$ files)
        if ( (line[0] == (char)0xEFu) && (line[1] == (char)0xBBu) && (line[2] == (char)0xBFu) )
            line.erase (line.begin(),line.begin()+3);
        curr->lexFull.push_back(line);
    }
    infile.close();

    out.push_back(curr);
}

/*TR_icdarImp::TR_icdarImp(const string &path)
{
    loadDataset(path);
}*/

void TR_icdarImp::load(const string &path)
{
    loadDataset(path);
}

void TR_icdarImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string train_path(path + "/train/");
    string test_path (path + "/test/");

    // loading 229 train images descriptions
    for (int i=1; i<230; i++)
        objParseFiles(train_path, i, train.back());

    // loading 233 test images descriptions
    for (int i=1; i<234; i++)
        objParseFiles(test_path, i, test.back());
}

Ptr<TR_icdar> TR_icdar::create()
{
    return Ptr<TR_icdarImp>(new TR_icdarImp);
}

}
}
