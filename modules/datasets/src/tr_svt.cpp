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

#include "opencv2/datasets/tr_svt.hpp"
#include "opencv2/datasets/util.hpp"

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "tinyxml2/tinyxml2.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace cv
{
namespace datasets
{

using namespace std;
using namespace tinyxml2;

class TR_svtImp CV_FINAL : public TR_svt
{
public:
    TR_svtImp() {}
    //TR_svtImp(const string &path);
    virtual ~TR_svtImp() CV_OVERRIDE {}

    virtual void load(const string &path) CV_OVERRIDE;

private:
    void loadDataset(const string &path);

    void xmlParse(const string &set, vector< Ptr<Object> > &out);
};

void TR_svtImp::xmlParse(const string &set, vector< Ptr<Object> > &out)
{
    XMLDocument doc;
    doc.LoadFile(set.c_str());
    XMLElement *root_ = doc.RootElement();
    string rootElem(root_->Name());
    if (rootElem == "tagset")
    {
        string strImage("image");
        XMLElement *child = root_->FirstChildElement(strImage.c_str());
        while (child)
        {
            string imageName = child->FirstChildElement("imageName")->GetText();
            string lex = child->FirstChildElement("lex")->GetText();

            Ptr<TR_svtObj> curr(new TR_svtObj);
            curr->fileName = imageName;
            split(lex, curr->lex, ',');

            XMLElement *childTaggeds = child->FirstChildElement("taggedRectangles");
            if (childTaggeds)
            {
                string strTagged("taggedRectangle");
                XMLElement *childTagged = childTaggeds->FirstChildElement(strTagged.c_str());
                while (childTagged)
                {
                    tag t;
                    t.value = childTagged->FirstChildElement("tag")->GetText();
                    t.height = atoi(childTagged->Attribute("height"));
                    t.width = atoi(childTagged->Attribute("width"));
                    t.x = atoi(childTagged->Attribute("x"));
                    t.y = atoi(childTagged->Attribute("y"));
                    curr->tags.push_back(t);

                    childTagged = childTagged->NextSiblingElement(strTagged.c_str());
                }
            }

            out.push_back(curr);

            child = child->NextSiblingElement(strImage.c_str());
        }
    }
}

/*TR_svtImp::TR_svtImp(const string &path)
{
    loadDataset(path);
}*/

void TR_svtImp::load(const string &path)
{
    loadDataset(path);
}

void TR_svtImp::loadDataset(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    string trainXml(path + "train.xml");
    string testXml(path + "test.xml");

    // loading train images description
    xmlParse(trainXml, train.back());

    // loading test images description
    xmlParse(testXml, test.back());
}

Ptr<TR_svt> TR_svt::create()
{
    return Ptr<TR_svtImp>(new TR_svtImp);
}

}
}
