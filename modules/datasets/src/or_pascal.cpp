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
// Copyright (C) 2015, Itseez Inc, all rights reserved.
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

#include "opencv2/datasets/or_pascal.hpp"
#include "opencv2/datasets/util.hpp"
#include "tinyxml2/tinyxml2.h"
#include <fstream>

namespace cv
{
namespace datasets
{

using namespace std;
using namespace tinyxml2;

class OR_pascalImp : public OR_pascal
{
public:
    OR_pascalImp() {}

    virtual void load(const string &path);

private:
    void loadDataset(const string &path, const string &nameImageSet, vector< Ptr<Object> > &imageSet);
    Ptr<Object> parseAnnotation(const string &path, const string &id);
    const char*  parseNodeText(XMLElement* node, const string &nodeName, const string &defaultValue);
};


void OR_pascalImp::load(const string &path)
{
    train.push_back(vector< Ptr<Object> >());
    test.push_back(vector< Ptr<Object> >());
    validation.push_back(vector< Ptr<Object> >());

    loadDataset(path, "train", train.back());
    loadDataset(path, "test", test.back());
    loadDataset(path, "val", validation.back());
}

void OR_pascalImp::loadDataset(const string &path, const string &nameImageSet, vector< Ptr<Object> > &imageSet)
{
    string pathImageSets(path + "ImageSets/Main/");
    string imageList = pathImageSets + nameImageSet + ".txt";

    ifstream in(imageList.c_str());
    string error_message = format("Image list not exists!\n%s", imageList.c_str());

    if (!in.is_open())
        CV_Error(Error::StsBadArg, error_message);

    string id = "";

    while( getline(in, id) )
    {
        if( strcmp(nameImageSet.c_str(), "test") == 0 ) // test set ground truth is not available
        {
            Ptr<OR_pascalObj> annotation(new OR_pascalObj);
            annotation->filename = path + "JPEGImages/" + id + ".jpg";
            imageSet.push_back(annotation);
        }
        else
        {
            imageSet.push_back(parseAnnotation(path, id));
        }
    }
}

const char* OR_pascalImp::parseNodeText(XMLElement* node, const string &nodeName, const string &defaultValue)
{
    XMLElement* child = node->FirstChildElement(nodeName.c_str());
    if ( child == 0 )
        return defaultValue.c_str();

    const char* e = child->GetText();
    if( e == 0 )
        return defaultValue.c_str();

    return e ;
}

Ptr<Object> OR_pascalImp::parseAnnotation(const string &path, const string &id)
{
    string pathAnnotations(path + "Annotations/");
    string pathImages(path + "JPEGImages/");
    Ptr<OR_pascalObj> annotation(new OR_pascalObj);

    XMLDocument doc;
    string xml_file = pathAnnotations + id + ".xml";

    XMLError error_code = doc.LoadFile(xml_file.c_str());
    string error_message = format("Parsing XML failed. Error code = %d. \nFile = %s", error_code, xml_file.c_str());
    switch (error_code)
    {
        case XML_SUCCESS:
            break;
        case XML_ERROR_FILE_NOT_FOUND:
            error_message = "XML file not found! " + error_message;
            CV_Error(Error::StsParseError, error_message);
            return annotation;
        default:
            CV_Error(Error::StsParseError, error_message);
            break;
    }

    // <annotation/>
    XMLElement *xml_ann = doc.RootElement();

    // <filename/>
    string img_name = xml_ann->FirstChildElement("filename")->GetText();
    annotation->filename = pathImages + img_name;

    // <size/>
    XMLElement *sz = xml_ann->FirstChildElement("size");
    int width = atoi(sz->FirstChildElement("width")->GetText());
    int height = atoi(sz->FirstChildElement("height")->GetText());
    int depth = atoi(sz->FirstChildElement("depth")->GetText());
    annotation->width = width;
    annotation->height = height;
    annotation->depth = depth;

    // <object/>
    vector<PascalObj> objects;
    XMLElement *xml_obj = xml_ann->FirstChildElement("object");

    while (xml_obj)
    {
        PascalObj pascal_obj;
        pascal_obj.name = xml_obj->FirstChildElement("name")->GetText();
        pascal_obj.pose = parseNodeText(xml_obj, "pose", "Unspecified");
        pascal_obj.truncated = atoi(parseNodeText(xml_obj, "truncated", "0")) > 0;
        pascal_obj.difficult = atoi(parseNodeText(xml_obj, "difficult", "0")) > 0;
        pascal_obj.occluded = atoi(parseNodeText(xml_obj, "occluded", "0")) > 0;

        // <bndbox/>
        XMLElement *xml_bndbox = xml_obj->FirstChildElement("bndbox");
        pascal_obj.xmin = atoi(xml_bndbox->FirstChildElement("xmin")->GetText());
        pascal_obj.ymin = atoi(xml_bndbox->FirstChildElement("ymin")->GetText());
        pascal_obj.xmax = atoi(xml_bndbox->FirstChildElement("xmax")->GetText());
        pascal_obj.ymax = atoi(xml_bndbox->FirstChildElement("ymax")->GetText());

        // <part/>
        vector<PascalPart> parts;
        XMLElement *xml_part = xml_obj->FirstChildElement("part");

        while (xml_part)
        {
            PascalPart pascal_part;
            pascal_part.name = xml_part->FirstChildElement("name")->GetText();

            xml_bndbox = xml_part->FirstChildElement("bndbox");
            pascal_part.xmin = atoi(xml_bndbox->FirstChildElement("xmin")->GetText());
            pascal_part.ymin = atoi(xml_bndbox->FirstChildElement("ymin")->GetText());
            pascal_part.xmax = atoi(xml_bndbox->FirstChildElement("xmax")->GetText());
            pascal_part.ymax = atoi(xml_bndbox->FirstChildElement("ymax")->GetText());
            parts.push_back(pascal_part);

            xml_part = xml_part->NextSiblingElement("part");
        }

        pascal_obj.parts = parts;
        objects.push_back(pascal_obj);

        xml_obj = xml_obj->NextSiblingElement("object");
    }

    annotation->objects = objects;

    return annotation;
}

Ptr<OR_pascal> OR_pascal::create()
{
    return Ptr<OR_pascalImp>(new OR_pascalImp);
}

}
}
