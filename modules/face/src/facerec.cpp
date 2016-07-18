/*
 * Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include "precomp.hpp"
#include "opencv2/face.hpp"

namespace cv
{
namespace face
{

std::vector<int> FaceRecognizer::getLabelsByString(const String &str) const
{
  std::vector<int> labels;
  for (std::map<int, String>::const_iterator it = _labelsInfo.begin(); it != _labelsInfo.end(); it++)
  {
      size_t found = (it->second).find(str);
      if (found != String::npos)
          labels.push_back(it->first);
  }
  return labels;
}

String FaceRecognizer::getLabelInfo(int label) const
{
    std::map<int, String>::const_iterator iter(_labelsInfo.find(label));
    return iter != _labelsInfo.end() ? iter->second : "";
}

void FaceRecognizer::setLabelInfo(int label, const String &strInfo)
{
    _labelsInfo[label] = strInfo;
}

void FaceRecognizer::update(InputArrayOfArrays src, InputArray labels)
{
    (void)src;
    (void)labels;
    String error_msg = format("This FaceRecognizer does not support updating, you have to use FaceRecognizer::train to update it.");
    CV_Error(Error::StsNotImplemented, error_msg);
}

void FaceRecognizer::load(const String &filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
    this->load(fs);
    fs.release();
}

void FaceRecognizer::save(const String &filename) const
{
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
        CV_Error(Error::StsError, "File can't be opened for writing!");
    this->save(fs);
    fs.release();
}

int FaceRecognizer::predict(InputArray src) const {
    int _label;
    double _dist;
    predict(src, _label, _dist);
    return _label;
}

void FaceRecognizer::predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const {
    Ptr<MinDistancePredictCollector> collector = MinDistancePredictCollector::create(getThreshold());
    predict(src, collector, 0);
    label = collector->getLabel();
    confidence = collector->getDist();
}

}
}

