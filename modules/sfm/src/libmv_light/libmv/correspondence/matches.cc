// Copyright (c) 2007, 2008 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "libmv/correspondence/matches.h"
#include "libmv/correspondence/feature.h"

namespace libmv {

Matches::~Matches() {}

void DeleteMatchFeatures(Matches *matches) {
  (void) matches;
  // XXX
  /*
  for (Correspondences::FeatureIterator it = correspondences->ScanAllFeatures();
       !it.Done(); it.Next()) {
    delete const_cast<Feature *>(it.feature());
  }
  */
}

int Matches::GetNumberOfMatches(ImageID id1,ImageID id2) const
{
    Features<Feature> features1 = InImage<Feature>(id1);
    Features<Feature> features2 = InImage<Feature>(id2);
    int count = 0;
    for(int i1=0;features1;++features1,++i1)
    {
        Features<Feature> temp = features2;
        for(int i2=0;temp;++temp,++i2)
        {
            if(features1.track() == temp.track())
            {
                ++count;
                break;
            }
        }
    }
    return count;
}

void Matches::DrawMatches(ImageID image_id1,const cv::Mat &image1,ImageID image_id2,const cv::Mat &image2, cv::Mat &out)const
{
    std::vector<cv::KeyPoint> points1;
    std::vector<cv::KeyPoint> points2;
    std::vector<cv::DMatch> matches;
    KeyPoints(image_id1,points1);
    KeyPoints(image_id2,points2);
    MatchesTwo(image_id1,image_id2,matches);
    cv::drawMatches(image1,points1,image2,points2,matches,out);
}

void Matches::MatchesTwo(ImageID image1,ImageID image2,std::vector<cv::DMatch> &matches)const
{
    Features<PointFeature> features1 = InImage<PointFeature>(image1);
    Features<PointFeature> features2 = InImage<PointFeature>(image2);
    for(int i1=0;features1;++features1,++i1)
    {
        Features<PointFeature> temp = features2;
        for(int i2=0;temp;++temp,++i2)
        {
            if(features1.track() == temp.track())
            {
                matches.push_back(cv::DMatch(i1,i2,std::numeric_limits<float>::max()));
                break;
            }
        }
    }
}

void Matches::KeyPoints(ImageID image,std::vector<cv::KeyPoint> &keypoints)const
{
    PointFeatures2KeyPoints(InImage<PointFeature>(image),keypoints);
}

void Matches::PointFeatures2KeyPoints(Features<PointFeature> features,std::vector<cv::KeyPoint> &keypoints)const
{
    for(;features;++features)
        keypoints.push_back(cv::KeyPoint(features.feature()->x(),features.feature()->y(),1));
}

}  // namespace libmv
