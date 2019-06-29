// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Augmenter::Augmenter() {}

void Augmenter::add(Ptr<Transform> transformation, float prob)
{
    transformations.push_back(transformation);
    probs.push_back(prob);
}

std::vector<Mat> Augmenter::applyImages(const std::vector<Mat>& imgs)
{
    std::vector<Mat> dstImgs;
    RNG rng;

    for (size_t i = 0; i < imgs.size(); i++)
    {
        Mat img = imgs[i].clone();

            for (size_t j = 0; j < transformations.size(); j++)
            {
                float prob = rng.uniform(0.f, 1.f);

                if (prob <= probs[j])
                {
                    transformations[j]->init(imgs[i]);
                    transformations[j]->image(img, img);
                }

            }

        dstImgs.push_back(img);
    }

    return dstImgs;
}


//std::tuple<std::vector<Mat>, std::vector<Mat>> 
//Augmenter::applyImagesWithMasks(const std::vector<Mat>& imgs, const std::vector<Mat>& masks)
//{
//    CV_Assert(imgs.size() == masks.size());
//    std::vector<Mat> dstImgs, dstMasks;
//    RNG rng;
//
//    for (size_t i = 0; i < imgs.size(); i++)
//    {
//        Mat img = imgs[i].clone();
//        Mat mask = masks[i].clone();
//
//        for (size_t j = 0; j < transformations.size(); j++)
//        {
//            float prob = rng.uniform(0.f, 1.f);
//
//            if (prob <= std::get<1>(transformations[j]))
//            {
//                std::get<0>(transformations[j])->init(imgs[i]);
//                std::get<0>(transformations[j])->image(img, img);
//                std::get<0>(transformations[j])->image(mask, mask);
//            }
//
//        }
//
//        dstImgs.push_back(img);
//        dstMasks.push_back(mask);
//    }
//
//    return std::make_tuple(dstImgs, dstMasks);
//}






}}
