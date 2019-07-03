// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <iostream>
namespace cv { namespace augment {

Augmenter::Augmenter() {}

void Augmenter::add(Ptr<Transform> transformation, float prob)
{
    transformations.push_back(transformation);
    probs.push_back(prob);
}

void Augmenter::applyImages(InputArrayOfArrays imgs, OutputArrayOfArrays dstImgs)
{
    dstImgs.create(imgs.cols(), 1, 0, -1, true);
    RNG rng;

    for (size_t i = 0; i < imgs.cols(); i++)
    {
        Mat originalImg = imgs.getMat(i);
        Mat img = originalImg.clone();

            for (size_t j = 0; j < transformations.size(); j++)
            {
                float prob = rng.uniform(0.f, 1.f);

                if (prob <= probs[j])
                {
                    transformations[j]->init(originalImg);
                    transformations[j]->image(img, img);
                }

            }

            dstImgs.create(img.size(), img.type(), i, true);
            Mat dstImg = dstImgs.getMat(i);
            img.copyTo(dstImg);

    }

}


void Augmenter::applyImagesWithMasks(InputArrayOfArrays imgs,
    InputArrayOfArrays masks,
    OutputArrayOfArrays dstImgs,
    OutputArrayOfArrays dstMasks)
{
    dstImgs.create(imgs.cols(), 1, 0, -1, true);
    dstMasks.create(masks.cols(), 1, 0, -1, true);
    RNG rng;

    for (size_t i = 0; i < imgs.cols(); i++)
    {
        Mat originalImg = imgs.getMat(i);
        Mat originalMask = masks.getMat(i);
        Mat augmentedImg = originalImg.clone();
        Mat augmentedMask = originalMask.clone();

        for (size_t j = 0; j < transformations.size(); j++)
        {
            float prob = rng.uniform(0.f, 1.f);

            if (prob <= probs[j])
            {
                transformations[j]->init(originalImg);
                transformations[j]->image(augmentedImg, augmentedImg);
                transformations[j]->mask(augmentedMask, augmentedMask);
            }

        }

        dstImgs.create(augmentedImg.size(), augmentedImg.type(), i, true);
        Mat dstImg = dstImgs.getMat(i);
        augmentedImg.copyTo(dstImg);

        dstMasks.create(augmentedMask.size(), augmentedMask.type(), i, true);
        Mat dstMask = dstMasks.getMat(i);
        augmentedMask.copyTo(dstMask);

    }
}


