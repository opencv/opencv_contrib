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

void Augmenter::applyImages(const std::vector<Mat>& imgs, OutputArrayOfArrays dstImgs)
{
    dstImgs.create(imgs.size(), 1, 0, -1, true);
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

            dstImgs.create(img.size(), img.type(), i, true);
            Mat dstImg = dstImgs.getMat(i);
            img.copyTo(dstImg);

    }

}


void Augmenter::applyImagesWithMasks(const std::vector<Mat>& imgs,
    const std::vector<Mat>& masks,
    OutputArrayOfArrays dstImgs,
    OutputArrayOfArrays dstMasks)
{
    dstImgs.create(imgs.size(), 1, 0, -1, true);
    dstMasks.create(masks.size(), 1, 0, -1, true);
    RNG rng;

    for (size_t i = 0; i < imgs.size(); i++)
    {
        Mat img = imgs[i].clone();
        Mat mask = masks[i].clone();

        for (size_t j = 0; j < transformations.size(); j++)
        {
            float prob = rng.uniform(0.f, 1.f);

            if (prob <= probs[j])
            {
                transformations[j]->init(imgs[i]);
                transformations[j]->image(img, img);
                transformations[j]->image(mask, mask);
            }

        }

        dstImgs.create(img.size(), img.type(), i, true);
        Mat dstImg = dstImgs.getMat(i);
        img.copyTo(dstImg);

        dstMasks.create(mask.size(), mask.type(), i, true);
        Mat dstMask = dstMasks.getMat(i);
        mask.copyTo(dstMask);

    }
}





}}
