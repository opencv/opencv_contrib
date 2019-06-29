// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
namespace cv { namespace augment {

Augmenter::Augmenter() {}

void Augmenter::add(Ptr<Transform> transformation, float prob)
{
    transformations.push_back(std::make_tuple(transformation, prob));
}

std::vector<Mat> Augmenter::applyImages(const std::vector<Mat>& imgs)
{
    std::vector<Mat> dst;
    RNG rng;

    for (size_t i = 0; i < imgs.size(); i++)
    {
        Mat img = imgs[i].clone();

            for (size_t j = 0; j < transformations.size(); j++)
            {
                float prob = rng.uniform(0.f, 1.f);

                if (prob <= std::get<1>(transformations[j]))
                {
                    std::get<0>(transformations[j])->init(imgs[i]);
                    std::get<0>(transformations[j])->image(img, img);
                }

            }

        dst.push_back(img);
    }

    return dst;
}






}}
