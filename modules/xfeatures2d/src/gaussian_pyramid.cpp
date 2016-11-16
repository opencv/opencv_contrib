/*IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

 By downloading, copying, installing or using the software you agree to this license.
 If you do not agree to this license, do not download, install,
 copy or use the software.


 License Agreement
 For Open Source Computer Vision Library

 Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
 Copyright (C) 2008-2010, Willow Garage Inc., all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistribution's of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.

 * Redistribution's in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 * The name of the copyright holders may not be used to endorse or promote products
 derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.*/

#include "precomp.hpp"
#include "gaussian_pyramid.hpp"
namespace cv{

Pyramid::Pyramid()
{
}
/**
 * Pyramid class constructor
 * octavesN_: number of octaves
 * layersN_: number of layers before subsampling layer
 * sigma0_: starting sigma (depends on detector's type, i.e. SIFT sigma0 = 1.6, Harris sigma0 = 1)
 * omin_: if omin<0 an octave is added before first octave. In this octave the image size is doubled
 * _DOG: if true, a DOG pyramid is build
 */
Pyramid::Pyramid(const Mat & img, int octavesN_, int layersN_, float sigma0_, int omin_, bool _DOG) :
    params(octavesN_, layersN_, sigma0_, omin_)
{
    build(img, _DOG);
}

/**
 * Build gaussian pyramid with layersN_ + 3 layers and 2^(1/layersN_) step between layers
 * each octave is downsampled of a factor of 2
 */

void Pyramid::build(const Mat& img, bool DOG)
{

    Size ksize(0, 0);
    int gsize;

    Size imgSize = img.size();
    int minSize = MIN(imgSize.width, imgSize.height);
    int octavesN = MIN(params.octavesN, floor(log2((double) minSize)));
    float sigma0 = params.sigma0;
    float sigma = sigma0;
    int layersN = params.layersN + 3;
    int omin = params.omin;
    float k = params.step;

    /*layer to downsample*/
    int down_lay = 1 / log(k);

    int octave, layer;
    double sigmaN = 0.5;

    std::vector<Mat> layers, DOG_layers;
    /* standard deviation of current layer*/
    float sigma_curr = sigma;
    /* standard deviation of previous layer*/
    float sigma_prev = sigma;

    if (omin < 0)
    {
        omin = -1;
        Mat tmp_img;
        Mat blurred_img;
        gsize = ceil(sigmaN * 3) * 2 + 1;
        GaussianBlur(img, blurred_img, Size(gsize,gsize), sigmaN);
        resize(blurred_img, tmp_img, ksize, 2, 2, INTER_AREA);
        layers.push_back(tmp_img);

        for (layer = 1; layer < layersN; layer++)
        {
            sigma_curr = getSigma(layer);
            sigma = sqrt(powf(sigma_curr, 2) - powf(sigma_prev, 2));
            Mat prev_lay = layers[layer - 1], curr_lay, DOG_lay;
            /* smoothing is applied on previous layer so sigma_curr^2 = sigma^2 + sigma_prev^2 */
            gsize = ceil(sigma * 3) * 2 + 1;
            GaussianBlur(prev_lay, curr_lay, Size(gsize,gsize), sigma);
            layers.push_back(curr_lay);
            if (DOG)
            {
                absdiff(curr_lay, prev_lay, DOG_lay);
                DOG_layers.push_back(DOG_lay);
            }
            sigma_prev = sigma_curr;

        }
        Octave tmp_oct(layers);
        octaves.push_back(tmp_oct);
        layers.clear();

        if (DOG)
        {
            DOGOctave tmp_DOG_Oct(DOG_layers);
            DOG_octaves.push_back(tmp_DOG_Oct);
            DOG_layers.clear();
        }

    }

    /* Presmoothing on first layer */
    double sb = sigmaN / powf(2.0f, omin);
    sigma = sigma0;
    if (sigma0 > sb)
        sigma = sqrt(sigma0 * sigma0 - sb * sb);

    /*1° step on image*/
    Mat tmpImg;
    gsize = ceil(sigma * 3) * 2 + 1;
    GaussianBlur(img, tmpImg, Size(gsize,gsize), sigma);
    layers.push_back(tmpImg);

    /*for every octave build layers*/
    sigma_prev = sigma;

    for (octave = 0; octave < octavesN; octave++)
    {
        for (layer = 1; layer < layersN; layer++)
        {
            sigma_curr = getSigma(layer);
            sigma = sqrt(powf(sigma_curr, 2) - powf(sigma_prev, 2));

            Mat prev_lay = layers[layer - 1], curr_lay, DOG_lay;
            gsize = ceil(sigma * 3) * 2 + 1;
            GaussianBlur(prev_lay, curr_lay, Size(gsize,gsize), sigma);
            layers.push_back(curr_lay);

            if (DOG)
            {
                absdiff(curr_lay, prev_lay, DOG_lay);
                DOG_layers.push_back(DOG_lay);
            }
            sigma_prev = sigma_curr;
        }

        Mat resized_lay;
        resize(layers[down_lay], resized_lay, ksize, 1.0f / 2, 1.0f / 2, INTER_AREA);

        Octave tmp_oct(layers);
        octaves.push_back(tmp_oct);
        if (DOG)
        {
            DOGOctave tmp_DOG_Oct(DOG_layers);
            DOG_octaves.push_back(tmp_DOG_Oct);
            DOG_layers.clear();
        }
        sigma_curr = sigma_prev = sigma0;
        layers.clear();
        layers.push_back(resized_lay);

    }

}
/**
 * Return layer at indicated octave and layer numbers
 */
Mat Pyramid::getLayer(int octave, int layer)
{
    return octaves[octave].getLayerAt(layer);
}

/**
 * Return DOG layer at indicated octave and layer numbers
 */
Mat Pyramid::getDOGLayer(int octave, int layer)
{
    CV_Assert(!DOG_octaves.empty());
    return DOG_octaves[octave].getLayerAt(layer);
}

/**
 * Return sigma value of indicated octave and layer
 */
float Pyramid::getSigma(int octave, int layer)
{

    return pow(2.0f, octave) * powf(params.step, layer) * params.sigma0;
}

/**
 * Return sigma value of indicated layer
 * sigma value of layer is the same at each octave
 * i.e. sigma of first layer at each octave is sigma0
 */
float Pyramid::getSigma(int layer)
{

    return powf(params.step, layer) * params.sigma0;
}

/**
 * Destructor of Pyramid class
 */
Pyramid::~Pyramid()
{
    clear();
}

/**
 * Clear octaves and params
 */
void Pyramid::clear()
{
    octaves.clear();
    params.clear();
}

/**
 * Empty Pyramid
 * @return
 */
bool Pyramid::empty()
{
    return octaves.empty();
}

Pyramid::Params::Params()
{
}

/**
 * Params for Pyramid class
 *
 */
Pyramid::Params::Params(int octavesN_, int layersN_, float sigma0_, int omin_) :
    octavesN(octavesN_), layersN(layersN_), sigma0(sigma0_), omin(omin_)
{
    CV_Assert(layersN > 0 && octavesN_>0);
    step = powf(2, 1.0f / layersN);
}

/**
 * Returns Pyramid's params
 */
Pyramid::Params Pyramid::getParams()
{
    return params;
}

/**
 * Set to zero all params
 */
void Pyramid::Params::clear()
{
    octavesN = 0;
    layersN = 0;
    sigma0 = 0;
    omin = 0;
    step = 0;
}

/**
 * Create an Octave with layers
 */
Pyramid::Octave::Octave(std::vector<Mat> layers)
{

    (*this).layers = layers;
}

/**
 * Return layers of the Octave
 */
std::vector<Mat> Pyramid::Octave::getLayers()
{
    return layers;
}

Pyramid::Octave::Octave()
{
}

/**
 * Return the Octave's layer at index i
 */
Mat Pyramid::Octave::getLayerAt(int i)
{
    CV_Assert(i < (int) layers.size());
    return layers[i];
}

Pyramid::Octave::~Octave()
{
}

Pyramid::DOGOctave::DOGOctave()
{
}

Pyramid::DOGOctave::DOGOctave(std::vector<Mat> layers)
{
    (*this).layers = layers;
}

Pyramid::DOGOctave::~DOGOctave()
{
}

std::vector<Mat> Pyramid::DOGOctave::getLayers()
{
    return layers;
}

Mat Pyramid::DOGOctave::getLayerAt(int i)
{
    CV_Assert(i < (int) layers.size());
    return layers[i];
}
} // namespace cv
