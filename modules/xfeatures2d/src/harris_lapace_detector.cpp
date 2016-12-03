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

namespace cv
{
namespace xfeatures2d
{

bool sort_func(KeyPoint kp1, KeyPoint kp2);

/*
 *  HarrisLaplaceFeatureDetector_Impl
 */
class HarrisLaplaceFeatureDetector_Impl : public HarrisLaplaceFeatureDetector
{
public:
    HarrisLaplaceFeatureDetector_Impl(
        int numOctaves=6,
        float corn_thresh=0.01,
        float DOG_thresh=0.01,
        int maxCorners=5000,
        int num_layers=4
    );
    virtual void read( const FileNode& fn );
    virtual void write( FileStorage& fs ) const;

protected:
    void detect( InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask=noArray() );

    int numOctaves;
    float corn_thresh;
    float DOG_thresh;
    int maxCorners;
    int num_layers;
};

Ptr<HarrisLaplaceFeatureDetector> HarrisLaplaceFeatureDetector::create(
    int numOctaves,
    float corn_thresh,
    float DOG_thresh,
    int maxCorners,
    int num_layers)
{
    return makePtr<HarrisLaplaceFeatureDetector_Impl>(numOctaves, corn_thresh, DOG_thresh, maxCorners, num_layers);
}

HarrisLaplaceFeatureDetector_Impl::HarrisLaplaceFeatureDetector_Impl(
    int _numOctaves,
    float _corn_thresh,
    float _DOG_thresh,
    int _maxCorners,
    int _num_layers
) :
    numOctaves(_numOctaves),
    corn_thresh(_corn_thresh),
    DOG_thresh(_DOG_thresh),
    maxCorners(_maxCorners),
    num_layers(_num_layers)
{
    CV_Assert(num_layers == 2 || num_layers==4);
}

void HarrisLaplaceFeatureDetector_Impl::read (const FileNode& fn)
{
    numOctaves = fn["numOctaves"];
    corn_thresh = fn["corn_thresh"];
    DOG_thresh = fn["DOG_thresh"];
    maxCorners = fn["maxCorners"];
    num_layers = fn["num_layers"];
}

void HarrisLaplaceFeatureDetector_Impl::write (FileStorage& fs) const
{
    fs << "numOctaves" << numOctaves;
    fs << "corn_thresh" << corn_thresh;
    fs << "DOG_thresh" << DOG_thresh;
    fs << "maxCorners" << maxCorners;
    fs << "num_layers" << num_layers;
}

/*
 * Detect method
 * The method detect Harris corners on scale space as described in
 * "K. Mikolajczyk and C. Schmid.
 * Scale & affine invariant interest point detectors.
 * International Journal of Computer Vision, 2004"
 */
void HarrisLaplaceFeatureDetector_Impl::detect(InputArray img, std::vector<KeyPoint>& keypoints, InputArray mask )
{
    Mat image = img.getMat();
    if( image.empty() )
    {
        keypoints.clear();
        return;
    }
    Mat_<float> dx2, dy2, dxy;
    Mat Lx, Ly;
    float si, sd;
    int gsize;
    Mat fimage;
    image.convertTo(fimage, CV_32F, 1.f/255);
    /*Build gaussian pyramid*/
    Pyramid pyr(fimage, numOctaves, num_layers, 1, -1, true);
    keypoints = std::vector<KeyPoint> (0);

    /*Find Harris corners on each layer*/
    for (int octave = 0; octave <= numOctaves; octave++)
    {
        for (int layer = 1; layer <= num_layers; layer++)
        {
            if (octave == 0)
                layer = num_layers;

            Mat Lxm2smooth, Lxmysmooth, Lym2smooth;

            si = pow(2, layer / (float) num_layers);
            sd = si * 0.7;

            Mat curr_layer;
            if (num_layers == 4)
            {
                if (layer == 1)
                {
                    Mat tmp = pyr.getLayer(octave - 1, num_layers - 1);
                    resize(tmp, curr_layer, Size(0, 0), 0.5, 0.5, INTER_AREA);

                } else
                    curr_layer = pyr.getLayer(octave, layer - 2);
            } else /*if num_layer==2*/
            {

                curr_layer = pyr.getLayer(octave, layer - 1);
            }

            /*Calculates second moment matrix*/

            /*Derivatives*/
            Sobel(curr_layer, Lx, CV_32F, 1, 0, 1);
            Sobel(curr_layer, Ly, CV_32F, 0, 1, 1);

            /*Normalization*/
            Lx = Lx * sd;
            Ly = Ly * sd;

            Mat Lxm2 = Lx.mul(Lx);
            Mat Lym2 = Ly.mul(Ly);
            Mat Lxmy = Lx.mul(Ly);

            gsize = ceil(si * 3) * 2 + 1;

            /*Convolution*/
            GaussianBlur(Lxm2, Lxm2smooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);
            GaussianBlur(Lym2, Lym2smooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);
            GaussianBlur(Lxmy, Lxmysmooth, Size(gsize, gsize), si, si, BORDER_REPLICATE);

            Mat cornern_mat(curr_layer.size(), CV_32F);

            /*Calculates cornerness in each pixel of the image*/
            for (int row = 0; row < curr_layer.rows; row++)
            {
                for (int col = 0; col < curr_layer.cols; col++)
                {
                    float dx2f = Lxm2smooth.at<float> (row, col);
                    float dy2f = Lym2smooth.at<float> (row, col);
                    float dxyf = Lxmysmooth.at<float> (row, col);
                    float det = dx2f * dy2f - dxyf * dxyf;
                    float tr = dx2f + dy2f;
                    float cornerness = det - (0.04f * tr * tr);
                    cornern_mat.at<float> (row, col) = cornerness;
                }
            }

            double maxVal = 0;
            Mat corn_dilate;

            /*Find max cornerness value and rejects all corners that are lower than a threshold*/
            minMaxLoc(cornern_mat, 0, &maxVal, 0, 0);
            threshold(cornern_mat, cornern_mat, maxVal * corn_thresh, 0, THRESH_TOZERO);
            dilate(cornern_mat, corn_dilate, Mat());

            Size imgsize = curr_layer.size();

            /*Verify for each of the initial points whether the DoG attains a maximum at the scale of the point*/
            Mat prevDOG, curDOG, succDOG;
            prevDOG = pyr.getDOGLayer(octave, layer - 1);
            curDOG = pyr.getDOGLayer(octave, layer);
            succDOG = pyr.getDOGLayer(octave, layer + 1);

            for (int y = 1; y < imgsize.height - 1; y++)
            {
                for (int x = 1; x < imgsize.width - 1; x++)
                {
                    float val = cornern_mat.at<float> (y, x);
                    if (val != 0 && val == corn_dilate.at<float> (y, x))
                    {

                        float curVal = curDOG.at<float> (y, x);
                        float prevVal =  prevDOG.at<float> (y, x);
                        float succVal = succDOG.at<float> (y, x);

                        KeyPoint kp(
                                Point(x * pow(2, octave - 1) + pow(2, octave - 1) / 2,
                                        y * pow(2, octave - 1) + pow(2, octave - 1) / 2),
                                3 * pow(2, octave - 1) * si * 2, 0, val, octave);

                        /*Check whether keypoint size is inside the image*/
                        float start_kp_x = kp.pt.x - kp.size / 2;
                        float start_kp_y = kp.pt.y - kp.size / 2;
                        float end_kp_x = start_kp_x + kp.size;
                        float end_kp_y = start_kp_y + kp.size;

                        if (curVal > prevVal && curVal > succVal && curVal >= DOG_thresh
                                && start_kp_x > 0 && start_kp_y > 0 && end_kp_x < image.cols
                                && end_kp_y < image.rows)
                            keypoints.push_back(kp);

                    }
                }
            }

        }

    }

    /*Sort keypoints in decreasing cornerness order*/
    sort(keypoints.begin(), keypoints.end(), sort_func);
    for (size_t i = 1; i < keypoints.size(); i++)
    {
        float max_diff = pow(2, keypoints[i].octave + 1.f / 2);

        if (keypoints[i].response == keypoints[i - 1].response && norm(
                keypoints[i].pt - keypoints[i - 1].pt) <= max_diff)
        {

            float x = (keypoints[i].pt.x + keypoints[i - 1].pt.x) / 2;
            float y = (keypoints[i].pt.y + keypoints[i - 1].pt.y) / 2;

            keypoints[i].pt = Point(x, y);
            --i;
            keypoints.erase(keypoints.begin() + i);

        }
    }

    /*Select strongest keypoints*/
    if (maxCorners > 0 && maxCorners < (int) keypoints.size())
        keypoints.resize(maxCorners);
}

bool sort_func(KeyPoint kp1, KeyPoint kp2)
{
    return (kp1.response > kp2.response);
}

}
}
