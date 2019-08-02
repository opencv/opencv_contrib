/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include "precomp.hpp"

using std::cerr;
using std::endl;
using std::vector;
using std::string;

namespace cv {
namespace xobjdetect {

static vector<Mat> sample_patches(
        const string& path,
        int n_rows,
        int n_cols,
        size_t n_patches)
{
    vector<String> filenames;
    glob(path, filenames);
    vector<Mat> patches;
    size_t patch_count = 0;
    for (size_t i = 0; i < filenames.size(); ++i) {
        Mat img = imread(filenames[i], IMREAD_GRAYSCALE);
        for (int row = 0; row + n_rows < img.rows; row += n_rows) {
            for (int col = 0; col + n_cols < img.cols; col += n_cols) {
                patches.push_back(img(Rect(col, row, n_cols, n_rows)).clone());
                patch_count += 1;
                if (patch_count == n_patches) {
                    goto sampling_finished;
                }
            }
        }
    }
sampling_finished:
    return patches;
}

static vector<Mat> read_imgs(const string& path)
{
    vector<String> filenames;
    glob(path, filenames);
    vector<Mat> imgs;
    for (size_t i = 0; i < filenames.size(); ++i) {
        imgs.push_back(imread(filenames[i], IMREAD_GRAYSCALE));
    }
    return imgs;
}

void WBDetectorImpl::read(const FileNode& node)
{
    boost_.read(node);
}


void WBDetectorImpl::write(FileStorage &fs) const
{
    boost_.write(fs);
}

void WBDetectorImpl::train(
    const string& pos_samples_path,
    const string& neg_imgs_path)
{

    vector<Mat> pos_imgs = read_imgs(pos_samples_path);
    vector<Mat> neg_imgs = sample_patches(neg_imgs_path, 24, 24, pos_imgs.size() * 10);

    assert(pos_imgs.size());
    assert(neg_imgs.size());

    int n_features;
    Mat pos_data, neg_data;

    Ptr<CvFeatureEvaluator> eval = CvFeatureEvaluator::create();
    eval->init(CvFeatureParams::create(), 1, Size(24, 24));
    n_features = eval->getNumFeatures();

    const int stages[] = {64, 128, 256, 512, 1024};
    const int stage_count = sizeof(stages) / sizeof(*stages);
    const int stage_neg = (int)(pos_imgs.size() * 5);
    const int max_per_image = 100;

    const float scales_arr[] = {.3f, .4f, .5f, .6f, .7f, .8f, .9f, 1.0f};
    const vector<float> scales(scales_arr,
            scales_arr + sizeof(scales_arr) / sizeof(*scales_arr));

    vector<String> neg_filenames;
    glob(neg_imgs_path, neg_filenames);


    for (int i = 0; i < stage_count; ++i) {

        cerr << "compute features" << endl;

        pos_data = Mat1b(n_features, (int)pos_imgs.size());
        neg_data = Mat1b(n_features, (int)neg_imgs.size());

        for (size_t k = 0; k < pos_imgs.size(); ++k) {
            eval->setImage(pos_imgs[k], +1, 0, boost_.get_feature_indices());
            for (int j = 0; j < n_features; ++j) {
                pos_data.at<uchar>(j, (int)k) = (uchar)(*eval)(j);
            }
        }

        for (size_t k = 0; k < neg_imgs.size(); ++k) {
            eval->setImage(neg_imgs[k], 0, 0, boost_.get_feature_indices());
            for (int j = 0; j < n_features; ++j) {
                neg_data.at<uchar>(j, (int)k) = (uchar)(*eval)(j);
            }
        }


        boost_.reset(stages[i]);
        boost_.fit(pos_data, neg_data);

        if (i + 1 == stage_count) {
            break;
        }

        int bootstrap_count = 0;
        size_t img_i = 0;
        for (; img_i < neg_filenames.size(); ++img_i) {
            cerr << "win " << bootstrap_count << "/" << stage_neg
                 << " img " << (img_i + 1) << "/" << neg_filenames.size() << "\r";
            Mat img = imread(neg_filenames[img_i], IMREAD_GRAYSCALE);
            vector<Rect> bboxes;
            Mat1f confidences;
            boost_.detect(eval, img, scales, bboxes, confidences);

            if (confidences.rows > 0) {
                Mat1i indices;
                sortIdx(confidences, indices,
                        CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING);

                int win_count = min(max_per_image, confidences.rows);
                win_count = min(win_count, stage_neg - bootstrap_count);
                Mat window;
                for (int k = 0; k < win_count; ++k) {
                    resize(img(bboxes[indices(k, 0)]), window, Size(24, 24), 0, 0, INTER_LINEAR_EXACT);
                    neg_imgs.push_back(window.clone());
                    bootstrap_count += 1;
                }
                if (bootstrap_count >= stage_neg) {
                    break;
                }
            }
        }
        cerr << "bootstrapped " << bootstrap_count << " windows from "
             << (img_i + 1) << " images" << endl;
    }
}

void WBDetectorImpl::detect(
    const Mat& img,
    vector<Rect> &bboxes,
    vector<double> &confidences)
{
    Mat test_img = img.clone();
    bboxes.clear();
    confidences.clear();
    vector<float> scales;
    for (float scale = 0.2f; scale < 1.2f; scale *= 1.1f) {
        scales.push_back(scale);
    }
    Ptr<CvFeatureParams> params = CvFeatureParams::create();
    Ptr<CvFeatureEvaluator> eval = CvFeatureEvaluator::create();
    eval->init(params, 1, Size(24, 24));
    boost_.detect(eval, img, scales, bboxes, confidences);
    assert(confidences.size() == bboxes.size());
}

Ptr<WBDetector>
WBDetector::create()
{
    return Ptr<WBDetector>(new WBDetectorImpl());
}

}
}
