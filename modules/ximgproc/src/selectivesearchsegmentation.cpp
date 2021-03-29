/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

/*******************************************************************************\
*                   Selective search segmentation                              *
* This code implements the segmentation method described in:                   *
* Jasper R. R. Uijlings, Koen E. A. van de Sande, Theo Gevers,                 *
* Arnold W. M. Smeulders: "Selective Search for Object Recognition "           *
* International Journal of Computer Vision, Volume 104 (2), page 154-171, 2013 *
* Author: Maximilien Cuony / LTS2 / EPFL / 2016                                *
********************************************************************************/

#include "precomp.hpp"
#include "opencv2/ximgproc/segmentation.hpp"

#include <iostream>

namespace cv {
    namespace ximgproc {
        namespace segmentation {

            // Helpers

            // Represent a regsion
            class Region {
                public:
                    int id;
                    int level;
                    int merged_to;
                    double rank;
                    Rect bounding_box;

                    Region() : id(0), level(0), merged_to(0), rank(0) {}

                    friend std::ostream& operator<<(std::ostream& os, const Region& n);

                    bool operator <(const Region& n) const {
                        return rank < n.rank;
                    }
            };

            // Comparator to sort cv::rect (used for a std::map).
            struct rectComparator {
                bool operator()(const cv::Rect_<int>& a, const cv::Rect_<int>& b) const {
                    if (a.x < b.x) {
                        return true;
                    }
                    if (a.x > b.x) {
                        return false;
                    }
                    if (a.y < b.y) {
                        return true;
                    }
                    if (a.y > b.y) {
                        return false;
                    }
                    if (a.width < b.width) {
                        return true;
                    }
                    if (a.width > b.width) {
                        return false;
                    }
                    if (a.height < b.height) {
                        return true;
                    }
                    if (a.height > b.height) {
                        return false;
                    }
                    return false;
                }
            };

            // Represent a neighboor
            class Neighbour {
                public:
                    int from;
                    int to;
                    float similarity;
                    friend std::ostream& operator<<(std::ostream& os, const Neighbour& n);

                    bool operator <(const Neighbour& n) const {
                        return similarity < n.similarity;
                    }
            };

            /****************************************
             * Stragegy / Color
             ***************************************/

            class SelectiveSearchSegmentationStrategyColorImpl CV_FINAL : public SelectiveSearchSegmentationStrategyColor {
                public:
                    SelectiveSearchSegmentationStrategyColorImpl() {
                        name_ = "SelectiveSearchSegmentationStrategyColor";
                        last_image_id = -1;
                    }

                    virtual void setImage(InputArray img, InputArray regions, InputArray sizes, int image_id = -1) CV_OVERRIDE;
                    virtual float get(int r1, int r2) CV_OVERRIDE;
                    virtual void merge(int r1, int r2) CV_OVERRIDE;

                private:
                    String name_;

                    Mat histograms; // [Region X Histogram]
                    Mat sizes;
                    int histogram_size;

                    int last_image_id; // If the image_id is not equal to -1 and the same as the previous call for setImage, computations are used again
                    Mat last_histograms;
            };


            void SelectiveSearchSegmentationStrategyColorImpl::setImage(InputArray img_, InputArray regions_, InputArray sizes_, int image_id) {

                Mat img = img_.getMat();
                Mat regions = regions_.getMat();
                sizes = sizes_.getMat();

                if (image_id == -1 || last_image_id != image_id) {

                    std::vector<Mat> img_planes;
                    split(img, img_planes);

                    int histogram_bins_size = 25;

                    float range[] = {0, 256};
                    const float* histogram_ranges = {range};

                    double min, max;
                    minMaxLoc(regions, &min, &max);
                    int nb_segs = (int)max + 1;

                    histogram_size = histogram_bins_size * img.channels();

                    histograms = Mat_<float>(nb_segs, histogram_size);

                    for (int r = 0; r < nb_segs; r++) {

                        // Generate mask
                        Mat mask = regions == r;

                        // Compute histogram for each channels
                        float tt = 0;

                        Mat tmp_hists = Mat(histogram_size, 1, CV_32F);
                        float *tmp_histogram = tmp_hists.ptr<float>(0);
                        int h_pos = 0;
                        Mat tmp_hist;

                        for (int p = 0; p < img.channels(); p++) {

                            calcHist(&img_planes[p], 1, 0, mask, tmp_hist, 1, &histogram_bins_size, &histogram_ranges);

                            float *tmp_hist_ = tmp_hist.ptr<float>(0);

                            // Copy local histogram to global histogram
                            for (int pos = 0; pos < histogram_bins_size; pos++) {
                                tmp_histogram[pos + h_pos] = tmp_hist_[pos];
                                tt += tmp_histogram[pos + h_pos];
                            }
                            h_pos += histogram_bins_size;
                        }

                        // Normalize historgrams
                        float* histogram = histograms.ptr<float>(r);

                        for (int h_pos2 = 0; h_pos2 < histogram_size; h_pos2++) {
                            histogram[h_pos2] = tmp_histogram[h_pos2] / tt;
                        }
                    }

                    // Save cache if we have an image id
                    if (image_id != -1) {
                        last_histograms = histograms.clone();
                        last_image_id = image_id;
                    }
                } else { // last_image_id == image_id
                    // Use cache
                    histograms = last_histograms.clone();
                }
            }

            float SelectiveSearchSegmentationStrategyColorImpl::get(int r1, int r2) {

                float r = 0;
                float* h1 = histograms.ptr<float>(r1);
                float* h2 = histograms.ptr<float>(r2);

                for (int i = 0; i < histogram_size; i++) {
                    r += min(h1[i], h2[i]);
                }

                return r;
            }

            void SelectiveSearchSegmentationStrategyColorImpl::merge(int r1, int r2) {
                int size_r1 = sizes.at<int>(r1);
                int size_r2 = sizes.at<int>(r2);

                float* h1 = histograms.ptr<float>(r1);
                float* h2 = histograms.ptr<float>(r2);

                for (int i = 0; i < histogram_size; i++) {
                    h1[i] = (h1[i] * size_r1 + h2[i] * size_r2) / (size_r1 + size_r2);
                    h2[i] = h1[i];
                }
            }


            Ptr<SelectiveSearchSegmentationStrategyColor> createSelectiveSearchSegmentationStrategyColor() {
                Ptr<SelectiveSearchSegmentationStrategyColor> s = makePtr<SelectiveSearchSegmentationStrategyColorImpl>();
                return s;
            }

            /****************************************
             * Stragegy / Multiple
             ***************************************/

            class SelectiveSearchSegmentationStrategyMultipleImpl CV_FINAL : public SelectiveSearchSegmentationStrategyMultiple {
                public:
                    SelectiveSearchSegmentationStrategyMultipleImpl() {
                        name_ = "SelectiveSearchSegmentationStrategyMultiple";
                        weights_total = 0;
                    }

                    virtual void setImage(InputArray img, InputArray regions, InputArray sizes, int image_id = -1) CV_OVERRIDE;
                    virtual float get(int r1, int r2) CV_OVERRIDE;
                    virtual void merge(int r1, int r2) CV_OVERRIDE;

                    virtual void addStrategy(Ptr<SelectiveSearchSegmentationStrategy> g, float weight) CV_OVERRIDE;
                    virtual void clearStrategies() CV_OVERRIDE;

                private:
                    String name_;
                    std::vector<Ptr<SelectiveSearchSegmentationStrategy> > strategies;
                    std::vector<float> weights;
                    float weights_total;
            };

            void SelectiveSearchSegmentationStrategyMultipleImpl::addStrategy(Ptr<SelectiveSearchSegmentationStrategy> g, float weight) {
                strategies.push_back(g);
                weights.push_back(weight);
                weights_total += weight;
            }

            void SelectiveSearchSegmentationStrategyMultipleImpl::clearStrategies() {
                strategies.clear();
                weights.clear();
                weights_total = 0;
            }

            void SelectiveSearchSegmentationStrategyMultipleImpl::setImage(InputArray img_, InputArray regions_, InputArray sizes_, int image_id) {
                for (unsigned int i = 0; i < strategies.size(); i++) {
                    strategies[i]->setImage(img_, regions_, sizes_, image_id);
                }
            }

            float SelectiveSearchSegmentationStrategyMultipleImpl::get(int r1, int r2) {
                float tt = 0;

                for (unsigned int i = 0; i < strategies.size(); i++) {
                    tt += weights[i] * strategies[i]->get(r1, r2);
                }

                return tt / weights_total;
            }

            void SelectiveSearchSegmentationStrategyMultipleImpl::merge(int r1, int r2) {
                for (unsigned int i = 0; i < strategies.size(); i++) {
                    strategies[i]->merge(r1, r2);
                }
            }

            Ptr<SelectiveSearchSegmentationStrategyMultiple> createSelectiveSearchSegmentationStrategyMultiple() {
                Ptr<SelectiveSearchSegmentationStrategyMultiple> s = makePtr<SelectiveSearchSegmentationStrategyMultipleImpl>();
                return s;
            }

            // Helpers to quickly create a multiple stragegy with 1 to 4 equal strageries
            Ptr<SelectiveSearchSegmentationStrategyMultiple> createSelectiveSearchSegmentationStrategyMultiple(Ptr<SelectiveSearchSegmentationStrategy> s1) {
                Ptr<SelectiveSearchSegmentationStrategyMultiple> s = makePtr<SelectiveSearchSegmentationStrategyMultipleImpl>();

                s->addStrategy(s1, 1.0f);

                return s;
            }

            Ptr<SelectiveSearchSegmentationStrategyMultiple> createSelectiveSearchSegmentationStrategyMultiple(Ptr<SelectiveSearchSegmentationStrategy> s1, Ptr<SelectiveSearchSegmentationStrategy> s2) {
                Ptr<SelectiveSearchSegmentationStrategyMultiple> s = makePtr<SelectiveSearchSegmentationStrategyMultipleImpl>();

                s->addStrategy(s1, 0.5f);
                s->addStrategy(s2, 0.5f);

                return s;
            }

            Ptr<SelectiveSearchSegmentationStrategyMultiple> createSelectiveSearchSegmentationStrategyMultiple(Ptr<SelectiveSearchSegmentationStrategy> s1, Ptr<SelectiveSearchSegmentationStrategy> s2, Ptr<SelectiveSearchSegmentationStrategy> s3) {
                Ptr<SelectiveSearchSegmentationStrategyMultiple> s = makePtr<SelectiveSearchSegmentationStrategyMultipleImpl>();

                s->addStrategy(s1, 0.3333f);
                s->addStrategy(s2, 0.3333f);
                s->addStrategy(s3, 0.3333f);

                return s;
            }

            Ptr<SelectiveSearchSegmentationStrategyMultiple> createSelectiveSearchSegmentationStrategyMultiple(Ptr<SelectiveSearchSegmentationStrategy> s1, Ptr<SelectiveSearchSegmentationStrategy> s2, Ptr<SelectiveSearchSegmentationStrategy> s3, Ptr<SelectiveSearchSegmentationStrategy> s4) {
                Ptr<SelectiveSearchSegmentationStrategyMultiple> s = makePtr<SelectiveSearchSegmentationStrategyMultipleImpl>();

                s->addStrategy(s1, 0.25f);
                s->addStrategy(s2, 0.25f);
                s->addStrategy(s3, 0.25f);
                s->addStrategy(s4, 0.25f);

                return s;
            }


            /****************************************
             * Stragegy / Size
             ***************************************/

            class SelectiveSearchSegmentationStrategySizeImpl CV_FINAL : public SelectiveSearchSegmentationStrategySize {
                public:
                    SelectiveSearchSegmentationStrategySizeImpl() {
                        name_ = "SelectiveSearchSegmentationStrategySize";
                    }

                    virtual void setImage(InputArray img, InputArray regions, InputArray sizes, int image_id = -1) CV_OVERRIDE;
                    virtual float get(int r1, int r2) CV_OVERRIDE;
                    virtual void merge(int r1, int r2) CV_OVERRIDE;

                private:
                    String name_;

                    Mat sizes;
                    int size_image;
            };


            void SelectiveSearchSegmentationStrategySizeImpl::setImage(InputArray img_, InputArray, InputArray sizes_, int /* image_id */) {
                Mat img = img_.getMat();
                size_image = img.rows * img.cols;
                sizes = sizes_.getMat();
            }

            float SelectiveSearchSegmentationStrategySizeImpl::get(int r1, int r2) {

                int size_r1 = sizes.at<int>(r1);
                int size_r2 = sizes.at<int>(r2);

                return max(min(1.0f - (float)(size_r1 + size_r2) / (float)(size_image), 1.0f), 0.0f);
            }

            void SelectiveSearchSegmentationStrategySizeImpl::merge(int /* r1 */, int /* r2 */) {
                // Nothing to do (sizes are merged at parent level)
            }


            Ptr<SelectiveSearchSegmentationStrategySize> createSelectiveSearchSegmentationStrategySize() {
                Ptr<SelectiveSearchSegmentationStrategySize> s = makePtr<SelectiveSearchSegmentationStrategySizeImpl>();
                return s;
            }


            /****************************************
             * Stragegy / Fill
             ***************************************/

            class SelectiveSearchSegmentationStrategyFillImpl CV_FINAL : public SelectiveSearchSegmentationStrategyFill {
                public:
                    SelectiveSearchSegmentationStrategyFillImpl() {
                        name_ = "SelectiveSearchSegmentationStrategyFill";
                    }

                    virtual void setImage(InputArray img, InputArray regions, InputArray sizes, int image_id = -1) CV_OVERRIDE;
                    virtual float get(int r1, int r2) CV_OVERRIDE;
                    virtual void merge(int r1, int r2) CV_OVERRIDE;

                private:
                    String name_;

                    Mat sizes;
                    int size_image;
                    std::vector<Rect> bounding_rects;
            };


            void SelectiveSearchSegmentationStrategyFillImpl::setImage(InputArray img_, InputArray regions_, InputArray sizes_, int /* image_id */) {
                Mat img = img_.getMat();
                sizes = sizes_.getMat();
                Mat regions = regions_.getMat();

                size_image = img.rows * img.cols;

                // Build initial bouding rects
                double min, max;
                minMaxLoc(regions, &min, &max);

                int nb_segs = (int)max + 1;

                // Build a list of points for each regions
                std::vector<std::vector<cv::Point> > points;

                points.resize(nb_segs);

                for (int i = 0; i < (int)regions.rows; i++) {
                    const int* p = regions.ptr<int>(i);

                    for (int j = 0; j < (int)regions.cols; j++) {
                        points[p[j]].push_back(cv::Point(j, i));
                    }
                }

                // Compute bounding rects for each regions
                bounding_rects.resize(nb_segs);

                for(int seg = 0; seg < nb_segs; seg++) {
                    bounding_rects[seg] = cv::boundingRect(points[seg]);
                }
            }

            float SelectiveSearchSegmentationStrategyFillImpl::get(int r1, int r2) {

                int size_r1 = sizes.at<int>(r1);
                int size_r2 = sizes.at<int>(r2);
                int bounding_rect_size = (bounding_rects[r1] | bounding_rects[r2]).area();

                return max(min(1.0f - (float)(bounding_rect_size - size_r1 - size_r2) / (float)(size_image), 1.0f), 0.0f);
            }

            void SelectiveSearchSegmentationStrategyFillImpl::merge(int r1, int r2) {
                bounding_rects[r1] = bounding_rects[r1] | bounding_rects[r2];
                bounding_rects[r2] = bounding_rects[r1];
            }


            Ptr<SelectiveSearchSegmentationStrategyFill> createSelectiveSearchSegmentationStrategyFill() {
                Ptr<SelectiveSearchSegmentationStrategyFill> s = makePtr<SelectiveSearchSegmentationStrategyFillImpl>();
                return s;
            }


            /****************************************
             * Stragegy / Texture
             ***************************************/

            class SelectiveSearchSegmentationStrategyTextureImpl CV_FINAL : public SelectiveSearchSegmentationStrategyTexture {
                public:
                    SelectiveSearchSegmentationStrategyTextureImpl() {
                        name_ = "SelectiveSearchSegmentationStrategyTexture";
                        last_image_id = -1;
                    }

                    virtual void setImage(InputArray img, InputArray regions, InputArray sizes, int image_id = -1) CV_OVERRIDE;
                    virtual float get(int r1, int r2) CV_OVERRIDE;
                    virtual void merge(int r1, int r2) CV_OVERRIDE;

                private:
                    String name_;

                    Mat histograms; //[Region X Histogram]
                    Mat sizes;
                    int histogram_size;

                    int last_image_id; // If the image_id is not equal to -1 and the same as the previous call for setImage, computations are used again
                    Mat last_histograms;
            };


            void SelectiveSearchSegmentationStrategyTextureImpl::setImage(InputArray img_, InputArray regions_, InputArray sizes_, int image_id) {

                Mat img = img_.getMat();
                Mat regions = regions_.getMat();
                sizes = sizes_.getMat();

                if (image_id == -1 || last_image_id != image_id) {

                    std::vector<Mat> img_planes;
                    split(img, img_planes);

                    int histogram_bins_size = 10;

                    float range[] = {0.0, 256.0};

                    double min, max;
                    minMaxLoc(regions, &min, &max);
                    int nb_segs = (int)max + 1;

                    histogram_size = histogram_bins_size * img.channels() * 8;

                    histograms = Mat_<float>(nb_segs, histogram_size);

                    // Compute, for each channels, the 8 gaussians
                    std::vector<Mat> img_gaussians;

                    for (int p = 0; p < img.channels(); p++) {

                        Mat tmp_gradiant;
                        Mat tmp_gradiant_pos, tmp_gradiant_neg;
                        Mat img_plane_rotated;
                        Mat tmp_rot;

                        // X, no rot
                        Scharr(img_planes[p], tmp_gradiant, CV_32F, 1, 0);
                        threshold(tmp_gradiant, tmp_gradiant_pos, 0, 0, THRESH_TOZERO);
                        threshold(tmp_gradiant, tmp_gradiant_neg, 0, 0, THRESH_TOZERO_INV);

                        img_gaussians.push_back(tmp_gradiant_pos.clone());
                        img_gaussians.push_back(tmp_gradiant_neg.clone());

                        // Y, no rot
                        Scharr(img_planes[p], tmp_gradiant, CV_32F, 0, 1);
                        threshold(tmp_gradiant, tmp_gradiant_pos, 0, 0, THRESH_TOZERO);
                        threshold(tmp_gradiant, tmp_gradiant_neg, 0, 0, THRESH_TOZERO_INV);

                        img_gaussians.push_back(tmp_gradiant_pos.clone());
                        img_gaussians.push_back(tmp_gradiant_neg.clone());

                        Point2f center(img.cols / 2.0f, img.rows / 2.0f);
                        Mat rot = cv::getRotationMatrix2D(center, 45.0, 1.0);
                        Rect bbox = cv::RotatedRect(center, img.size(), 45.0).boundingRect();
                        rot.at<double>(0,2) += bbox.width/2.0 - center.x;
                        rot.at<double>(1,2) += bbox.height/2.0 - center.y;

                        warpAffine(img_planes[p], img_plane_rotated, rot, bbox.size());

                        // X, rot
                        Scharr(img_plane_rotated, tmp_gradiant, CV_32F, 1, 0);

                        center = Point((int)(img_plane_rotated.cols / 2.0), (int)(img_plane_rotated.rows / 2.0));
                        rot = cv::getRotationMatrix2D(center, -45.0, 1.0);
                        // Using this bigger box avoids clipping the ends of narrow images
                        Rect bbox2 = cv::RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect();\
                        warpAffine(tmp_gradiant, tmp_rot, rot, bbox2.size());

                        // for narrow images, bbox might be less tall or wide than img
                        int start_x = std::max(0, (bbox.width - img.cols) / 2);
                        int start_y = std::max(0, (bbox.height - img.rows) / 2);
                        tmp_gradiant = tmp_rot(Rect(start_x, start_y, img.cols, img.rows));

                        threshold(tmp_gradiant, tmp_gradiant_pos, 0, 0, THRESH_TOZERO);
                        threshold(tmp_gradiant, tmp_gradiant_neg, 0, 0, THRESH_TOZERO_INV);

                        img_gaussians.push_back(tmp_gradiant_pos.clone());
                        img_gaussians.push_back(tmp_gradiant_neg.clone());

                        // Y, rot
                        Scharr(img_plane_rotated, tmp_gradiant, CV_32F, 0, 1);

                        center = Point((int)(img_plane_rotated.cols / 2.0), (int)(img_plane_rotated.rows / 2.0));
                        rot = cv::getRotationMatrix2D(center, -45.0, 1.0);
                        bbox2 = cv::RotatedRect(center, img_plane_rotated.size(), -45.0).boundingRect();\
                        warpAffine(tmp_gradiant, tmp_rot, rot, bbox2.size());

                        start_x = std::max(0, (bbox.width - img.cols) / 2);
                        start_y = std::max(0, (bbox.height - img.rows) / 2);
                        tmp_gradiant = tmp_rot(Rect(start_x, start_y, img.cols, img.rows));

                        threshold(tmp_gradiant, tmp_gradiant_pos, 0, 0, THRESH_TOZERO);
                        threshold(tmp_gradiant, tmp_gradiant_neg, 0, 0, THRESH_TOZERO_INV);

                        img_gaussians.push_back(tmp_gradiant_pos.clone());
                        img_gaussians.push_back(tmp_gradiant_neg.clone());

                    }

                    // Normalisze gaussiaans in 0-255 range (for faster computation of histograms)
                    for (int i = 0; i < img.channels() * 8; i++) {

                        double hmin, hmax;
                        minMaxLoc(img_gaussians[i], &hmin, &hmax);

                        Mat tmp;
                        img_gaussians[i].convertTo(tmp, CV_8U, (range[1] - 1) / (hmax - hmin), -(range[1] - 1) * hmin / (hmax - hmin));
                        img_gaussians[i] = tmp;

                    }

                    // We compute histograms manualy, directly addings bins based on the region instead of computing multiple histograms
                    // This speedup significantly computations

                    std::vector<int> totals;
                    totals.resize(nb_segs);

                    // Bins for histograms
                    Mat_<int> tmp_histograms = Mat_<int>::zeros(nb_segs, histogram_size);

                    int* regions_data = (int*)regions.data;

                    for (unsigned int x = 0; x < regions.total(); x++) {
                        int region = regions_data[x];

                        int* histogram = tmp_histograms.ptr<int>(region);

                        for (int p = 0; p < img.channels(); p++) {
                            for (unsigned int i = 0; i < 8; i++) {

                                int val = (int)((unsigned char*)img_gaussians[p * 8 + i].data)[x];

                                int bin = (int)((float)val / (range[1] / histogram_bins_size));

                                histogram[(p * 8 + i) * histogram_bins_size + bin]++;
                                totals[region]++;
                            }
                        }
                    }

                    // Normalisation per segments
                    for (int r = 0; r < nb_segs; r++) {

                        float* histogram = histograms.ptr<float>(r);
                        int* tmp_histogram = tmp_histograms.ptr<int>(r);

                        for (int h_pos2 = 0; h_pos2 < histogram_size; h_pos2++) {
                            histogram[h_pos2] = (float)tmp_histogram[h_pos2] / (float)totals[r];
                        }
                    }

                    if (image_id != -1) { // Save cache if it's apply
                        last_histograms = histograms.clone();
                        last_image_id = image_id;
                    }
                } else { // image_id == last_image_id
                    histograms = last_histograms.clone(); // Use cache
                }
            }

            float SelectiveSearchSegmentationStrategyTextureImpl::get(int r1, int r2) {

                float r = 0;
                float* h1 = histograms.ptr<float>(r1);
                float* h2 = histograms.ptr<float>(r2);

                for (int i = 0; i < histogram_size; i++) {
                    r += min(h1[i], h2[i]);
                }

                return r;
            }

            void SelectiveSearchSegmentationStrategyTextureImpl::merge(int r1, int r2) {
                int size_r1 = sizes.at<int>(r1);
                int size_r2 = sizes.at<int>(r2);

                float* h1 = histograms.ptr<float>(r1);
                float* h2 = histograms.ptr<float>(r2);

                for (int i = 0; i < histogram_size; i++) {
                    h1[i] = (h1[i] * size_r1 + h2[i] * size_r2) / (size_r1 + size_r2);
                    h2[i] = h1[i];
                }
            }


            Ptr<SelectiveSearchSegmentationStrategyTexture> createSelectiveSearchSegmentationStrategyTexture() {
                Ptr<SelectiveSearchSegmentationStrategyTexture> s = makePtr<SelectiveSearchSegmentationStrategyTextureImpl>();
                return s;
            }

            // Core

            class SelectiveSearchSegmentationImpl CV_FINAL : public SelectiveSearchSegmentation {
                public:
                    SelectiveSearchSegmentationImpl() {
                        name_ = "SelectiveSearchSegmentation";
                    }

                    ~SelectiveSearchSegmentationImpl() CV_OVERRIDE {
                    };

                    virtual void write(FileStorage& fs) const CV_OVERRIDE {
                        fs << "name" << name_;
                    }

                    virtual void read(const FileNode& fn) CV_OVERRIDE {
                        CV_Assert( (String)fn["name"] == name_);
                    }

                    virtual void setBaseImage(InputArray img) CV_OVERRIDE;

                    virtual void switchToSingleStrategy(int k = 200, float sigma = 0.8) CV_OVERRIDE;
                    virtual void switchToSelectiveSearchFast(int base_k = 150, int inc_k = 150, float sigma = 0.8) CV_OVERRIDE;
                    virtual void switchToSelectiveSearchQuality(int base_k = 150, int inc_k = 150, float sigma = 0.8) CV_OVERRIDE;

                    virtual void addImage(InputArray img) CV_OVERRIDE;
                    virtual void clearImages() CV_OVERRIDE;

                    virtual void addGraphSegmentation(Ptr<GraphSegmentation> g) CV_OVERRIDE;
                    virtual void clearGraphSegmentations() CV_OVERRIDE;

                    virtual void addStrategy(Ptr<SelectiveSearchSegmentationStrategy> s) CV_OVERRIDE;
                    virtual void clearStrategies() CV_OVERRIDE;

                    virtual void process(std::vector<Rect>& rects) CV_OVERRIDE;


                private:
                    String name_;

                    Mat base_image;
                    std::vector<Mat> images;
                    std::vector<Ptr<GraphSegmentation> > segmentations;
                    std::vector<Ptr<SelectiveSearchSegmentationStrategy> > strategies;

                    void hierarchicalGrouping(const Mat& img, Ptr<SelectiveSearchSegmentationStrategy>& s, const Mat& img_regions, const Mat_<char>& is_neighbour, const Mat_<int>& sizes, int& nb_segs, const std::vector<Rect>& bounding_rects, std::vector<Region>& regions, int region_id);
            };

            void SelectiveSearchSegmentationImpl::setBaseImage(InputArray img) {
                base_image = img.getMat();
            }

            void SelectiveSearchSegmentationImpl::addImage(InputArray img) {
                images.push_back(img.getMat());
            }

            void SelectiveSearchSegmentationImpl::clearImages() {
                images.clear();
            }

            void SelectiveSearchSegmentationImpl::addGraphSegmentation(Ptr<GraphSegmentation> g) {
                segmentations.push_back(g);
            }

            void SelectiveSearchSegmentationImpl::clearGraphSegmentations() {
                segmentations.clear();
            }

            void SelectiveSearchSegmentationImpl::addStrategy(Ptr<SelectiveSearchSegmentationStrategy> s) {
                strategies.push_back(s);
            }

            void SelectiveSearchSegmentationImpl::clearStrategies() {
                strategies.clear();
            }

            void SelectiveSearchSegmentationImpl::switchToSingleStrategy(int k, float sigma) {
                clearImages();
                clearGraphSegmentations();
                clearStrategies();

                Mat hsv;
                cvtColor(base_image, hsv, COLOR_BGR2HSV);
                addImage(hsv);

                Ptr<GraphSegmentation> gs = createGraphSegmentation();
                gs->setK((float)k);
                gs->setSigma(sigma);
                addGraphSegmentation(gs);

                Ptr<SelectiveSearchSegmentationStrategyColor> color = createSelectiveSearchSegmentationStrategyColor();
                Ptr<SelectiveSearchSegmentationStrategyFill> fill = createSelectiveSearchSegmentationStrategyFill();
                Ptr<SelectiveSearchSegmentationStrategyTexture> texture = createSelectiveSearchSegmentationStrategyTexture();
                Ptr<SelectiveSearchSegmentationStrategySize> size = createSelectiveSearchSegmentationStrategySize();

                Ptr<SelectiveSearchSegmentationStrategyMultiple> m = createSelectiveSearchSegmentationStrategyMultiple(color, fill, texture, size);

                addStrategy(m);

            }

            void SelectiveSearchSegmentationImpl::switchToSelectiveSearchFast(int base_k, int inc_k, float sigma) {
                clearImages();
                clearGraphSegmentations();
                clearStrategies();

                Mat hsv;
                cvtColor(base_image, hsv, COLOR_BGR2HSV);
                addImage(hsv);
                Mat lab;
                cvtColor(base_image, lab, COLOR_BGR2Lab);
                addImage(lab);

                for (int k = base_k; k <= base_k + inc_k * 2; k+= inc_k) {
                    Ptr<GraphSegmentation> gs = createGraphSegmentation();
                    gs->setK((float)k);
                    gs->setSigma(sigma);
                    addGraphSegmentation(gs);
                }

                Ptr<SelectiveSearchSegmentationStrategyColor> color = createSelectiveSearchSegmentationStrategyColor();
                Ptr<SelectiveSearchSegmentationStrategyFill> fill = createSelectiveSearchSegmentationStrategyFill();
                Ptr<SelectiveSearchSegmentationStrategyTexture> texture = createSelectiveSearchSegmentationStrategyTexture();
                Ptr<SelectiveSearchSegmentationStrategySize> size = createSelectiveSearchSegmentationStrategySize();

                Ptr<SelectiveSearchSegmentationStrategyMultiple> m = createSelectiveSearchSegmentationStrategyMultiple(color, fill, texture, size);

                addStrategy(m);

                Ptr<SelectiveSearchSegmentationStrategyFill> fill2 = createSelectiveSearchSegmentationStrategyFill();
                Ptr<SelectiveSearchSegmentationStrategyTexture> texture2 = createSelectiveSearchSegmentationStrategyTexture();
                Ptr<SelectiveSearchSegmentationStrategySize> size2 = createSelectiveSearchSegmentationStrategySize();

                Ptr<SelectiveSearchSegmentationStrategyMultiple> m2 = createSelectiveSearchSegmentationStrategyMultiple(fill2, texture2, size2);

                addStrategy(m2);

            }

            void SelectiveSearchSegmentationImpl::switchToSelectiveSearchQuality(int base_k, int inc_k, float sigma) {
                clearImages();
                clearGraphSegmentations();
                clearStrategies();


                Mat hsv;
                cvtColor(base_image, hsv, COLOR_BGR2HSV);
                addImage(hsv);
                Mat lab;
                cvtColor(base_image, lab, COLOR_BGR2Lab);
                addImage(lab);

                Mat I;
                cvtColor(base_image, I, COLOR_BGR2GRAY);
                addImage(I);

                Mat channel[3];
                split(hsv, channel);
                addImage(channel[0]);

                split(base_image, channel);
                std::vector<Mat> channel2 = {channel[2], channel[1], I};

                Mat rgI;
                merge(channel2, rgI);
                addImage(rgI);

                for (int k = base_k; k <= base_k + inc_k * 4; k+= inc_k) {
                    Ptr<GraphSegmentation> gs = createGraphSegmentation();
                    gs->setK((float)k);
                    gs->setSigma(sigma);
                    addGraphSegmentation(gs);
                }

                Ptr<SelectiveSearchSegmentationStrategyColor> color = createSelectiveSearchSegmentationStrategyColor();
                Ptr<SelectiveSearchSegmentationStrategyFill> fill = createSelectiveSearchSegmentationStrategyFill();
                Ptr<SelectiveSearchSegmentationStrategyTexture> texture = createSelectiveSearchSegmentationStrategyTexture();
                Ptr<SelectiveSearchSegmentationStrategySize> size = createSelectiveSearchSegmentationStrategySize();

                Ptr<SelectiveSearchSegmentationStrategyMultiple> m = createSelectiveSearchSegmentationStrategyMultiple(color, fill, texture, size);

                addStrategy(m);

                Ptr<SelectiveSearchSegmentationStrategyFill> fill2 = createSelectiveSearchSegmentationStrategyFill();
                Ptr<SelectiveSearchSegmentationStrategyTexture> texture2 = createSelectiveSearchSegmentationStrategyTexture();
                Ptr<SelectiveSearchSegmentationStrategySize> size2 = createSelectiveSearchSegmentationStrategySize();

                Ptr<SelectiveSearchSegmentationStrategyMultiple> m2 = createSelectiveSearchSegmentationStrategyMultiple(fill2, texture2, size2);

                addStrategy(m2);

                Ptr<SelectiveSearchSegmentationStrategyFill> fill3 = createSelectiveSearchSegmentationStrategyFill();
                addStrategy(fill3);

                Ptr<SelectiveSearchSegmentationStrategySize> size3 = createSelectiveSearchSegmentationStrategySize();
                addStrategy(size3);
            }

            void SelectiveSearchSegmentationImpl::process(std::vector<Rect>& rects) {

                std::vector<Region> all_regions;

                int image_id = 0;

                for(std::vector<Mat>::iterator image = images.begin(); image != images.end(); ++image) {
                    for(std::vector<Ptr<GraphSegmentation> >::iterator gs = segmentations.begin(); gs != segmentations.end(); ++gs) {

                        Mat img_regions;
                        Mat_<char> is_neighbour;
                        Mat_<int> sizes;

                        // Compute initial segmentation
                        (*gs)->processImage(*image, img_regions);

                        // Get number of regions
                        double min, max;
                        minMaxLoc(img_regions, &min, &max);
                        int nb_segs = (int)max + 1;

                        // Compute bouding rects and neighbours
                        std::vector<Rect> bounding_rects;
                        bounding_rects.resize(nb_segs);

                        std::vector<std::vector<cv::Point> > points;

                        points.resize(nb_segs);

                        is_neighbour = Mat::zeros(nb_segs, nb_segs, CV_8UC1);
                        sizes = Mat::zeros(nb_segs, 1, CV_32SC1);

                        const int* previous_p = NULL;

                        for (int i = 0; i < (int)img_regions.rows; i++) {
                            const int* p = img_regions.ptr<int>(i);

                            for (int j = 0; j < (int)img_regions.cols; j++) {

                                points[p[j]].push_back(cv::Point(j, i));
                                sizes.at<int>(p[j], 0) = sizes.at<int>(p[j], 0) + 1;

                                if (i > 0 && j > 0) {

                                    is_neighbour.at<char>(p[j], p[j - 1]) = 1;
                                    is_neighbour.at<char>(p[j], previous_p[j]) = 1;
                                    is_neighbour.at<char>(p[j], previous_p[j - 1]) = 1;

                                    is_neighbour.at<char>(p[j - 1], p[j]) = 1;
                                    is_neighbour.at<char>(previous_p[j], p[j]) = 1;
                                    is_neighbour.at<char>(previous_p[j - 1], p[j]) = 1;
                                }
                            }
                            previous_p = p;
                        }

                        for(int seg = 0; seg < nb_segs; seg++) {
                            bounding_rects[seg] = cv::boundingRect(points[seg]);
                        }

                        for(std::vector<Ptr<SelectiveSearchSegmentationStrategy> >::iterator strategy = strategies.begin(); strategy != strategies.end(); ++strategy) {
                            std::vector<Region> regions;
                            hierarchicalGrouping(*image, *strategy, img_regions, is_neighbour, sizes, nb_segs, bounding_rects, regions, image_id);

                            for(std::vector<Region>::iterator region = regions.begin(); region != regions.end(); ++region) {
                                all_regions.push_back(*region);
                            }
                        }

                        image_id++;
                    }
                }

                std::sort(all_regions.begin(), all_regions.end());

                std::map<Rect, char, rectComparator> processed_rect;

                rects.clear();

                // Remove duplicate in rect list
                for(std::vector<Region>::iterator region = all_regions.begin(); region != all_regions.end(); ++region) {
                    if (processed_rect.find((*region).bounding_box) == processed_rect.end()) {
                        processed_rect[(*region).bounding_box] = true;
                        rects.push_back((*region).bounding_box);
                    }
                }

            }

            void SelectiveSearchSegmentationImpl::hierarchicalGrouping(const Mat& img, Ptr<SelectiveSearchSegmentationStrategy>& s, const Mat& img_regions, const Mat_<char>& is_neighbour, const Mat_<int>& sizes_, int& nb_segs, const std::vector<Rect>& bounding_rects, std::vector<Region>& regions, int image_id) {

                Mat sizes = sizes_.clone();

                std::vector<Neighbour> similarities;
                regions.clear();

                /////////////////////////////////////////

                s->setImage(img, img_regions, sizes, image_id);

                // Compute initial similarities
                for (int i = 0; i < nb_segs; i++) {
                    Region r;

                    r.id = i;
                    r.level = 1;
                    r.merged_to = -1;
                    r.bounding_box = bounding_rects[i];

                    regions.push_back(r);

                    for (int j = i + 1; j < nb_segs; j++) {
                        if (is_neighbour.at<char>(i, j)) {
                            Neighbour n;
                            n.from = i;
                            n.to = j;
                            n.similarity = s->get(i, j);

                            similarities.push_back(n);
                        }
                    }
                }

                while(similarities.size() > 0) {

                    std::sort(similarities.begin(), similarities.end());

                    // for(std::vector<Neighbour>::iterator similarity = similarities.begin(); similarity != similarities.end(); ++similarity) {
                    //     std::cout << *similarity << std::endl;
                    // }

                    Neighbour p = similarities.back();
                    similarities.pop_back();

                    Region region_from = regions[p.from];
                    Region region_to = regions[p.to];

                    Region new_r;
                    new_r.id = std::min(region_from.id, region_to.id); // Should be the smalest, working ID
                    new_r.level = std::max(region_from.level, region_to.level) + 1;
                    new_r.merged_to = -1;
                    new_r.bounding_box = region_from.bounding_box | region_to.bounding_box;

                    regions.push_back(new_r);

                    regions[p.from].merged_to = (int)regions.size() - 1;
                    regions[p.to].merged_to = (int)regions.size() - 1;

                    // Merge
                    s->merge(region_from.id, region_to.id);

                    // Update size
                    sizes.at<int>(region_from.id, 0) += sizes.at<int>(region_to.id, 0);
                    sizes.at<int>(region_to.id, 0) = sizes.at<int>(region_from.id, 0);

                    std::vector<int> local_neighbours;

                    for(std::vector<Neighbour>::iterator similarity = similarities.begin(); similarity != similarities.end();) {
                        if ((*similarity).from == p.from || (*similarity).to == p.from || (*similarity).from == p.to || (*similarity).to == p.to) {
                            int from = 0;

                            if ((*similarity).from == p.from || (*similarity).from == p.to) {
                                from = (*similarity).to;
                            } else {
                                from = (*similarity).from;
                            }

                            bool already_neighboor = false;

                            for(std::vector<int>::iterator local_neighbour = local_neighbours.begin(); local_neighbour != local_neighbours.end(); local_neighbour++) {
                                if (*local_neighbour == from) {
                                    already_neighboor = true;
                                }
                            }

                            if (!already_neighboor) {
                                local_neighbours.push_back(from);
                            }

                            similarity = similarities.erase(similarity);
                        } else {
                            similarity++;
                        }
                    }

                    for(std::vector<int>::iterator local_neighbour = local_neighbours.begin(); local_neighbour != local_neighbours.end(); local_neighbour++) {

                        Neighbour n;
                        n.from = (int)regions.size() - 1;
                        n.to = *local_neighbour;
                        n.similarity = s->get(regions[n.from].id, regions[n.to].id);

                        similarities.push_back(n);
                    }
                }

                // Compute regions' rank
                for(std::vector<Region>::iterator region = regions.begin(); region != regions.end(); ++region) {
                    // Note: this is inverted from the paper, but we keep the lover region first so it's works
                    (*region).rank = ((double) rand() / (RAND_MAX)) * ((*region).level);
                }

            }

            Ptr<SelectiveSearchSegmentation> createSelectiveSearchSegmentation() {
                Ptr<SelectiveSearchSegmentation> s = makePtr<SelectiveSearchSegmentationImpl>();
                return s;
            }

            std::ostream& operator<<(std::ostream& os, const Neighbour& n) {
                os << "Neighbour[" << n.from << "->" << n.to << "," << n.similarity << "]";
                return os;
            }

            std::ostream& operator<<(std::ostream& os, const Region& r) {
                os << "Region[WID" << r.id << ", L" << r.level << ", merged to " << r.merged_to << ", R:" << r.rank << ", " << r.bounding_box << "]";
                return os;
            }
        }
    }
}
