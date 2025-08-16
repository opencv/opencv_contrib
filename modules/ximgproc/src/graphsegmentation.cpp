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

/******************************************************************************\
*                        Graph based segmentation                             *
* This code implements the segmentation method described in:                  *
* P. Felzenszwalb, D. Huttenlocher: "Graph-Based Image Segmentation"          *
* International Journal of Computer Vision, Vol. 59, No. 2, September 2004    *
*                                                                             *
* Author: Maximilien Cuony / LTS2 / EPFL / 2015                               *
*******************************************************************************/

#include "precomp.hpp"
#include "opencv2/ximgproc/segmentation.hpp"

#include <iostream>

namespace cv {
    namespace ximgproc {
        namespace segmentation {

            // Helpers

            // Represent an edge between two pixels
            class Edge {
                public:
                    int from;
                    int to;
                    float weight;

                    bool operator <(const Edge& e) const {
                        return weight < e.weight;
                    }
            };

            // A point in the sets of points
            class PointSetElement {
                public:
                    int p;
                    int size;

                    PointSetElement() { }

                    PointSetElement(int p_) {
                        p = p_;
                        size = 1;
                    }
            };

            // An object to manage set of points, who can be fusionned
            class PointSet {
                public:
                    PointSet(int nb_elements_);
                    ~PointSet();

                    int nb_elements;

                    // Return the main point of the point's set
                    int getBasePoint(int p);

                    // Join two sets of points, based on their main point
                    void joinPoints(int p_a, int p_b);

                    // Return the set size of a set (based on the main point)
                    int size(unsigned int p) { return mapping[p].size; }

                private:
                    PointSetElement* mapping;

            };

            class GraphSegmentationImpl : public GraphSegmentation {
                public:
                    GraphSegmentationImpl() {
                        sigma = 0.5;
                        k = 300;
                        min_size = 100;
                        name_ = "GraphSegmentation";
                    }

                    ~GraphSegmentationImpl() CV_OVERRIDE {
                    };

                    virtual void processImage(InputArray src, OutputArray dst) CV_OVERRIDE;

                    virtual void setSigma(double sigma_) CV_OVERRIDE { if (sigma_ <= 0) { sigma_ = 0.001; } sigma = sigma_; }
                    virtual double getSigma() CV_OVERRIDE { return sigma; }

                    virtual void setK(float k_) CV_OVERRIDE { k = k_; }
                    virtual float getK() CV_OVERRIDE { return k; }

                    virtual void setMinSize(int min_size_) CV_OVERRIDE { min_size = min_size_; }
                    virtual int getMinSize() CV_OVERRIDE { return min_size; }

                    virtual void write(FileStorage& fs) const CV_OVERRIDE {
                        fs << "name" << name_
                        << "sigma" << sigma
                        << "k" << k
                        << "min_size" << (int)min_size;
                    }

                    virtual void read(const FileNode& fn) CV_OVERRIDE {
                        CV_Assert( (String)fn["name"] == name_ );

                        sigma = (double)fn["sigma"];
                        k = (float)fn["k"];
                        min_size = (int)(int)fn["min_size"];
                    }

                private:
                    double sigma;
                    float k;
                    int min_size;
                    String name_;

                    // Pre-filter the image
                    void filter(const Mat &img, Mat &img_filtered);

                    // Build the graph between each pixels
                    void buildGraph(Edge **edges, int &nb_edges, const Mat &img_filtered);

                    // Segment the graph
                    void segmentGraph(Edge * edges, const int &nb_edges, const Mat & img_filtered, PointSet **es);

                    // Remove areas too small
                    void filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es);

                    // Map the segemented graph to a Mat with uniques, sequentials ids
                    void finalMapping(PointSet *es, Mat &output);
            };

            void GraphSegmentationImpl::filter(const Mat &img, Mat &img_filtered) {

                Mat img_converted;

                // Switch to [0,255] float
                double alpha = 1.0;

                const int depth = CV_MAT_DEPTH( img.type() );
                switch( depth )
                {
                    case CV_8U:  alpha = 1.;                  break;
                    case CV_16U: alpha = 255. / 65535. ;      break;
                    case CV_8S:  alpha = 255. / 127. ;        break;
                    case CV_16S: alpha = 255. / 32767. ;      break;
                    case CV_32F: alpha = 255. ;               break; // [0,1]->[0,255]
                    case CV_64F: alpha = 255. ;               break; // [0,1]->[0,255]
                    default: CV_Error(Error::StsBadArg,"Unsupported Mat type"); break;
                }

                // Check Negative Pixel Value.
                if(
                    ( depth != CV_8U ) && ( depth != CV_16U )
                )
                {
                    double min;
                    minMaxLoc( img, &min );
                    if ( min < 0.0 )
                    {
                        CV_Error(Error::StsBadArg,"Negative Pixel Valuea are contained.");
                    }
                }

#if 0
                // In test code, CV_16S results were slightly different from others.
                // Following code is workaround.
                if ( depth == CV_16S )
                {
                    Mat tmp;
                    img.convertTo(tmp, CV_8U, alpha );
                    tmp.convertTo(img_converted, CV_32F );
                }
                else
#endif
                img.convertTo(img_converted, CV_32F, alpha );

                // Apply gaussian filter
                GaussianBlur(img_converted, img_filtered, Size(0, 0), sigma, sigma);
            }

            void GraphSegmentationImpl::buildGraph(Edge **edges, int &nb_edges, const Mat &img_filtered) {

                *edges = new Edge[img_filtered.rows * img_filtered.cols * 4];

                nb_edges = 0;

                int nb_channels = img_filtered.channels();

                for (int i = 0; i < (int)img_filtered.rows; i++) {
                    const float* p = img_filtered.ptr<float>(i);

                    for (int j = 0; j < (int)img_filtered.cols; j++) {

                        //Take the right, left, top and down pixel
                        for (int delta = -1; delta <= 1; delta += 2) {
                            for (int delta_j = 0, delta_i = 1; delta_j <= 1; delta_j++ || delta_i--) {

                                int i2 = i + delta * delta_i;
                                int j2 = j + delta * delta_j;

                                if (i2 >= 0 && i2 < img_filtered.rows && j2 >= 0 && j2 < img_filtered.cols) {
                                    const float* p2 = img_filtered.ptr<float>(i2);

                                    float tmp_total = 0;

                                    for ( int channel = 0; channel < nb_channels; channel++) {
                                        float tmp_diff = p[j * nb_channels + channel] - p2[j2 * nb_channels + channel];
                                        tmp_total += tmp_diff * tmp_diff;
                                    }

                                    float diff = 0;
                                    diff = sqrt(tmp_total);

                                    (*edges)[nb_edges].weight = diff;
                                    (*edges)[nb_edges].from = i * img_filtered.cols +  j;
                                    (*edges)[nb_edges].to = i2 * img_filtered.cols + j2;

                                    nb_edges++;
                                }
                            }
                        }
                    }
                }
            }

            void GraphSegmentationImpl::segmentGraph(Edge *edges, const int &nb_edges, const Mat &img_filtered, PointSet **es) {

                int total_points = ( int)(img_filtered.rows * img_filtered.cols);

                // Sort edges
                std::sort(edges, edges + nb_edges);

                // Create a set with all point (by default mapped to themselves)
                *es = new PointSet(img_filtered.cols * img_filtered.rows);

                // Thresholds
                float* thresholds = new float[total_points];

                for (int i = 0; i < total_points; i++)
                    thresholds[i] = k;

                for ( int i = 0; i < nb_edges; i++) {

                    int p_a = (*es)->getBasePoint(edges[i].from);
                    int p_b = (*es)->getBasePoint(edges[i].to);

                    if (p_a != p_b) {
                        if (edges[i].weight <= thresholds[p_a] && edges[i].weight <= thresholds[p_b]) {
                            (*es)->joinPoints(p_a, p_b);
                            p_a = (*es)->getBasePoint(p_a);
                            thresholds[p_a] = edges[i].weight + k / (*es)->size(p_a);

                            edges[i].weight = 0;
                        }
                    }
                }

                delete [] thresholds;
            }

            void GraphSegmentationImpl::filterSmallAreas(Edge *edges, const int &nb_edges, PointSet *es) {

                for ( int i = 0; i < nb_edges; i++) {

                    if (edges[i].weight > 0) {

                        int p_a = es->getBasePoint(edges[i].from);
                        int p_b = es->getBasePoint(edges[i].to);

                        if (p_a != p_b && (es->size(p_a) < min_size || es->size(p_b) < min_size)) {
                            es->joinPoints(p_a, p_b);

                        }
                    }
                }

            }

            void GraphSegmentationImpl::finalMapping(PointSet *es, Mat &output) {

                int maximum_size = ( int)(output.rows * output.cols);

                int last_id = 0;
                int * mapped_id = new int[maximum_size];

                for ( int i = 0; i < maximum_size; i++)
                    mapped_id[i] = -1;

                int rows = output.rows;
                int cols = output.cols;

                if (output.isContinuous()) {
                    cols *= rows;
                    rows = 1;
                }

                for (int i = 0; i < rows; i++) {

                    int* p = output.ptr<int>(i);

                    for (int j = 0; j < cols; j++) {

                        int point = es->getBasePoint(i * cols + j);

                        if (mapped_id[point] == -1) {
                            mapped_id[point] = last_id;
                            last_id++;
                        }

                        p[j] = mapped_id[point];
                    }
                }

                delete [] mapped_id;
            }

            void GraphSegmentationImpl::processImage(InputArray src, OutputArray dst) {

                Mat img = src.getMat();

                dst.create(img.rows, img.cols, CV_32SC1);
                Mat output = dst.getMat();
                output.setTo(0);

                // Filter graph
                Mat img_filtered;
                filter(img, img_filtered);

                // Build graph
                Edge *edges;
                int nb_edges;

                buildGraph(&edges, nb_edges, img_filtered);

                // Segment graph
                PointSet *es;

                segmentGraph(edges, nb_edges, img_filtered, &es);

                // Remove small areas
                filterSmallAreas(edges, nb_edges, es);

                // Map to final output
                finalMapping(es, output);

                delete [] edges;
                delete es;

            }

            Ptr<GraphSegmentation> createGraphSegmentation(double sigma, float k, int min_size) {

                Ptr<GraphSegmentation> graphseg = makePtr<GraphSegmentationImpl>();

                graphseg->setSigma(sigma);
                graphseg->setK(k);
                graphseg->setMinSize(min_size);

                return graphseg;
            }

            PointSet::PointSet(int nb_elements_) {
                nb_elements = nb_elements_;

                mapping = new PointSetElement[nb_elements];

                for ( int i = 0; i < nb_elements; i++) {
                    mapping[i] = PointSetElement(i);
                }
            }

            PointSet::~PointSet() {
                delete [] mapping;
            }

            int PointSet::getBasePoint( int p) {

                 int base_p = p;

                while (base_p != mapping[base_p].p) {
                    base_p = mapping[base_p].p;
                }

                // Save mapping for faster acces later
                mapping[p].p = base_p;

                return base_p;
            }

            void PointSet::joinPoints(int p_a, int p_b) {

                // Always target smaller set, to avoid redirection in getBasePoint
                if (mapping[p_a].size < mapping[p_b].size)
                    swap(p_a, p_b);

                mapping[p_b].p = p_a;
                mapping[p_a].size += mapping[p_b].size;

                nb_elements--;
            }

        }
    }
}
