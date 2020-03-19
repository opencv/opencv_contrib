// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "precomp.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <cassert>
#include <cmath>
#include <math.h>
#include <utility>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <vector>
using namespace std;


#define PI 3.14159265

namespace cv {
namespace text {
    struct SWTPoint {
        int x;
        int y;
        float SWT;
    };

    struct Ray {
        SWTPoint p;
        SWTPoint q;
        std::vector<SWTPoint> points;
    };

    struct Component {
        SWTPoint BB_pointP;
        SWTPoint BB_pointQ;
        float cx;
        float cy;
        float median;
        float mean;
        int length, width;
        std::vector<SWTPoint> points;
    };

    struct ComponentAttr {
        float mean, variance, median;
        int xmin, ymin;
        int xmax, ymax;
        float length, width;
    };

    struct ChannelAverage {
        float Red, Green, Blue;
    };

    struct Direction {
        float x, y;
    };

    struct ChainedComponent {
        int chainIndexA;
        int chainIndexB;
        std::vector<int> componentIndices;
        float chainDist;
        Direction dir;
        bool merged;
    };

    const Scalar BLUE (255, 0, 0);
    const Scalar GREEN(0, 255, 0);
    const Scalar RED  (0, 0, 255);

    // A utility function to add an edge in an 
    // undirected graph. 
    void addEdge(std::vector<int> adj[], int u, int v) { 
        adj[u].push_back(v); 
        adj[v].push_back(u); 
    }

    void DFSUtil(int v, bool visited[], std::vector<int> adj[], int label, std::vector<int> &component_id) 
    { 
        // Mark the current node as visited and label it as belonging to the current component
        visited[v] = true; 
        component_id[v] = label;
        // Recur for all the vertices 
        // adjacent to this vertex 
        for (int i = 0; i < adj[v].size(); i++) {
            int neighbour = adj[v][i];
            if (!visited[neighbour]) {
                DFSUtil(neighbour, visited, adj, label, component_id); 
            }
        }
    } 

    int connected_components(std::vector<int> adj[], std::vector<int> &component_id, int num_vertices) {
        bool *visited = new bool[num_vertices]; 
        for(int v = 0; v < num_vertices; v++) 
            visited[v] = false; 
        int label = 0;
        for (int v=0; v<num_vertices; v++) 
        { 
            if (visited[v] == false) 
            { 
                DFSUtil(v, visited, adj, label, component_id); 
                label++;
            } 
        } 

        delete [] visited;
        return label;
    }

    void SWTFirstPass (Mat edgeImage, Mat gradientX, Mat gradientY, bool dark_on_light, Mat & SWTImage, std::vector<Ray> & rays) {
        for( int row = 0; row < SWTImage.rows; row++ ){
            float* ptr = (float*)SWTImage.ptr(row);
            for ( int col = 0; col < SWTImage.cols; col++ ){
                *ptr++ = -1;
            }
        }
        

        for( int row = 0; row < edgeImage.rows; row++ ){
            for ( int col = 0; col < edgeImage.cols; col++ ){
                uchar canny = edgeImage.at<uchar>(row, col);
                if (canny <= 0) continue;

                float dx = gradientX.at<float>(row, col);
                float dy = gradientY.at<float>(row, col);
                float mag = sqrt(dx * dx + dy * dy);
                dx = dx / mag;
                dy = dy / mag;

                if (dark_on_light){
                    dx = -dx;
                    dy = -dy;
                } 

                Ray ray;
                SWTPoint p;
                p.x = col;
                p.y = row;
                ray.p = p;
                std::vector<SWTPoint> points;
                points.push_back(p);
                float curPosX = (float) col + 0.5;
                float curPosY = (float) row + 0.5;
                int curPixX = col;
                int curPixY = row;
                float inc = 0.05;
                while (true) {
                    curPosX += inc * dx;
                    curPosY += inc * dy;
                    if ((int)(floor(curPosX)) != curPixX || (int)(floor(curPosY)) != curPixY) {
                        curPixX = (int)(floor(curPosX));
                        curPixY = (int)(floor(curPosY));
                        if (curPixX < 0 || (curPixX >= SWTImage.cols) || curPixY < 0 || (curPixY >= SWTImage.rows)) {
                            break;
                        }
                        SWTPoint pt;
                        pt.x = curPixX;
                        pt.y = curPixY;
                        points.push_back(pt);
                        if (edgeImage.at<uchar>(curPixY, curPixX) > 0) {
                            ray.q = pt;
                            float G_xt = gradientX.at<float>(curPixY,curPixX);
                            float G_yt = gradientY.at<float>(curPixY,curPixX);
                            mag = sqrt( (G_xt * G_xt) + (G_yt * G_yt) );
                            G_xt = G_xt / mag;
                            G_yt = G_yt / mag;
                            if (dark_on_light){
                                G_xt = -G_xt;
                                G_yt = -G_yt;
                            }

                            if (acos(dx * -G_xt + dy * -G_yt) < PI/2.0 ) {
                                float length = sqrt( ((float)ray.q.x - (float)ray.p.x)*((float)ray.q.x - (float)ray.p.x) + ((float)ray.q.y - (float)ray.p.y)*((float)ray.q.y - (float)ray.p.y));
                                for (std::vector<SWTPoint>::iterator pit = points.begin(); pit != points.end(); pit++) {
                                    if (SWTImage.at<float>(pit->y, pit->x) < 0) {
                                        SWTImage.at<float>(pit->y, pit->x) = length;
                                    } else {
                                        SWTImage.at<float>(pit->y, pit->x) = std::min(length, SWTImage.at<float>(pit->y, pit->x));
                                    }
                                }
                                ray.points = points;
                                rays.push_back(ray);
                            }
                            break; 
                        }
                    }
                }
                 
            }
        }
        
    }

    bool sortBySWT (const SWTPoint &lhs, const SWTPoint &rhs) {
        return lhs.SWT < rhs.SWT;
    }


    void SWTSecondPass (Mat & SWTImage, std::vector<Ray> & rays) {
        for (std::vector<Ray>::iterator rit = rays.begin(); rit != rays.end(); rit++) {
            for (std::vector<SWTPoint>::iterator pit = rit->points.begin(); pit != rit->points.end(); pit++) {
                pit->SWT = SWTImage.at<float>(pit->y, pit->x);
            }
            std::sort(rit->points.begin(), rit->points.end(), sortBySWT);
            float median = (rit -> points[rit -> points.size()/2]).SWT;
            for (std::vector<SWTPoint>::iterator pit = rit->points.begin(); pit != rit->points.end(); pit++) {
                SWTImage.at<float>(pit->y, pit->x) = std::min(pit->SWT, median);
            }
        }
    }
    
    void normalizeAndScale (const Mat& SWTImage, Mat& output) {
        assert (SWTImage.depth() == CV_32F );
        assert (SWTImage.channels() == 1);
        assert (output.depth() == CV_8U);
        assert (output.channels() == 1);

        Mat outputTemp(output.size(), CV_32FC1);

        float maxSWT = 0;
        float minSWT = 1e100;
        for(int row = 0; row < SWTImage.rows; row++){
            for (int col = 0; col < SWTImage.cols; col++){
                float val  = SWTImage.at<float>(row, col);
                if (val < 0) continue;
                maxSWT = std::max(val, maxSWT);
                minSWT = std::min(val, minSWT);
            }
        }

        float amplitude = maxSWT - minSWT;
        for(int row = 0; row < SWTImage.rows; row++){
            for (int col = 0; col < SWTImage.cols; col++){
                float val  = SWTImage.at<float>(row, col);
                if (val < 0) {
                    outputTemp.at<float>(row, col) = 1;
                }
                else {
                    outputTemp.at<float>(row, col) = (val - minSWT) / amplitude;
                }
            }
        }
        outputTemp.convertTo(output, CV_8UC1, 255);
    }

    std::vector<std::vector<SWTPoint>> getComponents (Mat& SWTImage, std::vector<Ray> & rays) {
        std::unordered_map<int, int> Pix2Node;
        std::unordered_map<int, SWTPoint> Node2Pix;

        int V = 2;
        
        int num_vertices = 0;

        for(int row = 0; row < SWTImage.rows; row++){
            for (int col = 0; col < SWTImage.cols; col++){
                float val  = SWTImage.at<float>(row, col);
                if (val < 0) {
                    continue;
                }
                else {
                    Pix2Node[row * SWTImage.cols + col] = num_vertices;
                    SWTPoint p;
                    p.x = col;
                    p.y = row;
                    Node2Pix[num_vertices] = p;
                    num_vertices++;
                }
            }
        }
        vector<int>* graph = new vector<int> [num_vertices]; 


        for(int row = 0; row < SWTImage.rows; row++){
            for (int col = 0; col < SWTImage.cols; col++){
                float val  = SWTImage.at<float>(row, col);
                if (val < 0) {
                    continue;
                }
                else {
                    int currentNode = Pix2Node[row * SWTImage.cols + col];
                    if (col+1 < SWTImage.cols) {
                        float right = SWTImage.at<float>(row, col+1);
                        if (right > 0 && (val/right <= 3.0 || right/val <= 3.0))
                            addEdge(graph, currentNode, Pix2Node.at(row * SWTImage.cols + col + 1));
                    }
                    if (row+1 < SWTImage.rows) {
                        if (col+1 < SWTImage.cols) {
                            float right_down = SWTImage.at<float>(row+1, col+1);
                            if (right_down > 0 && (val/right_down <= 3.0 || right_down/val <= 3.0))
                                addEdge(graph, currentNode, Pix2Node.at((row+1) * SWTImage.cols + col + 1));
                        }
                        float down = SWTImage.at<float>(row+1, col);
                        if (down > 0 && (val/down <= 3.0 || down/val <= 3.0))
                            addEdge(graph, currentNode, Pix2Node.at((row+1) * SWTImage.cols + col));
                        if (col-1 >= 0) {
                            float left_down = SWTImage.at<float>(row+1, col-1);
                            if (left_down > 0 && (val/left_down <= 3.0 || left_down/val <= 3.0))
                                addEdge(graph, currentNode, Pix2Node.at((row+1) * SWTImage.cols + col - 1));
                        }
                    }
                }
            }
        }

        std::vector<int> component_id(num_vertices);

        int num_comp = connected_components(graph, component_id, num_vertices);

        std::vector<std::vector<SWTPoint> > components;
        components.reserve(num_comp);

        for (int j = 0; j < num_comp; j++) {
            std::vector<SWTPoint> tmp;
            components.push_back(tmp);
        }
        for (int j = 0; j < num_vertices; j++) {
            SWTPoint p = Node2Pix[j];
            (components[component_id[j]]).push_back(p);
        }
        delete [] graph ;

        return components;
    }

    ComponentAttr getAttributes(vector<SWTPoint> component, Mat& SWTImage) {
        std::vector<float> temp;
        temp.reserve(component.size());
        ComponentAttr attributes;
        attributes.mean = 0;
        attributes.variance = 0;

        attributes.xmin = 100000;
        attributes.ymin = 100000;

        attributes.xmax = 0;
        attributes.ymax = 0;

        float sum = 0;
        
        for (int i = 0; i < component.size(); i++) {
            float val = SWTImage.at<float> (component[i].y, component[i].x);
            sum += val;
            temp.push_back(val);
            attributes.xmin = std::min (attributes.xmin, component[i].x);
            attributes.ymin = std::min (attributes.ymin, component[i].y);
            attributes.xmax = std::max (attributes.xmax, component[i].x);
            attributes.ymax = std::max (attributes.ymax, component[i].y);
        }
        attributes.mean = sum / ((float)component.size());
        for (int i = 0; i < component.size(); i++) {
            attributes.variance += (temp[i] - attributes.mean) * (temp[i] - attributes.mean);
        }
        
        attributes.variance = attributes.variance / ((float)component.size());
        std::sort(temp.begin(),temp.end());
        attributes.median = temp[temp.size()/2];

        attributes.length = (float) (attributes.xmax - attributes.xmin + 1);
        attributes.width = (float) (attributes.ymax - attributes.ymin + 1);
        return attributes;
    }

    void renderComponents (const Mat& SWTImage, std::vector<Component> components, Mat& output) {
        output.setTo(0);

        for (int i = 0; i < components.size(); i++) {
            Component component = components[i];
            for (int j = 0; j < component.points.size(); j++) {
                output.at<float>(component.points[j].y, component.points[j].x) = SWTImage.at<float>(component.points[j].y, component.points[j].x);
            }
        }
        for( int row = 0; row < output.rows; row++ ){
            float* ptr = (float*)output.ptr(row);
            for ( int col = 0; col < output.cols; col++ ){
                if (*ptr == 0) {
                    *ptr = -1;
                }
                ptr++;
            }
        }
        float maxVal = 0;
        float minVal = 1e100;
        for( int row = 0; row < output.rows; row++ ){
            const float* ptr = (const float*)output.ptr(row);
            for ( int col = 0; col < output.cols; col++ ){
                if (*ptr == 0) { }
                else {
                    maxVal = std::max(*ptr, maxVal);
                    minVal = std::min(*ptr, minVal);
                }
                ptr++;
            }
        }
        float difference = maxVal - minVal;
        for( int row = 0; row < output.rows; row++ ) {
            float* ptr = (float*)output.ptr(row);
            for ( int col = 0; col < output.cols; col++ ) {
                if (*ptr < 1) {
                    *ptr = 1;
                } else {
                    *ptr = ((*ptr) - minVal)/difference;
                }
                ptr++;
            }
        }

    }

    std::vector<Component> filterComponents(Mat& SWTImage, std::vector<std::vector<SWTPoint>> components, bool skipChecks){
        std::vector<Component> filteredComponents;
        filteredComponents.reserve(components.size());
        for (int i = 0; i < components.size(); i++) {
            vector<SWTPoint> component = components[i];
            ComponentAttr attributes = getAttributes(component, SWTImage);  
            if (!skipChecks && attributes.variance > 0.5 * attributes.mean) continue;
            if (!skipChecks && attributes.width > 300) continue;
            

            float area = attributes.length * attributes.width;
            float rminx = (float) attributes.xmin;
            float rmaxx = (float) attributes.xmax;
            float rminy = (float) attributes.ymin;
            float rmaxy = (float) attributes.ymax;
            // compute the rotated bounding box
            float increment = 1./36.;
            for (float theta = increment * PI; theta<PI/2.0; theta += increment * PI) {
                float xmin,xmax,ymin,ymax,xtemp,ytemp,ltemp,wtemp;
                    xmin = 1000000;
                    ymin = 1000000;
                    xmax = 0;
                    ymax = 0;
                for (unsigned int j = 0; j < components[i].size(); j++) {
                    xtemp = components[i][j].x * cos(theta) + components[i][j].y * -sin(theta);
                    ytemp = components[i][j].x * sin(theta) + components[i][j].y * cos(theta);
                    xmin = std::min(xtemp,xmin);
                    xmax = std::max(xtemp,xmax);
                    ymin = std::min(ytemp,ymin);
                    ymax = std::max(ytemp,ymax);
                }
                ltemp = xmax - xmin + 1;
                wtemp = ymax - ymin + 1;
                if (ltemp*wtemp < area) {
                    area = ltemp*wtemp;
                    attributes.length = ltemp;
                    attributes.width = wtemp;
                }
            }
            
            if (!skipChecks && (attributes.length/attributes.width < 1./10. || attributes.length/attributes.width > 10.)) continue;

            Component acceptedComponent;
            acceptedComponent.length = attributes.length;
            
            acceptedComponent.cx = ((float) (attributes.xmax+attributes.xmin)) / 2;
            acceptedComponent.cy = ((float) (attributes.ymax+attributes.ymin)) / 2;

            acceptedComponent.BB_pointP.x = attributes.xmin;
            acceptedComponent.BB_pointP.y = attributes.ymin;

            acceptedComponent.BB_pointQ.x = attributes.xmax;
            acceptedComponent.BB_pointQ.y = attributes.ymax;

            acceptedComponent.length = attributes.xmax - attributes.xmin + 1;
            acceptedComponent.width = attributes.ymax - attributes.ymin + 1;

            acceptedComponent.mean = attributes.mean;
            acceptedComponent.median = attributes.median;

            acceptedComponent.points = component;
            
            filteredComponents.push_back(acceptedComponent);
        }

        std::vector<Component> tempComp;
        tempComp.reserve(filteredComponents.size());
        
        for (unsigned int i = 0; i < filteredComponents.size(); i++) {
            int count = 0;
            Component compi = filteredComponents[i];
            for (unsigned int j = 0; j < filteredComponents.size(); j++) {
                if (i != j) {
                    Component compj = filteredComponents[j];
                    if (compi.BB_pointP.x <= compj.cx && compi.BB_pointQ.x >= compj.cx &&
                        compi.BB_pointP.y <= compj.cy && compi.BB_pointQ.y >= compj.cy) {
                        count++;
                    }
                }
            }
            if (count < 2) {
                tempComp.push_back(compi);
            }
        }
        filteredComponents = tempComp;


        return filteredComponents;
    };

    void renderComponentBBs (Mat& SWTImage, std::vector<Component> components, Mat& output) {
        Mat outTemp( output.size(), CV_8UC1 );

        output.convertTo(outTemp, CV_8UC1, 255.);
        cvtColor (outTemp, output, COLOR_GRAY2RGB);

        for (int i = 0; i < components.size(); i++) {
            Component compi = components[i];
            Scalar c;
            if (i % 3 == 0) {
                c = BLUE;
            }
            else if (i % 3 == 1) {
                c = GREEN;
            }
            else {
                c = RED;
            }
            rectangle(output, Point(compi.BB_pointP.x, compi.BB_pointP.y), Point(compi.BB_pointQ.x, compi.BB_pointQ.y), c, 2);
        }
    }

    vector<cv::Rect> getComponentBBs (std::vector<Component> components) {
        vector<cv::Rect> bbs;
        for (int i = 0; i < components.size(); i++) {
            Component compi = components[i];
            int wd = compi.BB_pointP.x - compi.BB_pointQ.x;
            int ht = compi.BB_pointP.y - compi.BB_pointQ.y;
            if (wd < 0) wd = -wd;
            if (ht < 0) ht = -ht;

            bbs.push_back(Rect(min(compi.BB_pointP.x, compi.BB_pointQ.x), min(compi.BB_pointP.y, compi.BB_pointQ.y), wd, ht));
        }
        return bbs;
    }

    bool chainSortDist (ChainedComponent Chainl, ChainedComponent Chainr) {
        return Chainl.chainDist < Chainr.chainDist;
    }

    bool chainSortLength (ChainedComponent Chainl, ChainedComponent Chainr) {
        return Chainl.componentIndices.size() < Chainr.componentIndices.size();
    }

    vector<cv::Rect> findValidChains(Mat input_image, Mat SWTImage, std::vector<Component> components, Mat & output) {
        std::vector<ChannelAverage> colorAverages;
        colorAverages.reserve(components.size());
        for (int i = 0; i < components.size(); i++) {
            Component compi = components[i];
            ChannelAverage avgCompi;
            avgCompi.Red = 0;
            avgCompi.Green = 0;
            avgCompi.Blue = 0;
            for (int j = 0; j < compi.points.size(); j++) {
                int x = compi.points[j].x;
                int y = compi.points[j].y;
                avgCompi.Red += (float) input_image.at<uchar>(y, x*3);
                avgCompi.Green += (float) input_image.at<uchar>(y, x*3+1);
                avgCompi.Blue += (float) input_image.at<uchar>(y, x*3+2);
            }
            avgCompi.Red /= compi.points.size();
            avgCompi.Green /= compi.points.size();
            avgCompi.Blue /= compi.points.size(); 
            colorAverages.push_back(avgCompi);      
        }
        int count = 0;
        std::vector<ChainedComponent> chains;
        for (int i = 0; i < components.size(); i++) {
            Component compi = components[i];
            for (int j = i+1; j < components.size(); j++) {
                Component compj = components[j];
                if ((compi.median / compj.median <= 2.0 || compj.median / compi.median <= 2.0) 
                    && (compi.width/compj.width <= 2.0 || compj.width/compi.width <= 2.0)) {
                        float dist = (compi.cx - compj.cx) * (compi.cx - compj.cx) +
                                     (compi.cy - compj.cy) * (compi.cy - compj.cy);
                        float colorDist = (colorAverages[i].Red - colorAverages[j].Red) * (colorAverages[i].Red - colorAverages[j].Red) +
                                        (colorAverages[i].Green - colorAverages[j].Green) * (colorAverages[i].Green - colorAverages[j].Green) +
                                        (colorAverages[i].Blue - colorAverages[j].Blue) * (colorAverages[i].Blue - colorAverages[j].Blue);
                        if (dist < 9*(float)(std::max(std::min(compi.length,compi.width),std::min(compj.length,compj.width)))
                            *(float)(std::max(std::min(compi.length,compi.width),std::min(compj.length,compj.width))) && colorDist < 1600) {
                                ChainedComponent chain;
                                chain.chainIndexA = i;
                                chain.chainIndexB = j;
                                vector <int> componentIndices;
                                componentIndices.push_back(i);
                                componentIndices.push_back(j);
                                chain.componentIndices = componentIndices;
                                chain.chainDist = dist;

                                float dx = compi.cx - compj.cx;
                                float dy = compi.cy - compj.cy;
                                float mod = sqrt(dx * dx + dy * dy);
                                dx = dx / mod;
                                dy = dy / mod;

                                Direction dir;
                                dir.x = dx;
                                dir.y = dy;
                                chain.dir = dir;
                                chains.push_back(chain);
                                count++;
                            }

                }
            }
        }
        
        std::sort(chains.begin(), chains.end(), chainSortDist);

        const float alignmentThreshold = PI / 6;
        int merges = 1;
        while (merges > 0) {
            for (unsigned int i = 0; i < chains.size(); i++) {
                chains[i].merged = false;
            }
            merges = 0;
            std::vector<ChainedComponent> chainsAfterMerging;
            for (int i = 0; i < chains.size(); i++) {
                for (int j = 0; j < chains.size(); j++){
                    if (i!=j && !chains[i].merged && !chains[j].merged) {
                        if (chains[i].chainIndexA == chains[j].chainIndexA) {
                            // cout << chains[i].dir.x << " " << chains[i].dir.y << " " << chains[j].dir.x << " " << chains[j].dir.y << endl;
                            if (acos(chains[i].dir.x * -chains[j].dir.x + chains[i].dir.y * -chains[j].dir.y) < alignmentThreshold) {
                                // std::cout << "here1" << endl;
                                chains[i].chainIndexA = chains[j].chainIndexB;
                                for (std::vector<int>::iterator it = chains[j].componentIndices.begin(); it != chains[j].componentIndices.end(); it++) {
                                    chains[i].componentIndices.push_back(*it);
                                }
                                float d_x = components[chains[i].chainIndexA].cx - components[chains[i].chainIndexB].cx;
                                float d_y = components[chains[i].chainIndexA].cy - components[chains[i].chainIndexB].cy;
                                chains[i].chainDist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Direction dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].dir = dir;
                                chains[j].merged = true;
                                merges++;
                            }
                        } else if (chains[i].chainIndexA == chains[j].chainIndexB) {
                            if (acos(chains[i].dir.x * chains[j].dir.x + chains[i].dir.y * chains[j].dir.y) < alignmentThreshold) {
                                // std::cout << "here2" << endl;
                                chains[i].chainIndexA = chains[j].chainIndexA;
                                for (std::vector<int>::iterator it = chains[j].componentIndices.begin(); it != chains[j].componentIndices.end(); it++) {
                                    chains[i].componentIndices.push_back(*it);
                                }
                                float d_x = components[chains[i].chainIndexA].cx - components[chains[i].chainIndexB].cx;
                                float d_y = components[chains[i].chainIndexA].cy - components[chains[i].chainIndexB].cy;
                                chains[i].chainDist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Direction dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].dir = dir;
                                chains[j].merged = true;
                                merges++;
                            }
                        } else if (chains[i].chainIndexB == chains[j].chainIndexA) {
                            if (acos(chains[i].dir.x * chains[j].dir.x + chains[i].dir.y * chains[j].dir.y) < alignmentThreshold) {
                                // std::cout << "here3" << endl;    
                                chains[i].chainIndexB = chains[j].chainIndexA;
                                for (std::vector<int>::iterator it = chains[j].componentIndices.begin(); it != chains[j].componentIndices.end(); it++) {
                                    chains[i].componentIndices.push_back(*it);
                                }
                                float d_x = components[chains[i].chainIndexA].cx - components[chains[i].chainIndexB].cx;
                                float d_y = components[chains[i].chainIndexA].cy - components[chains[i].chainIndexB].cy;
                                chains[i].chainDist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Direction dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].dir = dir;
                                chains[j].merged = true;
                                merges++;
                            }
                        } else if (chains[i].chainIndexB == chains[j].chainIndexB) {
                            if (acos(chains[i].dir.x * -chains[j].dir.x + chains[i].dir.y * -chains[j].dir.y) < alignmentThreshold) {
                                // std::cout << "here4" << endl;
                                chains[i].chainIndexB = chains[j].chainIndexA;
                                for (std::vector<int>::iterator it = chains[j].componentIndices.begin(); it != chains[j].componentIndices.end(); it++) {
                                    chains[i].componentIndices.push_back(*it);
                                }
                                float d_x = components[chains[i].chainIndexA].cx - components[chains[i].chainIndexB].cx;
                                float d_y = components[chains[i].chainIndexA].cy - components[chains[i].chainIndexB].cy;
                                chains[i].chainDist = d_x * d_x + d_y * d_y;

                                float mag = sqrt(d_x*d_x + d_y*d_y);
                                d_x = d_x / mag;
                                d_y = d_y / mag;
                                Direction dir;
                                dir.x = d_x;
                                dir.y = d_y;
                                chains[i].dir = dir;
                                chains[j].merged = true;
                                merges++;                                
                            }
                        }
                    }
                }
            }
            std::vector<ChainedComponent> newchains;
            for (int i = 0; i < chains.size(); i++) {
                if (!chains[i].merged) {
                    newchains.push_back(chains[i]);
                }
            }
            chains = newchains;
            std::stable_sort(chains.begin(), chains.end(), chainSortLength);                
        }

        std::vector<ChainedComponent> newchains;
        std::vector<std::vector<SWTPoint>> componentsPointsVector;
        vector<Component> finalComponents;
            for (int i = 0; i < chains.size(); i++) {
                if (chains[i].componentIndices.size() >= 3) {
                    newchains.push_back(chains[i]);
                    for (int j = 0; j < chains[i].componentIndices.size(); j++) { 
                        Component acceptedComponent = components[chains[i].componentIndices[j]];
                        std::vector<SWTPoint> componentPoints;
                        for (int k = 0; k < acceptedComponent.points.size(); k++) {
                            componentPoints.push_back(acceptedComponent.points[k]);
                        }
                        componentsPointsVector.push_back(componentPoints);
                    }
                }
            }
            finalComponents = filterComponents(SWTImage, componentsPointsVector, true);
            chains = newchains;
            std::stable_sort(chains.begin(), chains.end(), chainSortLength);

            // std::cout << chains.size() << " chains formed after merging" << std::endl;
            renderComponents(SWTImage, finalComponents, output);
            renderComponentBBs(SWTImage, finalComponents, output);
            return getComponentBBs(finalComponents);    
    }

    std::vector<cv::Rect> SWTTextDetection (const Mat& input_image, bool dark_on_light) {
        assert (input_image.depth() == CV_8U);
        assert (input_image.channels() == 3);

        // Convert to grayscale
        Mat grayImage(input_image.size(), CV_8UC1);
        cvtColor(input_image, grayImage, COLOR_RGB2GRAY);
        // Create Canny Image
        double threshold_low = 175;
        double threshold_high = 320;
        Mat canny_edge_image(input_image.size(), CV_8UC1);
        Canny (grayImage, canny_edge_image, threshold_low, threshold_high, 3);

        // Create gradient X, gradient Y
        Mat gaussianImage( input_image.size(), CV_32FC1);
        grayImage.convertTo(gaussianImage, CV_32FC1, 1./255.);
        
        
        Mat gradientX( input_image.size(), CV_32FC1 );
        Mat gradientY( input_image.size(), CV_32FC1 );
        GaussianBlur(gaussianImage, gaussianImage, Size(5, 5), 0);
        Scharr(gaussianImage, gradientX, -1, 1, 0);
        Scharr(gaussianImage, gradientY, -1, 0, 1);
        GaussianBlur(gradientX, gradientX, Size(3, 3), 0);
        GaussianBlur(gradientY, gradientY, Size(3, 3), 0);
        
        std::vector<Ray> rays;
        Mat SWTImage( input_image.size(), CV_32FC1 );
        
        SWTFirstPass (canny_edge_image, gradientX, gradientY, dark_on_light, SWTImage, rays );

        SWTSecondPass ( SWTImage, rays );

        Mat normalised_image (input_image.size(), CV_8UC1);
        normalizeAndScale(SWTImage, normalised_image);

        // Calculate legally connected components from SWT and gradient image.
        // return type is a vector of vectors, where each outer vector is a component and
        // the inner vector contains the (y,x) of each pixel in that component.
        std::vector<std::vector<SWTPoint> > components = getComponents(SWTImage, rays);
        std::vector<Component> validComponents = filterComponents(SWTImage, components, false);

        Mat outTemp( input_image.size(), CV_32FC1 );
        renderComponents(SWTImage, validComponents, outTemp);
        renderComponentBBs(SWTImage, validComponents, outTemp);
        
        Mat out( input_image.size(), CV_32FC1 );
        return findValidChains(input_image, SWTImage, validComponents, out);
    }
}
}


