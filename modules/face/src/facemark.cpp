// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/face/facemark.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
/*dataset parser*/
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>     /* atoi */

#undef BOILERPLATE_CODE
#define BOILERPLATE_CODE(name,classname)\
    if(facemarkType==name){\
        return classname::create();\
}
using namespace std;
namespace cv {
namespace face {
    bool getFacesHAAR( InputArray image, OutputArray faces, String face_cascade_name ){
        Mat gray;
        vector<Rect> roi;
        CascadeClassifier face_cascade;
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face_cascade\n"); return false; };
        cvtColor( image.getMat(), gray, COLOR_BGR2GRAY );
        equalizeHist( gray, gray );
        face_cascade.detectMultiScale( gray, roi, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
        Mat(roi).copyTo(faces);
        return true;
    }
    bool loadTrainingData(String filename, vector<String> & images, OutputArray _facePoints, char delim, float offset){
        string line;
        string item;
        vector<Point2f> pts;
        vector<float> raw;
        vector< vector<Point2f> > & facePoints =
            *(vector< vector<Point2f> >*) _facePoints.getObj();
        ifstream infile;
        infile.open(filename.c_str(), ios::in);
        if (!infile) {
           string error_message = "No valid input file was given, please check the given filename.";
           CV_ErrorNoReturn(Error::StsBadArg, error_message);
        }
        /*clear the output containers*/
        images.clear();
        facePoints.clear();
        /*the main loading process*/
        while (getline (infile, line)){
            istringstream ss(line); // string stream for the current line
            /*pop the image path*/
            getline (ss, item, delim);
            images.push_back(item);
            /*load all numbers*/
            raw.clear();
            while (getline (ss, item, delim)){
                raw.push_back((float)atof(item.c_str()));
            }
            /*convert to opencv points*/
            pts.clear();
            for(unsigned i = 0;i< raw.size();i+=2){
                pts.push_back(Point2f(raw[i]+offset,raw[i+1]+offset));
            }
            facePoints.push_back(pts);
        } // main loading process
        return true;
    }
    bool loadTrainingData(String imageList, String groundTruth, vector<String> & images, OutputArray _facePoints, float offset){
        string line;
        vector<Point2f> facePts;
        vector< vector<Point2f> > & facePoints =
                *(vector< vector<Point2f> >*) _facePoints.getObj();
        /*clear the output containers*/
        images.clear();
        facePoints.clear();
        /*load the images path*/
        ifstream infile;
        infile.open(imageList.c_str(), ios::in);
        if (!infile) {
           string error_message = "No valid input file was given, please check the given filename.";
           CV_ErrorNoReturn(Error::StsBadArg, error_message);
        }
        while (getline (infile, line)){
            images.push_back(line);
        }
        /*load the points*/
        ifstream ss_gt(groundTruth.c_str());
        while (getline (ss_gt, line)){
            facePts.clear();
            loadFacePoints(line, facePts, offset);
            facePoints.push_back(facePts);
        }
        return true;
    }
    bool loadFacePoints(String filename, OutputArray points, float offset){
        vector<Point2f> pts;
        string line, item;
        ifstream infile(filename.c_str());
        /*pop the version*/
        getline(infile, line);
        CV_Assert(line.compare(0,7,"version")==0);
        /*pop the number of points*/
        getline(infile, line);
        CV_Assert(line.compare(0,8,"n_points")==0);
        /*get the number of points*/
        string item_npts;
        int npts;
        istringstream linestream(line);
        linestream>>item_npts>>npts;
        /*pop out '{' character*/
        getline(infile, line);
        /*main process*/
        int cnt = 0;
        string x, y;
        pts.clear();
        while (getline(infile, line) && cnt<npts )
        {
            cnt+=1;
            istringstream ss(line);
            ss>>x>>y;
            pts.push_back(Point2f((float)atof(x.c_str())+offset,(float)atof(y.c_str())+offset));
        }
        Mat(pts).copyTo(points);
        return true;
    }
    bool loadTrainingData(vector<String> filename,vector< vector<Point2f> >
                              & trainlandmarks,vector<String> & trainimages)
    {
        string img;
        vector<Point2f> temp;
        string s;
        string tok;
        vector<string> coordinates;
        ifstream f1;
        for(unsigned long j=0;j<filename.size();j++){
            f1.open(filename[j].c_str(),ios::in);
            if(!f1.is_open()){
                cout<<filename[j]<<endl;
                CV_ErrorNoReturn(Error::StsError, "File can't be opened for reading!");
                return false;
            }
            //get the path of the image whose landmarks have to be detected
            getline(f1,img);
            //push the image paths in the vector
            trainimages.push_back(img);
            img.clear();
            while(getline(f1,s)){
                Point2f pt;
                stringstream ss(s); // Turn the string into a stream.
                while(getline(ss, tok,',')) {
                    coordinates.push_back(tok);
                    tok.clear();
                }
                pt.x = (float)atof(coordinates[0].c_str());
                pt.y = (float)atof(coordinates[1].c_str());
                coordinates.clear();
                temp.push_back(pt);
            }
            trainlandmarks.push_back(temp);
            temp.clear();
            f1.close();
        }
        return true;
    }
    void drawFacemarks(InputOutputArray image, InputArray points, Scalar color){
        Mat img = image.getMat();
        vector<Point2f> pts = points.getMat();
        for(size_t i=0;i<pts.size();i++){
            circle(img, pts[i],3, color,-1);
        }
    } //drawPoints
} /* namespace face */
} /* namespace cv */