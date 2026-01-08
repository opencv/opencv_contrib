// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
This file was part of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

#include "precomp.hpp"
#include "opencv2/face/facemark_train.hpp"

/*dataset parser*/
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>     /* atoi */

namespace cv {
namespace face {

using namespace std;

CParams::CParams(String s, double sf, int minN, Size minSz, Size maxSz){
    cascade = s;
    scaleFactor = sf;
    minNeighbors = minN;
    minSize = minSz;
    maxSize = maxSz;

    if (!face_cascade.load(cascade))
    {
        CV_Error_(Error::StsBadArg, ("Error loading face_cascade: %s", cascade.c_str()));
    }
}

bool getFaces(InputArray image, OutputArray faces, CParams* params)
{
    CV_Assert(params);
    Mat gray;
    std::vector<Rect> roi;

    cvtColor(image.getMat(), gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    params->face_cascade.detectMultiScale( gray, roi, params->scaleFactor, params->minNeighbors, CASCADE_SCALE_IMAGE, params->minSize, params->maxSize);

    Mat(roi).copyTo(faces);
    return true;
}

bool loadDatasetList(String imageList, String groundTruth, std::vector<String> & images, std::vector<String> & landmarks){
    std::string line;

    /*clear the output containers*/
    images.clear();
    landmarks.clear();

    /*open the files*/
    std::ifstream infile;
    infile.open(imageList.c_str(), std::ios::in);
    std::ifstream ss_gt;
    ss_gt.open(groundTruth.c_str(), std::ios::in);
    if ((!infile) || !(ss_gt)) {
       printf("No valid input file was given, please check the given filename.\n");
       return false;
    }

     /*load the images path*/
    while (getline (infile, line)){
        images.push_back(line);
    }

    /*load the points*/
    while (getline (ss_gt, line)){
        landmarks.push_back(line);
    }

    return true;
}

bool loadTrainingData(String filename, std::vector<String> & images, OutputArray _facePoints, char delim, float offset){
    std::string line;
    std::string item;
    std::vector<Point2f> pts;
    std::vector<float> raw;

    // FIXIT
    std::vector<std::vector<Point2f> > & facePoints =
        *(std::vector<std::vector<Point2f> >*) _facePoints.getObj();

    std::ifstream infile;
    infile.open(filename.c_str(), std::ios::in);
    if (!infile) {
        CV_Error_(Error::StsBadArg, ("No valid input file was given, please check the given filename: %s", filename.c_str()));
    }

    /*clear the output containers*/
    images.clear();
    facePoints.clear();

    /*the main loading process*/
    while (getline (infile, line)){
        std::istringstream ss(line); // string stream for the current line

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

bool loadTrainingData(String imageList, String groundTruth, std::vector<String> & images, OutputArray _facePoints, float offset){
    std::string line;
    std::vector<Point2f> facePts;

    // FIXIT
    std::vector<std::vector<Point2f> > & facePoints =
            *(std::vector<std::vector<Point2f> >*) _facePoints.getObj();

    /*clear the output containers*/
    images.clear();
    facePoints.clear();

    /*load the images path*/
    std::ifstream infile;
    infile.open(imageList.c_str(), std::ios::in);
    if (!infile) {
       CV_Error_(Error::StsBadArg, ("No valid input file was given, please check the given filename: %s", imageList.c_str()));
    }

    while (getline (infile, line)){
        images.push_back(line);
    }

    /*load the points*/
    std::ifstream ss_gt(groundTruth.c_str());
    while (getline (ss_gt, line)){
        facePts.clear();
        loadFacePoints(line, facePts, offset);
        facePoints.push_back(facePts);
    }

    return true;
}

bool loadFacePoints(String filename, OutputArray points, float offset){
    vector<Point2f> pts;

    std::string line, item;
    std::ifstream infile(filename.c_str());

    /*pop the version*/
    std::getline(infile, line);
    CV_Assert(line.compare(0,7,"version")==0);

    /*pop the number of points*/
    std::getline(infile, line);
    CV_Assert(line.compare(0,8,"n_points")==0);

    /*get the number of points*/
    std::string item_npts;
    int npts;

    std::istringstream linestream(line);
    linestream>>item_npts>>npts;

    /*pop out '{' character*/
    std::getline(infile, line);

    /*main process*/
    int cnt = 0;
    std::string x, y;
    pts.clear();
    while (std::getline(infile, line) && cnt<npts )
    {
        cnt+=1;

        std::istringstream ss(line);
        ss>>x>>y;
        pts.push_back(Point2f((float)atof(x.c_str())+offset,(float)atof(y.c_str())+offset));

    }

    Mat(pts).copyTo(points);
    return true;
}

bool getFacesHAAR(InputArray image, OutputArray faces, const String& face_cascade_name)
{
    Mat gray;
    vector<Rect> roi;
    CascadeClassifier face_cascade;
    CV_Assert(face_cascade.load(face_cascade_name) && "Can't loading face_cascade");
    cvtColor(image.getMat(), gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    face_cascade.detectMultiScale(gray, roi, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(roi).copyTo(faces);
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
            CV_Error(Error::StsError, "File can't be opened for reading!");
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
}
} /* namespace face */
} /* namespace cv */
