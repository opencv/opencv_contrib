#include "opencv2/face.hpp"
#include "opencv2/core.hpp"
#include "precomp.hpp"

/*dataset parser*/
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>     /* atoi */

#undef BOILERPLATE_CODE
#define BOILERPLATE_CODE(name,classname)\
    if(facemarkType==name){\
        return classname::create();\
}

namespace cv
{
    //namespace face {
    Facemark::~Facemark(){
    }

    void Facemark::training(String imageList, String groundTruth){
        trainingImpl(imageList, groundTruth);
    }

    bool Facemark::detect( InputArray image, std::vector<Point2f> & landmarks ){
        if( image.empty() )
        return false;

        return detectImpl( image.getMat(), landmarks );
    }

    bool Facemark::detect( InputArray image, Rect face, std::vector<Point2f> & landmarks ){
        Mat img = image.getMat();
        return detect(img(face), landmarks);
    }

    bool Facemark::detect( InputArray image, std::vector<Rect> faces, std::vector<std::vector<Point2f> > & landmarks ){
        Mat img = image.getMat();
        landmarks.resize(faces.size());

        for(unsigned i=0; i<faces.size();i++){
            detect(img(faces[i]), landmarks[i]);
        }

        return true;
    }

    Ptr<Facemark> Facemark::create( const String& facemarkType ){
        BOILERPLATE_CODE("AAM",FacemarkAAM);
        return Ptr<Facemark>();
    }


    bool Facemark::getFacesHaar( const Mat image, std::vector<Rect> & faces, String face_cascade_name ){
        Mat gray;

        CascadeClassifier face_cascade;
        if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face_cascade\n"); return false; };

        cvtColor( image, gray, CV_BGR2GRAY );
        equalizeHist( gray, gray );
        face_cascade.detectMultiScale( gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        return true;
    }

    bool Facemark::setFaceDetector(bool(*f)(const Mat , std::vector<Rect> & )){
        faceDetector = f;
        isSetDetector = true;
        printf("face detector is configured\n");
        return true;
    }


    bool Facemark::getFaces( InputArray image , std::vector<Rect> & faces){

        if(!isSetDetector){
            return false;
        }

        faceDetector(image.getMat(), faces);
        printf("getfces %i\n",(int)faces.size());
        return true;
    }

    bool Facemark::process(InputArray image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks ){
        if(!isSetDetector){
            return false;
        }

        faceDetector(image.getMat(), faces);
        printf("process::face detected %i\n",(int)faces.size());
        detect(image.getMat(), faces, landmarks);
        return true;
    }

    bool Facemark::process(InputArray image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks, String haarModel ){

        getFacesHaar(image.getMat(), faces, haarModel);
        printf("process::face detected %i\n",(int)faces.size());
        detect(image.getMat(), faces, landmarks);

        return true;
    }

    bool Facemark::loadTrainingData(String filename, std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints, char delim){
        std::string line;
        std::string item;
        std::vector<Point2f> pts;
        std::vector<float> raw;
        // TODO: throw error if file not exist
        std::ifstream infile(filename.c_str());

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
                pts.push_back(Point2f(raw[i],raw[i+1]));
            }
            facePoints.push_back(pts);
        } // main loading process

        return true;
    }

    bool Facemark::loadTrainingData(String imageList, String groundTruth, std::vector<String> & images, std::vector<std::vector<Point2f> > & facePoints){
        std::string line;
        std::vector<Point2f> facePts;

        /*clear the output containers*/
        images.clear();
        facePoints.clear();

        /*load the images path*/
        std::ifstream infile(imageList.c_str());
        while (getline (infile, line)){
            images.push_back(line);
        }

        /*load the points*/
        std::ifstream ss_gt(groundTruth.c_str());
        while (getline (ss_gt, line)){
            facePts.clear();
            loadFacePoints(line, facePts);
            facePoints.push_back(facePts);
        }

        return true;
    }

    bool Facemark::loadFacePoints(String filename, std::vector<Point2f> & pts){
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
            pts.push_back(Point2f((float)atof(x.c_str()),(float)atof(y.c_str())));

        }

        return true;
    }

//  } /* namespace face */
} /* namespace cv */
