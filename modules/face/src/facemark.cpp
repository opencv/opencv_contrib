#include "opencv2/face.hpp"
#include "opencv2/core.hpp"
#include "precomp.hpp"

#undef BOILERPLATE_CODE
#define BOILERPLATE_CODE(name,classname)\
    if(trackerType==name){\
        return classname::create();\
}

namespace cv
{
    //namespace face {
    Facemark::Facemark(){
        isSetDetector =false;
    }

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

    Ptr<Facemark> Facemark::create( const String& trackerType ){
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


    bool Facemark:: getFaces( InputArray image , std::vector<Rect> & faces){

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

//  } /* namespace face */
} /* namespace cv */
