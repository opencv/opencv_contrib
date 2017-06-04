
#ifndef __OPENCV_FACELANDMARK_HPP__
#define __OPENCV_FACELANDMARK_HPP__

#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/imgproc/types_c.h"

#undef BOILERPLATE_CODE
#define BOILERPLATE_CODE(name,classname) \
    static Ptr<classname> create(const classname::Params &parameters); \
    CV_WRAP static Ptr<classname> create(); \
    virtual ~classname(){};

namespace cv
{
    //namespace face {
    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:

        virtual ~Facemark();
        virtual void read( const FileNode& fn )=0;
        virtual void write( FileStorage& fs ) const=0;

        /**
        * \brief training the facemark model, input are the file names of image list and landmark annotation
        */
        virtual bool training(String imageList, String groundTruth);

        /**
        * \brief extract landmark points from a face
        */
        // CV_WRAP bool detect( InputArray image, Rect2d& boundingBox );
        bool detect( InputArray image, std::vector<Point2f> & landmarks );//!< from a face
        bool detect( InputArray image, Rect face, std::vector<Point2f> & landmarks );//!< from an ROI
        bool detect( InputArray image, std::vector<Rect> faces, std::vector<std::vector<Point2f> >& landmarks );//!< from many ROIs

        static Ptr<Facemark> create( const String& trackerType );

        //!<  default face detector
        bool getFacesHaar( const Mat image , std::vector<Rect> & faces, String face_cascade_name);

        //!<  set the custom face detector
        bool setFaceDetector(bool(*f)(const Mat , std::vector<Rect> & ));
        //!<  get faces using the custom detector
        bool getFaces( InputArray image , std::vector<Rect> & faces);

        //!<  get faces and then extract landmarks for each of them
        bool process(InputArray image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks );

        //!<  using the default face detector (haarClassifier), xml of the model should be provided
        bool process(InputArray image,std::vector<Rect> & faces, std::vector<std::vector<Point2f> >& landmarks, String haarModel );

    protected:
        virtual bool detectImpl( InputArray image, std::vector<Point2f> & landmarks )=0;
        virtual bool trainingImpl(String imageList, String groundTruth)=0;


        /*circumventable face extractor function*/
        bool(*faceDetector)(const Mat , std::vector<Rect> &  ) ;
        bool isSetDetector;

    }; /* Facemark*/

    class CV_EXPORTS_W FacemarkAAM : public Facemark
    {
    public:
        struct CV_EXPORTS Params
        {
            /**
            * \brief Constructor
            */
            Params();

            /*read only parameters - just for example*/
            double detect_thresh;         //!<  detection confidence threshold
            double sigma;                 //!<  another parameter

            /**
            * \brief Read parameters from file, currently unused
            */
            void read(const FileNode& /*fn*/);

            /**
            * \brief Read parameters from file, currently unused
            */
            void write(FileStorage& /*fs*/) const;
        };


        /**
        * \brief this BOILERPLATE_CODE is equivalent to the following snippet
        * (see the definition at the top)
        * static Ptr<FacemarkAAM> create(const FacemarkAAM::Params &parameters);
        * CV_WRAP static Ptr<FacemarkAAM> create();
        * virtual ~FacemarkAAM() {}
        */
        BOILERPLATE_CODE("AAM",FacemarkAAM);
    }; /* AAM */

//  } /* namespace face */
} /* namespace cv */


#endif //__OPENCV_FACELANDMARK_HPP__
