#ifndef __OPENCV_FACEMARK_HPP__
#define __OPENCV_FACEMARK_HPP__
#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace cv {
namespace face {
//! @addtogroup face
//! @{
    CV_EXPORTS_W bool getFacesHAAR( InputArray image,
                                    OutputArray faces,
                                    String face_cascade_name );
    CV_EXPORTS_W bool loadTrainingData( String filename , std::vector<String> & images,
                                        OutputArray facePoints,
                                        char delim = ' ', float offset = 0.0);
    CV_EXPORTS_W bool loadTrainingData( String imageList, String groundTruth,
                                        std::vector<String> & images,
                                        OutputArray facePoints,
                                        float offset = 0.0);
    /** @brief This function extracts the data for training from .txt files which contains the corresponding image name and landmarks.
    *The first file in each file should give the path of the image whose
    *landmarks are being described in the file. Then in the subsequent
    *lines there should be coordinates of the landmarks in the image
    *i.e each line should be of the form x,y
    *where x represents the x coordinate of the landmark and y represents
    *the y coordinate of the landmark.
    *
    *For reference you can see the files as provided in the
    *<a href="http://www.ifp.illinois.edu/~vuongle2/helen/">HELEN dataset</a>
    *
    * @param filename A vector of type cv::String containing name of the .txt files.
    * @param trainlandmarks A vector of type cv::Point2f that would store shape or landmarks of all images.
    * @param trainimages A vector of type cv::String which stores the name of images whose landmarks are tracked
    * @returns A boolean value. It returns true when it reads the data successfully and false otherwise
    */
    CV_EXPORTS_W bool loadTrainingData(std::vector<String> filename,std::vector< std::vector<Point2f> >
                              &trainlandmarks,std::vector<String> & trainimages);

    CV_EXPORTS_W bool loadFacePoints( String filename, OutputArray points,
                                      float offset = 0.0);

    CV_EXPORTS_W void drawFacemarks( InputOutputArray image, InputArray points,
                                     Scalar color = Scalar(255,0,0));

    /** @brief Abstract base class for all facemark models

    All facemark models in OpenCV are derived from the abstract base class Facemark, which
    provides a unified access to all facemark algorithms in OpenCV.

    ### Description

    Facemark is a base class which provides universal access to any specific facemark algorithm.
    Therefore, the users should declare a desired algorithm before they can use it in their application.

    Here is an example on how to declare facemark algorithm:
    @code
    // Using Facemark in your code:
    Ptr<Facemark> facemark = FacemarkLBF::create();
    @endcode

    The typical pipeline for facemark detection is listed as follows:
      (Non-mandatory) Set a user defined face detection using Facemark::setFaceDetector.
      The facemark algorithms are desgined to fit the facial points into a face.
      Therefore, the face information should be provided to the facemark algorithm.
      Some algorithms might provides a default face recognition function.
      However, the users might prefer to use their own face detector to obtains the best possible detection result.
      (Non-mandatory) Training the model for a specific algorithm using Facemark::training.
      In this case, the model should be automatically saved by the algorithm.
      If the user already have a trained model, then this part can be omitted.
      Load the trained model using Facemark::loadModel.
      Perform the fitting via the Facemark::fit.
    */
    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:
        /**
        * \brief training the facemark model, input are the file names of image list and landmark annotation
        */
        virtual void training(String imageList, String groundTruth)=0;
        /** @brief This function is used to train the model using gradient boosting to get a cascade of regressors
        *which can then be used to predict shape.
        *@param images A vector of type cv::Mat which stores the images which are used in training samples.
        *@param landmarks A vector of vectors of type cv::Point2f which stores the landmarks detected in a particular image.
        *@param scale A size of type cv::Size to which all images and landmarks have to be scaled to.
        *@param configfile A variable of type std::string which stores the name of the file storing parameters for training the model.
        *@param modelFilename A variable of type std::string which stores the name of the trained model file that has to be saved.
        *@returns A boolean value. The function returns true if the model is trained properly or false if it is not trained.
        */
        virtual bool training(std::vector<Mat>& images, std::vector< std::vector<Point2f> >& landmarks,std::string configfile,Size scale,std::string modelFilename = "face_landmarks.dat")=0;
        /** @brief This function is used to load the trained model..
        *@param filename A variable of type cv::String which stores the name of the file in which trained model is stored.
        */
        virtual void loadModel(String filename)=0;
        /** @brief This functions retrieves a centered and scaled face shape, according to the bounding rectangle.
        *@param image A variable of type cv::InputArray which stores the image whose landmarks have to be found
        *@param faces A variable of type cv::InputArray which stores the bounding boxes of faces found in a given image.
        *@param landmarks A variable of type cv::InputOutputArray which stores the landmarks of all the faces found in the image
        */
        virtual bool fit( InputArray image, InputArray faces, InputOutputArray landmarks )=0;//!< from many ROIs
        virtual bool setFaceDetector(bool(*f)(InputArray , OutputArray ))=0;
        //!<  set the custom face detector
        virtual bool getFaces( InputArray image , OutputArray faces)=0;
        //!<  get faces using the custom detector
    }; /* Facemark*/
//! @}
} /* namespace face */
} /* namespace cv */
#endif //__OPENCV_FACELANDMARK_HPP__