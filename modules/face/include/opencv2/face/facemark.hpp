
#ifndef __OPENCV_FACELANDMARK_HPP__
#define __OPENCV_FACELANDMARK_HPP__

/**
@defgroup face Face Recognition
- @ref tutorial_table_of_content_facemark
*/

#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/imgproc/types_c.h"

namespace cv {
namespace face {

//! @addtogroup face
//! @{

    /** @brief Default face detector
    This function is mainly utilized by the implementation of a Facemark Algorithm.
    End users are advised to use function Facemark::getFaces which can be manually defined
    and circumvented to the algorithm by Facemark::setFaceDetector.

    @param image The input image to be processed.
    @param faces Output of the function which represent region of interest of the detected faces.
    Each face is stored in cv::Rect container.
    @param face_cascade_model The filename of a cascade model for face detection.

    <B>Example of usage</B>
    @code
    std::vector<cv::Rect> roi;
    cv::face::getFacesHaar(frame, roi, "haarcascade_frontalface_alt.xml");
    for(int j=0;j<rects.size();j++){
        cv::rectangle(frame, rects[j], cv::Scalar(255,0,255));
    }
    @endcode
    */
    CV_EXPORTS_W bool getFacesHaar( InputArray image,
                                    OutputArray faces,
                                    String face_cascade_model );


    /** @brief A utility to load list of paths to training image and annotation file.
    @param imageList The specified file contains paths to the training images.
    @param annotationList The specified file contains paths to the training annotations.
    @param images The loaded paths of training images.
    @param annotations The loaded paths of annotation files.

    Example of usage:
    @code
    String imageFiles = "images_path.txt";
    String ptsFiles = "annotations_path.txt";
    std::vector<String> images_train;
    std::vector<String> landmarks_train;
    loadDatasetList(imageFiles,ptsFiles,images_train,landmarks_train);
    @endcode

    */
    CV_EXPORTS_W bool loadDatasetList(String imageList,
                                      String annotationList,
                                      std::vector<String> & images,
                                      std::vector<String> & annotations);

    /** @brief A utility to load facial landmark dataset from a single file.

    @param filename The filename of a file that contains the dataset information.
    Each line contains the filename of an image followed by
    pairs of x and y values of facial landmarks points separated by a space.
    Example
    @code
    /home/user/ibug/image_003_1.jpg 336.820955 240.864510 334.238298 260.922709 335.266918 ...
    /home/user/ibug/image_005_1.jpg 376.158428 230.845712 376.736984 254.924635 383.265403 ...
    @endcode
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.
    @param delim Delimiter between each element, the default value is a whitespace.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0);
    @endcode
    */

    CV_EXPORTS_W bool loadTrainingData( String filename , std::vector<String> & images,
                                        OutputArray facePoints,
                                        char delim = ' ', float offset = 0.0);

    /** @brief A utility to load facial landmark information from the dataset.

    @param imageList A file contains the list of image filenames in the training dataset.
    @param groundTruth A file contains the list of filenames
    where the landmarks points information are stored.
    The content in each file should follow the standard format (see face::loadFacePoints).
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0);
    @endcode

    example of content in the images_train.txt
    @code
    /home/user/ibug/image_003_1.jpg
    /home/user/ibug/image_004_1.jpg
    /home/user/ibug/image_005_1.jpg
    /home/user/ibug/image_006.jpg
    @endcode

    example of content in the points_train.txt
    @code
    /home/user/ibug/image_003_1.pts
    /home/user/ibug/image_004_1.pts
    /home/user/ibug/image_005_1.pts
    /home/user/ibug/image_006.pts
    @endcode
    */

    CV_EXPORTS_W bool loadTrainingData( String imageList, String groundTruth,
                                        std::vector<String> & images,
                                        OutputArray facePoints,
                                        float offset = 0.0);

    /** @brief A utility to load facial landmark information from a given file.

    @param filename The filename of file contains the facial landmarks data.
    @param points The loaded facial landmark points.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    std::vector<Point2f> points;
    face::loadFacePoints("filename.txt", points, 0.0);
    @endcode

    The annotation file should follow the default format which is
    @code
    version: 1
    n_points:  68
    {
    212.716603 499.771793
    230.232816 566.290071
    ...
    }
    @endcode
    where n_points is the number of points considered
    and each point is represented as its position in x and y.
    */

    CV_EXPORTS_W bool loadFacePoints( String filename, OutputArray points,
                                      float offset = 0.0);

    /** @brief Utility to draw the detected facial landmark points

    @param image The input image to be processed.
    @param points Contains the data of points which will be drawn.
    @param color The color of points in BGR format represented by cv::Scalar.

    <B>Example of usage</B>
    @code
    std::vector<Rect> faces;
    std::vector<std::vector<Point2f> > landmarks;
    facemark->getFaces(img, faces);
    facemark->fit(img, faces, landmarks);
    for(int j=0;j<rects.size();j++){
        face::drawFacemarks(frame, landmarks[j], Scalar(0,0,255));
    }
    @endcode
    */
    CV_EXPORTS_W void drawFacemarks( InputOutputArray image, InputArray points,
                                     Scalar color = Scalar(255,0,0));

    /** @brief Abstract base class for all facemark models

    All facemark models in OpenCV are derived from the abstract base class Facemark, which
    provides a unified access to all facemark algorithms in OpenCV.

    To utilize this API in your program, please take a look at the @ref tutorial_table_of_content_facemark
    ### Description

    Facemark is a base class which provides universal access to any specific facemark algorithm.
    Therefore, the users should declare a desired algorithm before they can use it in their application.

    Here is an example on how to declare facemark algorithm:
    @code
    // Using Facemark in your code:
    Ptr<Facemark> facemark = FacemarkLBF::create();
    @endcode

    The typical pipeline for facemark detection is listed as follows:
    - (Non-mandatory) Set a user defined face detection using Facemark::setFaceDetector.
      The facemark algorithms are desgined to fit the facial points into a face.
      Therefore, the face information should be provided to the facemark algorithm.
      Some algorithms might provides a default face recognition function.
      However, the users might prefer to use their own face detector to obtains the best possible detection result.
    - (Non-mandatory) Training the model for a specific algorithm using Facemark::training.
      In this case, the model should be automatically saved by the algorithm.
      If the user already have a trained model, then this part can be omitted.
    - Load the trained model using Facemark::loadModel.
    - Perform the fitting via the Facemark::fit.
    */
    class CV_EXPORTS_W Facemark : public virtual Algorithm
    {
    public:

        // virtual void read( const FileNode& fn )=0;
        // virtual void write( FileStorage& fs ) const=0;

        /** @brief Add one training sample to the trainer.

        @param image Input image.
        @param landmarks The ground-truth of facial landmarks points corresponds to the image.

        <B>Example of usage</B>
        @code
        String imageFiles = "../data/images_train.txt";
        String ptsFiles = "../data/points_train.txt";
        std::vector<String> images_train;
        std::vector<String> landmarks_train;

        // load the list of dataset: image paths and landmark file paths
        loadDatasetList(imageFiles,ptsFiles,images_train,landmarks_train);

        Mat image;
        std::vector<Point2f> facial_points;
        for(size_t i=0;i<images_train.size();i++){
            image = imread(images_train[i].c_str());
            loadFacePoints(landmarks_train[i],facial_points);
            facemark->addTrainingSample(image, facial_points);
        }
        @endcode

        The contents in the training files should follows the standard format.
        Here are examples for the contents in these files.
        example of content in the images_train.txt
        @code
        /home/user/ibug/image_003_1.jpg
        /home/user/ibug/image_004_1.jpg
        /home/user/ibug/image_005_1.jpg
        /home/user/ibug/image_006.jpg
        @endcode

        example of content in the points_train.txt
        @code
        /home/user/ibug/image_003_1.pts
        /home/user/ibug/image_004_1.pts
        /home/user/ibug/image_005_1.pts
        /home/user/ibug/image_006.pts
        @endcode

        */
        virtual bool addTrainingSample(InputArray image, InputArray landmarks)=0;

        /** @brief Trains a Facemark algorithm using the given dataset.
        Before the training process, training samples should be added to the trainer
        using face::addTrainingSample function.

        <B>Example of usage</B>
        @code
        FacemarkLBF::Params params;
        params.model_filename = "ibug68.model"; // filename to save the trained model
        Ptr<Facemark> facemark = FacemarkLBF::create(params);

        // add training samples (see Facemark::addTrainingSample)

        facemark->training();
        @endcode
        */

        virtual void training()=0;

        /** @brief A function to load the trained model before the fitting process.

        @param model A string represent the filename of a trained model.

        <B>Example of usage</B>
        @code
        facemark->loadModel("../data/lbf.model");
        @endcode
        */
        virtual void loadModel(String model)=0;
        // virtual void saveModel(String fs)=0;

        /** @brief Trains a Facemark algorithm using the given dataset.

        @param image Input image.
        @param faces Output of the function which represent region of interest of the detected faces.
        Each face is stored in cv::Rect container.
        @param landmarks The detected landmark points for each faces.

        <B>Example of usage</B>
        @code
        Mat image = imread("image.jpg");
        std::vector<Rect> faces;
        std::vector<std::vector<Point2f> > landmarks;
        facemark->fit(image, faces, landmarks);
        @endcode
        */
        virtual bool fit( InputArray image, InputArray faces, InputOutputArray landmarks )=0;

        /** @brief Set a user defined face detector for the Facemark algorithm.

        <B>Example of usage</B>
        @code
        facemark->setFaceDetector(myDetector);
        @endcode

        Example of a user defined face detector
        @code
        bool myDetector( InputArray image, OutputArray ROIs ){
            std::vector<Rect> & faces = *(std::vector<Rect>*) ROIs.getObj();
            faces.clear();

            Mat img = image.getMat();

            // -------- do something --------
        }
        @endcode
        */
        virtual bool setFaceDetector(bool(*f)(InputArray , OutputArray ))=0;

        /** @brief Detect faces from a given image using default or user defined face detector.
        Some Algorithm might not provide a default face detector.

        @param image Input image.
        @param faces Output of the function which represent region of interest of the detected faces.
        Each face is stored in cv::Rect container.

        <B>Example of usage</B>
        @code
        std::vector<cv::Rect> faces;
        facemark->getFaces(img, faces);
        for(int j=0;j<faces.size();j++){
            cv::rectangle(img, faces[j], cv::Scalar(255,0,255));
        }
        @endcode
        */
        virtual bool getFaces( InputArray image , OutputArray faces)=0;

    }; /* Facemark*/

//! @}

} /* namespace face */
} /* namespace cv */


#endif //__OPENCV_FACELANDMARK_HPP__
