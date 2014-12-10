// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (c) 2011,2012. Philipp Wagner <bytefish[at]gmx[dot]de>.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_FACEREC_HPP__
#define __OPENCV_FACEREC_HPP__

#include "opencv2/core.hpp"

namespace cv { namespace face {

//! @addtogroup face
//! @{

/** @brief Abstract base class for all face recognition models

All face recognition models in OpenCV are derived from the abstract base class FaceRecognizer, which
provides a unified access to all face recongition algorithms in OpenCV.

### Description

I'll go a bit more into detail explaining FaceRecognizer, because it doesn't look like a powerful
interface at first sight. But: Every FaceRecognizer is an Algorithm, so you can easily get/set all
model internals (if allowed by the implementation). Algorithm is a relatively new OpenCV concept,
which is available since the 2.4 release. I suggest you take a look at its description.

Algorithm provides the following features for all derived classes:

-   So called “virtual constructor”. That is, each Algorithm derivative is registered at program
    start and you can get the list of registered algorithms and create instance of a particular
    algorithm by its name (see Algorithm::create). If you plan to add your own algorithms, it is
    good practice to add a unique prefix to your algorithms to distinguish them from other
    algorithms.
-   Setting/Retrieving algorithm parameters by name. If you used video capturing functionality from
    OpenCV highgui module, you are probably familar with cv::cvSetCaptureProperty,
ocvcvGetCaptureProperty, VideoCapture::set and VideoCapture::get. Algorithm provides similar
    method where instead of integer id's you specify the parameter names as text Strings. See
    Algorithm::set and Algorithm::get for details.
-   Reading and writing parameters from/to XML or YAML files. Every Algorithm derivative can store
    all its parameters and then read them back. There is no need to re-implement it each time.

Moreover every FaceRecognizer supports the:

-   **Training** of a FaceRecognizer with FaceRecognizer::train on a given set of images (your face
    database!).
-   **Prediction** of a given sample image, that means a face. The image is given as a Mat.
-   **Loading/Saving** the model state from/to a given XML or YAML.
-   **Setting/Getting labels info**, that is stored as a string. String labels info is useful for
    keeping names of the recognized people.

@note When using the FaceRecognizer interface in combination with Python, please stick to Python 2.
Some underlying scripts like create_csv will not work in other versions, like Python 3. Setting the
Thresholds +++++++++++++++++++++++

Sometimes you run into the situation, when you want to apply a threshold on the prediction. A common
scenario in face recognition is to tell, whether a face belongs to the training dataset or if it is
unknown. You might wonder, why there's no public API in FaceRecognizer to set the threshold for the
prediction, but rest assured: It's supported. It just means there's no generic way in an abstract
class to provide an interface for setting/getting the thresholds of *every possible* FaceRecognizer
algorithm. The appropriate place to set the thresholds is in the constructor of the specific
FaceRecognizer and since every FaceRecognizer is a Algorithm (see above), you can get/set the
thresholds at runtime!

Here is an example of setting a threshold for the Eigenfaces method, when creating the model:

@code
// Let's say we want to keep 10 Eigenfaces and have a threshold value of 10.0
int num_components = 10;
double threshold = 10.0;
// Then if you want to have a cv::FaceRecognizer with a confidence threshold,
// create the concrete implementation with the appropiate parameters:
Ptr<FaceRecognizer> model = createEigenFaceRecognizer(num_components, threshold);
@endcode

Sometimes it's impossible to train the model, just to experiment with threshold values. Thanks to
Algorithm it's possible to set internal model thresholds during runtime. Let's see how we would
set/get the prediction for the Eigenface model, we've created above:

@code
// The following line reads the threshold from the Eigenfaces model:
double current_threshold = model->getDouble("threshold");
// And this line sets the threshold to 0.0:
model->set("threshold", 0.0);
@endcode

If you've set the threshold to 0.0 as we did above, then:

@code
//
Mat img = imread("person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
// Get a prediction from the model. Note: We've set a threshold of 0.0 above,
// since the distance is almost always larger than 0.0, you'll get -1 as
// label, which indicates, this face is unknown
int predicted_label = model->predict(img);
// ...
@endcode

is going to yield -1 as predicted label, which states this face is unknown.

### Getting the name of a FaceRecognizer

Since every FaceRecognizer is a Algorithm, you can use Algorithm::name to get the name of a
FaceRecognizer:

@code
// Create a FaceRecognizer:
Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
// And here's how to get its name:
String name = model->name();
@endcode

 */
class CV_EXPORTS_W FaceRecognizer : public Algorithm
{
public:
    //! virtual destructor
    virtual ~FaceRecognizer() {}

    /** @brief Trains a FaceRecognizer with given data and associated labels.

    @param src The training images, that means the faces you want to learn. The data has to be
    given as a vector\<Mat\>.
    @param labels The labels corresponding to the images have to be given either as a vector\<int\>
    or a

    The following source code snippet shows you how to learn a Fisherfaces model on a given set of
    images. The images are read with imread and pushed into a std::vector\<Mat\>. The labels of each
    image are stored within a std::vector\<int\> (you could also use a Mat of type CV_32SC1). Think of
    the label as the subject (the person) this image belongs to, so same subjects (persons) should have
    the same label. For the available FaceRecognizer you don't have to pay any attention to the order of
    the labels, just make sure same persons have the same label:

    @code
    // holds images and labels
    vector<Mat> images;
    vector<int> labels;
    // images for first person
    images.push_back(imread("person0/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    images.push_back(imread("person0/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    images.push_back(imread("person0/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
    // images for second person
    images.push_back(imread("person1/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
    images.push_back(imread("person1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
    images.push_back(imread("person1/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
    @endcode

    Now that you have read some images, we can create a new FaceRecognizer. In this example I'll create
    a Fisherfaces model and decide to keep all of the possible Fisherfaces:

    @code
    // Create a new Fisherfaces model and retain all available Fisherfaces,
    // this is the most common usage of this specific FaceRecognizer:
    //
    Ptr<FaceRecognizer> model =  createFisherFaceRecognizer();
    @endcode

    And finally train it on the given dataset (the face images and labels):

    @code
    // This is the common interface to train all of the available cv::FaceRecognizer
    // implementations:
    //
    model->train(images, labels);
    @endcode
     */
    CV_WRAP virtual void train(InputArrayOfArrays src, InputArray labels) = 0;

    /** @brief Updates a FaceRecognizer with given data and associated labels.

    @param src The training images, that means the faces you want to learn. The data has to be given
    as a vector\<Mat\>.
    @param labels The labels corresponding to the images have to be given either as a vector\<int\> or
    a

    This method updates a (probably trained) FaceRecognizer, but only if the algorithm supports it. The
    Local Binary Patterns Histograms (LBPH) recognizer (see createLBPHFaceRecognizer) can be updated.
    For the Eigenfaces and Fisherfaces method, this is algorithmically not possible and you have to
    re-estimate the model with FaceRecognizer::train. In any case, a call to train empties the existing
    model and learns a new model, while update does not delete any model data.

    @code
    // Create a new LBPH model (it can be updated) and use the default parameters,
    // this is the most common usage of this specific FaceRecognizer:
    //
    Ptr<FaceRecognizer> model =  createLBPHFaceRecognizer();
    // This is the common interface to train all of the available cv::FaceRecognizer
    // implementations:
    //
    model->train(images, labels);
    // Some containers to hold new image:
    vector<Mat> newImages;
    vector<int> newLabels;
    // You should add some images to the containers:
    //
    // ...
    //
    // Now updating the model is as easy as calling:
    model->update(newImages,newLabels);
    // This will preserve the old model data and extend the existing model
    // with the new features extracted from newImages!
    @endcode

    Calling update on an Eigenfaces model (see createEigenFaceRecognizer), which doesn't support
    updating, will throw an error similar to:

    @code
    OpenCV Error: The function/feature is not implemented (This FaceRecognizer (FaceRecognizer.Eigenfaces) does not support updating, you have to use FaceRecognizer::train to update it.) in update, file /home/philipp/git/opencv/modules/contrib/src/facerec.cpp, line 305
    terminate called after throwing an instance of 'cv::Exception'
    @endcode

    @note The FaceRecognizer does not store your training images, because this would be very
    memory intense and it's not the responsibility of te FaceRecognizer to do so. The caller is
    responsible for maintaining the dataset, he want to work with.
     */
    CV_WRAP virtual void update(InputArrayOfArrays src, InputArray labels) = 0;

    /** @overload */
    virtual int predict(InputArray src) const = 0;

    /** @brief Predicts a label and associated confidence (e.g. distance) for a given input image.

    @param src Sample image to get a prediction from.
    @param label The predicted label for the given image.
    @param confidence Associated confidence (e.g. distance) for the predicted label.

    The suffix const means that prediction does not affect the internal model state, so the method can
    be safely called from within different threads.

    The following example shows how to get a prediction from a trained model:

    @code
    using namespace cv;
    // Do your initialization here (create the cv::FaceRecognizer model) ...
    // ...
    // Read in a sample image:
    Mat img = imread("person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    // And get a prediction from the cv::FaceRecognizer:
    int predicted = model->predict(img);
    @endcode

    Or to get a prediction and the associated confidence (e.g. distance):

    @code
    using namespace cv;
    // Do your initialization here (create the cv::FaceRecognizer model) ...
    // ...
    Mat img = imread("person1/3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    // Some variables for the predicted label and associated confidence (e.g. distance):
    int predicted_label = -1;
    double predicted_confidence = 0.0;
    // Get the prediction and associated confidence from the model
    model->predict(img, predicted_label, predicted_confidence);
    @endcode
     */
    CV_WRAP virtual void predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) const = 0;

    /** @brief Saves a FaceRecognizer and its model state.

    Saves this model to a given filename, either as XML or YAML.
    @param filename The filename to store this FaceRecognizer to (either XML/YAML).

    Every FaceRecognizer overwrites FaceRecognizer::save(FileStorage& fs) to save the internal model
    state. FaceRecognizer::save(const String& filename) saves the state of a model to the given
    filename.

    The suffix const means that prediction does not affect the internal model state, so the method can
    be safely called from within different threads.
     */
    CV_WRAP virtual void save(const String& filename) const = 0;

    /** @brief Loads a FaceRecognizer and its model state.

    Loads a persisted model and state from a given XML or YAML file . Every FaceRecognizer has to
    overwrite FaceRecognizer::load(FileStorage& fs) to enable loading the model state.
    FaceRecognizer::load(FileStorage& fs) in turn gets called by
    FaceRecognizer::load(const String& filename), to ease saving a model.
     */
    CV_WRAP virtual void load(const String& filename) = 0;

    /** @overload
    Saves this model to a given FileStorage.
    @param fs The FileStorage to store this FaceRecognizer to.
    */
    virtual void save(FileStorage& fs) const = 0;

    /** @overload */
    virtual void load(const FileStorage& fs) = 0;

    /** @brief Sets string info for the specified model's label.

    The string info is replaced by the provided value if it was set before for the specified label.
     */
    virtual void setLabelInfo(int label, const String& strInfo) = 0;

    /** @brief Gets string information by label.

    If an unknown label id is provided or there is no label information associated with the specified
    label id the method returns an empty string.
     */
    virtual String getLabelInfo(int label) const = 0;

    /** @brief Gets vector of labels by string.

    The function searches for the labels containing the specified sub-string in the associated string
    info.
     */
    virtual std::vector<int> getLabelsByString(const String& str) const = 0;
};

/**
@param num_components The number of components (read: Eigenfaces) kept for this Principal
Component Analysis. As a hint: There's no rule how many components (read: Eigenfaces) should be
kept for good reconstruction capabilities. It is based on your input data, so experiment with the
number. Keeping 80 components should almost always be sufficient.
@param threshold The threshold applied in the prediction.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.
-   **THE EIGENFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.
-   This model does not support updating.

### Model internal data:

-   num_components see createEigenFaceRecognizer.
-   threshold see createEigenFaceRecognizer.
-   eigenvalues The eigenvalues for this Principal Component Analysis (ordered descending).
-   eigenvectors The eigenvectors for this Principal Component Analysis (ordered by their
    eigenvalue).
-   mean The sample mean calculated from the training data.
-   projections The projections of the training data.
-   labels The threshold applied in the prediction. If the distance to the nearest neighbor is
    larger than the threshold, this method returns -1.
 */
CV_EXPORTS_W Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
/**
@param num_components The number of components (read: Fisherfaces) kept for this Linear
Discriminant Analysis with the Fisherfaces criterion. It's useful to keep all components, that
means the number of your classes c (read: subjects, persons you want to recognize). If you leave
this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the
correct number (c-1) automatically.
@param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
is larger than the threshold, this method returns -1.

### Notes:

-   Training and prediction must be done on grayscale images, use cvtColor to convert between the
    color spaces.
-   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
    SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
    input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
    the images.
-   This model does not support updating.

### Model internal data:

-   num_components see createFisherFaceRecognizer.
-   threshold see createFisherFaceRecognizer.
-   eigenvalues The eigenvalues for this Linear Discriminant Analysis (ordered descending).
-   eigenvectors The eigenvectors for this Linear Discriminant Analysis (ordered by their
    eigenvalue).
-   mean The sample mean calculated from the training data.
-   projections The projections of the training data.
-   labels The labels corresponding to the projections.
 */
CV_EXPORTS_W Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
/**
@param radius The radius used for building the Circular Local Binary Pattern. The greater the
radius, the
@param neighbors The number of sample points to build a Circular Local Binary Pattern from. An
appropriate value is to use `8` sample points. Keep in mind: the more sample points you include,
the higher the computational cost.
@param grid_x The number of cells in the horizontal direction, 8 is a common value used in
publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
feature vector.
@param grid_y The number of cells in the vertical direction, 8 is a common value used in
publications. The more cells, the finer the grid, the higher the dimensionality of the resulting
feature vector.
@param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
is larger than the threshold, this method returns -1.

### Notes:

-   The Circular Local Binary Patterns (used in training and prediction) expect the data given as
    grayscale images, use cvtColor to convert between the color spaces.
-   This model supports updating.

### Model internal data:

-   radius see createLBPHFaceRecognizer.
-   neighbors see createLBPHFaceRecognizer.
-   grid_x see createLBPHFaceRecognizer.
-   grid_y see createLBPHFaceRecognizer.
-   threshold see createLBPHFaceRecognizer.
-   histograms Local Binary Patterns Histograms calculated from the given training data (empty if
    none was given).
-   labels Labels corresponding to the calculated Local Binary Patterns Histograms.
 */
CV_EXPORTS_W Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

bool initModule_facerec();

//! @}

}} //namespace cv::face

#endif //__OPENCV_FACEREC_HPP__
