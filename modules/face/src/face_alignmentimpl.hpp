// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_FACE_ALIGNMENTIMPL_HPP__
#define __OPENCV_FACE_ALIGNMENTIMPL_HPP__
#include "opencv2/face.hpp"
#include <string>
#include <sstream>
#include <vector>
#include <fstream>
#include <queue>
#include <algorithm>
#include <ctime>

using namespace std;
namespace cv{
namespace face{
/**@brief structure determining split in regression tree
*/
struct splitr{
        //!index1 Index of the first coordinates among the test coordinates for deciding split.
        uint64_t index1;
        //! index2 index of the second coordinate among the test coordinates for deciding split.
        uint64_t index2;
        //! thresh threshold for deciding the split.
        float thresh;
};
/** @brief represents a node of the regression tree*/
struct node_info{
    //First pixel coordinate of split
    long index1;
    //Second pixel coordinate .split
    long index2;
    long depth;
    long node_no;
};
/** @brief regression tree structure. Each leaf node is a vector storing residual shape.
* The tree is represented as vector of leaves.
*/
struct tree_node{
    splitr split;
    std::vector<Point2f> leaf;
};
struct regtree{
    std::vector<tree_node> nodes;
};
/** @brief Represents a training sample
*It contains current shape, difference between actual shape
*and current shape. It also stores the image whose shape is being
*detected.
*/
struct training_sample{
    //! shapeResiduals vector which stores the residual shape remaining to be corrected.
    std::vector<Point2f> shapeResiduals;
    //! current_shape vector containing current estimate of the shape
    std::vector<Point2f> current_shape;
    //! actual_shape vector containing the actual shape of the face or the ground truth.
    std::vector<Point2f> actual_shape;
    //! image A mat object which stores the image.
    Mat image ;
    //! pixel_intensities vector containing pixel intensities of the coordinates chosen for testing
    std::vector<int> pixel_intensities;
    //! pixel_coordinates vector containing pixel coordinates used for testing
    std::vector<Point2f> pixel_coordinates;
    //! bound Rectangle enclosing the face found  in the image for training
    Rect bound;
};
class FacemarkKazemiImpl : public FacemarkKazemi{

public:
    FacemarkKazemiImpl(const FacemarkKazemi::Params& parameters);
    void loadModel(String fs) CV_OVERRIDE;
    bool setFaceDetector(FN_FaceDetector f, void* userdata) CV_OVERRIDE;
    bool getFaces(InputArray image, OutputArray faces) CV_OVERRIDE;
    bool fit(InputArray image, InputArray faces, OutputArrayOfArrays landmarks ) CV_OVERRIDE;
    void training(String imageList, String groundTruth);
    bool training(vector<Mat>& images, vector< vector<Point2f> >& landmarks,string filename,Size scale,string modelFilename) CV_OVERRIDE;
    // Destructor for the class.
    virtual ~FacemarkKazemiImpl() CV_OVERRIDE;

    virtual void read( const FileNode& ) CV_OVERRIDE {}
    virtual void write( FileStorage& ) const CV_OVERRIDE {}

protected:
    FacemarkKazemi::Params params;
    float minmeanx;
    float maxmeanx;
    float minmeany;
    float maxmeany;
    bool isModelLoaded;
    /* meanshape This is a vector which stores the mean shape of all the images used in training*/
    std::vector<Point2f> meanshape;
    std::vector< std::vector<regtree> > loaded_forests;
    std::vector< std::vector<Point2f> > loaded_pixel_coordinates;
    FN_FaceDetector faceDetector;
    void* faceDetectorData;
    bool findNearestLandmarks(std::vector< std::vector<int> >& nearest);
    /*Extract left node of the current node in the regression tree*/
    unsigned long left(unsigned long index);
    // Extract the right node of the current node in the regression tree
    unsigned long right(unsigned long index);
    // This function randomly  generates test splits to get the best split.
    splitr getTestSplits(std::vector<Point2f> pixel_coordinates,int seed);
    // This function writes a split node to the XML file storing the trained model
    void writeSplit(std::ofstream& os,const splitr split);
    // This function writes a leaf node to the binary file storing the trained model
    void writeLeaf(std::ofstream& os, const std::vector<Point2f> &leaf);
    // This function writes a tree to the binary file containing the model
    void writeTree(std::ofstream &f,regtree tree);
    // This function saves the pixel coordinates to a binary file
    void writePixels(std::ofstream& f,int index);
    // This function saves model to the binary file
    bool saveModel(String filename);
    // This funcrion reads pixel coordinates from the model file
    void readPixels(std::ifstream& is,uint64_t index);
    //This function reads the split node of the tree from binary file
    void readSplit(std::ifstream& is, splitr &vec);
    //This function reads a leaf node of the tree.
    void readLeaf(std::ifstream& is, std::vector<Point2f> &leaf);
    /* This function generates pixel intensities of the randomly generated test coordinates used to decide the split.
    */
    bool getPixelIntensities(Mat img,std::vector<Point2f> pixel_coordinates_,std::vector<int>& pixel_intensities_,Rect face);
    //This function initialises the training parameters.
    bool setTrainingParameters(String filename);
    //This function finds a warp matrix that warp the pixels from the normalised space to the actual space
    bool convertToActual(Rect r,Mat &warp);
    //This function finds a warp matrix that warps the pixels from the actual space to normaluised space
    bool convertToUnit(Rect r,Mat &warp);
    /** @brief This function calculates mean shape while training.
    * This function is only called when new training data is supplied by the train function.
    *@param trainlandmarks A vector of type cv::Point2f which stores the landmarks of corresponding images.
    *@param trainimages A vector of type cv::Mat which stores the images which serve as training data.
    *@param faces A vector of type cv::Rect which stores the bounding recatngle of each training image
    *@returns A boolean value. It returns true if mean shape is found successfully else returns false.
    */
    bool calcMeanShape(std::vector< std::vector<Point2f> > & trainlandmarks,std::vector<Mat>& trainimages,std::vector<Rect>& faces);
    /** @brief This functions scales the annotations to a common size which is considered same for all images.
    * @param trainlandmarks A vector of type cv::Point2f stores the landmarks of the corresponding training images.
    * @param trainimages A vector of type cv::Mat which stores the images which are to be scaled.
    * @param s A variable of type cv::Size stores the common size to which all the images are scaled.
    * @returns A boolean value. It returns true when data is scaled properly else returns false.
    */
    bool scaleData(std::vector< std::vector<Point2f> >& trainlandmarks,
                                    std::vector<Mat>& trainimages , Size s=Size(460,460) );
    // This function gets the landmarks in the meanshape nearest to the pixel coordinates.
    unsigned long getNearestLandmark (Point2f pixels );
    // This function gets the relative position of the test pixel coordinates relative to the current shape.
    bool getRelativePixels(std::vector<Point2f> sample,std::vector<Point2f>& pixel_coordinates , std::vector<int> nearest_landmark = std::vector<int>());
    // This function partitions samples according to the split
    unsigned long divideSamples (splitr split,std::vector<training_sample>& samples,unsigned long start,unsigned long end);
    // This function fits a regression tree according to the shape residuals calculated to give weak learners for GBT algorithm.
    bool buildRegtree(regtree &tree,std::vector<training_sample>& samples,std::vector<Point2f> pixel_coordinates);
    // This function greedily decides the best split among the test splits generated.
    bool getBestSplit(std::vector<Point2f> pixel_coordinates, std::vector<training_sample>& samples,unsigned long start ,
                                        unsigned long end,splitr& split,std::vector< std::vector<Point2f> >& sum,long node_no);
    // This function randomly generates test coordinates for each level of cascade.
    void getTestCoordinates ();
    // This function implements gradient boosting by fitting regression trees
    std::vector<regtree> gradientBoosting(std::vector<training_sample>& samples,std::vector<Point2f> pixel_coordinates);
    // This function creates training sample by randomly assigning a current shape from set of shapes available.
    void createLeafNode(regtree& tree,long node_no,std::vector<Point2f> assign);
    // This function creates a split node in the regression tree.
    void createSplitNode(regtree& tree, splitr split,long node_no);
    // This function prepares the training samples
    bool createTrainingSamples(std::vector<training_sample> &samples,std::vector<Mat> images,std::vector< std::vector<Point2f> > landmarks,
    std::vector<Rect> rectangle);
    //This function generates a split
    bool generateSplit(std::queue<node_info>& curr,std::vector<Point2f> pixel_coordinates, std::vector<training_sample>& samples,
                                        splitr &split , std::vector< std::vector<Point2f> >& sum);
    bool setMeanExtreme();
    //friend class getRelShape;
    friend class getRelPixels;
};
}//face
}//cv
#endif
