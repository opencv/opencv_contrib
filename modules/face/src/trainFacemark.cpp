// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "face_alignmentimpl.hpp"
#include "opencv2/calib3d.hpp"
#include <climits>

using namespace std;
namespace cv{
namespace face{
// Threading helper classes
class getDiffShape : public ParallelLoopBody
{
    public:
        getDiffShape(vector<training_sample>* samples_) :
        samples(samples_)
        {
        }
        virtual void operator()( const cv::Range& range) const CV_OVERRIDE
        {
            for(size_t j = (size_t)range.start; j < (size_t)range.end; ++j){
                (*samples)[j].shapeResiduals.resize((*samples)[j].current_shape.size());
                for(unsigned long k=0;k<(*samples)[j].current_shape.size();k++)
                    (*samples)[j].shapeResiduals[k]=(*samples)[j].actual_shape[k]-(*samples)[j].current_shape[k];
            }
        }
    private:
        vector<training_sample>* samples;
};
class getRelPixels : public ParallelLoopBody
{
    public:
        getRelPixels(vector<training_sample>* samples_,FacemarkKazemiImpl& object_) :
        samples(samples_),
        object(object_)
        {
        }
        virtual void operator()( const cv::Range& range) const CV_OVERRIDE
        {
            for (size_t j = (size_t)range.start; j < (size_t)range.end; ++j){
                object.getRelativePixels(((*samples)[j]).current_shape,((*samples)[j]).pixel_coordinates);
            }
        }
    private:
        vector<training_sample>* samples;
        FacemarkKazemiImpl& object;
};
//This function initialises the training parameters.
bool FacemarkKazemiImpl::setTrainingParameters(String filename){
    cout << "Reading Training Parameters " << endl;
    FileStorage fs;
    fs.open(filename, FileStorage::READ);
    if (!fs.isOpened())
    {   String error_message = "Error while opening configuration file.Aborting..";
        CV_Error(Error::StsBadArg, error_message);
    }
    int cascade_depth_;
    int tree_depth_;
    int num_trees_per_cascade_level_;
    float learning_rate_;
    int oversampling_amount_;
    int num_test_coordinates_;
    float lambda_;
    int num_test_splits_;
    fs["cascade_depth"]>> cascade_depth_;
    fs["tree_depth"]>> tree_depth_;
    fs["num_trees_per_cascade_level"] >> num_trees_per_cascade_level_;
    fs["learning_rate"] >> learning_rate_;
    fs["oversampling_amount"] >> oversampling_amount_;
    fs["num_test_coordinates"] >> num_test_coordinates_;
    fs["lambda"] >> lambda_;
    fs["num_test_splits"] >> num_test_splits_;
    params.cascade_depth = (unsigned long)cascade_depth_;
    params.tree_depth = (unsigned long) tree_depth_;
    params.num_trees_per_cascade_level = (unsigned long) num_trees_per_cascade_level_;
    params.learning_rate = (float) learning_rate_;
    params.oversampling_amount = (unsigned long) oversampling_amount_;
    params.num_test_coordinates = (unsigned  long) num_test_coordinates_;
    params.lambda = (float) lambda_;
    params.num_test_splits = (unsigned long) num_test_splits_;
    fs.release();
    cout<<"Parameters loaded"<<endl;
    return true;
}
void FacemarkKazemiImpl::getTestCoordinates ()
{
    for(unsigned long i = 0; i < params.cascade_depth; ++i){
        vector<Point2f> temp;
        RNG rng = theRNG();
        for(unsigned long j = 0; j < params.num_test_coordinates; ++j)
        {
            Point2f pt;
            pt.x = (float)rng.uniform(minmeanx,maxmeanx);
            pt.y = (float)rng.uniform(minmeany,maxmeany);
            temp.push_back(pt);
        }
        loaded_pixel_coordinates.push_back(temp);
    }
}
unsigned long FacemarkKazemiImpl::  getNearestLandmark(Point2f pixel)
{
    if(meanshape.empty()) {
            // throw error if no data (or simply return -1?)
            String error_message = "The data is not loaded properly by train function. Aborting...";
            CV_Error(Error::StsBadArg, error_message);
    }
    float dist=float(INT_MAX);
    unsigned long index =0;
    for(unsigned long i=0;i<meanshape.size();i++){
        Point2f pt = meanshape[i]-pixel;
        if(sqrt(pt.x*pt.x+pt.y*pt.y)<dist){
            dist=sqrt(pt.x*pt.x+pt.y*pt.y);
            index = i;
        }
    }
    return index;
}
bool FacemarkKazemiImpl :: getRelativePixels(vector<Point2f> sample,vector<Point2f>& pixel_coordinates,std::vector<int> nearest){
    if(sample.size()!=meanshape.size()){
        String error_message = "Error while finding relative shape. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
    }
    Mat transform_mat;
    transform_mat = estimateAffinePartial2D(meanshape, sample);
    unsigned long index;
    for (unsigned long i = 0;i<pixel_coordinates.size();i++) {
        if(!nearest.empty())
            index = nearest[i];
        index = getNearestLandmark(pixel_coordinates[i]);
        pixel_coordinates[i] = pixel_coordinates[i] - meanshape[index];
        Mat C = (Mat_<double>(3,1) << pixel_coordinates[i].x, pixel_coordinates[i].y, 0);
        if(!transform_mat.empty()){
            Mat D =transform_mat*C;
            pixel_coordinates[i].x = float((D.at<double>(0,0)));
            pixel_coordinates[i].y = float((D.at<double>(1,0)));
        }
        pixel_coordinates[i] = pixel_coordinates[i] + sample[index];
    }
    return true;
}
bool FacemarkKazemiImpl::getPixelIntensities(Mat img,vector<Point2f> pixel_coordinates,vector<int>& pixel_intensities,Rect face){
    if(pixel_coordinates.size()==0){
        String error_message = "No pixel coordinates found. Aborting.....";
        CV_Error(Error::StsBadArg, error_message);
    }
    Mat transform_mat;
    convertToActual(face,transform_mat);
    Mat dst = img.clone();
    Mat C,D;
    for(size_t j=0;j<pixel_coordinates.size();j++){
        C = (Mat_<double>(3,1) << pixel_coordinates[j].x, pixel_coordinates[j].y, 1);
        D = transform_mat*C;
        pixel_coordinates[j].x = float(D.at<double>(0,0));
        pixel_coordinates[j].y = float(D.at<double>(1,0));
    }
    int val;
    for(unsigned long j=0;j<pixel_coordinates.size();j++){
        if(pixel_coordinates[j].x>0&&pixel_coordinates[j].x<img.cols&&pixel_coordinates[j].y>0&&pixel_coordinates[j].y<img.rows){
            Vec3b val1 = img.at<Vec3b>((int)pixel_coordinates[j].y,(int)pixel_coordinates[j].x);
            val = (int)(val1[0]+val1[1]+val1[2])/3;
        }
        else
            val = 0;
        pixel_intensities.push_back(val);
    }
    return true;
}
vector<regtree> FacemarkKazemiImpl::gradientBoosting(vector<training_sample>& samples,vector<Point2f> pixel_coordinates){
    vector<regtree> forest;
    vector<Point2f> meanresidual;
    meanresidual.resize(samples[0].shapeResiduals.size());
    for(unsigned long i=0;i<samples.size();i++){
        for(unsigned long j=0;j<samples[i].shapeResiduals.size();j++){
            meanresidual[j]=meanresidual[j]+samples[i].shapeResiduals[j];
        }
    }
    for(unsigned long i=0;i<meanresidual.size();i++){
        meanresidual[i].x=(meanresidual[i].x)/samples.size();
        meanresidual[i].y=(meanresidual[i].y)/samples.size();
    }
    for(unsigned long i=0;i<samples.size();i++){
        for(unsigned long j=0;j<samples[i].shapeResiduals.size();j++)
            samples[i].shapeResiduals[j]=samples[i].shapeResiduals[j]-meanresidual[j];
    }
    for(unsigned long i=0;i<params.num_trees_per_cascade_level;i++){
            regtree tree;
            buildRegtree(tree,samples,pixel_coordinates);
            forest.push_back(tree);
    }
    return forest;
}
bool FacemarkKazemiImpl::createTrainingSamples(vector<training_sample> &samples,vector<Mat> images,vector< vector<Point2f> > landmarks,vector<Rect> rectangle){
    unsigned long in=0;
    samples.resize(params.oversampling_amount*images.size());
    for(unsigned long i=0;i<images.size();i++){
        for(unsigned long j=0;j<params.oversampling_amount;j++){
            samples[in].image=images[i];
            samples[in].actual_shape = landmarks[i];
            samples[in].bound = rectangle[i];
            unsigned long  rindex=i;
            if(in%2==0)
                samples[in].current_shape = meanshape;
            else{
                RNG rng(in);
                rindex =(unsigned long)rng.uniform(0,(int)landmarks.size()-1);
                samples[in].current_shape = landmarks[rindex];
            }
            in++;
        }
    }
    parallel_for_(Range(0,(int)samples.size()),getDiffShape(&samples));
    return true;
}
void FacemarkKazemiImpl :: writeLeaf(ofstream& os, const vector<Point2f> &leaf)
{
    uint64_t size = leaf.size();
    os.write((char*)&size, sizeof(size));
    os.write((char*)&leaf[0], leaf.size() * sizeof(Point2f));
}
void FacemarkKazemiImpl :: writeSplit(ofstream& os, const splitr& vec)
{
    os.write((char*)&vec.index1, sizeof(vec.index1));
    os.write((char*)&vec.index2, sizeof(vec.index2));
    os.write((char*)&vec.thresh, sizeof(vec.thresh));
    uint32_t dummy_ = 0;
    os.write((char*)&dummy_, sizeof(dummy_)); // buggy original writer structure alignment
    CV_CheckEQ((int)(sizeof(vec.index1) + sizeof(vec.index2) + sizeof(vec.thresh) + sizeof(dummy_)), 24, "Invalid build configuration");

}
void FacemarkKazemiImpl :: writeTree(ofstream &f,regtree tree)
{
    string s("num_nodes");
    uint64_t len = s.size();
    f.write((char*)&len, sizeof(len));
    f.write(s.c_str(), len);
    uint64_t num_nodes = tree.nodes.size();
    f.write((char*)&num_nodes,sizeof(num_nodes));
    for(size_t i=0;i<tree.nodes.size();i++){
        if(tree.nodes[i].leaf.empty()){
            s = string("split");
            len = s.size();
            f.write((char*)&len, sizeof(len));
            f.write(s.c_str(), len);
            writeSplit(f,tree.nodes[i].split);
        }
        else{
            s = string("leaf");
            len = s.size();
            f.write((char*)&len, sizeof(len));
            f.write(s.c_str(), len);
            writeLeaf(f,tree.nodes[i].leaf);
        }
    }
}
void FacemarkKazemiImpl :: writePixels(ofstream& f,int index){
    f.write((char*)&loaded_pixel_coordinates[index][0], loaded_pixel_coordinates[index].size() * sizeof(Point2f));
}
bool FacemarkKazemiImpl :: saveModel(String filename){
    ofstream f(filename.c_str(),ios::binary);
    if(!f.is_open()){
        String error_message = "Error while opening file to write model. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
    }
    if(loaded_forests.size()!=loaded_pixel_coordinates.size()){
        String error_message = "Incorrect training data. Aborting....";
        CV_Error(Error::StsBadArg, error_message);
    }
    string s("cascade_depth");
    uint64_t len = s.size();
    f.write((char*)&len, sizeof(len));
    f.write(s.c_str(), len);
    uint64_t cascade_size = loaded_forests.size();
    f.write((char*)&cascade_size,sizeof(cascade_size));
    s = string("pixel_coordinates");
    len = s.size();
    f.write((char*)&len, sizeof(len));
    f.write(s.c_str(), len);
    uint64_t num_pixels = loaded_pixel_coordinates[0].size();
    f.write((char*)&num_pixels,sizeof(num_pixels));
    for(unsigned long i=0;i< loaded_pixel_coordinates.size();i++){
        writePixels(f,i);
    }
    s = string("mean_shape");
    uint64_t len1 = s.size();
    f.write((char*)&len1, sizeof(len1));
    f.write(s.c_str(), len1);
    uint64_t mean_shape_size = meanshape.size();
    f.write((char*)&mean_shape_size,sizeof(mean_shape_size));
    f.write((char*)&meanshape[0], meanshape.size() * sizeof(Point2f));
    s = string("num_trees");
    len = s.size();
    f.write((char*)&len, sizeof(len));
    f.write(s.c_str(), len);
    uint64_t num_trees = loaded_forests[0].size();
    f.write((char*)&num_trees,sizeof(num_trees));
    for(unsigned long i=0 ; i<loaded_forests.size() ; i++){
        for(unsigned long j=0 ; j<loaded_forests[i].size() ; j++){
            writeTree(f,loaded_forests[i][j]);
       }
    }
    return true;
}
void FacemarkKazemiImpl::training(String imageList, String groundTruth){
    imageList.clear();
    groundTruth.clear();
    String error_message = "Less arguments than required";
    CV_Error(Error::StsBadArg, error_message);
}
bool FacemarkKazemiImpl::training(vector<Mat>& images, vector< vector<Point2f> >& landmarks,string filename,Size scale,string modelFilename){
    if(!setTrainingParameters(filename)){
        String error_message = "Error while loading training parameters";
        CV_Error(Error::StsBadArg, error_message);
    }
    vector<Rect> rectangles;
    scaleData(landmarks,images,scale);
    calcMeanShape(landmarks,images,rectangles);
    if(images.size()!=landmarks.size()){
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly. Aborting training function....";
        CV_Error(Error::StsBadArg, error_message);
    }
    vector<training_sample> samples;
    getTestCoordinates();
    createTrainingSamples(samples,images,landmarks,rectangles);
    images.clear();
    landmarks.clear();
    rectangles.clear();
    for(unsigned long i=0;i< params.cascade_depth;i++){
        cout<<"Training regressor "<<i<<"..."<<endl;
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
            (*it).pixel_coordinates = loaded_pixel_coordinates[i];
        }
        parallel_for_(Range(0,(int)samples.size()),getRelPixels(&samples,*this));
        for (std::vector<training_sample>::iterator it = samples.begin(); it != samples.end(); it++) {
            getPixelIntensities((*it).image,(*it).pixel_coordinates,(*it).pixel_intensities,(*it).bound);
        }
        loaded_forests.push_back(gradientBoosting(samples,loaded_pixel_coordinates[i]));
    }
    saveModel(modelFilename);
    return true;
}
}//cv
}//face
