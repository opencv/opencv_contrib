/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.
                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)
Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.
This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

This file was part of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

#include "precomp.hpp"
#include "opencv2/face.hpp"
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdarg>

namespace cv {
namespace face {

#define TIMER_BEGIN { double __time__ = (double)getTickCount();
#define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
#define TIMER_END   }

#define SIMILARITY_TRANSFORM(x, y, scale, rotate) do {            \
    double x_tmp = scale * (rotate(0, 0)*x + rotate(0, 1)*y); \
    double y_tmp = scale * (rotate(1, 0)*x + rotate(1, 1)*y); \
    x = x_tmp; y = y_tmp;                                     \
} while(0)

FacemarkLBF::Params::Params(){

    cascade_face = "";
    shape_offset = 0.0;
    n_landmarks = 68;
    initShape_n = 10;
    stages_n=5;
    tree_n=6;
    tree_depth=5;
    bagging_overlap = 0.4;
    model_filename = "";
    save_model = true;
    verbose = true;
    seed = 0;

    int _pupils[][6] = { { 36, 37, 38, 39, 40, 41 }, { 42, 43, 44, 45, 46, 47 } };
    for (int i = 0; i < 6; i++) {
        pupils[0].push_back(_pupils[0][i]);
        pupils[1].push_back(_pupils[1][i]);
    }

    int _feats_m[] = { 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
    double _radius_m[] = { 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
    for (int i = 0; i < 10; i++) {
        feats_m.push_back(_feats_m[i]);
        radius_m.push_back(_radius_m[i]);
    }

    detectROI = Rect(-1,-1,-1,-1);
}

void FacemarkLBF::Params::read( const cv::FileNode& fn ){
    *this = FacemarkLBF::Params();

    if (!fn["verbose"].empty())
        fn["verbose"] >> verbose;

}

void FacemarkLBF::Params::write( cv::FileStorage& fs ) const{
    fs << "verbose" << verbose;
}

class FacemarkLBFImpl : public FacemarkLBF {
public:
    FacemarkLBFImpl( const FacemarkLBF::Params &parameters = FacemarkLBF::Params() );

    void read( const FileNode& /*fn*/ ) CV_OVERRIDE;
    void write( FileStorage& /*fs*/ ) const CV_OVERRIDE;

    void loadModel(String fs) CV_OVERRIDE;

    bool setFaceDetector(bool(*f)(InputArray , OutputArray, void * extra_params ), void* userData) CV_OVERRIDE;
    bool getFaces(InputArray image, OutputArray faces) CV_OVERRIDE;
    bool getData(void * items) CV_OVERRIDE;

    Params params;

protected:

    bool fit(InputArray image, InputArray faces, OutputArrayOfArrays landmarks) CV_OVERRIDE;
    bool fitImpl( const Mat image, std::vector<Point2f> & landmarks );//!< from a face

    bool addTrainingSample(InputArray image, InputArray landmarks) CV_OVERRIDE;
    void training(void* parameters) CV_OVERRIDE;

    Rect getBBox(Mat &img, const Mat_<double> shape);
    void prepareTrainingData(Mat img, std::vector<Point2f> facePoints,
        std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes);
    void data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);
    Mat getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);

    bool defaultFaceDetector(const Mat& image, std::vector<Rect>& faces);

    CascadeClassifier face_cascade;
    FN_FaceDetector faceDetector;
    void* faceDetectorData;

    /*training data*/
    std::vector<std::vector<Point2f> > data_facemarks; //original position
    std::vector<Mat> data_faces; //face ROI
    std::vector<BBox> data_boxes;
    std::vector<Mat> data_shapes; //position in the face ROI

private:
    bool isModelTrained;

    /*---------------LBF Class---------------------*/
    class LBF {
    public:
        void calcSimilarityTransform(const Mat &shape1, const Mat &shape2, double &scale, Mat &rotate);
        std::vector<Mat> getDeltaShapes(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                                   std::vector<BBox> &bboxes, Mat &mean_shape);
        double calcVariance(const Mat &vec);
        double calcVariance(const std::vector<double> &vec);
        double calcMeanError(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes, int landmark_n , std::vector<int> &left, std::vector<int> &right );

    };

    /*---------------RandomTree Class---------------------*/
    class RandomTree : public LBF {
    public:
        RandomTree(){};
        ~RandomTree(){};

        void initTree(int landmark_id, int depth, std::vector<int>, std::vector<double>);
        void train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                   std::vector<Mat> &delta_shapes, Mat &mean_shape, std::vector<int> &index, int stage);
        void splitNode(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes,
                      cv::Mat &delta_shapes, cv::Mat &mean_shape, std::vector<int> &root, int idx, int stage);

        void write(FileStorage fs, int forestId, int i, int j);
        void read(FileStorage fs, int forestId, int i, int j);

        int depth;
        int nodes_n;
        int landmark_id;
        cv::Mat_<double> feats;
        std::vector<int> thresholds;

        std::vector<int> params_feats_m;
        std::vector<double> params_radius_m;
    };
    /*---------------RandomForest Class---------------------*/
    class RandomForest : public LBF {
    public:
        RandomForest(){};
        ~RandomForest(){};

        void initForest(int landmark_n, int trees_n, int tree_depth, double ,  std::vector<int>, std::vector<double>, bool);
        void train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, \
                   std::vector<BBox> &bboxes, std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, int stage);
        Mat generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape);

        void write(FileStorage fs, int forestId);
        void read(FileStorage fs, int forestId);

        bool verbose;
        int landmark_n;
        int trees_n, tree_depth;
        double overlap_ratio;
        std::vector<std::vector<RandomTree> > random_trees;

        std::vector<int> feats_m;
        std::vector<double> radius_m;
    };
    /*---------------Regressor Class---------------------*/
    class Regressor  : public LBF {
    protected:
        struct feature_node{
            int index;
            double value;
        };
    public:
        Regressor(){};
        ~Regressor(){};

        void initRegressor(Params);
        void trainRegressor(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
                   std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
                   cv::Mat &mean_shape, int start_from, Params );
        Mat globalRegressionPredict(const Mat &lbf, int stage);
        Mat predict(Mat &img, BBox &bbox);

        void write(FileStorage fs, Params config);
        void read(FileStorage fs, Params & config);

        void globalRegressionTrain(
            std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes,
            int stage, Params config
        );

        Mat supportVectorRegression(
            feature_node **x, double *y, int nsamples, int feat_size, bool verbose=0
        );

        int stages_n;
        int landmark_n;
        cv::Mat mean_shape;
        std::vector<RandomForest> random_forests;
        std::vector<cv::Mat> gl_regression_weights;

    }; // LBF

    Regressor regressor;
}; // class

/*
* Constructor
*/
Ptr<FacemarkLBF> FacemarkLBF::create(const FacemarkLBF::Params &parameters){
    return Ptr<FacemarkLBFImpl>(new FacemarkLBFImpl(parameters));
}
/*
* Constructor
*/
Ptr<Facemark> createFacemarkLBF(){
    const FacemarkLBF::Params parameters;
    return Ptr<FacemarkLBFImpl>(new FacemarkLBFImpl(parameters));
}

FacemarkLBFImpl::FacemarkLBFImpl( const FacemarkLBF::Params &parameters ) :
    faceDetector(NULL), faceDetectorData(NULL)
{
    isModelTrained = false;
    params = parameters;
}

bool FacemarkLBFImpl::setFaceDetector(bool(*f)(InputArray , OutputArray, void * extra_params ), void* userData){
    faceDetector = f;
    faceDetectorData = userData;
    return true;
}

bool FacemarkLBFImpl::getFaces(InputArray image, OutputArray faces_)
{
    if (!faceDetector)
    {
        std::vector<Rect> faces;
        defaultFaceDetector(image.getMat(), faces);
        Mat(faces).copyTo(faces_);
        return true;
    }
    return faceDetector(image, faces_, faceDetectorData);
}

bool FacemarkLBFImpl::defaultFaceDetector(const Mat& image, std::vector<Rect>& faces){
    Mat gray;

    faces.clear();

    if (image.channels() > 1)
    {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    }
    else
    {
        gray = image;
    }

    equalizeHist(gray, gray);

    if (face_cascade.empty())
    {
        { /* check the cascade classifier file */
            std::ifstream infile;
            infile.open(params.cascade_face.c_str(), std::ios::in);
            if (!infile)
                CV_Error_(Error::StsBadArg, ("The cascade classifier model is not found: %s", params.cascade_face.c_str()));
        }
        face_cascade.load(params.cascade_face.c_str());
        CV_Assert(!face_cascade.empty());
    }
    face_cascade.detectMultiScale(gray, faces, 1.05, 2, CASCADE_SCALE_IMAGE, Size(30, 30) );
    return true;
}

bool FacemarkLBFImpl::getData(void * items){
    CV_UNUSED(items);
    return false;
}

bool FacemarkLBFImpl::addTrainingSample(InputArray image, InputArray landmarks){
    // FIXIT
    std::vector<Point2f> & _landmarks = *(std::vector<Point2f>*)landmarks.getObj();
    prepareTrainingData(image.getMat(), _landmarks, data_faces, data_shapes, data_boxes);
    return true;
}

void FacemarkLBFImpl::training(void* parameters){
    CV_UNUSED(parameters);

    if (data_faces.empty())
    {
        CV_Error(Error::StsBadArg, "Training data is not provided. Consider to add using addTrainingSample() function!");
    }

    if (params.model_filename.empty() && params.save_model)
    {
        CV_Error(Error::StsBadArg, "The parameter model_filename should be set!");
    }

    // flip the image and swap the landmark position
    data_augmentation(data_faces, data_shapes, data_boxes);

    Mat mean_shape = getMeanShape(data_shapes, data_boxes);

    int N = (int)data_faces.size();
    int L = N*params.initShape_n;
    std::vector<Mat> imgs(L), gt_shapes(L), current_shapes(L);
    std::vector<BBox> bboxes(L);
    RNG rng(params.seed);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < params.initShape_n; j++) {
            int idx = i*params.initShape_n + j;
            int k = rng.uniform(0, N - 1);
            k = (k >= i) ? k + 1 : k; // require k != i
            imgs[idx] = data_faces[i];
            gt_shapes[idx] = data_shapes[i];
            bboxes[idx] = data_boxes[i];
            current_shapes[idx] = data_boxes[i].reproject(data_boxes[k].project(data_shapes[k]));
        }
    }

    regressor.initRegressor(params);
    regressor.trainRegressor(imgs, gt_shapes, current_shapes, bboxes, mean_shape, 0, params);

    if(params.save_model){
        FileStorage fs(params.model_filename.c_str(),FileStorage::WRITE_BASE64);
        regressor.write(fs, params);
    }

    isModelTrained = true;
}

/**
 * @brief Copy the contents of a corners vector to an OutputArray, settings its size.
 */
static void _copyVector2Output(std::vector< std::vector< Point2f > > &vec, OutputArrayOfArrays out)
{
    out.create((int)vec.size(), 1, CV_32FC2);

    if (out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(68, 1, CV_32FC2, i);
            Mat &m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if (out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(68, 1, CV_32FC2, i);
            UMat &m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else if (out.kind() == _OutputArray::STD_VECTOR_VECTOR) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(68, 1, CV_32FC2, i);
            Mat m = out.getMat(i);
            Mat(Mat(vec[i]).t()).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
            "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}

bool FacemarkLBFImpl::fit(InputArray image, InputArray roi, OutputArrayOfArrays _landmarks)
{
    Mat roimat = roi.getMat();
    std::vector<Rect> faces = roimat.reshape(4, roimat.rows);
    if (faces.empty()) return false;

    std::vector<std::vector<Point2f> > landmarks;

    landmarks.resize(faces.size());

    for(unsigned i=0; i<faces.size();i++){
        params.detectROI = faces[i];
        fitImpl(image.getMat(), landmarks[i]);
    }
    _copyVector2Output(landmarks, _landmarks);
    return true;
}

bool FacemarkLBFImpl::fitImpl( const Mat image, std::vector<Point2f>& landmarks){
    if (landmarks.size()>0)
        landmarks.clear();

    if (!isModelTrained) {
        CV_Error(Error::StsBadArg, "The LBF model is not trained yet. Please provide a trained model.");
    }

    Mat img;
    if(image.channels()>1){
        cvtColor(image,img,COLOR_BGR2GRAY);
    }else{
        img = image;
    }

    Rect box;
    if (params.detectROI.width>0){
        box = params.detectROI;
    }else{
        std::vector<Rect> rects;

        if (!getFaces(img, rects)) return 0;
        if (rects.empty())  return 0; //failed to get face
        box = rects[0];
    }

    double min_x, min_y, max_x, max_y;
    min_x = std::max(0., (double)box.x - box.width / 2);
    max_x = std::min(img.cols - 1., (double)box.x+box.width + box.width / 2);
    min_y = std::max(0., (double)box.y - box.height / 2);
    max_y = std::min(img.rows - 1., (double)box.y + box.height + box.height / 2);

    double w = max_x - min_x;
    double h = max_y - min_y;

    BBox bbox(box.x - min_x, box.y - min_y, box.width, box.height);
    Mat crop = img(Rect((int)min_x, (int)min_y, (int)w, (int)h)).clone();
    Mat shape = regressor.predict(crop, bbox);

    if(params.detectROI.width>0){
        landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
        params.detectROI.width = -1;
    }else{
        landmarks = Mat(shape.reshape(2)+Scalar(min_x, min_y));
    }

    return 1;
}

void FacemarkLBFImpl::read( const cv::FileNode& fn ){
    params.read( fn );
}

void FacemarkLBFImpl::write( cv::FileStorage& fs ) const {
    params.write( fs );
}

void FacemarkLBFImpl::loadModel(String s){
    if(params.verbose) printf("loading data from : %s\n", s.c_str());
    std::ifstream infile;
    infile.open(s.c_str(), std::ios::in);
    if (!infile) {
        CV_Error(Error::StsBadArg, "No valid input file was given, please check the given filename.");
    }

    FileStorage fs(s.c_str(),FileStorage::READ);
    regressor.read(fs, params);

    isModelTrained = true;
}

Rect FacemarkLBFImpl::getBBox(Mat &img, const Mat_<double> shape) {
    std::vector<Rect> rects;

    if(!faceDetector){
        defaultFaceDetector(img, rects);
    }else{
        faceDetector(img, rects, faceDetectorData);
    }

    if (rects.size() == 0) return Rect(-1, -1, -1, -1);
    double center_x=0, center_y=0, min_x, max_x, min_y, max_y;

    min_x = shape(0, 0);
    max_x = shape(0, 0);
    min_y = shape(0, 1);
    max_y = shape(0, 1);

    for (int i = 0; i < shape.rows; i++) {
        center_x += shape(i, 0);
        center_y += shape(i, 1);
        min_x = std::min(min_x, shape(i, 0));
        max_x = std::max(max_x, shape(i, 0));
        min_y = std::min(min_y, shape(i, 1));
        max_y = std::max(max_y, shape(i, 1));
    }
    center_x /= shape.rows;
    center_y /= shape.rows;

    for (int i = 0; i < (int)rects.size(); i++) {
        Rect r = rects[i];
        if (max_x - min_x > r.width*1.5) continue;
        if (max_y - min_y > r.height*1.5) continue;
        if (abs(center_x - (r.x + r.width / 2)) > r.width / 2) continue;
        if (abs(center_y - (r.y + r.height / 2)) > r.height / 2) continue;
        return r;
    }
    return Rect(-1, -1, -1, -1);
}

void FacemarkLBFImpl::prepareTrainingData(Mat img, std::vector<Point2f> facePoints,
    std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes)
{
    Mat shape;
    Mat _shape = Mat(facePoints).reshape(1);
    Rect box = getBBox(img, _shape);

    if(img.channels()>1){
        cvtColor(img,img,COLOR_BGR2GRAY);
    }

    if(box.x != -1){
        _shape.convertTo(shape, CV_64FC1);
        Mat sx = shape.col(0);
        Mat sy = shape.col(1);
        double min_x, max_x, min_y, max_y;
        minMaxIdx(sx, &min_x, &max_x);
        minMaxIdx(sy, &min_y, &max_y);

        min_x = std::max(0., min_x - (double)box.width / 2.);
        max_x = std::min(img.cols - 1., max_x + (double)box.width / 2.);
        min_y = std::max(0., min_y - (double)box.height / 2.);
        max_y = std::min(img.rows - 1., max_y + (double)box.height / 2.);

        double w = max_x - min_x;
        double h = max_y - min_y;

        shape = Mat(shape.reshape(2)-Scalar(min_x, min_y)).reshape(1);

        boxes.push_back(BBox(box.x - min_x, box.y - min_y, box.width, box.height));
        Mat crop = img(Rect((int)min_x, (int)min_y, (int)w, (int)h)).clone();
        cropped.push_back(crop);
        shapes.push_back(shape);

    } // if box is valid
} // prepareTrainingData

void FacemarkLBFImpl::data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {
    int N = (int)imgs.size();
    imgs.reserve(2 * N);
    gt_shapes.reserve(2 * N);
    bboxes.reserve(2 * N);
    for (int i = 0; i < N; i++) {
        Mat img_flipped;
        Mat_<double> gt_shape_flipped(gt_shapes[i].size());
        flip(imgs[i], img_flipped, 1);
        int w = img_flipped.cols - 1;
        // int h = img_flipped.rows - 1;
        for (int k = 0; k < gt_shapes[i].rows; k++) {
            gt_shape_flipped(k, 0) = w - gt_shapes[i].at<double>(k, 0);
            gt_shape_flipped(k, 1) = gt_shapes[i].at<double>(k, 1);
        }
        int x_b, y_b, w_b, h_b;
        x_b = w - (int)bboxes[i].x - (int)bboxes[i].width;
        y_b = (int)bboxes[i].y;
        w_b = (int)bboxes[i].width;
        h_b = (int)bboxes[i].height;
        BBox bbox_flipped(x_b, y_b, w_b, h_b);

        imgs.push_back(img_flipped);
        gt_shapes.push_back(gt_shape_flipped);
        bboxes.push_back(bbox_flipped);

    }
#define SWAP(shape, i, j) do { \
        double tmp = shape.at<double>(i-1, 0); \
        shape.at<double>(i-1, 0) = shape.at<double>(j-1, 0); \
        shape.at<double>(j-1, 0) = tmp; \
        tmp =  shape.at<double>(i-1, 1); \
        shape.at<double>(i-1, 1) = shape.at<double>(j-1, 1); \
        shape.at<double>(j-1, 1) = tmp; \
    } while(0)

    if (params.n_landmarks == 29) {
        for (int i = N; i < (int)gt_shapes.size(); i++) {
            SWAP(gt_shapes[i], 1, 2);
            SWAP(gt_shapes[i], 3, 4);
            SWAP(gt_shapes[i], 5, 7);
            SWAP(gt_shapes[i], 6, 8);
            SWAP(gt_shapes[i], 13, 15);
            SWAP(gt_shapes[i], 9, 10);
            SWAP(gt_shapes[i], 11, 12);
            SWAP(gt_shapes[i], 17, 18);
            SWAP(gt_shapes[i], 14, 16);
            SWAP(gt_shapes[i], 19, 20);
            SWAP(gt_shapes[i], 23, 24);
        }
    }
    else if (params.n_landmarks == 68) {
        for (int i = N; i < (int)gt_shapes.size(); i++) {
            for (int k = 1; k <= 8; k++) SWAP(gt_shapes[i], k, 18 - k);
            for (int k = 18; k <= 22; k++) SWAP(gt_shapes[i], k, 45 - k);
            for (int k = 37; k <= 40; k++) SWAP(gt_shapes[i], k, 83 - k);
            SWAP(gt_shapes[i], 42, 47);
            SWAP(gt_shapes[i], 41, 48);
            SWAP(gt_shapes[i], 32, 36);
            SWAP(gt_shapes[i], 33, 35);
            for (int k = 49; k <= 51; k++) SWAP(gt_shapes[i], k, 104 - k);
            SWAP(gt_shapes[i], 60, 56);
            SWAP(gt_shapes[i], 59, 57);
            SWAP(gt_shapes[i], 61, 65);
            SWAP(gt_shapes[i], 62, 64);
            SWAP(gt_shapes[i], 68, 66);
        }
    }
    else {
        printf("Wrong n_landmarks, currently only 29 and 68 landmark points are supported");
    }

#undef SWAP

}

FacemarkLBFImpl::BBox::BBox() {}
FacemarkLBFImpl::BBox::~BBox() {}

FacemarkLBFImpl::BBox::BBox(double _x, double _y, double w, double h) {
    x = _x;
    y = _y;
    width = w;
    height = h;
    x_center = x + w / 2.;
    y_center = y + h / 2.;
    x_scale = w / 2.;
    y_scale = h / 2.;
}

// Project absolute shape to relative shape binding to this bbox
Mat FacemarkLBFImpl::BBox::project(const Mat &shape) const {
    Mat res(shape.rows, shape.cols, CV_64FC1);
    for (int i = 0; i < shape.rows; i++) {
        res.at<double>(i, 0) = (shape.at<double>(i, 0) - x_center) / x_scale;
        res.at<double>(i, 1) = (shape.at<double>(i, 1) - y_center) / y_scale;
    }
    return res;
}

// Project relative shape to absolute shape binding to this bbox
Mat FacemarkLBFImpl::BBox::reproject(const Mat &shape) const {
    Mat res(shape.rows, shape.cols, CV_64FC1);
    for (int i = 0; i < shape.rows; i++) {
        res.at<double>(i, 0) = shape.at<double>(i, 0)*x_scale + x_center;
        res.at<double>(i, 1) = shape.at<double>(i, 1)*y_scale + y_center;
    }
    return res;
}

Mat FacemarkLBFImpl::getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {

    int N = (int)gt_shapes.size();
    Mat mean_shape = Mat::zeros(gt_shapes[0].rows, 2, CV_64FC1);
    for (int i = 0; i < N; i++) {
        mean_shape += bboxes[i].project(gt_shapes[i]);
    }
    mean_shape /= N;
    return mean_shape;
}

// Similarity Transform, project shape2 to shape1
// p1 ~= scale * rotate * p2, p1 and p2 are vector in math
void FacemarkLBFImpl::LBF::calcSimilarityTransform(const Mat &shape1, const Mat &shape2, double &scale, Mat &rotate) {
    Mat_<double> rotate_(2, 2);
    double x1_center, y1_center, x2_center, y2_center;
    x1_center = cv::mean(shape1.col(0))[0];
    y1_center = cv::mean(shape1.col(1))[0];
    x2_center = cv::mean(shape2.col(0))[0];
    y2_center = cv::mean(shape2.col(1))[0];

    Mat temp1(shape1.rows, shape1.cols, CV_64FC1);
    Mat temp2(shape2.rows, shape2.cols, CV_64FC1);
    temp1.col(0) = shape1.col(0) - x1_center;
    temp1.col(1) = shape1.col(1) - y1_center;
    temp2.col(0) = shape2.col(0) - x2_center;
    temp2.col(1) = shape2.col(1) - y2_center;

    Mat_<double> covar1, covar2;
    Mat_<double> mean1, mean2;
    calcCovarMatrix(temp1, covar1, mean1, COVAR_COLS);
    calcCovarMatrix(temp2, covar2, mean2, COVAR_COLS);

    double s1 = sqrt(cv::norm(covar1));
    double s2 = sqrt(cv::norm(covar2));
    scale = s1 / s2;
    temp1 /= s1;
    temp2 /= s2;

    double num = temp1.col(1).dot(temp2.col(0)) - temp1.col(0).dot(temp2.col(1));
    double den = temp1.col(0).dot(temp2.col(0)) + temp1.col(1).dot(temp2.col(1));
    double normed = sqrt(num*num + den*den);
    double sin_theta = num / normed;
    double cos_theta = den / normed;
    rotate_(0, 0) = cos_theta; rotate_(0, 1) = -sin_theta;
    rotate_(1, 0) = sin_theta; rotate_(1, 1) = cos_theta;
    rotate = rotate_;
}

// Get relative delta_shapes for predicting target
std::vector<Mat> FacemarkLBFImpl::LBF::getDeltaShapes(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                           std::vector<BBox> &bboxes, Mat &mean_shape) {
    std::vector<Mat> delta_shapes;
    int N = (int)gt_shapes.size();
    delta_shapes.resize(N);
    double scale;
    Mat_<double> rotate;
    for (int i = 0; i < N; i++) {
        delta_shapes[i] = bboxes[i].project(gt_shapes[i]) - bboxes[i].project(current_shapes[i]);
        calcSimilarityTransform(mean_shape, bboxes[i].project(current_shapes[i]), scale, rotate);
        // delta_shapes[i] = scale * delta_shapes[i] * rotate.t(); // the result is better without this part
    }
    return delta_shapes;
}

double FacemarkLBFImpl::LBF::calcVariance(const Mat &vec) {
    double m1 = cv::mean(vec)[0];
    double m2 = cv::mean(vec.mul(vec))[0];
    double variance = m2 - m1*m1;
    return variance;
}

double FacemarkLBFImpl::LBF::calcVariance(const std::vector<double> &vec) {
    if (vec.size() == 0) return 0.;
    Mat_<double> vec_(vec);
    double m1 = cv::mean(vec_)[0];
    double m2 = cv::mean(vec_.mul(vec_))[0];
    double variance = m2 - m1*m1;
    return variance;
}

double FacemarkLBFImpl::LBF::calcMeanError(std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes, int landmark_n , std::vector<int> &left, std::vector<int> &right ) {
    int N = (int)gt_shapes.size();

    double e = 0;
    // every train data
    for (int i = 0; i < N; i++) {
        const Mat_<double> &gt_shape = (Mat_<double>)gt_shapes[i];
        const Mat_<double> &current_shape = (Mat_<double>)current_shapes[i];
        double x1, y1, x2, y2;
        x1 = x2 = y1 = y2 = 0;
        for (int j = 0; j < (int)left.size(); j++) {
            x1 += gt_shape(left[j], 0);
            y1 += gt_shape(left[j], 1);
        }
        for (int j = 0; j < (int)right.size(); j++) {
            x2 += gt_shape(right[j], 0);
            y2 += gt_shape(right[j], 1);
        }
        x1 /= left.size(); y1 /= left.size();
        x2 /= right.size(); y2 /= right.size();
        double pupils_distance = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
        // every landmark
        double e_ = 0;
        for (int j = 0; j < landmark_n; j++) {
            e_ += norm(gt_shape.row(j) - current_shape.row(j));
        }
        e += e_ / pupils_distance;
    }
    e /= N*landmark_n;
    return e;
}

/*---------------RandomTree Implementation---------------------*/
void FacemarkLBFImpl::RandomTree::initTree(int _landmark_id, int _depth, std::vector<int> feats_m, std::vector<double> radius_m) {
    landmark_id = _landmark_id;
    depth = _depth;
    nodes_n = 1 << depth;
    feats = Mat::zeros(nodes_n, 4, CV_64FC1);
    thresholds.resize(nodes_n);

    params_feats_m = feats_m;
    params_radius_m = radius_m;
}

void FacemarkLBFImpl::RandomTree::train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                       std::vector<Mat> &delta_shapes, Mat &mean_shape, std::vector<int> &index, int stage) {
    Mat_<double> delta_shapes_((int)delta_shapes.size(), 2);
    for (int i = 0; i < (int)delta_shapes.size(); i++) {
        delta_shapes_(i, 0) = delta_shapes[i].at<double>(landmark_id, 0);
        delta_shapes_(i, 1) = delta_shapes[i].at<double>(landmark_id, 1);
    }
    splitNode(imgs, current_shapes, bboxes, delta_shapes_, mean_shape, index, 1, stage);
}

void FacemarkLBFImpl::RandomTree::splitNode(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                           Mat &delta_shapes, Mat &mean_shape, std::vector<int> &root, int idx, int stage) {

    int N = (int)root.size();
    if (N == 0) {
        thresholds[idx] = 0;
        feats.row(idx).setTo(0);
        std::vector<int> left, right;
        // split left and right child in DFS
        if (2 * idx < feats.rows / 2)
            splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
        if (2 * idx + 1 < feats.rows / 2)
            splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
        return;
    }

    int feats_m = params_feats_m[stage];
    double radius_m = params_radius_m[stage];
    Mat_<double> candidate_feats(feats_m, 4);
    RNG rng(getTickCount());
    // generate feature pool
    for (int i = 0; i < feats_m; i++) {
        double x1, y1, x2, y2;
        x1 = rng.uniform(-1., 1.); y1 = rng.uniform(-1., 1.);
        x2 = rng.uniform(-1., 1.); y2 = rng.uniform(-1., 1.);
        if (x1*x1 + y1*y1 > 1. || x2*x2 + y2*y2 > 1.) {
            i--;
            continue;
        }
        candidate_feats[i][0] = x1 * radius_m;
        candidate_feats[i][1] = y1 * radius_m;
        candidate_feats[i][2] = x2 * radius_m;
        candidate_feats[i][3] = y2 * radius_m;
    }
    // calc features
    Mat_<int> densities(feats_m, N);
    for (int i = 0; i < N; i++) {
        double scale;
        Mat_<double> rotate;
        const Mat_<double> &current_shape = (Mat_<double>)current_shapes[root[i]];
        BBox &bbox = bboxes[root[i]];
        Mat &img = imgs[root[i]];
        calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);
        for (int j = 0; j < feats_m; j++) {
            double x1 = candidate_feats(j, 0);
            double y1 = candidate_feats(j, 1);
            double x2 = candidate_feats(j, 2);
            double y2 = candidate_feats(j, 3);
            SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
            SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

            x1 = x1*bbox.x_scale + current_shape(landmark_id, 0);
            y1 = y1*bbox.y_scale + current_shape(landmark_id, 1);
            x2 = x2*bbox.x_scale + current_shape(landmark_id, 0);
            y2 = y2*bbox.y_scale + current_shape(landmark_id, 1);
            x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
            x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
            densities(j, i) = (int)img.at<uchar>(int(y1), int(x1)) - (int)img.at<uchar>(int(y2), int(x2));
        }
    }
    Mat_<int> densities_sorted;
    cv::sort(densities, densities_sorted, SORT_EVERY_ROW + SORT_ASCENDING);
    //select a feat which reduces maximum variance
    double variance_all = (calcVariance(delta_shapes.col(0)) + calcVariance(delta_shapes.col(1)))*N;
    double variance_reduce_max = 0;
    int threshold = 0;
    int feat_id = 0;
    std::vector<double> left_x, left_y, right_x, right_y;
    left_x.reserve(N); left_y.reserve(N);
    right_x.reserve(N); right_y.reserve(N);
    for (int j = 0; j < feats_m; j++) {
        left_x.clear(); left_y.clear();
        right_x.clear(); right_y.clear();
        int threshold_ = densities_sorted(j, (int)(N*rng.uniform(0.05, 0.95)));
        for (int i = 0; i < N; i++) {
            if (densities(j, i) < threshold_) {
                left_x.push_back(delta_shapes.at<double>(root[i], 0));
                left_y.push_back(delta_shapes.at<double>(root[i], 1));
            }
            else {
                right_x.push_back(delta_shapes.at<double>(root[i], 0));
                right_y.push_back(delta_shapes.at<double>(root[i], 1));
            }
        }
        double variance_ = (calcVariance(left_x) + calcVariance(left_y))*left_x.size() + \
            (calcVariance(right_x) + calcVariance(right_y))*right_x.size();
        double variance_reduce = variance_all - variance_;
        if (variance_reduce > variance_reduce_max) {
            variance_reduce_max = variance_reduce;
            threshold = threshold_;
            feat_id = j;
        }
    }
    thresholds[idx] = threshold;
    feats(idx, 0) = candidate_feats(feat_id, 0); feats(idx, 1) = candidate_feats(feat_id, 1);
    feats(idx, 2) = candidate_feats(feat_id, 2); feats(idx, 3) = candidate_feats(feat_id, 3);
    // generate left and right child
    std::vector<int> left, right;
    left.reserve(N);
    right.reserve(N);
    for (int i = 0; i < N; i++) {
        if (densities(feat_id, i) < threshold) left.push_back(root[i]);
        else right.push_back(root[i]);
    }
    // split left and right child in DFS
    if (2 * idx < feats.rows / 2)
        splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, left, 2 * idx, stage);
    if (2 * idx + 1 < feats.rows / 2)
        splitNode(imgs, current_shapes, bboxes, delta_shapes, mean_shape, right, 2 * idx + 1, stage);
}

void FacemarkLBFImpl::RandomTree::write(FileStorage fs, int k, int i, int j) {

    String x;
    x = cv::format("tree_%i_%i_%i",k,i,j);
    fs << x << feats;
    x = cv::format("thresholds_%i_%i_%i",k,i,j);
    fs << x << thresholds;
}

void FacemarkLBFImpl::RandomTree::read(FileStorage fs, int k, int i, int j) {
    String x;
    x = cv::format("tree_%i_%i_%i",k,i,j);
    fs[x] >> feats;
    x = cv::format("thresholds_%i_%i_%i",k,i,j);
    fs[x] >> thresholds;
}


/*---------------RandomForest Implementation---------------------*/
void FacemarkLBFImpl::RandomForest::initForest(
    int _landmark_n,
    int _trees_n,
    int _tree_depth,
    double _overlap_ratio,
    std::vector<int>_feats_m,
    std::vector<double>_radius_m,
    bool verbose_mode
) {
    trees_n = _trees_n;
    landmark_n = _landmark_n;
    tree_depth = _tree_depth;
    overlap_ratio = _overlap_ratio;

    feats_m = _feats_m;
    radius_m = _radius_m;

    verbose = verbose_mode;

    random_trees.resize(landmark_n);
    for (int i = 0; i < landmark_n; i++) {
        random_trees[i].resize(trees_n);
        for (int j = 0; j < trees_n; j++) random_trees[i][j].initTree(i, tree_depth, feats_m, radius_m);
    }
}

void FacemarkLBFImpl::RandomForest::train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, \
                         std::vector<BBox> &bboxes, std::vector<Mat> &delta_shapes, Mat &mean_shape, int stage) {
    int N = (int)imgs.size();
    int Q = int(N / ((1. - overlap_ratio) * trees_n));

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < landmark_n; i++) {
    TIMER_BEGIN
        std::vector<int> root;
        for (int j = 0; j < trees_n; j++) {
            int start = max(0, int(floor(j*Q - j*Q*overlap_ratio)));
            int end = min(int(start + Q + 1), N);
            int L = end - start;
            root.resize(L);
            for (int k = 0; k < L; k++) root[k] = start + k;
            random_trees[i][j].train(imgs, current_shapes, bboxes, delta_shapes, mean_shape, root, stage);
        }
        if(verbose) printf("Train %2dth of %d landmark Done, it costs %.4lf s\n", i+1, landmark_n, TIMER_NOW);
    TIMER_END
    }
}

Mat FacemarkLBFImpl::RandomForest::generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape) {
    Mat lbf_feat(1, landmark_n*trees_n, CV_32SC1);
    double scale;
    Mat_<double> rotate;
    calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);

    int base = 1 << (tree_depth - 1);

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            RandomTree &tree = random_trees[i][j];
            int code = 0;
            int idx = 1;
            for (int k = 1; k < tree.depth; k++) {
                double x1 = tree.feats(idx, 0);
                double y1 = tree.feats(idx, 1);
                double x2 = tree.feats(idx, 2);
                double y2 = tree.feats(idx, 3);
                SIMILARITY_TRANSFORM(x1, y1, scale, rotate);
                SIMILARITY_TRANSFORM(x2, y2, scale, rotate);

                x1 = x1*bbox.x_scale + current_shape.at<double>(i, 0);
                y1 = y1*bbox.y_scale + current_shape.at<double>(i, 1);
                x2 = x2*bbox.x_scale + current_shape.at<double>(i, 0);
                y2 = y2*bbox.y_scale + current_shape.at<double>(i, 1);
                x1 = max(0., min(img.cols - 1., x1)); y1 = max(0., min(img.rows - 1., y1));
                x2 = max(0., min(img.cols - 1., x2)); y2 = max(0., min(img.rows - 1., y2));
                int density = img.at<uchar>(int(y1), int(x1)) - img.at<uchar>(int(y2), int(x2));
                code <<= 1;
                if (density < tree.thresholds[idx]) {
                    idx = 2 * idx;
                }
                else {
                    code += 1;
                    idx = 2 * idx + 1;
                }
            }
            lbf_feat.at<int>(i*trees_n + j) = (i*trees_n + j)*base + code;
        }
    }
    return lbf_feat;
}

void FacemarkLBFImpl::RandomForest::write(FileStorage fs, int k) {
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            random_trees[i][j].write(fs,k,i,j);
        }
    }
}

void FacemarkLBFImpl::RandomForest::read(FileStorage fs,int k)
{
    for (int i = 0; i < landmark_n; i++) {
        for (int j = 0; j < trees_n; j++) {
            random_trees[i][j].initTree(i, tree_depth, feats_m, radius_m);
            random_trees[i][j].read(fs,k,i,j);
        }
    }
}

/*---------------Regressor Implementation---------------------*/
void FacemarkLBFImpl::Regressor::initRegressor(Params config) {
    stages_n = config.stages_n;
    landmark_n = config.n_landmarks;

    random_forests.resize(stages_n);
    for (int i = 0; i < stages_n; i++)
        random_forests[i].initForest(
            config.n_landmarks,
            config.tree_n,
            config.tree_depth,
            config.bagging_overlap,
            config.feats_m,
            config.radius_m,
            config.verbose
        );

    mean_shape.create(config.n_landmarks, 2, CV_64FC1);

    gl_regression_weights.resize(stages_n);
    int F = config.n_landmarks * config.tree_n * (1 << (config.tree_depth - 1));

    for (int i = 0; i < stages_n; i++) {
        gl_regression_weights[i].create(2 * config.n_landmarks, F, CV_64FC1);
    }
}

void FacemarkLBFImpl::Regressor::trainRegressor(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                        std::vector<BBox> &bboxes, Mat &mean_shape_, int start_from, Params config) {
    CV_Assert(start_from >= 0 && start_from < stages_n);
    mean_shape = mean_shape_;
    int N = (int)imgs.size();

    for (int k = start_from; k < stages_n; k++) {
        std::vector<Mat> delta_shapes = getDeltaShapes(gt_shapes, current_shapes, bboxes, mean_shape);

        // train random forest
        if(config.verbose) printf("training random forest %dth of %d stages, ",k+1, stages_n);
        TIMER_BEGIN
            random_forests[k].train(imgs, current_shapes, bboxes, delta_shapes, mean_shape, k);
            if(config.verbose) printf("costs %.4lf s\n",  TIMER_NOW);
        TIMER_END

        // generate lbf of every train data
        std::vector<Mat> lbfs;
        lbfs.resize(N);
        for (int i = 0; i < N; i++) {
            lbfs[i] = random_forests[k].generateLBF(imgs[i], current_shapes[i], bboxes[i], mean_shape);
        }

        // global regression
        if(config.verbose) printf("start train global regression of %dth stage\n", k);
        TIMER_BEGIN
            globalRegressionTrain(lbfs, delta_shapes, k, config);
            if(config.verbose) printf("end of train global regression of %dth stage, costs %.4lf s\n", k, TIMER_NOW);
        TIMER_END

        // update current_shapes
        double scale;
        Mat rotate;
        for (int i = 0; i < N; i++) {
            Mat delta_shape = globalRegressionPredict(lbfs[i], k);
            calcSimilarityTransform(bboxes[i].project(current_shapes[i]), mean_shape, scale, rotate);
            current_shapes[i] = bboxes[i].reproject(bboxes[i].project(current_shapes[i]) + scale * delta_shape * rotate.t());
        }

        // calc mean error
        double e = calcMeanError(gt_shapes, current_shapes, config.n_landmarks, config.pupils[0],config.pupils[1]);
        if(config.verbose) printf("Train %dth stage Done with Error = %lf\n", k, e);

    } // for int k
}//Regressor::training

void FacemarkLBFImpl::Regressor::globalRegressionTrain(
    std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes,
    int stage, Params config
) {

    int N = (int)lbfs.size();
    int M = lbfs[0].cols;
    int F = config.n_landmarks*config.tree_n*(1 << (config.tree_depth - 1));
    int landmark_n_ = delta_shapes[0].rows;
    feature_node **X = (feature_node **)malloc(N * sizeof(feature_node *));
    double **Y = (double **)malloc(landmark_n_ * 2 * sizeof(double *));
    for (int i = 0; i < N; i++) {
        X[i] = (feature_node *)malloc((M + 1) * sizeof(feature_node));
        for (int j = 0; j < M; j++) {
            X[i][j].index = lbfs[i].at<int>(0, j) + 1; // index starts from 1
            X[i][j].value = 1;
        }
        X[i][M].index = -1;
        X[i][M].value = -1;
    }
    for (int i = 0; i < landmark_n_; i++) {
        Y[2 * i] = (double *)malloc(N*sizeof(double));
        Y[2 * i + 1] = (double *)malloc(N*sizeof(double));
        for (int j = 0; j < N; j++) {
            Y[2 * i][j] = delta_shapes[j].at<double>(i, 0);
            Y[2 * i + 1][j] = delta_shapes[j].at<double>(i, 1);
        }
    }

    double *y;
    Mat weights;
    for(int i=0; i< landmark_n_; i++){
        y =  Y[2 * i];
        Mat wx = supportVectorRegression(X,y,N,F,config.verbose);
        weights.push_back(wx);

        y = Y[2 * i + 1];
        Mat wy = supportVectorRegression(X,y,N,F,config.verbose);
        weights.push_back(wy);
    }

    gl_regression_weights[stage] = weights;

    // free
    for (int i = 0; i < N; i++) free(X[i]);
    for (int i = 0; i < 2 * landmark_n_; i++) free(Y[i]);
    free(X);
    free(Y);
} // Regressor:globalRegressionTrain

/*adapted from the liblinear library*/
/* TODO: change feature_node to MAT
* as the index in feature_node is only used for "counter"
*/
Mat FacemarkLBFImpl::Regressor::supportVectorRegression(
    feature_node **x, double *y, int nsamples, int feat_size, bool verbose
){
    #define GETI(i) ((int) y[i])

    std::vector<double> w;
    w.resize(feat_size);

    RNG rng(0);
    int l = nsamples; // n-samples
    double C = 1./(double)nsamples;
    double p = 0;
    int w_size = feat_size; //feat size
    double eps =  0.00001;
    int i, s, iter = 0;
    int max_iter = 1000;
    int active_size = l;
    std::vector<int> index(l);

    double d, G, H;
    double Gmax_old = HUGE_VAL;
    double Gmax_new, Gnorm1_new;
    double Gnorm1_init = -1.0; // Gnorm1_init is initialized at the first iteration
    std::vector<double> beta(l);
    std::vector<double> QD(l);

    double lambda[1], upper_bound[1];
    lambda[0] = 0.5/C;
    upper_bound[0] = HUGE_VAL;

    // Initial beta can be set here. Note that
    // -upper_bound <= beta[i] <= upper_bound
    for(i=0; i<l; i++)
        beta[i] = 0;

    for(i=0; i<w_size; i++)
        w[i] = 0;

    for(i=0; i<l; i++){
        QD[i] = 0;
        feature_node *xi = x[i];
        while(xi->index != -1){
            double val = xi->value;
            QD[i] += val*val;
            w[xi->index-1] += beta[i]*val;
            xi++;
        }
        index[i] = i;
    }

    while(iter < max_iter){
        Gmax_new = 0;
        Gnorm1_new = 0;

        for(i=0; i<active_size; i++){
            int j = i+rng.uniform(0,RAND_MAX)%(active_size-i);
            swap(index[i], index[j]);
        }

        for(s=0; s<active_size; s++){
            i = index[s];
            G = -y[i] + lambda[GETI(i)]*beta[i];
            H = QD[i] + lambda[GETI(i)];

            feature_node *xi = x[i];
            while(xi->index != -1){
                int ind = xi->index-1;
                double val = xi->value;
                G += val*w[ind];
                xi++;
            }

            double Gp = G+p;
            double Gn = G-p;
            double violation = 0;
            if(beta[i] == 0){
                if(Gp < 0)
                    violation = -Gp;
                else if(Gn > 0)
                    violation = Gn;
                else if(Gp>Gmax_old && Gn<-Gmax_old){
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
            }else if(beta[i] >= upper_bound[GETI(i)]){
                if(Gp > 0)
                    violation = Gp;
                else if(Gp < -Gmax_old){
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
            }else if(beta[i] <= -upper_bound[GETI(i)]){
                if(Gn < 0)
                    violation = -Gn;
                else if(Gn > Gmax_old){
                    active_size--;
                    swap(index[s], index[active_size]);
                    s--;
                    continue;
                }
            }else if(beta[i] > 0)
                violation = fabs(Gp);
            else
                violation = fabs(Gn);

            Gmax_new = max(Gmax_new, violation);
            Gnorm1_new += violation;

            // obtain Newton direction d
            if(Gp < H*beta[i])
                d = -Gp/H;
            else if(Gn > H*beta[i])
                d = -Gn/H;
            else
                d = -beta[i];

            if(fabs(d) < 1.0e-12)
                continue;

            double beta_old = beta[i];
            beta[i] = min(max(beta[i]+d, -upper_bound[GETI(i)]), upper_bound[GETI(i)]);
            d = beta[i]-beta_old;

            if(d != 0){
                xi = x[i];
                while(xi->index != -1){
                    w[xi->index-1] += d*xi->value;
                    xi++;
                }
            }
        }// for s<active_size

        if(iter == 0)
            Gnorm1_init = Gnorm1_new;
        iter++;

        if(Gnorm1_new <= eps*Gnorm1_init){
            if(active_size == l)
                break;
            else{
                active_size = l;
                Gmax_old = HUGE_VAL;
                continue;
            }
        }

        Gmax_old = Gmax_new;
    } //while <max_iter

    if(verbose) printf("optimization finished, #iter = %d\n", iter);
    if(iter >= max_iter && verbose)
        printf("WARNING: reaching max number of iterations\n");

    // calculate objective value
    double v = 0;
    int nSV = 0;
    for(i=0; i<w_size; i++)
        v += w[i]*w[i];
    v = 0.5*v;
    for(i=0; i<l; i++){
        v += p*fabs(beta[i]) - y[i]*beta[i] + 0.5*lambda[GETI(i)]*beta[i]*beta[i];
        if(beta[i] != 0)
            nSV++;
    }

    if(verbose) printf("Objective value = %lf\n", v);
    if(verbose) printf("nSV = %d\n",nSV);

    return Mat(Mat(w).t()).clone();

}//end

Mat FacemarkLBFImpl::Regressor::globalRegressionPredict(const Mat &lbf, int stage) {
    const Mat_<double> &weight = (Mat_<double>)gl_regression_weights[stage];
    Mat delta_shape(weight.rows / 2, 2, CV_64FC1);
    const double *w_ptr = NULL;
    const int *lbf_ptr = lbf.ptr<int>(0);

    //#pragma omp parallel for num_threads(2) private(w_ptr)
    for (int i = 0; i < delta_shape.rows; i++) {
        w_ptr = weight.ptr<double>(2 * i);
        double y = 0;
        for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape.at<double>(i, 0) = y;

        w_ptr = weight.ptr<double>(2 * i + 1);
        y = 0;
        for (int j = 0; j < lbf.cols; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape.at<double>(i, 1) = y;
    }
    return delta_shape;
} // Regressor::globalRegressionPredict

Mat FacemarkLBFImpl::Regressor::predict(Mat &img, BBox &bbox) {
    Mat current_shape = bbox.reproject(mean_shape);
    double scale;
    Mat rotate;
    Mat lbf_feat;
    for (int k = 0; k < stages_n; k++) {
        // generate lbf
        lbf_feat = random_forests[k].generateLBF(img, current_shape, bbox, mean_shape);
        // update current_shapes
        Mat delta_shape = globalRegressionPredict(lbf_feat, k);
        delta_shape = delta_shape.reshape(0, landmark_n);
        calcSimilarityTransform(bbox.project(current_shape), mean_shape, scale, rotate);
        current_shape = bbox.reproject(bbox.project(current_shape) + scale * delta_shape * rotate.t());
    }
    return current_shape;
} // Regressor::predict

void FacemarkLBFImpl::Regressor::write(FileStorage fs, Params config) {

    fs << "stages_n" << config.stages_n;
    fs << "tree_n" << config.tree_n;
    fs << "tree_depth" << config.tree_depth;
    fs << "n_landmarks" << config.n_landmarks;

    fs << "regressor_meanshape" <<mean_shape;

    // every stages
    String x;
    for (int k = 0; k < config.stages_n; k++) {
        if(config.verbose) printf("Write %dth stage\n", k);
        random_forests[k].write(fs,k);
        x = cv::format("weights_%i",k);
        fs << x << gl_regression_weights[k];
    }
}

void FacemarkLBFImpl::Regressor::read(FileStorage fs, Params & config){
    fs["stages_n"] >>  config.stages_n;
    fs["tree_n"] >>  config.tree_n;
    fs["tree_depth"] >>  config.tree_depth;
    fs["n_landmarks"] >>  config.n_landmarks;

    stages_n = config.stages_n;
    landmark_n = config.n_landmarks;

    initRegressor(config);

    fs["regressor_meanshape"] >>  mean_shape;

    // every stages
    String x;
    for (int k = 0; k < stages_n; k++) {
        random_forests[k].initForest(
            config.n_landmarks,
            config.tree_n,
            config.tree_depth,
            config.bagging_overlap,
            config.feats_m,
            config.radius_m,
            config.verbose
        );
        random_forests[k].read(fs,k);

        x = cv::format("weights_%i",k);
        fs[x] >> gl_regression_weights[k];
    }
}

#undef TIMER_BEGIN
#undef TIMER_NOW
#undef TIMER_END
#undef SIMILARITY_TRANSFORM
} /* namespace face */
} /* namespace cv */
