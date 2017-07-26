#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"
#include "liblinear.hpp"
#include <fstream>

namespace cv
{
    #define TIMER_BEGIN { double __time__ = getTickCount();
    #define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
    #define TIMER_END   }

    #define SIMILARITY_TRANSFORM(x, y, scale, rotate) do {            \
        double x_tmp = scale * (rotate(0, 0)*x + rotate(0, 1)*y); \
        double y_tmp = scale * (rotate(1, 0)*x + rotate(1, 1)*y); \
        x = x_tmp; y = y_tmp;                                     \
    } while(0)

    FacemarkLBF::Params::Params(){
        detect_thresh = 0.5;
        sigma=0.2;
    }

    void FacemarkLBF::Params::read( const cv::FileNode& fn ){
        *this = FacemarkLBF::Params();

        if (!fn["detect_thresh"].empty())
            fn["detect_thresh"] >> detect_thresh;

        if (!fn["sigma"].empty())
            fn["sigma"] >> sigma;

    }

    void FacemarkLBF::Params::write( cv::FileStorage& fs ) const{
        fs << "detect_thresh" << detect_thresh;
        fs << "sigma" << sigma;
    }

    class FacemarkLBFImpl : public FacemarkLBF {
    public:
        FacemarkLBFImpl( const FacemarkLBF::Params &parameters = FacemarkLBF::Params() );

        void read( const FileNode& /*fn*/ );
        void write( FileStorage& /*fs*/ ) const;

        void saveModel(FileStorage& fs);
        void loadModel(FileStorage& fs);

    protected:

        bool fitImpl( const Mat, std::vector<Point2f> & landmarks );
        bool fitImpl( const Mat, std::vector<Point2f>& , Mat R, Point2f T, float scale );
        void trainingImpl(String imageList, String groundTruth, const FacemarkLBF::Params &parameters);
        void trainingImpl(String imageList, String groundTruth);

        Rect getBBox(Mat &img, const Mat_<double> shape, CascadeClassifier cc);
        void prepareTrainingData(std::vector<String> images, std::vector<std::vector<Point2f> > & facePoints, std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes, CascadeClassifier cc);
        void data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);
        Mat getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes);

        FacemarkLBF::Params params;
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

        };

        /*---------------RandomTree Class---------------------*/
        class RandomTree : public LBF {
        public:
            RandomTree(){};
            ~RandomTree(){};

            void init(int landmark_id, int depth, std::vector<int>, std::vector<double>);
            void train(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                       std::vector<Mat> &delta_shapes, Mat &mean_shape, std::vector<int> &index, int stage);
            void splitNode(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes,
                          cv::Mat &delta_shapes, cv::Mat &mean_shape, std::vector<int> &root, int idx, int stage);
            void write(FILE *fd);

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

            void init(int landmark_n, int trees_n, int tree_depth, double ,  std::vector<int>, std::vector<double>);
            void train(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, std::vector<cv::Mat> &current_shapes, \
                       std::vector<BBox> &bboxes, std::vector<cv::Mat> &delta_shapes, cv::Mat &mean_shape, int stage);
            Mat generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape);
            void write(FILE *fd);

            int landmark_n;
            int trees_n, tree_depth;
            double overlap_ratio;
            std::vector<std::vector<RandomTree> > random_trees;

            std::vector<int> feats_m;
            std::vector<double> radius_m;
        };
        /*---------------Regressor Class---------------------*/
        class Regressor  : public LBF {
        public:
            Regressor(){};
            ~Regressor(){};

            void init(Params);
            void training(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &gt_shapes, \
                       std::vector<cv::Mat> &current_shapes, std::vector<BBox> &bboxes, \
                       cv::Mat &mean_shape, int start_from, Params );
            void globalRegressionTrain(std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes, int stage, Params);
            void write(FILE *fd, Params params);

            int stages_n;
            int landmark_n;
            cv::Mat mean_shape;
            std::vector<RandomForest> random_forests;
            std::vector<cv::Mat> gl_regression_weights;
        }; // LBF

    }; // class

    /*
    * Constructor
    */
    Ptr<FacemarkLBF> FacemarkLBF::create(const FacemarkLBF::Params &parameters){
        return Ptr<FacemarkLBFImpl>(new FacemarkLBFImpl(parameters));
    }

    Ptr<FacemarkLBF> FacemarkLBF::create(){
        return Ptr<FacemarkLBFImpl>(new FacemarkLBFImpl());
    }

    FacemarkLBFImpl::FacemarkLBFImpl( const FacemarkLBF::Params &parameters ) :
        params( parameters )
    {
        isSetDetector =false;
        isModelTrained = false;
        params.cascade_face = "../data/haarcascade_frontalface_alt.xml";
        params.shape_offset = 0.0;
        params.n_landmarks = 68;
        params.initShape_n = 10;
        params.stages_n=2;//TODO: 5;
        params.tree_n=6;
        params.tree_depth=5;
        params.bagging_overlap = 0.4;
        params.saved_file_name = "ibug.model";

        int pupils[][6] = { { 36, 37, 38, 39, 40, 41 }, { 42, 43, 44, 45, 46, 47 } };
        for (int i = 0; i < 6; i++) {
            params.pupils[0].push_back(pupils[0][i]);
            params.pupils[1].push_back(pupils[1][i]);
        }

        int feats_m[] = { 500, 500, 500, 300, 300, 300, 200, 200, 200, 100 };
        double radius_m[] = { 0.3, 0.2, 0.15, 0.12, 0.10, 0.10, 0.08, 0.06, 0.06, 0.05 };
        for (int i = 0; i < 10; i++) {
            params.feats_m.push_back(feats_m[i]);
            params.radius_m.push_back(radius_m[i]);
        }
    }

    void FacemarkLBFImpl::trainingImpl(String imageList, String groundTruth){
        std::vector<String> images;
        std::vector<std::vector<Point2f> > facePoints;

        loadTrainingData(imageList, groundTruth, images, facePoints, params.shape_offset);

        std::vector<Mat> cropped;
        std::vector<BBox> boxes;
        std::vector<Mat> shapes;

        /*check the cascade classifier file*/
        std::ifstream infile;
        infile.open(params.cascade_face.c_str(), std::ios::in);
        if (!infile) {
           std::string error_message = "The cascade classifier model is not found.";
           CV_Error(CV_StsBadArg, error_message);
        }

        CascadeClassifier cc(params.cascade_face.c_str());
        prepareTrainingData(images, facePoints, cropped, shapes, boxes, cc);

        // flip the image and swap the landmark position
        data_augmentation(cropped, shapes, boxes);

        Mat mean_shape = getMeanShape(shapes, boxes);

        int N = cropped.size();
        int L = N*params.initShape_n;
        std::vector<Mat> imgs(L), gt_shapes(L), current_shapes(L);
        std::vector<BBox> bboxes(L);
        RNG rng(getTickCount());
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < params.initShape_n; j++) {
                int idx = i*params.initShape_n + j;
                int k = 0;
                do {
                    k = rng.uniform(0, N);
                } while (k == i);
                imgs[idx] = cropped[i];
                gt_shapes[idx] = shapes[i];
                bboxes[idx] = boxes[i];
                current_shapes[idx] = boxes[i].reproject(boxes[k].project(shapes[k]));
            }
        }

        // random shuffle
        std::srand(std::time(0));
        std::random_shuffle(imgs.begin(), imgs.end());
        std::srand(std::time(0));
        std::random_shuffle(gt_shapes.begin(), gt_shapes.end());
        std::srand(std::time(0));
        std::random_shuffle(bboxes.begin(), bboxes.end());
        std::srand(std::time(0));
        std::random_shuffle(current_shapes.begin(), current_shapes.end());

        Regressor lbf;
        lbf.init(params);
        lbf.training(imgs, gt_shapes, current_shapes, bboxes, mean_shape, 0, params);

        FILE *fd = fopen(params.saved_file_name.c_str(), "wb");
        assert(fd);
        lbf.write(fd, params);
        fclose(fd);
    }

    bool FacemarkLBFImpl::fitImpl( const Mat image, std::vector<Point2f>& landmarks){
        Mat R =  Mat::eye(2, 2, CV_32F);
        Point2f t = Point2f(0,0);
        float scale = 1.0;

        return fitImpl(image, landmarks, R, t, scale);
    }

    bool FacemarkLBFImpl::fitImpl( const Mat image, std::vector<Point2f>& landmarks, Mat R, Point2f T, float scale ){
        printf("fitting\n");
        return 0;
    }

    void FacemarkLBFImpl::read( const cv::FileNode& fn ){
        params.read( fn );
    }

    void FacemarkLBFImpl::write( cv::FileStorage& fs ) const {
        params.write( fs );
    }

    void FacemarkLBFImpl::saveModel(FileStorage& fs){

    }

    void FacemarkLBFImpl::loadModel(FileStorage& fs){

    }

    Rect FacemarkLBFImpl::getBBox(Mat &img, const Mat_<double> shape, CascadeClassifier cc) {
        std::vector<Rect> rects;
        cc.detectMultiScale(img, rects, 1.05, 2, CV_HAAR_SCALE_IMAGE, Size(30, 30));
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

    void FacemarkLBFImpl::prepareTrainingData(std::vector<String> images, std::vector<std::vector<Point2f> > & facePoints, std::vector<Mat> & cropped, std::vector<Mat> & shapes, std::vector<BBox> &boxes, CascadeClassifier cc){
        std::vector<std::vector<Point2f> > facePts;
        boxes.clear();
        cropped.clear();
        shapes.clear();

        int N = 10;//TODO: images.size();
        for(int i=0; i<N;i++){
            printf("image #%i/%i\n", i, N);
            Mat img = imread(images[i].c_str(), 0);
            Rect box = getBBox(img, Mat(facePoints[i]).reshape(1), cc);
            if(box.x != -1){
                Mat _shape = Mat(facePoints[i]).reshape(1);
                Mat shape;
                _shape.convertTo(shape, CV_64FC1);
                Mat sx = shape.col(0);
                Mat sy = shape.col(1);
                double min_x, max_x, min_y, max_y;
                minMaxIdx(sx, &min_x, &max_x);
                minMaxIdx(sy, &min_y, &max_y);

                min_x = std::max(0., min_x - box.width / 2);
                max_x = std::min(img.cols - 1., max_x + box.width / 2);
                min_y = std::max(0., min_y - box.height / 2);
                max_y = std::min(img.rows - 1., max_y + box.height / 2);

                double w = max_x - min_x;
                double h = max_y - min_y;

                facePts.push_back(facePoints[i]);
                boxes.push_back(BBox(box.x - min_x, box.y - min_y, box.width, box.height));
                Mat crop = img(Rect(min_x, min_y, w, h)).clone();
                cropped.push_back(crop);
                shapes.push_back(shape);
            }
        }//images.size()

        facePoints = facePts;
    }

    void FacemarkLBFImpl::data_augmentation(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {
        int N = imgs.size();
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
            x_b = w - bboxes[i].x - bboxes[i].width;
            y_b = bboxes[i].y;
            w_b = bboxes[i].width;
            h_b = bboxes[i].height;
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
            printf("Wrong n_landmarks, it must be 29 or 68");
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
        Mat_<double> res(shape.rows, shape.cols);
        const Mat_<double> &shape_ = (Mat_<double>)shape;
        for (int i = 0; i < shape.rows; i++) {
            res(i, 0) = (shape_(i, 0) - x_center) / x_scale;
            res(i, 1) = (shape_(i, 1) - y_center) / y_scale;
        }
        return res;
    }

    // Project relative shape to absolute shape binding to this bbox
    Mat FacemarkLBFImpl::BBox::reproject(const Mat &shape) const {
        Mat_<double> res(shape.rows, shape.cols);
        const Mat_<double> &shape_ = (Mat_<double>)shape;
        for (int i = 0; i < shape.rows; i++) {
            res(i, 0) = shape_(i, 0)*x_scale + x_center;
            res(i, 1) = shape_(i, 1)*y_scale + y_center;
        }
        return res;
    }

    Mat FacemarkLBFImpl::getMeanShape(std::vector<Mat> &gt_shapes, std::vector<BBox> &bboxes) {

        int N = gt_shapes.size();
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
        calcCovarMatrix(temp1, covar1, mean1, CV_COVAR_COLS);
        calcCovarMatrix(temp2, covar2, mean2, CV_COVAR_COLS);

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
        int N = gt_shapes.size();
        delta_shapes.resize(N);
        double scale;
        Mat_<double> rotate;
        for (int i = 0; i < N; i++) {
            delta_shapes[i] = bboxes[i].project(gt_shapes[i]) - bboxes[i].project(current_shapes[i]);
            calcSimilarityTransform(mean_shape, bboxes[i].project(current_shapes[i]), scale, rotate);
            // delta_shapes[i] = scale * delta_shapes[i] * rotate.t();
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

    /*---------------RandomTree Implementation---------------------*/
    void FacemarkLBFImpl::RandomTree::init(int _landmark_id, int _depth, std::vector<int> feats_m, std::vector<double> radius_m) {
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
        Mat_<double> delta_shapes_(delta_shapes.size(), 2);
        for (int i = 0; i < (int)delta_shapes.size(); i++) {
            delta_shapes_(i, 0) = delta_shapes[i].at<double>(landmark_id, 0);
            delta_shapes_(i, 1) = delta_shapes[i].at<double>(landmark_id, 1);
        }
        splitNode(imgs, current_shapes, bboxes, delta_shapes_, mean_shape, index, 1, stage);
    }

    void FacemarkLBFImpl::RandomTree::splitNode(std::vector<Mat> &imgs, std::vector<Mat> &current_shapes, std::vector<BBox> &bboxes,
                               Mat &delta_shapes, Mat &mean_shape, std::vector<int> &root, int idx, int stage) {

        int N = root.size();
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


    void FacemarkLBFImpl::RandomTree::write(FILE *fd) {
        for (int i = 1; i < nodes_n / 2; i++) {
            fwrite(feats.ptr<double>(i), sizeof(double), 4, fd);
            fwrite(&thresholds[i], sizeof(int), 1, fd);
        }
    }

    /*---------------RandomForest Implementation---------------------*/
    void FacemarkLBFImpl::RandomForest::init(int _landmark_n, int _trees_n, int _tree_depth, double _overlap_ratio, std::vector<int>_feats_m, std::vector<double>_radius_m) {
        trees_n = _trees_n;
        landmark_n = _landmark_n;
        tree_depth = _tree_depth;
        overlap_ratio = _overlap_ratio;

        feats_m = _feats_m;
        radius_m = _radius_m;

        random_trees.resize(landmark_n);
        for (int i = 0; i < landmark_n; i++) {
            random_trees[i].resize(trees_n);
            for (int j = 0; j < trees_n; j++) random_trees[i][j].init(i, tree_depth, feats_m, radius_m);
        }
    }

    void FacemarkLBFImpl::RandomForest::train(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes, \
                             std::vector<BBox> &bboxes, std::vector<Mat> &delta_shapes, Mat &mean_shape, int stage) {
        int N = imgs.size();
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
            printf("Train %2dth of %d landmark Done, it costs %.4lf s\n", i+1, landmark_n, TIMER_NOW);
        TIMER_END
        }
    }

    Mat FacemarkLBFImpl::RandomForest::generateLBF(Mat &img, Mat &current_shape, BBox &bbox, Mat &mean_shape) {
        Mat_<int> lbf(1, landmark_n*trees_n);
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
                lbf(i*trees_n + j) = (i*trees_n + j)*base + code;
            }
        }
        return lbf;
    }


    void FacemarkLBFImpl::RandomForest::write(FILE *fd) {
        for (int i = 0; i < landmark_n; i++) {
            for (int j = 0; j < trees_n; j++) {
                random_trees[i][j].write(fd);
            }
        }
    }

    /*---------------Regressor Implementation---------------------*/
    void FacemarkLBFImpl::Regressor::init(Params params) {
        stages_n = params.stages_n;
        landmark_n = params.n_landmarks;

        random_forests.resize(stages_n);
        for (int i = 0; i < stages_n; i++)
            random_forests[i].init(params.n_landmarks, params.tree_n, params.tree_depth, params.bagging_overlap, params.feats_m, params.radius_m);

        mean_shape.create(params.n_landmarks, 2, CV_64FC1);

        gl_regression_weights.resize(stages_n);
        int F = params.n_landmarks * params.tree_n * (1 << (params.tree_depth - 1));

        for (int i = 0; i < stages_n; i++) {
            gl_regression_weights[i].create(2 * params.n_landmarks, F, CV_64FC1);
        }
    }

    void FacemarkLBFImpl::Regressor::training(std::vector<Mat> &imgs, std::vector<Mat> &gt_shapes, std::vector<Mat> &current_shapes,
                            std::vector<BBox> &bboxes, Mat &mean_shape_, int start_from, Params params) {
        assert(start_from >= 0 && start_from < stages_n);
        mean_shape = mean_shape_;
        int N = imgs.size();

        for (int k = start_from; k < stages_n; k++) {
            std::vector<Mat> delta_shapes = getDeltaShapes(gt_shapes, current_shapes, bboxes, mean_shape);

            // train random forest
            printf("training random forest %dth of %d stages, ",k+1, stages_n);
            TIMER_BEGIN
                random_forests[k].train(imgs, gt_shapes, current_shapes, bboxes, delta_shapes, mean_shape, k);
                printf("costs %.4lf s\n",  TIMER_NOW);
            TIMER_END

            // generate lbf of every train data
            std::vector<Mat> lbfs;
            lbfs.resize(N);
            for (int i = 0; i < N; i++) {
                lbfs[i] = random_forests[k].generateLBF(imgs[i], current_shapes[i], bboxes[i], mean_shape);
            }

            // global regression
            printf("start train global regression of %dth stage\n", k);
            TIMER_BEGIN
                globalRegressionTrain(lbfs, delta_shapes, k, params);
                printf("end of train global regression of %dth stage, costs %.4lf s\n", k, TIMER_NOW);
            TIMER_END

        } // for int k
    }//Regressor::training

    // Global Regression to predict delta shape with LBF
    void FacemarkLBFImpl::Regressor::globalRegressionTrain(std::vector<Mat> &lbfs, std::vector<Mat> &delta_shapes, int stage, Params params) {
        int N = lbfs.size();
        int M = lbfs[0].cols;
        int F = params.n_landmarks*params.tree_n*(1 << (params.tree_depth - 1));
        int landmark_n = delta_shapes[0].rows;
        // prepare linear regression params X and Y
        struct liblinear::feature_node **X = (struct liblinear::feature_node **)malloc(N * sizeof(struct liblinear::feature_node *));
        double **Y = (double **)malloc(landmark_n * 2 * sizeof(double *));
        for (int i = 0; i < N; i++) {
            X[i] = (struct liblinear::feature_node *)malloc((M + 1) * sizeof(struct liblinear::feature_node));
            for (int j = 0; j < M; j++) {
                X[i][j].index = lbfs[i].at<int>(0, j) + 1; // index starts from 1
                X[i][j].value = 1;
            }
            X[i][M].index = X[i][M].value = -1;
        }
        for (int i = 0; i < landmark_n; i++) {
            Y[2 * i] = (double *)malloc(N*sizeof(double));
            Y[2 * i + 1] = (double *)malloc(N*sizeof(double));
            for (int j = 0; j < N; j++) {
                Y[2 * i][j] = delta_shapes[j].at<double>(i, 0);
                Y[2 * i + 1][j] = delta_shapes[j].at<double>(i, 1);
            }
        }
        // train every landmark
        struct liblinear::problem prob;
        struct liblinear::parameter param;
        prob.l = N;
        prob.n = F;
        prob.x = X;
        prob.bias = -1;
        param.solver_type = liblinear::L2R_L2LOSS_SVR_DUAL;
        param.C = 1. / N;
        param.p = 0;
        param.eps = 0.00001;

        Mat_<double> weight(2 * landmark_n, F);

        #pragma omp parallel for
        for (int i = 0; i < landmark_n; i++) {

        #define FREE_MODEL(model)   \
        free(model->w);         \
        free(model->label);     \
        free(model)

            printf("train %2dth landmark\n", i);
            struct liblinear::problem prob_ = prob;
            prob_.y = Y[2 * i];
            liblinear::check_parameter(&prob_, &param);
            struct liblinear::model *model = liblinear::train(&prob_, &param);
            for (int j = 0; j < F; j++) weight(2 * i, j) = liblinear::get_decfun_coef(model, j + 1, 0);
            FREE_MODEL(model);

            prob_.y = Y[2 * i + 1];
            liblinear::check_parameter(&prob_, &param);
            model = liblinear::train(&prob_, &param);
            for (int j = 0; j < F; j++) weight(2 * i + 1, j) = liblinear::get_decfun_coef(model, j + 1, 0);
            FREE_MODEL(model);

        #undef FREE_MODEL

        }

        gl_regression_weights[stage] = weight;

        // free
        for (int i = 0; i < N; i++) free(X[i]);
        for (int i = 0; i < 2 * landmark_n; i++) free(Y[i]);
        free(X);
        free(Y);
    } // Regressor:globalRegressionTrain

    void FacemarkLBFImpl::Regressor::write(FILE *fd, Params params) {
        // global parameters
        fwrite(&params.stages_n, sizeof(int), 1, fd);
        fwrite(&params.tree_n, sizeof(int), 1, fd);
        fwrite(&params.tree_depth, sizeof(int), 1, fd);
        fwrite(&params.n_landmarks, sizeof(int), 1, fd);
        // mean_shape
        double *ptr = NULL;
        for (int i = 0; i < mean_shape.rows; i++) {
            ptr = mean_shape.ptr<double>(i);
            fwrite(ptr, sizeof(double), mean_shape.cols, fd);
        }
        // every stages
        for (int k = 0; k < params.stages_n; k++) {
            printf("Write %dth stage\n", k);
            random_forests[k].write(fd);
            for (int i = 0; i < 2 * params.n_landmarks; i++) {
                ptr = gl_regression_weights[k].ptr<double>(i);
                fwrite(ptr, sizeof(double), gl_regression_weights[k].cols, fd);
            }
        }
    }

    #undef TIMER_BEGIN
    #undef TIMER_NOW
    #undef TIMER_END
    #undef SIMILARITY_TRANSFORM
} /* namespace cv */
