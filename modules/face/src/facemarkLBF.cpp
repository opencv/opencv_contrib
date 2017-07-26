#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"
#include <fstream>

namespace cv
{
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
        FacemarkLBF::Params params;
    private:
        bool isModelTrained;
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

        for (int i = 0; i < rects.size(); i++) {
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
                Mat shape = Mat(facePoints[i]).reshape(1);
                shape.convertTo(shape, CV_64FC1);
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
            int h = img_flipped.rows - 1;
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

    FacemarkLBFImpl::BBox::BBox(double x, double y, double w, double h) {
        this->x = x; this->y = y;
        this->width = w; this->height = h;
        this->x_center = x + w / 2.;
        this->y_center = y + h / 2.;
        this->x_scale = w / 2.;
        this->y_scale = h / 2.;
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
} /* namespace cv */
