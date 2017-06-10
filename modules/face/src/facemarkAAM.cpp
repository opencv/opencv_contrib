#include "opencv2/face.hpp"
#include "opencv2/core.hpp"

#include <iostream>
using namespace std;

namespace cv
{
    //namespace face {

    /*
    * Parameters
    */
    FacemarkAAM::Params::Params(){
        detect_thresh = 0.5;
        sigma=0.2;
    }

    void FacemarkAAM::Params::read( const cv::FileNode& fn ){
        *this = FacemarkAAM::Params();

        if (!fn["detect_thresh"].empty())
            fn["detect_thresh"] >> detect_thresh;

        if (!fn["sigma"].empty())
            fn["sigma"] >> sigma;

    }

    void FacemarkAAM::Params::write( cv::FileStorage& fs ) const{
        fs << "detect_thresh" << detect_thresh;
        fs << "sigma" << sigma;
    }

    class FacemarkAAMImpl : public FacemarkAAM {
    public:
        FacemarkAAMImpl( const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
        void read( const FileNode& /*fn*/ );
        void write( FileStorage& /*fs*/ ) const;

        void saveTrainedModel(String filename);
        void loadTrainedModel(String filename);

    protected:

        bool detectImpl( InputArray image, std::vector<Point2f> & landmarks );
        void trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters);
        void trainingImpl(String imageList, String groundTruth);

        Mat procrustes(std::vector<Point2f> P, std::vector<Point2f> Q, Mat & rot, Scalar & trans, float & scale);
        void calcMeanShape(std::vector<std::vector<Point2f> > shapes,std::vector<Point2f> & mean);
        void procrustesAnalysis(std::vector<std::vector<Point2f> > shapes, std::vector<std::vector<Point2f> > & normalized, std::vector<Point2f> & new_mean);

        FacemarkAAM::Params params;

    private:
        int test;
    };

    /*
    * Constructor
    */
    Ptr<FacemarkAAM> FacemarkAAM::create(const FacemarkAAM::Params &parameters){
        return Ptr<FacemarkAAMImpl>(new FacemarkAAMImpl(parameters));
    }

    Ptr<FacemarkAAM> FacemarkAAM::create(){
        return Ptr<FacemarkAAMImpl>(new FacemarkAAMImpl());
    }

    FacemarkAAMImpl::FacemarkAAMImpl( const FacemarkAAM::Params &parameters ) :
        params( parameters )
    {
        isSetDetector =false;
        test = 11;
    }

    void FacemarkAAMImpl::read( const cv::FileNode& fn ){
        params.read( fn );
    }

    void FacemarkAAMImpl::write( cv::FileStorage& fs ) const {
        params.write( fs );
    }


    void FacemarkAAM::training(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
        trainingImpl(imageList, groundTruth, parameters);
    }

    void FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth, const FacemarkAAM::Params &parameters){
        params = parameters;
        trainingImpl(imageList, groundTruth);
    }

    void FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth){
        printf("inside the training func %s %s\n", imageList.c_str(), groundTruth.c_str());
        std::vector<String> images;
        std::vector<std::vector<Point2f> > facePoints;

        // load dataset
        if(groundTruth==""){
            loadTrainingData(imageList, images, facePoints);
        }else{
            loadTrainingData(imageList, groundTruth, images, facePoints);
        }
        std::cout<<images.size()<<std::endl;

        // calculate base shape
        std::vector<std::vector<Point2f> > normalized_shapes;
        std::vector<Point2f> s0;
        procrustesAnalysis(facePoints, normalized_shapes,s0);
        cout<<s0<<endl;
    }

    bool FacemarkAAMImpl::detectImpl( InputArray image, std::vector<Point2f>& landmarks ){
        if (landmarks.size()>0)
            landmarks.clear();

        landmarks.push_back(Point2f(2.0,3.3));
        landmarks.push_back(Point2f(1.5,2.2));
        Mat img = image.getMat();
        printf("detect::rows->%i landmarks-> %i\n",(int)img.rows,(int)landmarks.size());
        return true;
    }

    void FacemarkAAMImpl::saveTrainedModel(String filename){
        printf("save training result %s\n",filename.c_str());
    }

    void FacemarkAAMImpl::loadTrainedModel(String filename){
        printf("load trained model %s\n",filename.c_str());
    }

    Mat FacemarkAAMImpl::procrustes(std::vector<Point2f> P, std::vector<Point2f> Q, Mat & rot, Scalar & trans, float & scale){

        // calculate average
        Scalar mx = mean(P);
        Scalar my = mean(Q);

        // zero centered data
        Mat X0 = Mat(P) - mx;
        Mat Y0 = Mat(Q) - my;

        // calculate magnitude
        Mat Xs, Ys;
        multiply(X0,X0,Xs);
        multiply(Y0,Y0,Ys);

        // cout<<Xs<<endl;

        // calculate the sum
        Mat sumXs, sumYs;
        reduce(Xs,sumXs, 0, CV_REDUCE_SUM);
        reduce(Ys,sumYs, 0, CV_REDUCE_SUM);

        //calculate the normrnd
        double normX = sqrt(sumXs.at<float>(0)+sumXs.at<float>(1));
        double normY = sqrt(sumYs.at<float>(0)+sumYs.at<float>(1));

        //normalization
        X0 = X0/normX;
        Y0 = Y0/normY;

        //reshape, convert to 2D Matrix
        Mat Xn=X0.reshape(1);
        Mat Yn=Y0.reshape(1);

        //calculate the covariance matrix
        Mat M = Xn.t()*Yn;

        // decompose
        Mat U,S,Vt;
        SVD::compute(M, S, U, Vt);

        // extract the transformations
        scale = (S.at<float>(0)+S.at<float>(1))*(float)normX/(float)normY;
        rot = Vt.t()*U.t();

        Mat muX(mx),mX; muX.pop_back();muX.pop_back();
        Mat muY(my),mY; muY.pop_back();muY.pop_back();
        muX.convertTo(mX,CV_32FC1);
        muY.convertTo(mY,CV_32FC1);

        Mat t = mX.t()-scale*mY.t()*rot;
        trans[0] = t.at<float>(0);
        trans[1] = t.at<float>(1);

        // calculate the recovered form
        Mat Qmat = Mat(Q).reshape(1);

        return scale*Qmat*rot+trans;
    }

    void FacemarkAAMImpl::procrustesAnalysis(std::vector<std::vector<Point2f> > shapes, std::vector<std::vector<Point2f> > & normalized, std::vector<Point2f> & new_mean){

        std::vector<Scalar> mean_every_shape;
        mean_every_shape.resize(shapes.size());

        Point2f temp;

        // calculate the mean of every shape
        for(unsigned i=0; i< shapes.size();i++){
            mean_every_shape[i] = mean(shapes[i]);
            // cout<<mean_every_shape[i]<<endl;
        }

        //normalize every shapes
        Mat tShape;
        normalized.clear();
        for(unsigned i=0; i< shapes.size();i++){
            // tShape = Mat(shapes[i]) - mean_every_shape[i];
            normalized.push_back((Mat)(Mat(shapes[i]) - mean_every_shape[i]));
        }

        // calculate the mean shape
        std::vector<Point2f> mean_shape;
        calcMeanShape(normalized, mean_shape);

        // update the mean shape and normalized shapes iteratively
        int maxIter = 100;
        Mat R;
        Scalar t;
        float s;
        Mat aligned;
        for(int i=0;i<maxIter;i++){
            // align
            for(unsigned k=0;k< normalized.size();k++){
                aligned=procrustes(mean_shape, normalized[k], R, t, s);
                aligned.reshape(2).copyTo(normalized[k]);
            }

            //calc new mean
            calcMeanShape(normalized, new_mean);
            // align the new mean
            aligned=procrustes(mean_shape, new_mean, R, t, s);
            // update
            aligned.reshape(2).copyTo(mean_shape);
        }
        // cout<<mean_shape<<endl;

    }

    void FacemarkAAMImpl::calcMeanShape(std::vector<std::vector<Point2f> > shapes,std::vector<Point2f> & mean){
        mean.resize(shapes[0].size());
        Point2f tmp;
        for(unsigned i=0;i<shapes[0].size();i++){
            tmp.x=0;
            tmp.y=0;
            for(unsigned k=0;k< shapes.size();k++){
                tmp.x+= shapes[k][i].x;
                tmp.y+= shapes[k][i].y;
            }
            tmp.x/=shapes.size();
            tmp.y/=shapes.size();
            mean[i] = tmp;
        }
    }

//  } /* namespace face */
} /* namespace cv */
