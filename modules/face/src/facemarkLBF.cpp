#include "opencv2/face.hpp"
#include "opencv2/imgcodecs.hpp"
#include "precomp.hpp"

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
    }

    void FacemarkLBFImpl::trainingImpl(String imageList, String groundTruth){
        printf("training\n");
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
} /* namespace cv */
