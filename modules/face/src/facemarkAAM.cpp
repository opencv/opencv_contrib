#include "opencv2/face.hpp"
#include "opencv2/core.hpp"

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

    protected:

        bool detectImpl( InputArray image, std::vector<Point2f> & landmarks );
        bool trainingImpl(String imageList, String groundTruth);
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

    bool FacemarkAAMImpl::trainingImpl(String imageList, String groundTruth){
        printf("inside the training func %s %s\n", imageList.c_str(), groundTruth.c_str());
        return true;
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

//  } /* namespace face */
} /* namespace cv */
