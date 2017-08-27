Adding a new algorithm to the Facemark API {#tutorial_facemark_add_algorithm}
==========================================================

Goals
----

In this tutorial you will learn how to:
- integrate a new algorithm of facial landmark detector into the Facemark API
- compile a specific contrib module


Explanation
-----------

-  **Add the class header**

    The class header should be added to facemark.hpp file.
    Here is the template that you can use to integrate a new algorithm.

        @code{.cpp}
        class CV_EXPORTS_W FacemarkLBF : public Facemark {
        public:
            struct CV_EXPORTS Params {
                Params();

                /*read only parameters - just for example*/
                double detect_thresh;         //!<  detection confidence threshold
                double sigma;                 //!<  another parameter

                void read(const FileNode& /*fn*/);
                void write(FileStorage& /*fs*/) const;
            };

            BOILERPLATE_CODE("LBF",FacemarkLBF);
        };
        @endcode

-  **Add the code for pointing out the implementation**

    You should modify the face/src/facemark.cpp file using the following template.

        @code{.cpp}
        Ptr<Facemark> Facemark::create( const String& facemarkType ){
            BOILERPLATE_CODE("AAM",FacemarkAAM);
            BOILERPLATE_CODE("LBF",FacemarkLBF); // declare the new algorithm here!
            return Ptr<Facemark>();
        }
        @endcode

-  **Add the implementation code**

    Create a new file in the source folder with name representing the new algorithm.
    Here is the template that you can use.

        @code{.cpp}
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
        }

        @endcode

-  **Compiling the code**

    Clear the build folder and then rebuild the entire library.
    Note that you can deactivate the compilation of other contrib modules by adding "-D BUILD_opencv_<MODULE_NAME>=OFF" flag to the cmake.
    After that you can execute make command in "<build_folder>/modules/face" to speed up the compiling process.
