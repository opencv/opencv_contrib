Adding a new algorithm to the Facemark API {#tutorial_facemark_add_algorithm}
==========================================================

Goals
----

In this tutorial you will learn how to:
- integrate a new algorithm of facial landmark detector into the Facemark API
- compile a specific contrib module
- using extra parameters in a function


Explanation
-----------

-  **Add the class header**

    The class header for a new algorithm should be added to a new file in include/opencv2/face.
    Here is the template that you can use to integrate a new algorithm, change the FacemarkNEW to a representative name of the new algorithm and save it using a representative filename accordingly.

        @code{.cpp}
        class CV_EXPORTS_W FacemarkNEW : public Facemark {
        public:
            struct CV_EXPORTS Config {
                Config();

                /*read only parameters - just for example*/
                double detect_thresh;         //!<  detection confidence threshold
                double sigma;                 //!<  another parameter

                void read(const FileNode& /*fn*/);
                void write(FileStorage& /*fs*/) const;
            };

            /*Builder and destructor*/
            static Ptr<FacemarkNEW> create(const FacemarkNEW::Config &conf = FacemarkNEW::Config() );
            virtual ~FacemarkNEW(){};
        };
        @endcode


-  **Add the implementation code**

    Create a new file in the source folder with name representing the new algorithm.
    Here is the template that you can use.

        @code{.cpp}
        #include "opencv2/face.hpp"
        #include "precomp.hpp"

        namespace cv
        {
            FacemarkNEW::Config::Config(){
                detect_thresh = 0.5;
                sigma=0.2;
            }

            void FacemarkNEW::Config::read( const cv::FileNode& fn ){
                *this = FacemarkNEW::Config();

                if (!fn["detect_thresh"].empty())
                    fn["detect_thresh"] >> detect_thresh;

                if (!fn["sigma"].empty())
                    fn["sigma"] >> sigma;

            }

            void FacemarkNEW::Config::write( cv::FileStorage& fs ) const{
                fs << "detect_thresh" << detect_thresh;
                fs << "sigma" << sigma;
            }

            /*implementation of the algorithm is in this class*/
            class FacemarkNEWImpl : public FacemarkNEW {
            public:
                FacemarkNEWImpl( const FacemarkNEW::Config &conf = FacemarkNEW::Config() );

                void read( const FileNode& /*fn*/ );
                void write( FileStorage& /*fs*/ ) const;

                void loadModel(String filename);

                bool setFaceDetector(bool(*f)(InputArray , OutputArray, void * extra_params));
                bool getFaces( InputArray image , OutputArray faces, void * extra_params);

                Config config;

            protected:

                bool addTrainingSample(InputArray image, InputArray landmarks);
                void training();
                bool fit(InputArray image, InputArray faces, InputOutputArray landmarks, void * runtime_params);

                Config config; // configurations

                /*proxy to the user defined face detector function*/
                bool(*faceDetector)(InputArray , OutputArray, void * );
            }; // class

            Ptr<FacemarkNEW> FacemarkNEW::create(const FacemarkNEW::Config &conf){
                return Ptr<FacemarkNEWImpl>(new FacemarkNEWImpl(conf));
            }

            FacemarkNEWImpl::FacemarkNEWImpl( const FacemarkNEW::Config &conf ) :
                config( conf )
            {
                // other initialization
            }

            bool FacemarkNEWImpl::addTrainingSample(InputArray image, InputArray landmarks){
                // pre-process and save the new training sample
                return true;
            }

            void FacemarkNEWImpl::training(){
                printf("training\n");
            }

            bool FacemarkNEWImpl::fit(
                InputArray image,
                InputArray faces,
                InputOutputArray landmarks,
                void * runtime_params)
            {
                if(runtime_params!=0){
                    // do something based on the extra parameters
                }

                printf("fitting\n");
                return 0;
            }

            void FacemarkNEWImpl::read( const cv::FileNode& fn ){
                config.read( fn );
            }

            void FacemarkNEWImpl::write( cv::FileStorage& fs ) const {
                config.write( fs );
            }

            void FacemarkNEWImpl::loadModel(String filename){
                // load the model
            }

            bool FacemarkNEWImpl::setFaceDetector(bool(*f)(InputArray , OutputArray, void * extra_params )){
                faceDetector = f;
                isSetDetector = true;
                return true;
            }

            bool FacemarkNEWImpl::getFaces( InputArray image , OutputArray roi, void * extra_params){
                if(!isSetDetector){
                    return false;
                }

                if(extra_params!=0){
                    //extract the extra parameters
                }

                std::vector<Rect> & faces = *(std::vector<Rect>*)roi.getObj();
                faces.clear();

                faceDetector(image.getMat(), faces, extra_params);

                return true;
            }
        }

        @endcode

-  **Compiling the code**

    Clear the build folder and then rebuild the entire library.
    Note that you can deactivate the compilation of other contrib modules by adding "-D BUILD_opencv_<MODULE_NAME>=OFF" flag to the cmake.
    After that you can execute make command in "<build_folder>/modules/face" to speed up the compiling process.

Best Practice
-----------
- **Handling the extra parameters**
    To handle the extra parameters, a new struct should be created to holds all the required parameters.
    Here is an example of of a parameters container
    @code
    struct CV_EXPORTS Params
    {
        Params( Mat rot = Mat::eye(2,2,CV_32F),
                Point2f trans = Point2f(0.0,0.0),
                float scaling = 1.0
        );

        Mat R;
        Point2f t;
        float scale;
    };
    @endcode

    Here is a snippet to extract the extra parameters:
    @code
    if(runtime_params!=0){
        Telo*  conf = (Telo*)params;
        Params* params
        std::vector<Params> params = *(std::vector<Params>*)runtime_params;
        for(size_t i=0; i<params.size();i++){
            fit(img, landmarks[i], params[i].R,params[i].t, params[i].scale);
        }
    }else{
        // do something
    }
    @endcode

    And here is an example to pass the extra parameter into fit function
    @code
    FacemarkAAM::Params * params = new FacemarkAAM::Params(R,T,scale);
    facemark->fit(image, faces, landmarks, params)
    @endcode

    In order to understand this scheme, here is a simple example that you can try to compile and see how it works.
    @code
    struct Params{
        int x,y;
        Params(int _x, int _y);
    };
    Params::Params(int _x,int _y){
        x = _x;
        y = _y;
    }

    void test(int a, void * params=0){
        printf("a:%i\n", a);
        if(params!=0){
            Params*  params = (Params*)params;
            printf("extra parameters:%i %i\n", params->x, params->y);
        }
    }

    int main(){
        Params* params = new Params(7,22);
        test(99, params);
        return 0;
    }
    @endcode

- **Minimize the dependency**
    It is highly recomended to keep the code as small as possible when compiled. For this purpose, the developers are ecouraged to avoid the needs of heavy dependency such as `imgcodecs` and `highgui`.

- **Documentation and examples**
    Please update the documentation whenever needed and put example code for the new algorithm.

- **Test codes**
    An algorithm should be accompanied with its corresponding test code to ensure that the algorithm is compatible with various types of environment (Linux, Windows64, Windows32, Android, etc). There are several basic test that should be performed as demonstrated in the test/test_facemark_lbf.cpp file including cration of its instance, add training data, perform the training process, load a trained model, and perform the fitting to obtain facial landmarks.

- **Data organization**
    It is advised to divide the data for a new algorithm into 3 parts :
    @code
    class CV_EXPORTS_W FacemarkNEW : public Facemark {
    public:
        struct CV_EXPORTS Params
        {
            // variables utilized as extra parameters
        }
        struct CV_EXPORTS Config
        {
            // variables used to configure the algorithm
        }
        struct CV_EXPORTS Model
        {
            // variables to store the information of model
        }

        static Ptr<FacemarkNEW> create(const FacemarkNEW::Config &conf = FacemarkNEW::Config() );
        virtual ~FacemarkNEW(){};
    }
    @endcode
