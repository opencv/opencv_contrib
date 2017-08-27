#ifndef __OPENCV_FACEMARK_AAM_HPP__
#define __OPENCV_FACEMARK_AAM_HPP__

#include "opencv2/face/facemark.hpp"
namespace cv {
namespace face {

//! @addtogroup face
//! @{

    class CV_EXPORTS_W FacemarkAAM : public Facemark
    {
    public:
        struct CV_EXPORTS Params
        {
            /**
            * \brief Constructor
            */
            Params();

            // /**
            // * \brief Read parameters from file, currently unused
            // */
            // // void read(const FileNode& /*fn*/);
            //
            // /**
            // * \brief Read parameters from file, currently unused
            // */
            // // void write(FileStorage& /*fs*/) const;

            std::string model_filename;
            int m;
            int n;
            int n_iter;
            bool verbose;
        };

        /**
        * \brief The model of AAM Algorithm
        */
        struct CV_EXPORTS Model{
            int npts; //!<  unused delete
            int max_n; //!<  unused delete
            std::vector<int>scales;
            //!<  defines the scales considered to build the model

            /*warping*/
            std::vector<Vec3i> triangles;
            //!<  each element contains 3 values, represent index of facemarks that construct one triangle (obtained using delaunay triangulation)

            struct Texture{
                int max_m; //!<  unused delete
                Rect resolution;
                //!<  resolution of the current scale
                Mat A;
                //!<  gray values from all face region in the dataset, projected in PCA space
                Mat A0;
                //!<  average of gray values from all face region in the dataset
                Mat AA;
                //!<  gray values from all erorded face region in the dataset, projected in PCA space
                Mat AA0;
                //!<  average of gray values from all erorded face region in the dataset

                std::vector<std::vector<Point> > textureIdx;
                //!<  index for warping of each delaunay triangle region constructed by 3 facemarks
                std::vector<Point2f> base_shape;
                //!<  basic shape, normalized to be fit in an image with current detection resolution
                std::vector<int> ind1;
                //!<  index of pixels for mapping process to obtains the grays values of face region
                std::vector<int> ind2;
                //!<  index of pixels for mapping process to obtains the grays values of eroded face region
            };
            std::vector<Texture> textures;
            //!<  a container to holds the texture data for each scale of fitting

            /*shape*/
            std::vector<Point2f> s0;
            //!<  the basic shape obtained from training dataset
            Mat S,Q;
            //!<  the encoded shapes from training data

        };

        /** @brief A custom fitting function designed for AAM algorithm.
        AAM fitting relies on basic shape as initializer. Therefore,
        transformation paramters are needed to adjust the initial points in the fitting.

        @param image Input image.
        @param landmarks The fitted facial landmarks.
        @param R Rotation matrix.
        @param T Translation vector.
        @param scale scaling factor.

        <B>Example of usage</B>
        @code
        Mat R =  Mat::eye(2, 2, CV_32F);
        Point2f t = Point2f(0,0);
        float scale = 1.0;
        std::vector<Point2f> landmarks;
        facemark->fitSingle(image, landmarks, R,T, scale);
        @endcode

        */
        virtual bool fitSingle( InputArray image, OutputArray landmarks, Mat R, Point2f T, float scale )=0;
        virtual void getParams(Model & params) = 0;
        //!<  initializer
        static Ptr<FacemarkAAM> create(const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
        virtual ~FacemarkAAM() {}

    }; /* AAM */

//! @}

} /* namespace face */
} /* namespace cv */
#endif
