// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
This file contains results of GSoC Project: Facemark API for OpenCV
Final report: https://gist.github.com/kurnianggoro/74de9121e122ad0bd825176751d47ecc
Student: Laksono Kurnianggoro
Mentor: Delia Passalacqua
*/

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

        /**
        * \brief Read parameters from file, currently unused
        */
        void read(const FileNode& /*fn*/);

        /**
        * \brief Read parameters from file, currently unused
        */
        void write(FileStorage& /*fs*/) const;

        std::string model_filename;
        int m;
        int n;
        int n_iter;
        bool verbose;
        bool save_model;
        int max_m, max_n, texture_max_m;
        std::vector<float>scales;
    };

    /**
    * \brief Optional parameter for fitting process.
    */
    struct CV_EXPORTS Config
    {
        Config( Mat rot = Mat::eye(2,2,CV_32F),
                Point2f trans = Point2f(0.0f,0.0f),
                float scaling = 1.0f,
                int scale_id=0
        );

        Mat R;
        Point2f t;
        float scale;
        int model_scale_idx;

    };

    /**
    * \brief Data container for the facemark::getData function
    */
    struct CV_EXPORTS Data
    {
        std::vector<Point2f> s0;
    };

    /**
    * \brief The model of AAM Algorithm
    */
    struct CV_EXPORTS Model
    {
        int npts; //!<  unused delete
        int max_n; //!<  unused delete
        std::vector<float>scales;
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

    //!<  initializer
    static Ptr<FacemarkAAM> create(const FacemarkAAM::Params &parameters = FacemarkAAM::Params() );
    virtual ~FacemarkAAM() {}

}; /* AAM */

//! @}

} /* namespace face */
} /* namespace cv */
#endif
