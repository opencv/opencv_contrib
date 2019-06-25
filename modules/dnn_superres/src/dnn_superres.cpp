// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/dnn_superres.hpp"

namespace cv
{
    namespace dnn
    {
        namespace dnn_superres
        {

            DnnSuperResImpl::DnnSuperResImpl()
            {
                //emtpy constructor
            }

            DnnSuperResImpl::DnnSuperResImpl(std::string algo, int scale)
            {
                setModel(algo, scale);
            }

            void DnnSuperResImpl::setModel(std::string algo, int scale)
            {
                std::string filename_pb;
                std::string filename_pbtxt;
                bool modelFlag = true;
                int s = scale;

                if( algo == "espcn" )
                {
                    if( s == 2 )
                    {
                        //filename_b = ...
                        //filename_pbtxt = ...
                    }
                    else if( s == 3 )
                    {
                        //
                    }
                    else{
                        //
                    }
                }
                else if( algo == "lapsrn" )
                {
                    //filename_b = ...
                    //filename_pbtxt = ...
                }
                else if( algo == "fsrcnn" )
                {
                    //filename_b = ...
                    //filename_pbtxt = ...
                }
                else if( algo == "edsr" )
                {
                    //filename_b = ...
                    //filename_pbtxt = ...
                }
                else
                {
                    std::cout << "Algorithm is not recognized. No model set. \n";
                    modelFlag = false;
                }

                if( filename_pb.size() && filename_pbtxt.size() )
                {
                    net = readNetFromTensorflow(filename_pb, filename_pbtxt);
                    std::cout << "Successfully loaded model. \n";
                }
                else{
                    if( modelFlag == true )
                    {
                        std::cout << "Requested algorithm not implemented yet. \n";
                    }
                }
            }

            void DnnSuperResImpl::upsample(Mat img, Mat img_new)
            {
                if( !net.empty() )
                {
                    //get blob
                    Mat blob = blobFromImage (img, 1.0);
                    std::cout << "Made a blob. \n";

                    //get prediction
                    net.setInput(blob);
                    img_new = net.forward();
                    std::cout << "Made a Prediction. \n";
                }
                else
                {
                    std::cout << "Model not specified. Please set model via setModel(). \n";
                }
            }
        }
    }
}
