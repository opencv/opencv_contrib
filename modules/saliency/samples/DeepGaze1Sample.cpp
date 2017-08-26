// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/saliency.hpp>
#include <opencv2/datasets/saliency_mit1003.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::saliency;
using namespace cv::datasets;


int main(int argc, char* argv[])
{
    const char *keys =
        "{ help h usage ? |     | show this message }"
        "{ train          |0    | set training on }"
        "{ default        |1    | use default deep net(AlexNet) and default weights }"
        "{ AUC            |0    | calculate AUC with input fixation map }"
        "{ img_path       |     | path to folder with img }"
        "{ fix_path       |     | path to folder with fixation img for compute AUC }"
        "{ model_path     |bvlc_alexnet.caffemodel   | path to your caffe model }"
        "{ proto_path     |deploy.prototxt   | path to your deep net caffe prototxt }"
        "{ dataset_path d |./   | path to Dataset for training }";

    CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string img_path = parser.get<string>("img_path");
    string model_path = parser.get<string>("model_path");
    string proto_path = parser.get<string>("proto_path");
    string dataset_path = parser.get<string>("dataset_path");
    string fix_path = parser.get<string>("fix_path");

    DeepGaze1 g;
    if ( parser.get<bool>( "default" ) )
    {
        g = DeepGaze1( proto_path, model_path );
    }
    else
    {
        g = DeepGaze1( proto_path, model_path, vector<string>(1, "conv5"), 257 );
    }

//Download mit1003 saliency dataset in the working directory
//ALLSTIMULI folder store images
//ALLFIXATIONMAPS foler store training eye fixation

    if ( parser.get<bool>( "train" ) )
    {
        Ptr<SALIENCY_mit1003> datasetConnector = SALIENCY_mit1003::create();
        datasetConnector->load( dataset_path );
        vector<vector<Mat> > dataset( datasetConnector->getDataset() );

        g.training( dataset[0], dataset[1], 1, 200, 0.9, 0.000001, 0.01);
    }

    ofstream file;
    Mat res2;
    g.computeSaliency( imread( img_path ), res2 );
    resize( res2, res2, Size( 1024, 768 ) );
    if ( parser.get<bool>( "AUC") )
        cout << "AUC = " << g.computeAUC( res2, imread( fix_path, 0 ) ) << endl;;
    g.saliencyMapVisualize( res2 );
    file.open( "saliency.csv" );
    for ( int i = 0; i < res2.rows; i++)
    {
        for ( int j=0; j < res2.cols; j++)
        {
            file << res2.at<double>( i, j ) << " ";
        }
        file << endl;
    }
    file.close();
    return 0;
} //main
