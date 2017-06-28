// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/dnn.hpp>
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
/* Find best class for the blob (i. e. class with maximal probability) */

int main()
{
    DeepGaze1 g = DeepGaze1();
    vector<Mat> images;
    vector<Mat> fixs;

//Download mit1003 saliency dataset in the working directory
//ALLSTIMULI folder store images
//ALLFIXATIONMAPS foler store training eye fixation
//************ Code only work in linux platform ****
    string dataset_path;

    cin >> dataset_path;
    Ptr<SALIENCY_mit1003> datasetConnector = SALIENCY_mit1003::create();
    datasetConnector->load( dataset_path );
    vector<vector<Mat> > dataset( datasetConnector->getDataset() );

    //g.training( dataset[0], dataset[1] );

    ofstream file;
    Mat res2;
    g.computeSaliency( dataset[0][0], res2 );
    resize( res2, res2, Size( 1024, 768 ) );
    cout << "AUC = " << g.computeAUC( res2, dataset[1][0] ) << endl;
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
