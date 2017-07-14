/*----------------------------------------------
 * Usage:
 * facemark_demo_aam <image_id>
 *
 * Example:
 * facemark_demo_aam 87
 *
 * Notes:
 * the user should provides the list of training images_train
 * accompanied by their corresponding landmarks location in separated files.
 * example of contents for images_train.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 * example of contents for points_train.txt:
 * ../trainset/image_0001.pts
 * ../trainset/image_0002.pts
 * where the image_xxxx.pts contains the position of each face landmark.
 * example of the contents:
 *  version: 1
 *  n_points:  68
 *  {
 *  115.167660 220.807529
 *  116.164839 245.721357
 *  120.208690 270.389841
 *  ...
 *  }
 * example of the dataset is available at http://www.ifp.illinois.edu/~vuongle2/helen/
 *--------------------------------------------------*/

#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

Mat loadCSV(std::string filename);

int main(int argc, char** argv )
{
    Ptr<Facemark> facemark = FacemarkAAM::create();

    /*--------------- TRAINING -----------------*/
    String imageFiles = "../data/images_train.txt";
    String ptsFiles = "../data/points_train.txt";

    /* trained model will be saved to AAM.yml */
    facemark->training(imageFiles, ptsFiles);

    /*--------------- FITTING -----------------*/
    imageFiles = "../data/images_test.txt";
    ptsFiles = "../data/points_test.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    /*load the list of test
    *files, offest -1.0 is used since in opencv index is started from 0
    */
    facemark->loadTrainingData(imageFiles, ptsFiles, images, facePoints, -1.0);

    /*load the selected image*/
    int tId = 0;
    if(argc>1)tId = atoi(argv[1]);
    Mat image = imread(images[tId]);

    /*load the face detection result from another code
    *alternatively, custom face detector can be utilized
    */
    Mat initial = loadCSV("../data/faces.csv");
    float scale = initial.at<float>(tId,0);
    Point2f T = Point2f(initial.at<float>(tId,1),initial.at<float>(tId,2));
    Mat R=Mat::eye(2, 2, CV_32F);

    /*fitting process*/
    std::vector<Point2f> landmarks;
    facemark->fit(image, landmarks, R,T, scale);
    facemark->drawPoints(image, landmarks);
    imshow("fitting", image);
    waitKey(0);
}

Mat loadCSV(std::string filename){
    ifstream inputfile(filename.c_str());
    std::string current_line;
    // vector allows you to add data without knowing the exact size beforehand
    vector< vector<float> > all_data;
    // Start reading lines as long as there are lines in the file
    while(getline(inputfile, current_line)){
       // Now inside each line we need to seperate the cols
       vector<float> values;
       stringstream temp(current_line);
       string single_value;
       while(getline(temp,single_value,',')){
            // convert the string element to a integer value
            values.push_back((float)atof(single_value.c_str()));
       }
       // add the row to the complete data vector
       all_data.push_back(values);
    }

    // Now add all the data into a Mat element
    Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32F);
    // Loop over vectors and add the data
    for(int rows = 0; rows < (int)all_data.size(); rows++){
       for(int cols= 0; cols< (int)all_data[0].size(); cols++){
          vect.at<float>(rows,cols) = all_data[rows][cols];
       }
    }
    inputfile.close();
    return vect.clone();
}
