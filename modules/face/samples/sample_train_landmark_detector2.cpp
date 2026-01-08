/*----------------------------------------------
 * the user should provide the list of training images_train,
 * accompanied by their corresponding landmarks location in separated files.
 * example of contents for images.txt:
 * ../trainset/image_0001.png
 * ../trainset/image_0002.png
 * example of contents for annotation.txt:
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
 * example of the dataset is available at https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
 *--------------------------------------------------*/
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::face;

static bool myDetector(InputArray image, OutputArray faces, CascadeClassifier *face_cascade)
{
    Mat gray;

    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        gray = image.getMat().clone();

    equalizeHist(gray, gray);

    std::vector<Rect> faces_;
    face_cascade->detectMultiScale(gray, faces_, 1.4, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
    Mat(faces_).copyTo(faces);
    return true;
}

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
   CommandLineParser parser(argc, argv,
        "{ help h usage ? |      | give the following arguments in following format }"
        "{ images i       |      | (required) path to images txt file      [example - /data/images.txt] }"
        "{ annotations a  |.     | (required) path to annotations txt file [example - /data/annotations.txt] }"
        "{ config c       |      | (required) path to configuration xml file containing parameters for training.[example - /data/config.xml] }"
        "{ model m        |      | (required) path to file containing trained model for face landmark detection[example - /data/model.dat] }"
        "{ width w        |  460 | The width which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ height h       |  460 | The height which you want all images to get to scale the annotations. large images are slow to process [default = 460] }"
        "{ face_cascade f |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string annotations(parser.get<string>("annotations"));
    string imagesList(parser.get<string>("images"));
    //default initialisation
    Size scale(460,460);
    scale = Size(parser.get<int>("width"),parser.get<int>("height"));
    if (annotations.empty()){
        parser.printMessage();
        cerr << "Name for annotations file not  found. Aborting...." << endl;
        return -1;
    }
    if (imagesList.empty()){
        parser.printMessage();
        cerr << "Name for file containing image list not found. Aborting....." << endl;
        return -1;
    }
    string configfile_name(parser.get<string>("config"));
    if (configfile_name.empty()){
        parser.printMessage();
        cerr << "No configuration file name found which contains the parameters for training" << endl;
        return -1;
    }
    string modelfile_name(parser.get<string>("model"));
    if (modelfile_name.empty()){
        parser.printMessage();
        cerr << "No name  for the model_file found in which the trained model has to be saved" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "The name of the cascade classifier to be loaded to detect faces is not found" << endl;
        return -1;
    }
    //create a pointer to call the base class
    //pass the face cascade xml file which you want to pass as a detector
    CascadeClassifier face_cascade;
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    params.configfile = configfile_name;
    Ptr<FacemarkKazemi> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector((FN_FaceDetector)myDetector, &face_cascade);

    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imagesList, annotations, images, facePoints, 0.0);
    //gets landmarks and corresponding image names in both the vectors
    vector<Mat> Trainimages;
    std::vector<std::vector<Point2f> > Trainlandmarks;
    //vector to store images
    Mat src;
    for(unsigned long i=0;i<images.size();i++){
        src = imread(images[i]);
        if(src.empty()){
            cout<<images[i]<<endl;
            cerr<<string("Image not found\n.Aborting...")<<endl;
            continue;
        }
        Trainimages.push_back(src);
        Trainlandmarks.push_back(facePoints[i]);
    }
    cout<<"Got data"<<endl;
    facemark->training(Trainimages,Trainlandmarks,configfile_name,scale,modelfile_name);
    cout<<"Training complete"<<endl;
    return 0;
}
