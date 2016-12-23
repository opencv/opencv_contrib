/*
Sample of using OpenCV dnn module with Torch ENet model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
using namespace std;

const String keys =
        "{help h    || Sample app for loading ENet Torch model. "
                       "The model and class names list can be downloaded here: "
                       "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AAD5YkUa8XgMpHs2gCRgmCVCa }"
        "{model m   || path to Torch .net model file (model_best.net) }"
        "{image i   || path to image file }"
        "{i_blob    | .0 | input blob name) }"
        "{o_blob    || output blob name) }"
        "{c_names c || path to file with classnames for channels (categories.txt) }"
        "{result r  || path to save output blob (optional, binary format, NCHW order) }"
        ;

std::vector<String> readClassNames(const char *filename);

int main(int argc, char **argv)
{
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelFile = parser.get<String>("model");
    String imageFile = parser.get<String>("image");
    String inBlobName = parser.get<String>("i_blob");
    String outBlobName = parser.get<String>("o_blob");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    String classNamesFile = parser.get<String>("c_names");
    String resultFile = parser.get<String>("result");

    //! [Create the importer of TensorFlow model]
    Ptr<dnn::Importer> importer;
    try                                     //Try to import TensorFlow AlexNet model
    {
        importer = dnn::createTorchImporter(modelFile);
    }
    catch (const cv::Exception &err)        //Importer can throw errors, we will catch them
    {
        std::cerr << err.msg << std::endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        std::cerr << "Can't load network by using the mode file: " << std::endl;
        std::cerr << modelFile << std::endl;
        exit(-1);
    }

    //! [Initialize network]
    dnn::Net net;
    importer->populateNet(net);
    importer.release();                     //We don't need importer anymore
    //! [Initialize network]

    //! [Prepare blob]
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    cv::Size inputImgSize = cv::Size(512, 512);

    if (inputImgSize != img.size())
        resize(img, img, inputImgSize);       //Resize image to input size

    if(img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    img.convertTo(img, CV_32F, 1/255.0);

    dnn::Blob inputBlob = dnn::Blob::fromImages(img);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob(inBlobName, inputBlob);        //set the network input
    //! [Set input blob]

    cv::TickMeter tm;
    tm.start();

    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]

    tm.stop();

    //! [Gather output]
    dnn::Blob prob = net.getBlob(outBlobName);   //gather output of "prob" layer

    Mat& result = prob.matRef();

    BlobShape shape = prob.shape();

    if (!resultFile.empty()) {
        CV_Assert(result.isContinuous());

        ofstream fout(resultFile.c_str(), ios::out | ios::binary);
        fout.write((char*)result.data, result.total() * sizeof(float));
        fout.close();
    }

    std::cout << "Output blob shape " << shape  << std::endl;
    std::cout << "Inference time, ms: " << tm.getTimeMilli()  << std::endl;

    std::vector<String> classNames;
    if(!classNamesFile.empty()) {
        classNames = readClassNames(classNamesFile.c_str());
        if (classNames.size() > prob.channels())
            classNames = std::vector<String>(classNames.begin() + classNames.size() - prob.channels(),
                                             classNames.end());
    }

    for(int i_c = 0; i_c < prob.channels(); i_c++) {
        ostringstream convert;
        convert << "Channel #" << i_c;

        if(classNames.size() == prob.channels())
            convert << ": " << classNames[i_c];

        imshow(convert.str().c_str(), prob.getPlane(0, i_c));
    }
    waitKey();

    return 0;
} //main


std::vector<String> readClassNames(const char *filename)
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }

    fp.close();
    return classNames;
}
