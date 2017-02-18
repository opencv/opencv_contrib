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
        "{c_names c || path to file with classnames for channels (optional, categories.txt) }"
        "{result r  || path to save output blob (optional, binary format, NCHW order) }"
        "{show s    || whether to show all output channels or not}"
        "{o_blob    || output blob's name. If empty, last blob's name in net is used}"
        ;

std::vector<String> readClassNames(const char *filename);
static void colorizeSegmentation(Blob &score, Mat &segm,
                                 Mat &legend, vector<String> &classNames);

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
    Mat img = imread(imageFile), input;
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    cv::Size inputImgSize = cv::Size(512, 512);

    if (inputImgSize != img.size())
        resize(img, img, inputImgSize);       //Resize image to input size

    if(img.channels() == 3)
        cv::cvtColor(img, input, cv::COLOR_BGR2RGB);

    input.convertTo(input, CV_32F, 1/255.0);

    dnn::Blob inputBlob = dnn::Blob::fromImages(input);   //Convert Mat to dnn::Blob image batch
    //! [Prepare blob]

    //! [Set input blob]
    net.setBlob("", inputBlob);        //set the network input
    //! [Set input blob]

    cv::TickMeter tm;
    tm.start();

    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]

    tm.stop();

    //! [Gather output]

    String oBlob = net.getLayerNames().back();
    if (!parser.get<String>("o_blob").empty())
    {
        oBlob = parser.get<String>("o_blob");
    }

    dnn::Blob prob = net.getBlob(oBlob);   //gather output of "prob" layer

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

    if (parser.has("show"))
    {
        std::vector<String> classNames;
        if(!classNamesFile.empty()) {
            classNames = readClassNames(classNamesFile.c_str());
            if (classNames.size() > prob.channels())
                classNames = std::vector<String>(classNames.begin() + classNames.size() - prob.channels(),
                                                 classNames.end());
        }
        Mat segm, legend;
        colorizeSegmentation(prob, segm, legend, classNames);

        Mat show;
        addWeighted(img, 0.2, segm, 0.8, 0.0, show);

        imshow("Result", show);
        if(classNames.size())
            imshow("Legend", legend);
        waitKey();
    }

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

static void colorizeSegmentation(Blob &score, Mat &segm, Mat &legend, vector<String> &classNames)
{
    const int rows = score.rows();
    const int cols = score.cols();
    const int chns = score.channels();

    vector<Vec3i> colors;
    RNG rng(12345678);

    cv::Mat maxCl(rows, cols, CV_8UC1);
    cv::Mat maxVal(rows, cols, CV_32FC1);
    for (int ch = 0; ch < chns; ch++)
    {
        colors.push_back(Vec3i(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)));
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptrf(0, ch, row);
            uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        cv::Vec3b *ptrSegm = segm.ptr<cv::Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }

    if (classNames.size() == colors.size())
    {
        int blockHeight = 30;
        legend.create(blockHeight*classNames.size(), 200, CV_8UC3);
        for(int i = 0; i < classNames.size(); i++)
        {
            cv::Mat block = legend.rowRange(i*blockHeight, (i+1)*blockHeight);
            block = colors[i];
            putText(block, classNames[i], Point(0, blockHeight/2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
        }
    }
}
